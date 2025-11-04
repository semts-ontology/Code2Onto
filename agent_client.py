import os
import sys
import json
import inspect
import logging
from typing import Any, Dict, List, Optional, Callable
from mcp.client.session import ClientSession
from local_agent_server import LocalAgentServer

try:
    from mcp.client.stdio import stdio_client
except ImportError as e:
    raise ImportError("MCP client stdio not available. Please upgrade 'mcp'.") from e

try:
    from mcp.client.stdio import StdioServerParameters
    HAS_PARAMS = True
except Exception:
    StdioServerParameters = None
    HAS_PARAMS = False

from langchain_core.tools import tool


class _CompatServerParams:
    """Compatibility wrapper for server parameters."""
    def __init__(self, command_list: List[str]):
        if not command_list:
            raise ValueError("command_list must be non-empty")
        self.command: str = command_list[0]
        self.args: List[str] = command_list[1:]
        self.env: Optional[Dict[str, str]] = None
        self.cwd: Optional[str] = None


class ValidationLimiter:
    """Tracks validation tool usage to enforce limits."""
    def __init__(self, limit: int):
        self.current = 0
        self.limit = limit
    
    def check_and_increment(self) -> tuple[bool, int, int]:
        """Returns (allowed, current, limit)"""
        self.current += 1
        return self.current <= self.limit, self.current, self.limit
    
    def reset(self):
        """Reset counter for new analysis."""
        self.current = 0


class UnifiedAgentClient:
    """
    Unified agent serving as proxy for both MCP (remote) and Local servers.
    Provides a single interface for all tools regardless of their location.
    """

    def __init__(self, server_command: Optional[List[str]] = None):
        if server_command is None:
            server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_agent_server.py")
            self.server_command = [sys.executable, "-u", server_path]
        else:
            self.server_command = server_command

        self._conn_cm = None
        self._conn = None
        self._session_cm = None
        self._session: Optional[ClientSession] = None
        
        self.local_server = LocalAgentServer()
        
        self._local_tools: Dict[str, Callable] = {}
        self._register_local_tools()

    def _register_local_tools(self):
        """Register local server tools."""
        self._local_tools = {
            "resolve_runtime_context_values": self.local_server.resolve_runtime_context_values,
            "create_ttl_file": self.local_server.create_ttl_file,
            "list_available_variables": self.local_server.list_available_variables
        }

    def set_direct_variables(self, variables: Dict[str, Any]):
        """Set variables for local analysis."""
        self.local_server.set_direct_variables(variables)

    def get_direct_variables(self) -> Optional[Dict[str, Any]]:
        """Get the stored variables."""
        return self.local_server.get_direct_variables()

    def _build_stdio_client_cm(self):
        """Build stdio client context manager with compatibility handling."""
        try:
            sig = inspect.signature(stdio_client)
            names = set(sig.parameters.keys())

            if "server" in names:
                params = None
                if HAS_PARAMS:
                    params = StdioServerParameters(
                        command=self.server_command[0],
                        args=self.server_command[1:],
                        env=None,
                        cwd=None,
                    )
                else:
                    params = _CompatServerParams(self.server_command)
                return stdio_client(server=params)

            if "command" in names:
                return stdio_client(command=self.server_command)
            if "argv" in names:
                return stdio_client(argv=self.server_command)
            if "args" in names and "executable" in names:
                return stdio_client(executable=self.server_command[0], args=self.server_command[1:])
            if "args" in names:
                return stdio_client(args=self.server_command)

            return stdio_client(self.server_command)
        except Exception as e:
            raise RuntimeError(f"Failed to create stdio client: {e}")

    async def __aenter__(self):
        """Enter async context."""
        try:
            self._conn_cm = self._build_stdio_client_cm()
            self._conn = await self._conn_cm.__aenter__()
            read_stream, write_stream = self._conn
            self._session_cm = ClientSession(read_stream, write_stream)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()
            return self
        except Exception as e:
            if self._session_cm:
                try:
                    await self._session_cm.__aexit__(type(e), e, e.__traceback__)
                except:
                    pass
            if self._conn_cm:
                try:
                    await self._conn_cm.__aexit__(type(e), e, e.__traceback__)
                except:
                    pass
            raise

    async def __aexit__(self, exc_type, exc, tb):
        """Exit async context."""
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(exc_type, exc, tb)
        except Exception as e:
            print(f"Warning: Error closing session: {e}")

        try:
            if self._conn_cm is not None:
                await self._conn_cm.__aexit__(exc_type, exc, tb)
        except Exception as e:
            print(f"Warning: Error closing connection: {e}")

    @staticmethod
    def _extract_text_content(result) -> str:
        """Extract text content from MCP result."""
        try:
            if isinstance(result, dict):
                content_list = result.get("content", [])
            else:
                content_list = getattr(result, "content", [])

            if content_list and len(content_list) > 0:
                first_item = content_list[0]
                if isinstance(first_item, dict):
                    return first_item.get("text", "")
                else:
                    return getattr(first_item, "text", "")
            return ""
        except Exception:
            return str(result) if result else ""

    @staticmethod
    def _extract_json_content(result) -> Dict[str, Any]:
        """Extract and parse JSON content from MCP result."""
        text = UnifiedAgentClient._extract_text_content(result)
        try:
            parsed = json.loads(text) if text else {}
            if isinstance(parsed, dict) and 'content' in parsed and isinstance(parsed.get('content'), list):
                return UnifiedAgentClient._extract_json_content(parsed)
            return parsed
        except json.JSONDecodeError:
            return {"raw_text": text}

    def is_local_tool(self, tool_name: str) -> bool:
        """Check if a tool is handled locally."""
        return tool_name in self._local_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Route tool calls to appropriate server (local or MCP).
        """
        if self.is_local_tool(tool_name):
            try:
                result = self._local_tools[tool_name](**arguments)
                return {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                    "isError": False
                }
            except Exception as e:
                return {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)}, indent=2)}],
                    "isError": True
                }
        
        if self._session is None:
            raise RuntimeError("Agent not initialized. Use 'async with' context.")
        return await self._session.call_tool(tool_name, arguments)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from both local and MCP servers."""
        tools = []
        
        local_tool_schemas = {
            "resolve_runtime_context_values": {
                "name": "resolve_runtime_context_values",
                "description": "Resolve runtime context locally using variables held by the orchestrator",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "variable_mapping": {
                            "type": "string",
                            "description": "JSON mapping specifying variables to analyze"
                        },
                        "resolver_type": {
                            "type": "string",
                            "description": "Type of resolver to use (default, custom, etc.)",
                            "default": "default"
                        }
                    },
                    "required": ["variable_mapping"]
                }
            },
            "create_ttl_file": {
                "name": "create_ttl_file",
                "description": "Create a TTL file at specified path with given TTL code",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ttl_code": {"type": "string", "description": "TTL code content"},
                        "path": {"type": "string", "description": "File path where to save the TTL file"}
                    },
                    "required": ["ttl_code", "path"]
                }
            },
            "list_available_variables": {
                "name": "list_available_variables",
                "description": "List all available variables with basic type information",
                "inputSchema": {"type": "object", "properties": {}}
            }
        }
        
        for tool_name in self._local_tools:
            tools.append(local_tool_schemas[tool_name])
        
        if self._session is not None:
            mcp_tools_result = await self._session.list_tools()
            for tool in mcp_tools_result.tools:
                tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
                tools.append(tool_dict)
        
        return tools

    async def get_general_ontology_infos(self) -> Dict[str, Any]:
        result = await self.call_tool("get_general_ontology_infos", {})
        return self._extract_json_content(result)

    async def get_ontology(self) -> Dict[str, Any]:
        result = await self.call_tool("get_ontology", {})
        return self._extract_json_content(result)

    async def get_vocabularies(self) -> Dict[str, Any]:
        result = await self.call_tool("get_vocabularies", {})
        return self._extract_json_content(result)

    async def get_ontology_examples(self) -> str:
        result = await self.call_tool("get_ontology_examples", {})
        return self._extract_text_content(result)

    async def validate_ttl_code(self, ttl_code: str) -> Dict[str, Any]:
        result = await self.call_tool("validate_ttl_code", {"ttl_code": ttl_code})
        return self._extract_json_content(result)

    async def get_complete_ontology_info(self) -> Dict[str, Any]:
        general_info = await self.get_general_ontology_infos()
        ontology = await self.get_ontology()
        return {
            "general_info": general_info,
            "ontology": ontology
        }

    async def get_all_knowledge_resources(self) -> Dict[str, Any]:
        general_info = await self.get_general_ontology_infos()
        ontology = await self.get_ontology()
        vocabularies = await self.get_vocabularies()
        examples = await self.get_ontology_examples()
        return {
            "general_info": general_info,
            "ontology": ontology,
            "vocabularies": vocabularies,
            "examples": examples
        }

    async def validate_and_create_ttl(self, ttl_code: str, output_path: str) -> Dict[str, Any]:
        validation_result = await self.validate_ttl_code(ttl_code)
        result = {
            "validation": validation_result,
            "file_created": False,
            "file_path": None
        }
        if validation_result.get("overall_valid", False):
            create_result_raw = await self.call_tool("create_ttl_file", {
                "ttl_code": ttl_code,
                "path": output_path
            })
            create_result = self._extract_json_content(create_result_raw)
            result["file_created"] = create_result.get("success", False)
            result["file_path"] = create_result.get("file_path")
            result["creation_details"] = create_result
        return result

    def create_langchain_tools(self, logger: logging.Logger, validation_limit: int = 5) -> List:
        """
        Create LangChain Tools that proxy to local or MCP server via this agent.
        Orchestrator should call this method to get tools for the LLM agent.
        """
        limiter = ValidationLimiter(validation_limit)

        @tool
        async def get_general_ontology_infos() -> str:
            """Get general ontology information including prefixes, abstract, and description."""
            result = await self.get_general_ontology_infos()
            return json.dumps(result, indent=2)

        @tool
        async def get_ontology() -> str:
            """Get complete ontology structure with classes and properties."""
            result = await self.get_ontology()
            return json.dumps(result, indent=2)

        @tool
        async def get_vocabularies() -> str:
            """Get knowledge vocabularies with concepts and hierarchies."""
            result = await self.get_vocabularies()
            return json.dumps(result, indent=2)

        @tool
        async def get_ontology_examples() -> str:
            """Get TTL examples showing how to use ontology classes."""
            result = await self.get_ontology_examples()
            return result

        @tool
        async def resolve_runtime_context_values(variable_mapping: str, resolver_type: str = "default") -> str:
            """
            Resolve runtime context locally using variables held by the agent's local server.

            Args:
                variable_mapping: JSON mapping specifying variables to analyze
                resolver_type: Type of resolver to use (default, custom, etc.)
            """
            raw = await self.call_tool("resolve_runtime_context_values", {
                "variable_mapping": variable_mapping,
                "resolver_type": resolver_type
            })
            return self._extract_text_content(raw)

        @tool
        async def validate_ttl_code(ttl_code: str) -> str:
            """
            Validate TTL code using OWL axiom validation and SHACL.
            Limited to a configured number of calls per analysis session.
            """
            allowed, current, limit = limiter.check_and_increment()
            logger.info(f"\n🔍 VALIDATION CALL {current}/{limit}")

            raw_result = await self.call_tool("validate_ttl_code", {"ttl_code": ttl_code})
            validation_data = self._extract_json_content(raw_result)
            result_str = json.dumps(validation_data, indent=2)

            logger.info("\n" + "=" * 80)
            logger.info(f"VALIDATION EXECUTED ({current}/{limit})")
            logger.info("=" * 80)

            try:
                syntax_ok = bool(validation_data.get('syntax_valid', False))
                owl_val = validation_data.get('owl_validation', {}) or {}
                shacl_val = validation_data.get('shacl_validation', {}) or {}

                if not syntax_ok:
                    logger.info("Syntax invalid; OWL and SHACL validators skipped.")
                else:
                    if isinstance(owl_val, dict):
                        if 'error' in owl_val:
                            logger.info(f"OWL Validation Error: {owl_val['error']}")
                        else:
                            violations = owl_val.get('violations', []) or []
                            logger.info(f"OWL Violations: {len(violations)}")
                            if violations:
                                logger.info("First few OWL violations:")
                                for v in violations[:3]:
                                    logger.info(f"  - {v.get('type', 'unknown')}: {v.get('issue', 'no details')}")
                    if isinstance(shacl_val, dict):
                        if 'error' in shacl_val:
                            logger.info(f"SHACL Validation Error: {shacl_val['error']}")
                        else:
                            conforms = shacl_val.get('conforms', True)
                            logger.info(f"SHACL Conforms: {conforms}")
                            if not conforms:
                                summary = shacl_val.get('summary', '')
                                if isinstance(summary, dict):
                                    shacl_violation_count = summary.get('total_violations', 0)
                                elif isinstance(summary, str):
                                    import re
                                    match = re.search(r'Results \((\d+)\)', summary)
                                    shacl_violation_count = int(match.group(1)) if match else 0
                                else:
                                    shacl_violation_count = 0
                                logger.info(f"SHACL Violations: {shacl_violation_count}")
                                if isinstance(summary, dict) and 'violations' in summary:
                                    violations = summary['violations'][:3]
                                    logger.info("First few SHACL violations:")
                                    for v in violations:
                                        logger.info(f"  - {v.get('constraint_type', 'unknown')}: {v.get('message', 'no details')}")
            except Exception as e:
                logger.error(f"Error parsing validation result: {e}")

            logger.info("=" * 80 + "\n")

            if not allowed:
                error_msg = (
                    f"VALIDATION LIMIT EXCEEDED ({current}/{limit})\n"
                    f"You have already used all {limit} allowed validation attempts.\n"
                    f"Please proceed with creating the TTL file using the last validated version."
                )
                logger.warning(error_msg)
                return json.dumps({
                    "error": "validation_limit_exceeded",
                    "message": error_msg,
                    "attempts_used": current,
                    "limit": limit
                }, indent=2)
            
            return result_str

        @tool
        async def create_ttl_file(ttl_code: str, path: str) -> str:
            """
            Create a TTL file at the specified path with the given TTL content.
            Validates the Turtle syntax before writing.
            """
            raw = await self.call_tool("create_ttl_file", {"ttl_code": ttl_code, "path": path})
            result_str = self._extract_text_content(raw)

            logger.info("\n" + "=" * 80)
            logger.info("FILE CREATION EXECUTED")
            logger.info(f"Target path: {path}")
            try:
                result_data = json.loads(result_str)
                logger.info(f"Success: {result_data.get('success', False)}")
                if result_data.get('success'):
                    logger.info(f"File path: {result_data.get('file_path')}")
                    logger.info(f"File size: {result_data.get('file_size')} bytes")
            except Exception:
                pass
            logger.info("=" * 80 + "\n")

            return result_str

        return [
            get_general_ontology_infos,
            get_ontology,
            get_vocabularies,
            get_ontology_examples,
            resolve_runtime_context_values,
            validate_ttl_code,
            create_ttl_file
        ]