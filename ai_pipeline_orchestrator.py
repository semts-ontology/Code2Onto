import inspect
import json
import datetime
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.callbacks.base import BaseCallbackHandler

from agent_client import UnifiedAgentClient
from tool_modules.runtime_resolver import ResolverFactory

from config import SYSTEM_PROMPT, MAPPING_EXAMPLE, USER_PROMPT, DEFAULT_VALIDATION_LIMIT



# Logging and Utilities

def _setup_logging(log_dir: Optional[str] = None) -> Tuple[logging.Logger, str]:
    """Setup logging with file handler."""
    log_dir = log_dir or "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"agent_{timestamp}.log"

    logger = logging.getLogger(f"agent_{timestamp}")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger, str(log_file)


def _capture_user_code_only(caller_frame) -> str:
    """Capture user's code, excluding orchestrator imports."""
    try:
        filename = caller_frame.f_code.co_filename
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        call_line = caller_frame.f_lineno
        user_code_lines = []

        for i, line in enumerate(lines, 1):
            if any(exclude in line for exclude in [
                'from ai_pipeline_orchestrator import analyze_code',
                'import ai_pipeline_orchestrator',
                'analyze_code('
            ]):
                continue

            if i >= call_line and 'analyze_code(' in line:
                break

            user_code_lines.append(line)

        return ''.join(user_code_lines).strip()

    except Exception as e:
        return f"# Error capturing user code: {e}"


def _capture_user_variables_only(caller_frame, resolver=None) -> Dict[str, Any]:
    """Capture user-defined variables for context using resolver for relevance and type suggestions."""
    resolver = resolver or ResolverFactory.create_resolver("default")

    context = {
        "available_variables": [],
        "frame_info": {
            "filename": caller_frame.f_code.co_filename if caller_frame else "unknown",
            "function_name": caller_frame.f_code.co_name if caller_frame else "unknown",
            "line_number": caller_frame.f_lineno if caller_frame else 0
        }
    }

    if caller_frame is None:
        return context

    EXCLUDE_VARS = {
        '__builtins__', '__name__', '__doc__', '__file__', '__package__',
        '__spec__', '__loader__', '__cached__', '__annotations__',
        'UnifiedAgent', 'analyze_code', 'analyze_code_quick'
    }

    try:
        variables_to_check = {}

        if caller_frame.f_locals:
            variables_to_check.update(caller_frame.f_locals)

        if caller_frame.f_globals:
            for name, obj in caller_frame.f_globals.items():
                if name not in EXCLUDE_VARS and not name.startswith('_'):
                    if not (inspect.ismodule(obj) or inspect.isfunction(obj) or inspect.isclass(obj)):
                        variables_to_check[name] = obj

        for var_name, var_obj in variables_to_check.items():
            if var_name in EXCLUDE_VARS or var_name.startswith('_'):
                continue
            if resolver.is_relevant_variable(var_obj):
                var_basic_info = {
                    "name": var_name,
                    "type": type(var_obj).__name__,
                    "module": getattr(type(var_obj), "__module__", None),
                    "suggested_analysis": resolver.suggest_analysis_type(var_obj)
                }
                context["available_variables"].append(var_basic_info)

    except Exception as e:
        context["_error"] = f"Error capturing user variables: {e}"

    return context



class MCPLoggingCallback(BaseCallbackHandler):
    """Enhanced callback handler for comprehensive logging."""

    def __init__(self, logger):
        self.logger = logger

    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs) -> None:
        self.logger.info("=" * 80)
        self.logger.info("LLM Call Started")
        self.logger.info(f"Model: {serialized.get('name', 'unknown')}")
        self.logger.info(f"Number of prompts: {len(prompts)}")
        for i, prompt in enumerate(prompts, 1):
            self.logger.info(f"\n--- Prompt {i} ---")
            self.logger.info(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        self.logger.info("=" * 80)

    def on_llm_end(self, response, **kwargs) -> None:
        self.logger.info("-" * 80)
        self.logger.info("LLM Call Completed")
        if hasattr(response, 'generations'):
            for i, gen in enumerate(response.generations, 1):
                if gen:
                    text = gen[0].text if hasattr(gen[0], 'text') else str(gen[0])
                    self.logger.info(f"\n--- Response {i} ---")
                    self.logger.info(text[:500] + "..." if len(text) > 500 else text)
        self.logger.info("-" * 80)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.logger.error(f"LLM Error: {error}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown")
        self.logger.info(f"TOOL CALL: {tool_name}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        self.logger.info("<" * 80)
        self.logger.info(f"Output preview: {output[:300]}..." if len(output) > 300 else f"Output: {output}")
        self.logger.info("<" * 80 + "\n")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        self.logger.error(f"Tool Error: {error}")
        self.logger.error(f"Error type: {type(error).__name__}")

    def on_agent_action(self, action, **kwargs) -> None:
        self.logger.info("\n" + "~" * 80)
        self.logger.info(f"AGENT ACTION: {action.tool}")
        try:
            payload = json.dumps(action.tool_input, indent=2)
        except TypeError:
            payload = str(action.tool_input)
        self.logger.info(f"Action Input: {payload}")
        self.logger.info("~" * 80)

    def on_agent_finish(self, finish, **kwargs) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("AGENT FINISHED")
        self.logger.info(f"Final Output: {finish.return_values.get('output', 'N/A')[:300]}...")
        self.logger.info("=" * 80 + "\n")


def _create_llm(callbacks: Optional[List[BaseCallbackHandler]] = None) -> AzureChatOpenAI:
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not all([endpoint, api_key, deployment]):
        raise RuntimeError("Missing Azure OpenAI environment variables. Check .env file.")

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment,
        reasoning_effort="medium",
        callbacks=callbacks
    )



# Main Analysis Pipeline

async def _run_analysis_agent(
    source_code: str,
    minimal_context: Dict[str, Any],
    output_path: str,
    agent_client: UnifiedAgentClient,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run the analysis agent with unified tool access."""

    tools = agent_client.create_langchain_tools(logger, validation_limit=DEFAULT_VALIDATION_LIMIT)
    cb = MCPLoggingCallback(logger)
    llm = _create_llm(callbacks=[cb])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=24,
        return_intermediate_steps=True,
        callbacks=[cb]
    )

    try:
        result = await agent_executor.ainvoke({
            "source_code": source_code,
            "minimal_context": json.dumps(minimal_context, indent=2, default=str),
            "output_path": output_path,
            "validation_limit": DEFAULT_VALIDATION_LIMIT,
            "mapping_example": MAPPING_EXAMPLE
        })

        return {
            "success": True,
            "result": result.get("output", ""),
            "iterations": len(result.get("intermediate_steps", [])),
            "output_path": output_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output_path": output_path
        }


def analyze_code(
    output_path: str = "instances.ttl",
    server_command: Optional[List[str]] = None,
    log_dir: Optional[str] = None,
    variables: Optional[Dict[str, Any]] = None
) -> str:
    logger, log_file = _setup_logging(log_dir) 
    logger.info("=" * 80)
    logger.info("AGENT ANALYSIS SESSION")
    logger.info("=" * 80)
    logger.info(f"Session started: {datetime.datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output path: {output_path}")
    logger.info("=" * 80 + "\n")

    resolver = ResolverFactory.create_resolver("default")

    if variables is None:
        caller_frame = inspect.currentframe().f_back
        source_code = _capture_user_code_only(caller_frame)
        minimal_context = _capture_user_variables_only(caller_frame, resolver=resolver)

        safe_variables = {}
        if caller_frame:
            for var_name, var_obj in caller_frame.f_locals.items():
                try:
                    if (not var_name.startswith('_') and
                        not inspect.ismodule(var_obj) and
                        not inspect.isfunction(var_obj) and
                        not inspect.isclass(var_obj) and
                        not inspect.isbuiltin(var_obj) and
                        resolver.is_relevant_variable(var_obj)):
                        safe_variables[var_name] = var_obj
                except Exception:
                    continue

    else:
        caller_frame = None
        source_code = "# Variables provided directly"
        safe_variables = variables
        minimal_context = {
            "available_variables": [
                {
                    "name": var_name,
                    "type": type(var_obj).__name__,
                    "module": getattr(type(var_obj), "__module__", None),
                    "suggested_analysis": resolver.suggest_analysis_type(var_obj)
                }
                for var_name, var_obj in variables.items()
            ],
            "frame_info": {
                "filename": "direct_variables",
                "function_name": "analyze_code",
                "line_number": 0
            }
        }

    logger.info("SOURCE CODE CAPTURE")
    logger.info("-" * 80)
    logger.info(source_code)
    logger.info("-" * 80 + "\n")

    logger.info("RUNTIME CONTEXT CAPTURE")
    logger.info(f"Captured {len(minimal_context.get('available_variables', []))} user variables:")
    for var in minimal_context.get('available_variables', []):
        logger.info(f"  - {var['name']}: {var['type']} (suggested: {var.get('suggested_analysis')})")
    logger.info("\n")

    async def _run_pipeline():
        """Main analysis pipeline."""
        try:
            async with UnifiedAgentClient(server_command) as agent_client:
                logger.info("Connected to MCP and Local servers\n")

                agent_client.set_direct_variables(safe_variables)
                logger.info(f"Registered {len(safe_variables)} variables with local server\n")

                result = await _run_analysis_agent(
                    source_code=source_code,
                    minimal_context=minimal_context,
                    output_path=output_path,
                    agent_client=agent_client,
                    logger=logger
                )

                result["log_file"] = log_file
                
                logger.info("\n" + "=" * 80)
                logger.info("ANALYSIS SESSION COMPLETED")
                logger.info("=" * 80)
                logger.info(f"Success: {result['success']}")
                logger.info(f"Session ended: {datetime.datetime.now().isoformat()}")
                logger.info(f"Log file: {log_file}")
                logger.info("=" * 80 + "\n")
                
                return result

        except Exception as e:
            logger.error("\n" + "=" * 80)
            logger.error("PIPELINE FAILURE")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Session ended: {datetime.datetime.now().isoformat()}")
            logger.error("=" * 80 + "\n")
            
            return {
                "success": False,
                "error": str(e),
                "log_file": log_file
            }

    try:
        result = asyncio.run(_run_pipeline())
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        error_result = {
            "success": False,
            "error": f"Pipeline execution failed: {e}",
            "log_file": log_file
        }
        return json.dumps(error_result, indent=2)


def analyze_code_quick(variables: Dict[str, Any], output_path: str = "instances.ttl") -> str:
    return analyze_code(output_path=output_path, variables=variables)