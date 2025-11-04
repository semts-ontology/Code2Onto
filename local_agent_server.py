import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from rdflib import Graph

from tool_modules.runtime_resolver import ResolverFactory


class LocalAgentServer:
    """Local server providing file operations and runtime context resolution."""

    def __init__(self):
        self.direct_variables: Optional[Dict[str, Any]] = None

    def set_direct_variables(self, variables: Dict[str, Any]):
        """Set variables for local analysis."""
        self.direct_variables = variables

    def get_direct_variables(self) -> Optional[Dict[str, Any]]:
        """Get the stored variables."""
        return self.direct_variables

    def resolve_runtime_context_values(
        self,
        variable_mapping: str,
        resolver_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Resolve runtime context locally using stored variables.
        
        Args:
            variable_mapping: JSON mapping specifying variables to analyze
            resolver_type: Type of resolver to use (default, custom, etc.)
        
        Returns:
            Dictionary with resolved context information
        """
        try:
            mapping_dict = json.loads(variable_mapping)
            resolver = ResolverFactory.create_resolver(resolver_type)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in variable_mapping"}
        except ValueError as e:
            return {"error": str(e)}
        
        direct_vars = self.direct_variables or {}
        output = {
            "requested_mapping": mapping_dict,
            "available_variables": list(direct_vars.keys()),
            "resolver_used": resolver_type,
            "resolved": {}
        }
        
        for alias, spec in mapping_dict.items():
            base = spec.get("base_variable")
            attr_path = spec.get("attribute_path", []) or []
            requested = spec.get("requested_attributes", [])
            
            if base not in direct_vars:
                output["resolved"][alias] = {"error": f"Variable '{base}' not found"}
                continue
            
            analysis = resolver.analyze_variable(
                direct_vars,
                base,
                attr_path,
                requested
            )
            
            output["resolved"][alias] = analysis
        
        return output

    def create_ttl_file(self, ttl_code: str, path: str) -> Dict[str, Any]:
        """
        Create a TTL file at specified path with given TTL code.
        
        Args:
            ttl_code: TTL content to write
            path: File path where to save the TTL file
        
        Returns:
            Dictionary with operation result
        """
        try:
            test_graph = Graph()
            test_graph.parse(data=ttl_code, format="turtle")
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(ttl_code, encoding="utf-8")

            return {
                "success": True,
                "file_path": str(file_path.resolve()),
                "file_size": len(ttl_code.encode('utf-8')),
                "message": f"TTL file successfully created at {path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating TTL file: {str(e)}"
            }

    def list_available_variables(self) -> Dict[str, Any]:
        """
        List all available variables with basic type information.
        
        Returns:
            Dictionary with variable information
        """
        if not self.direct_variables:
            return {"variables": [], "count": 0}
        
        variables_info = []
        for name, obj in self.direct_variables.items():
            variables_info.append({
                "name": name,
                "type": type(obj).__name__,
                "module": getattr(type(obj), "__module__", None)
            })
        
        return {
            "variables": variables_info,
            "count": len(variables_info)
        }