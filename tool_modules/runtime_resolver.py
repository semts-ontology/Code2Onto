from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class RuntimeContextResolver(ABC):
    """Abstract base class for runtime context resolvers."""
    
    @abstractmethod
    def analyze_variable(
        self, 
        variables: Dict[str, Any], 
        var_name: str, 
        attribute_path: List[str] = None,
        requested_attributes: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a specific variable and return context information.
        
        Args:
            variables: Dictionary of available variables
            var_name: Name of the variable to analyze
            attribute_path: Path to nested attributes (e.g., ['model', 'params'])
            requested_attributes: Specific attributes to extract (optional filtering)
            
        Returns:
            Dictionary with analysis results or error information
        """
        pass
    
    @abstractmethod
    def is_relevant_variable(self, var_obj: Any) -> bool:
        """Determine if a variable is relevant for analysis."""
        pass
    
    @abstractmethod
    def suggest_analysis_type(self, var_obj: Any) -> str:
        """Suggest the type of analysis for a variable."""
        pass


class DefaultRuntimeResolver(RuntimeContextResolver):
    """
    Default resolver supporting pandas, numpy, scikit-learn, and basic Python types.
    This is the direct replacement for RuntimeContextAnalyzer.
    """
    def analyze_variable(
        self,
        variables: Dict[str, Any],
        var_name: str,
        attribute_path: List[str] = None,
        requested_attributes: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze a specific variable from a variables dictionary."""
        attribute_path = attribute_path or []
        
        try:
            if var_name not in variables:
                return {"error": f"Variable '{var_name}' not found in provided variables"}
            
            var_obj = variables[var_name]
            
            current_obj = var_obj
            for attr in attribute_path:
                if hasattr(current_obj, attr):
                    current_obj = getattr(current_obj, attr)
                else:
                    return {"error": f"Attribute '{attr}' not found in path {'.'.join(attribute_path)}"}
            
            analysis = self._analyze_variable_detailed(var_name, current_obj, attribute_path)
            
            if requested_attributes and "analysis_results" in analysis:
                filtered = {k: v for k, v in analysis["analysis_results"].items() 
                           if k in requested_attributes}
                analysis["filtered_results"] = filtered
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing variable '{var_name}': {str(e)}"}
    
    def is_relevant_variable(self, var_obj: Any) -> bool:
        """Check if variable should be included in analysis."""
        import inspect
        if inspect.isbuiltin(var_obj):
            return False
        if inspect.ismodule(var_obj):
            return False
        if inspect.isfunction(var_obj):
            return False
        if inspect.isclass(var_obj):
            return False
        return True
    
    def suggest_analysis_type(self, var_obj: Any) -> str:
        """Suggest analysis type based on object type."""
        if pd is not None and isinstance(var_obj, pd.DataFrame):
            return "dataframe"
        elif pd is not None and isinstance(var_obj, pd.Series):
            return "series"
        elif hasattr(var_obj, 'get_params') and hasattr(var_obj, 'fit'):
            return "ml_model"
        elif np is not None and isinstance(var_obj, np.ndarray):
            return "numpy_array"
        elif isinstance(var_obj, (list, tuple)):
            return "collection"
        elif isinstance(var_obj, dict):
            return "dictionary"
        elif isinstance(var_obj, (int, float)):
            return "numeric"
        elif isinstance(var_obj, str):
            return "string"
        elif isinstance(var_obj, bool):
            return "boolean"
        elif hasattr(var_obj, 'read') or hasattr(var_obj, 'write'):
            return "file_object"
        elif hasattr(var_obj, '__fspath__'):
            return "path_object"
        elif hasattr(var_obj, 'strftime'):
            return "datetime_object"
        else:
            return "generic_object"
    
    def _analyze_variable_detailed(self, var_name: str, var_obj: Any, attribute_path: List[str] = None) -> Dict[str, Any]:
        """
        Perform detailed analysis of a variable object.
        This is the exact logic from RuntimeContextAnalyzer._analyze_variable_detailed.
        """
        attribute_path = attribute_path or []
        
        var_info = {
            "name": var_name,
            "type": type(var_obj).__name__,
            "module": getattr(type(var_obj), "__module__", None),
            "attribute_path": attribute_path,
            "analysis_results": {}
        }
        
        try:
            if pd is not None:
                if isinstance(var_obj, pd.DataFrame):
                    var_info["analysis_results"] = {
                        "object_type": "pandas_dataframe",
                        "shape": var_obj.shape,
                        "columns": list(var_obj.columns),
                        "dtypes": {str(col): str(dtype) for col, dtype in var_obj.dtypes.items()},
                        "index_type": type(var_obj.index).__name__,
                        "index_name": var_obj.index.name,
                        "memory_usage": var_obj.memory_usage(deep=True).to_dict() if hasattr(var_obj, 'memory_usage') else None,
                        "null_counts": var_obj.isnull().sum().to_dict(),
                        "numeric_columns": list(var_obj.select_dtypes(include=[np.number]).columns) if np is not None else [],
                        "datetime_columns": list(var_obj.select_dtypes(include=['datetime64']).columns),
                        "categorical_columns": list(var_obj.select_dtypes(include=['category']).columns)
                    }
                    
                    datetime_cols = var_info["analysis_results"]["datetime_columns"]
                    if datetime_cols:
                        for col in datetime_cols[:1]:
                            try:
                                var_info["analysis_results"]["temporal_analysis"] = {
                                    "column": col,
                                    "min_date": str(var_obj[col].min()),
                                    "max_date": str(var_obj[col].max()),
                                    "date_range_days": (var_obj[col].max() - var_obj[col].min()).days,
                                    "frequency_hint": pd.infer_freq(var_obj[col]) if len(var_obj) > 1 else None
                                }
                            except Exception:
                                pass
                    
                    return var_info
                
                elif isinstance(var_obj, pd.Series):
                    var_info["analysis_results"] = {
                        "object_type": "pandas_series",
                        "shape": var_obj.shape,
                        "dtype": str(var_obj.dtype),
                        "name": var_obj.name,
                        "null_count": var_obj.isnull().sum(),
                        "unique_count": var_obj.nunique(),
                        "is_numeric": pd.api.types.is_numeric_dtype(var_obj),
                        "is_datetime": pd.api.types.is_datetime64_any_dtype(var_obj)
                    }
                    
                    if var_info["analysis_results"]["is_numeric"]:
                        var_info["analysis_results"]["statistics"] = {
                            "mean": float(var_obj.mean()),
                            "std": float(var_obj.std()),
                            "min": float(var_obj.min()),
                            "max": float(var_obj.max()),
                            "median": float(var_obj.median())
                        }
                    
                    return var_info
            
            if np is not None and isinstance(var_obj, np.ndarray):
                var_info["analysis_results"] = {
                    "object_type": "numpy_array",
                    "shape": var_obj.shape,
                    "dtype": str(var_obj.dtype),
                    "ndim": var_obj.ndim,
                    "size": var_obj.size,
                    "memory_usage_bytes": var_obj.nbytes
                }
                
                if np.issubdtype(var_obj.dtype, np.number):
                    var_info["analysis_results"]["statistics"] = {
                        "mean": float(np.mean(var_obj)),
                        "std": float(np.std(var_obj)),
                        "min": float(np.min(var_obj)),
                        "max": float(np.max(var_obj))
                    }
                
                return var_info
            
            if hasattr(var_obj, "get_params"):
                try:
                    params = var_obj.get_params()
                    var_info["analysis_results"] = {
                        "object_type": "ml_model",
                        "model_class": type(var_obj).__name__,
                        "model_params": {k: str(v) for k, v in params.items()},
                        "is_fitted": hasattr(var_obj, "fit") and any(
                            hasattr(var_obj, attr) for attr in ["coef_", "feature_importances_", "support_", "classes_"]
                        )
                    }
                    
                    if hasattr(var_obj, "feature_importances_"):
                        var_info["analysis_results"]["has_feature_importances"] = True
                    if hasattr(var_obj, "coef_"):
                        var_info["analysis_results"]["has_coefficients"] = True
                    if hasattr(var_obj, "classes_"):
                        var_info["analysis_results"]["classes"] = list(var_obj.classes_)
                    
                    return var_info
                except Exception:
                    pass
            
            if isinstance(var_obj, (int, float)):
                var_info["analysis_results"] = {
                    "object_type": "numeric_primitive",
                    "value": var_obj,
                    "is_integer": isinstance(var_obj, int),
                    "is_positive": var_obj > 0,
                    "magnitude": abs(var_obj)
                }
                return var_info
            
            elif isinstance(var_obj, str):
                var_info["analysis_results"] = {
                    "object_type": "string_primitive",
                    "value": var_obj[:200],
                    "length": len(var_obj),
                    "is_numeric": var_obj.isdigit(),
                    "contains_path": "/" in var_obj or "\\" in var_obj,
                    "contains_url": var_obj.startswith(("http://", "https://"))
                }
                return var_info
            
            elif isinstance(var_obj, bool):
                var_info["analysis_results"] = {
                    "object_type": "boolean_primitive",
                    "value": var_obj
                }
                return var_info
            
            elif isinstance(var_obj, (list, tuple)):
                var_info["analysis_results"] = {
                    "object_type": "collection",
                    "collection_type": type(var_obj).__name__,
                    "length": len(var_obj),
                    "is_empty": len(var_obj) == 0
                }
                
                if var_obj and len(var_obj) > 0:
                    var_info["analysis_results"]["element_types"] = list(set(type(item).__name__ for item in var_obj[:10]))
                    var_info["analysis_results"]["first_element"] = str(var_obj[0])[:100]
                
                return var_info
            
            elif isinstance(var_obj, dict):
                var_info["analysis_results"] = {
                    "object_type": "dictionary",
                    "length": len(var_obj),
                    "keys": list(var_obj.keys())[:10],
                    "key_types": list(set(type(k).__name__ for k in var_obj.keys())),
                    "value_types": list(set(type(v).__name__ for v in var_obj.values()))
                }
                return var_info
            
            else:
                var_info["analysis_results"] = {
                    "object_type": "generic_object",
                    "class_name": type(var_obj).__name__,
                    "module": getattr(type(var_obj), "__module__", None),
                    "attributes": [attr for attr in dir(var_obj) if not attr.startswith('_')][:20],
                    "callable": callable(var_obj),
                    "has_len": hasattr(var_obj, '__len__'),
                    "has_iter": hasattr(var_obj, '__iter__')
                }
                return var_info
        
        except Exception as e:
            var_info["analysis_results"] = {
                "object_type": "analysis_error",
                "error": str(e)
            }
            return var_info


class ResolverFactory:
    """Factory for creating and managing runtime context resolvers."""
    
    _resolvers: Dict[str, type] = {
        "default": DefaultRuntimeResolver,
    }
    
    @classmethod
    def register_resolver(cls, name: str, resolver_class: type):
        """
        Register a custom resolver.
        
        Args:
            name: Identifier for the resolver
            resolver_class: Class inheriting from RuntimeContextResolver
        """
        if not issubclass(resolver_class, RuntimeContextResolver):
            raise TypeError(f"Resolver must inherit from RuntimeContextResolver, got {resolver_class}")
        cls._resolvers[name] = resolver_class
    
    @classmethod
    def create_resolver(cls, name: str = "default") -> RuntimeContextResolver:
        """
        Create a resolver instance by name.
        
        Args:
            name: Resolver identifier
            
        Returns:
            Instance of the requested resolver
        """
        if name not in cls._resolvers:
            raise ValueError(f"Unknown resolver: {name}. Available: {list(cls._resolvers.keys())}")
        return cls._resolvers[name]()
    
    @classmethod
    def list_resolvers(cls) -> List[str]:
        """List all registered resolver names."""
        return list(cls._resolvers.keys())