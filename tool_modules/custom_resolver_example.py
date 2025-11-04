from runtime_resolver import RuntimeContextResolver, ResolverFactory

class CustomResolver(RuntimeContextResolver):
    
    def analyze_variable(self, variables, var_name, attribute_path=None, requested_attributes=None):
        pass
    
    def is_relevant_variable(self, var_obj):
        return None
    
    def suggest_analysis_type(self, var_obj):
        return None

ResolverFactory.register_resolver("custom", CustomResolver)