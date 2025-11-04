import os

"""
Configuration for orchestrator.
"""

DEFAULT_VALIDATION_LIMIT = int(os.getenv("VALIDATION_LIMIT", "5"))

MAPPING_EXAMPLE = """
{
  "my_dataframe": {
    "base_variable": "df",
    "analysis_type": "dataframe",
    "requested_attributes": ["shape", "columns", "dtypes", "index_type"],
    "attribute_path": [],
    "ontology_mapping_intent": "map to semts:TimeSeriesSegment"
  },
  "trained_model": {
    "base_variable": "model",
    "analysis_type": "ml_model",
    "requested_attributes": ["model_params", "model_type"],
    "attribute_path": [],
    "ontology_mapping_intent": "map to semts:Model"
  }
}
"""

SYSTEM_PROMPT = """You are a SemTS (Semantic Time Series) analysis expert. Your task is to analyze Python code and minimal runtime context to generate semantic annotations in TTL format using the SemTS ontology.

You have access to tools that provide:
- Ontology information (descriptions)
- Ontology schema (classes, properties)
- Knowledge vocabularies with concepts
- Targeted runtime context variable resolution
- Instance examples in TTL format
- TTL validation capabilities
- TTL file creation

Your workflow should be:
1. Use get_general_ontology_infos to understand ontology metadata and prefixes
2. Use get_ontology to explore available classes and properties
3. Use get_vocabularies to understand vocabulary concepts
4. Use get_ontology_examples to see ontology patterns in examples
5. ANALYZE THE CODE to identify variables that need detailed runtime context resolution with respect to the ontology
6. CREATE A MAPPING specifying exactly what context information you need for each variable
7. Use resolve_runtime_context_values with your specific mapping to get detailed analysis
8. Generate TTL annotations mapping code variables to SemTS concepts
9. Validate the TTL using validate_ttl_code (use the validate_ttl_code min 1 time, but max {validation_limit} times!)
10. Use create_ttl_file to save the final result

For step 6, create mappings like:
{mapping_example}

Important:
- Only request the specific runtime context information you need for semantic mapping.
"""

USER_PROMPT = """
Analyze the following Python code and minimal runtime context to generate SemTS TTL annotations.

Source Code:
{source_code}

Available Runtime Variables:
{minimal_context}

Target Output Path: {output_path}

Please:
1. Explore the ontology structure and vocabulary concepts
2. Analyze the code and available variables
3. Create a specific variable mapping for runtime context resolution
4. Generate appropriate TTL annotations
5. Validate the generated TTL
6. Save the final result

"""