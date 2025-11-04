import json
from typing import Dict, List, Any
from pathlib import Path
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import SKOS, RDF, RDFS, OWL
from importlib.util import find_spec
from tool_modules.ontology_validation import OWLValidator, SHACLValidator
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import ListToolsResult, Tool


# Namespaces

try:
    SEMTS = Namespace("https://w3id.org/semts/ontology#")
except TypeError:
    from rdflib import URIRef as _URIRef
    class SimpleNamespace:
        def __init__(self, base_uri):
            self.base_uri = base_uri
        def __getitem__(self, item):
            return _URIRef(self.base_uri + item)
        def __getattr__(self, item):
            return _URIRef(self.base_uri + item)
    SEMTS = SimpleNamespace("https://w3id.org/semts/ontology#")

try:
    DCAT = Namespace("http://www.w3.org/ns/dcat#")
    TIME = Namespace("http://www.w3.org/2006/time#")
    QUDT = Namespace("http://qudt.org/schema/qudt#")
    QUDT_UNIT = Namespace("http://qudt.org/vocab/unit/")
    EX = Namespace("http://example.org/")
    MLS = Namespace("http://www.w3.org/ns/mls#")
    PROV = Namespace("http://www.w3.org/ns/prov#")
    SEMTS_DV = Namespace("https://w3id.org/semts/vocabulary/data-knowledge#")
    SEMTS_SV = Namespace("https://w3id.org/semts/vocabulary/scenario-knowledge#")
except TypeError:
    DCAT = SimpleNamespace("http://www.w3.org/ns/dcat#")
    TIME = SimpleNamespace("http://www.w3.org/2006/time#")
    QUDT = SimpleNamespace("http://qudt.org/schema/qudt#")
    QUDT_UNIT = SimpleNamespace("http://qudt.org/vocab/unit/")
    EX = SimpleNamespace("http://example.org/")
    MLS = SimpleNamespace("http://www.w3.org/ns/mls#")
    PROV = SimpleNamespace("http://www.w3.org/ns/prov#")
    SEMTS_DV = SimpleNamespace("https://w3id.org/semts/vocabulary/data-knowledge#")
    SEMTS_SV = SimpleNamespace("https://w3id.org/semts/vocabulary/scenario-knowledge#")


OWL_VALIDATION_AVAILABLE = find_spec("owlready2") is not None
SHACL_VALIDATION_AVAILABLE = find_spec("pyshacl") is not None


def _text_result(text: str, is_error: bool = False) -> dict:
    """Return a plain dict in the shape the MCP server expects for a tool call result."""
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error
    }


class RemoteAgentServer:
    """MCP Server providing SemTS ontology and knowledge services (no runtime context resolution)."""

    def __init__(self, ontology_path: str = None, vocab_paths: List[str] = None, examples_path: str = None):
        self.server = Server("semts-ontology-server")
        self.ontology_path = ontology_path or "semts/ontology.ttl"
        self.vocab_paths = vocab_paths or [
            "semts/vocab_data_knowledge.ttl",
            "semts/vocab_scenario_knowledge.ttl"
        ]
        self.examples_path = examples_path or "semts/semts_class_examples.ttl"
        self.ontology_graph = Graph()
        self.vocab_graphs: List[Graph] = []

        self._convert_ontology_to_nt(self.ontology_path)
        self.owlready_ontology_path = "semts/semts_class_examples.nt"
        self._register_tools()

    def _convert_ontology_to_nt(self, src: str) -> str:
        try:
            if not Path(src).exists():
                print(f"Warning: Ontology file not found: {src}")
                return src
            g = Graph()
            g.parse(src, format="turtle")
            out = Path(src).with_suffix(".nt")
            out.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(destination=str(out), format="nt")
        except Exception as e:
            print(f"Warning: TTL→NT conversion failed ({src}): {e}")

    def _register_tools(self):
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            tools = [
                Tool(
                    name="get_general_ontology_infos",
                    description="Load prefixes, abstract and description from ontology",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_ontology",
                    description="Load ontology as a dictionary with classes and properties",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_vocabularies",
                    description="Load vocabularies as dictionaries",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_ontology_examples",
                    description="Load instance examples in their raw TTL form",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="validate_ttl_code",
                    description="Validate TTL code using OWL axiom validation and SHACL",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ttl_code": {"type": "string", "description": "TTL code to validate"}
                        },
                        "required": ["ttl_code"]
                    }
                )
            ]
            return ListToolsResult(tools=tools)

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> dict:
            try:
                await self._ensure_loaded()

                if name == "get_general_ontology_infos":
                    return await self._get_general_ontology_infos()
                elif name == "get_ontology":
                    return await self._get_ontology()
                elif name == "get_vocabularies":
                    return await self._get_vocabularies()
                elif name == "get_ontology_examples":
                    return await self._get_ontology_examples()
                elif name == "validate_ttl_code":
                    return await self._validate_ttl_code(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                return _text_result(f"Error: {str(e)}", is_error=True)

    async def _ensure_loaded(self):
        """Ensure ontology and vocabulary files are loaded."""
        if not self.ontology_graph or not self.vocab_graphs:
            await self._load_files()

    async def _load_files(self):
        """Load ontology and vocabulary files."""
        self.ontology_graph = Graph()
        self.vocab_graphs = []

        if Path(self.ontology_path).exists():
            try:
                self.ontology_graph.parse(self.ontology_path, format="turtle")
            except Exception as e:
                print(f"Warning: Could not load ontology file {self.ontology_path}: {e}")

        for vocab_path in self.vocab_paths:
            if Path(vocab_path).exists():
                try:
                    vg = Graph()
                    vg.parse(vocab_path, format="turtle")
                    self.vocab_graphs.append(vg)
                except Exception as e:
                    print(f"Warning: Could not load vocabulary file {vocab_path}: {e}")

    async def _get_general_ontology_infos(self) -> dict:
        """Load prefixes, abstract and description from ontology."""
        try:
            result = {"prefixes": {}, "abstract": None, "description": None}

            for prefix, namespace in self.ontology_graph.namespaces():
                result["prefixes"][prefix] = str(namespace)

            for subj in self.ontology_graph.subjects(RDF.type, OWL.Ontology):
                for abstract in self.ontology_graph.objects(subj, URIRef("http://purl.org/dc/terms/abstract")):
                    result["abstract"] = str(abstract)
                    break
                for desc in self.ontology_graph.objects(subj, URIRef("http://purl.org/dc/terms/description")):
                    result["description"] = str(desc)
                    break
                break

            return _text_result(json.dumps(result, indent=2))
        except Exception as e:
            return _text_result(f"Error loading ontology info: {str(e)}", is_error=True)

    async def _get_ontology(self) -> dict:
        """Load ontology as a dictionary with classes and properties."""
        try:
            ontology_dict = {"classes": {}, "object_properties": {}, "data_properties": {}}

            for subj in self.ontology_graph.subjects(RDF.type, OWL.Class):
                if isinstance(subj, URIRef):
                    class_name = str(subj).split('#')[-1] if '#' in str(subj) else str(subj).split('/')[-1]
                    class_info = {
                        "uri": str(subj),
                        "name": class_name,
                        "label": None,
                        "comment": None,
                        "superclasses": []
                    }

                    for label in self.ontology_graph.objects(subj, RDFS.label):
                        class_info["label"] = str(label)
                        break

                    for comment in self.ontology_graph.objects(subj, RDFS.comment):
                        class_info["comment"] = str(comment)
                        break

                    for superclass in self.ontology_graph.objects(subj, RDFS.subClassOf):
                        if isinstance(superclass, URIRef):
                            super_name = str(superclass).split('#')[-1] if '#' in str(superclass) else str(superclass).split('/')[-1]
                            class_info["superclasses"].append(super_name)

                    ontology_dict["classes"][class_name] = class_info

            for prop in self.ontology_graph.subjects(RDF.type, OWL.ObjectProperty):
                if isinstance(prop, URIRef):
                    prop_name = str(prop).split('#')[-1] if '#' in str(prop) else str(prop).split('/')[-1]
                    prop_info = {"uri": str(prop), "name": prop_name, "domain": None, "range": None, "comment": None}

                    for domain in self.ontology_graph.objects(prop, RDFS.domain):
                        if isinstance(domain, URIRef):
                            domain_name = str(domain).split('#')[-1] if '#' in str(domain) else str(domain).split('/')[-1]
                            prop_info["domain"] = domain_name
                            break

                    for range_val in self.ontology_graph.objects(prop, RDFS.range):
                        if isinstance(range_val, URIRef):
                            range_name = str(range_val).split('#')[-1] if '#' in str(range_val) else str(range_val).split('/')[-1]
                            prop_info["range"] = range_name
                            break

                    for comment in self.ontology_graph.objects(prop, RDFS.comment):
                        prop_info["comment"] = str(comment)
                        break

                    ontology_dict["object_properties"][prop_name] = prop_info

            for prop in self.ontology_graph.subjects(RDF.type, OWL.DatatypeProperty):
                if isinstance(prop, URIRef):
                    prop_name = str(prop).split('#')[-1] if '#' in str(prop) else str(prop).split('/')[-1]
                    prop_info = {"uri": str(prop), "name": prop_name, "domain": None, "range": None, "comment": None}

                    for domain in self.ontology_graph.objects(prop, RDFS.domain):
                        if isinstance(domain, URIRef):
                            domain_name = str(domain).split('#')[-1] if '#' in str(domain) else str(domain).split('/')[-1]
                            prop_info["domain"] = domain_name
                            break

                    for range_val in self.ontology_graph.objects(prop, RDFS.range):
                        prop_info["range"] = str(range_val)
                        break

                    for comment in self.ontology_graph.objects(prop, RDFS.comment):
                        prop_info["comment"] = str(comment)
                        break

                    ontology_dict["data_properties"][prop_name] = prop_info

            return _text_result(json.dumps(ontology_dict, indent=2))
        except Exception as e:
            return _text_result(f"Error loading ontology: {str(e)}", is_error=True)

    async def _get_vocabularies(self) -> dict:
        """Load vocabularies as dictionaries."""
        try:
            vocabularies: Dict[str, Any] = {}

            for i, vg in enumerate(self.vocab_graphs):
                vocab_name = f"vocabulary_{i}"
                if i < len(self.vocab_paths):
                    vocab_file = Path(self.vocab_paths[i]).stem
                    vocab_name = vocab_file

                concepts: Dict[str, Any] = {}
                knowledge_concept_uri = URIRef("https://w3id.org/semts/ontology#KnowledgeConcept")

                for concept in vg.subjects(RDF.type, knowledge_concept_uri):
                    if isinstance(concept, URIRef):
                        concept_name = str(concept).split('#')[-1] if '#' in str(concept) else str(concept).split('/')[-1]
                        concept_info = {
                            "uri": str(concept),
                            "name": concept_name,
                            "prefLabel": None,
                            "definition": None,
                            "broader": [],
                            "narrower": []
                        }

                        for label in vg.objects(concept, SKOS.prefLabel):
                            concept_info["prefLabel"] = str(label)
                            break

                        for definition in vg.objects(concept, SKOS.definition):
                            concept_info["definition"] = str(definition)
                            break

                        for broader in vg.objects(concept, SKOS.broader):
                            if isinstance(broader, URIRef):
                                broader_name = str(broader).split('#')[-1] if '#' in str(broader) else str(broader).split('/')[-1]
                                concept_info["broader"].append(broader_name)

                        for narrower in vg.subjects(SKOS.broader, concept):
                            if isinstance(narrower, URIRef):
                                narrower_name = str(narrower).split('#')[-1] if '#' in str(narrower) else str(narrower).split('/')[-1]
                                concept_info["narrower"].append(narrower_name)

                        concepts[concept_name] = concept_info

                vocabularies[vocab_name] = concepts

            return _text_result(json.dumps(vocabularies, indent=2))
        except Exception as e:
            return _text_result(f"Error loading vocabularies: {str(e)}", is_error=True)

    async def _get_ontology_examples(self) -> dict:
        """Load instance examples in their raw TTL form."""
        try:
            if not Path(self.examples_path).exists():
                return _text_result(f"Examples file '{self.examples_path}' not found. Please ensure the file exists.", is_error=False)

            ttl_content = Path(self.examples_path).read_text(encoding="utf-8")
            return _text_result(ttl_content)
        except Exception as e:
            return _text_result(f"Error loading examples: {str(e)}", is_error=True)

    async def _validate_ttl_code(self, ttl_code: str) -> dict:
        try:
            validation_results = {
                "syntax_valid": False,
                "owl_validation": None,
                "shacl_validation": None,
                "overall_valid": False
            }

            try:
                test_graph = Graph()
                test_graph.parse(data=ttl_code, format="turtle")
                validation_results["syntax_valid"] = True
            except Exception as e:
                validation_results["syntax_error"] = str(e)
                validation_results["owl_validation"] = {"note": "OWL validation skipped due to syntax error"}
                validation_results["shacl_validation"] = {"note": "SHACL validation skipped due to syntax error"}
                return _text_result(json.dumps(validation_results, indent=2))

            try:
                from tool_modules.ontology_validation import OWLValidator
                
                with OWLValidator(self.ontology_path) as validator:
                    validator.load_instances(ttl_code)
                    is_valid, violations, inferred_facts = validator.validate_all_axioms()

                    validation_results["owl_validation"] = {
                        "valid": bool(is_valid),
                        "violations": (violations or []),
                        "inferred_facts": (inferred_facts or [])
                    }
                    
            except Exception as e:
                validation_results["owl_validation"] = {"error": str(e)}

            if SHACL_VALIDATION_AVAILABLE:
                shapes_file = "semts/shacl_shapes.ttl"
                if Path(shapes_file).exists():
                    try:
                        from tool_modules.ontology_validation import SHACLValidator
                        validator = SHACLValidator(shapes_file)
                        conforms, summary = validator.validate_data(ttl_code, self.ontology_path)
                        validation_results["shacl_validation"] = {
                            "conforms": bool(conforms),
                            "summary": summary
                        }
                    except Exception as e:
                        validation_results["shacl_validation"] = {"error": str(e)}
                else:
                    validation_results["shacl_validation"] = {"note": f"SHACL shapes file not found: {shapes_file}"}
            else:
                validation_results["shacl_validation"] = {"note": "SHACL validation not available"}

            owl_info = validation_results["owl_validation"] or {}
            shacl_info = validation_results["shacl_validation"] or {}
            owl_valid = owl_info.get("valid", True) if "error" not in owl_info else False
            shacl_valid = shacl_info.get("conforms", True) if "error" not in shacl_info else False
            validation_results["overall_valid"] = validation_results["syntax_valid"] and owl_valid and shacl_valid

            return _text_result(json.dumps(validation_results, indent=2))
        except Exception as e:
            return _text_result(f"Error during validation: {str(e)}", is_error=True)


async def main():
    """Run the MCP server."""
    server = RemoteAgentServer()
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="semts-ontology-server",
                server_version="2.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())