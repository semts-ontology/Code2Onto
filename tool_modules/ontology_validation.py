import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import rdflib
from rdflib import Namespace
from rdflib.namespace import SKOS, RDF, RDFS, OWL, XSD

# Use rdflib.Graph explicitly everywhere to avoid clashes with owlready2.triplelite.Graph
RDFGraph = rdflib.Graph

try:
    import owlready2
    from owlready2 import (
        World,
        sync_reasoner_pellet,
        FunctionalProperty,
        InverseFunctionalProperty,
        OwlReadyInconsistentOntologyError,
    )
    OWL_VALIDATION_AVAILABLE = True
except ImportError:
    OWL_VALIDATION_AVAILABLE = False

try:
    import pyshacl
    SHACL_VALIDATION_AVAILABLE = True
except ImportError:
    SHACL_VALIDATION_AVAILABLE = False


def _patch_owlready_graph_destructor():
    """
    Make owlready2 triplelite.Graph.__del__ safe.

    Prevents errors like:
    Exception ignored in: <function Graph.__del__ ...>
    AttributeError: 'Graph' object has no attribute 'db'
    or
    AttributeError: 'NoneType' object has no attribute 'close'
    """
    if not OWL_VALIDATION_AVAILABLE:
        return

    try:
        import owlready2.triplelite as tl

        if getattr(getattr(tl, "Graph", object), "__del__", None) and getattr(tl.Graph.__del__, "_semts_patched", False):
            return

        def _safe_del(self):
            try:
                db = getattr(self, "db", None)
                if db is not None and hasattr(db, "close"):
                    try:
                        db.close()
                    except Exception:
                        pass
            except Exception:
                pass

        _safe_del._semts_patched = True
        tl.Graph.__del__ = _safe_del
    except Exception:
        pass


class OWLValidator:
    """Comprehensive OWL axiom validator using owlready2 with robust cleanup."""

    def __init__(self, ontology_file: str):
        if not OWL_VALIDATION_AVAILABLE:
            raise ImportError("owlready2 is required for OWL validation. Install with: pip install owlready2")

        _patch_owlready_graph_destructor()

        self.world: Optional[World] = None
        self.onto: Optional[Any] = None
        self.ontology_file = ontology_file
        self._temp_files: List[str] = []
        self._world_closed = False

    def __enter__(self):
        """Context manager entry"""
        self._initialize_world()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        self._cleanup()
        return False

    def _initialize_world(self):
        """Initialize the owlready2 world and load ontology"""
        try:
            self.world = World()

            ontology_path = Path(self.ontology_file)
            if not ontology_path.exists():
                raise FileNotFoundError(f"Ontology file not found: {self.ontology_file}")

            abs_path = ontology_path.resolve()

            if abs_path.suffix.lower() in ['.ttl', '.turtle']:
                self.onto = self._load_turtle_ontology(str(abs_path))
            elif abs_path.suffix.lower() in ['.owl', '.rdf', '.xml']:
                self.onto = self.world.get_ontology(f"file://{abs_path}").load()
            elif abs_path.suffix.lower() == '.nt':
                self.onto = self._load_ntriples_ontology(str(abs_path))
            else:
                self.onto = self.world.get_ontology(f"file://{abs_path}").load()

        except Exception as e:
            self._cleanup()
            raise Exception(f"Failed to initialize OWL validator: {str(e)}")

    def _load_turtle_ontology(self, ttl_path: str):
        """Load TTL ontology by converting to OWL/XML format"""
        try:
            g = RDFGraph()
            g.parse(ttl_path, format="turtle")

            temp_owl = tempfile.NamedTemporaryFile(mode='w', suffix='.owl', delete=False, encoding='utf-8')
            self._temp_files.append(temp_owl.name)

            owl_content = g.serialize(format="xml")
            if isinstance(owl_content, bytes):
                owl_content = owl_content.decode('utf-8')

            temp_owl.write(owl_content)
            temp_owl.close()
            g.close()

            return self.world.get_ontology(f"file://{temp_owl.name}").load()

        except Exception as e:
            raise Exception(f"Failed to load TTL ontology: {str(e)}")

    def _load_ntriples_ontology(self, nt_path: str):
        """Load N-Triples ontology by converting to OWL/XML format"""
        try:
            g = RDFGraph()
            g.parse(nt_path, format="nt")

            temp_owl = tempfile.NamedTemporaryFile(mode='w', suffix='.owl', delete=False, encoding='utf-8')
            self._temp_files.append(temp_owl.name)

            owl_content = g.serialize(format="xml")
            if isinstance(owl_content, bytes):
                owl_content = owl_content.decode('utf-8')

            temp_owl.write(owl_content)
            temp_owl.close()
            g.close()

            return self.world.get_ontology(f"file://{temp_owl.name}").load() 
        
        except Exception as e:
            raise Exception(f"Failed to load N-Triples ontology: {str(e)}")

    def load_instances(self, instances_content: str):
        """Load instance data from TTL content"""
        if not self.world:
            raise RuntimeError("Validator not initialized. Use as context manager.")

        try:
            g = RDFGraph()
            g.parse(data=instances_content, format="turtle")

            temp_instances = tempfile.NamedTemporaryFile(mode='w', suffix='.owl', delete=False, encoding='utf-8')
            self._temp_files.append(temp_instances.name)

            owl_content = g.serialize(format="xml")
            if isinstance(owl_content, bytes):
                owl_content = owl_content.decode('utf-8')

            temp_instances.write(owl_content)
            temp_instances.close()
            g.close()

            self.world.get_ontology(f"file://{temp_instances.name}").load()

        except Exception as e:
            raise Exception(f"Failed to load instances: {str(e)}")

    def load_instances_from_file(self, file_path: str, format: str = "turtle"):
        """Load instance data from file"""
        if not self.world:
            raise RuntimeError("Validator not initialized. Use as context manager.")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.load_instances(content)
        except Exception as e:
            raise Exception(f"Failed to load instances from file {file_path}: {str(e)}")

    def validate_all_axioms(self):
        """
        Validate all types of OWL axioms automatically
        Returns: (is_valid, violations, inferred_facts)
        """
        if not self.world or not self.onto:
            raise RuntimeError("Validator not initialized. Use as context manager.")

        violations = []
        inferred_facts: List[Dict[str, Any]] = []

        try:
            with self.onto:
                sync_reasoner_pellet(
                    self.world,
                    infer_property_values=True,
                    infer_data_property_values=True
                )

                for individual in self.world.individuals():
                    if hasattr(individual, 'is_a') and hasattr(individual, '__class__'):
                        try:
                            original_classes = set(getattr(individual.__class__, 'is_a', []))
                            current_classes = set(getattr(individual, 'is_a', []))

                            for cls in current_classes - original_classes:
                                if hasattr(cls, 'name') and getattr(cls, 'name', None):
                                    inferred_facts.append({
                                        'individual': getattr(individual, 'name', str(individual)),
                                        'inferred_class': cls.name,
                                        'type': 'class_inference'
                                    })
                        except Exception:
                            continue

                violations.extend(self._check_property_violations())

                return True, violations, inferred_facts

        except OwlReadyInconsistentOntologyError:
            violations = self._extract_inconsistencies()
            return False, violations, []
        except Exception as e:
            violations.append({
                'type': 'validation_error',
                'issue': f'Validation failed: {str(e)}',
                'fix_suggestion': 'Check ontology and instance syntax'
            })
            return False, violations, []

    def _check_property_violations(self):
        """Check for property-specific violations"""
        violations = []

        try:
            for prop in self.world.properties():
                if isinstance(prop, FunctionalProperty):
                    violations.extend(self._check_functional_property(prop))
                if isinstance(prop, InverseFunctionalProperty):
                    violations.extend(self._check_inverse_functional_property(prop))
        except Exception as e:
            violations.append({
                'type': 'property_check_error',
                'issue': f'Property validation failed: {str(e)}',
                'fix_suggestion': 'Check property definitions and usage'
            })

        return violations

    def _check_functional_property(self, prop):
        """Check functional property constraints"""
        violations = []
        try:
            for individual in self.world.individuals():
                if hasattr(individual, prop.python_name):
                    values = getattr(individual, prop.python_name, [])
                    if isinstance(values, list) and len(values) > 1:
                        violations.append({
                            'type': 'functional_property_violation',
                            'individual': getattr(individual, 'name', str(individual)),
                            'property': getattr(prop, 'name', str(prop)),
                            'issue': f'Functional property has {len(values)} values, must have at most 1',
                            'fix_suggestion': f'Remove duplicate values for property {prop.name if hasattr(prop, "name") else str(prop)}'
                        })
        except Exception:
            pass
        return violations

    def _check_inverse_functional_property(self, prop):
        """Check inverse functional property constraints"""
        violations = []
        try:
            value_to_individuals: Dict[Any, List[str]] = {}

            for individual in self.world.individuals():
                if hasattr(individual, prop.python_name):
                    values = getattr(individual, prop.python_name, [])
                    if not isinstance(values, list):
                        values = [values]

                    for value in values:
                        if value not in value_to_individuals:
                            value_to_individuals[value] = []
                        value_to_individuals[value].append(getattr(individual, 'name', str(individual)))

            for value, individuals in value_to_individuals.items():
                if len(individuals) > 1:
                    violations.append({
                        'type': 'inverse_functional_property_violation',
                        'property': getattr(prop, 'name', str(prop)),
                        'value': str(value),
                        'individuals': individuals,
                        'issue': f'Inverse functional property value shared by {len(individuals)} individuals',
                        'fix_suggestion': f'Ensure property {prop.name if hasattr(prop, "name") else str(prop)} has unique values across individuals'
                    })
        except Exception:
            pass

        return violations

    def _extract_inconsistencies(self):
        """Extract detailed inconsistency information"""
        violations = []

        try:
            for cls in self.world.inconsistent_classes():
                violations.append({
                    'type': 'inconsistent_class',
                    'class': getattr(cls, 'name', str(cls)),
                    'issue': 'Class leads to logical contradiction',
                    'fix_suggestion': 'Check class axioms and individual assertions for contradictions'
                })
        except Exception:
            violations.append({
                'type': 'general_inconsistency',
                'issue': 'Ontology contains logical contradictions',
                'fix_suggestion': 'Review class hierarchies, property domains/ranges, and individual assertions'
            })

        return violations

    def _cleanup(self):
        """Clean up resources and temporary files"""
        if self._world_closed:
            return

        if self.world:
            try:
                graph = getattr(self.world, 'graph', None)
                if graph and hasattr(graph, 'close'):
                    try:
                        graph.close()
                    except Exception:
                        pass

                self.world = None
                self.onto = None
                self._world_closed = True

            except Exception as e:
                print(f"Warning: Error during world cleanup: {e}")

        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_file}: {e}")

        self._temp_files.clear()

    def __del__(self):
        """Destructor to ensure cleanup even if context manager not used"""
        try:
            self._cleanup()
        except Exception:
            pass


class SHACLValidator:
    """Comprehensive SHACL constraint validator using pyshacl."""

    def __init__(self, shapes_file: str):
        if not SHACL_VALIDATION_AVAILABLE:
            raise ImportError("pyshacl is required for SHACL validation. Install with: pip install pyshacl")

        if not Path(shapes_file).exists():
            raise FileNotFoundError(f"SHACL shapes file not found: {shapes_file}")

        self.shapes_graph = RDFGraph()
        self.shapes_graph.parse(shapes_file, format="turtle")
        self.shapes_file = shapes_file

    def validate_data(self, ttl_content: str, ontology_file: str = None):
        """
        Validate TTL content against SHACL shapes
        Returns: (conforms, summary)
        """
        data_graph = RDFGraph()
        data_graph.parse(data=ttl_content, format="turtle")
        ontology_graph = None
        if ontology_file and Path(ontology_file).exists():
            ontology_graph = RDFGraph()

            tried = False
            for fmt in ("turtle", "xml", "nt"):
                try:
                    ontology_graph.parse(ontology_file, format=fmt)
                    tried = True
                    break
                except Exception:
                    continue
            if not tried:
                ontology_graph = None

        conforms, results_graph, results_text = pyshacl.validate(
            data_graph=data_graph,
            shacl_graph=self.shapes_graph,
            ont_graph=ontology_graph,
            advanced=True,
            inference='rdfs',
            debug=False
        )

        summary = {
            "conforms": conforms,
            "validation_type": "SHACL",
            "shapes_file": self.shapes_file
        }

        if not conforms:
            violations = self._parse_validation_results(results_graph)
            summary["total_violations"] = len(violations)
            summary["violations"] = violations
        else:
            summary["total_violations"] = 0
            summary["violations"] = []

        return conforms, summary

    def _parse_validation_results(self, results_graph: RDFGraph):
        """Parse SHACL validation results into LLM-friendly format"""
        violations = []
        SHACL_NS = Namespace("http://www.w3.org/ns/shacl#")

        for result in results_graph.subjects(RDF.type, SHACL_NS.ValidationResult):
            violation: Dict[str, Any] = {}

            for focus_node in results_graph.objects(result, SHACL_NS.focusNode):
                violation['focus_node'] = self._clean_uri(str(focus_node))

            for path in results_graph.objects(result, SHACL_NS.resultPath):
                violation['property'] = self._clean_uri(str(path))

            for severity in results_graph.objects(result, SHACL_NS.resultSeverity):
                violation['severity'] = self._clean_uri(str(severity))

            for message in results_graph.objects(result, SHACL_NS.resultMessage):
                violation['message'] = str(message)

            for value in results_graph.objects(result, SHACL_NS.value):
                violation['violating_value'] = str(value)

            for constraint in results_graph.objects(result, SHACL_NS.sourceConstraintComponent):
                constraint_name = self._clean_uri(str(constraint))
                violation['constraint_type'] = constraint_name
                violation['fix_suggestion'] = self._generate_fix_suggestion(
                    constraint_name, violation
                )

            for shape in results_graph.objects(result, SHACL_NS.sourceShape):
                violation['source_shape'] = self._clean_uri(str(shape))

            violations.append(violation)

        return violations

    def _clean_uri(self, uri: str) -> str:
        """Extract the local name from a URI"""
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.split('/')[-1]
        return uri

    def _generate_fix_suggestion(self, constraint_type: str, violation: dict) -> str:
        """Generate specific fix suggestions based on constraint type"""
        property_name = violation.get('property', 'unknown property')
        focus_node = violation.get('focus_node', 'the node')

        if 'minCount' in constraint_type:
            return f"Add required values for property '{property_name}' on {focus_node}"

        elif 'maxCount' in constraint_type:
            return f"Remove excess values for property '{property_name}' on {focus_node}"

        elif 'datatype' in constraint_type:
            expected_type = self._extract_expected_datatype(violation)
            return f"Change '{property_name}' value to correct datatype: {expected_type}"

        elif 'class' in constraint_type:
            return f"Ensure {focus_node} is declared as instance of the correct class"

        elif 'nodeKind' in constraint_type:
            return f"Ensure '{property_name}' value is the correct node type (IRI, Literal, or BlankNode)"

        elif 'pattern' in constraint_type:
            return f"Ensure '{property_name}' value matches the required pattern/format"

        elif 'minLength' in constraint_type:
            return f"Increase length of '{property_name}' value"

        elif 'maxLength' in constraint_type:
            return f"Reduce length of '{property_name}' value"

        elif 'minInclusive' in constraint_type or 'minExclusive' in constraint_type:
            return f"Increase '{property_name}' value to meet minimum requirement"

        elif 'maxInclusive' in constraint_type or 'maxExclusive' in constraint_type:
            return f"Decrease '{property_name}' value to meet maximum requirement"

        elif 'in' in constraint_type:
            return f"Use one of the allowed values for '{property_name}'"

        elif 'hasValue' in constraint_type:
            return f"Set '{property_name}' to the required specific value"

        else:
            return f"Fix {constraint_type} constraint violation for '{property_name}'"

    def _extract_expected_datatype(self, violation: dict) -> str:
        """Extract expected datatype from violation context"""
        return "expected datatype (check SHACL shape for details)"

    def get_shape_info(self):
        """Get information about available SHACL shapes"""
        SHACL_NS = Namespace("http://www.w3.org/ns/shacl#")
        shapes_info: Dict[str, Any] = {}

        for shape in self.shapes_graph.subjects(RDF.type, SHACL_NS.NodeShape):
            shape_name = self._clean_uri(str(shape))
            shape_info: Dict[str, Any] = {"uri": str(shape), "type": "NodeShape"}

            for target_class in self.shapes_graph.objects(shape, SHACL_NS.targetClass):
                shape_info["target_class"] = self._clean_uri(str(target_class))

            properties: List[str] = []
            for prop_shape in self.shapes_graph.objects(shape, SHACL_NS.property):
                for path in self.shapes_graph.objects(prop_shape, SHACL_NS.path):
                    properties.append(self._clean_uri(str(path)))
            shape_info["properties"] = properties

            shapes_info[shape_name] = shape_info

        return shapes_info