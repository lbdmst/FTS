import json
import unittest
from pathlib import Path

import atomic_timeseries as ats


ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "primitive_catalog.json"


class PrimitiveCatalogTests(unittest.TestCase):
    def test_catalog_matches_public_add_primitives(self) -> None:
        self.assertTrue(CATALOG_PATH.exists(), "primitive_catalog.json must be generated")
        catalog = json.loads(CATALOG_PATH.read_text())
        catalog_names = {entry["function_name"] for entry in catalog}
        public_names = {name for name in ats.__all__ if name.startswith("add_")}
        self.assertEqual(catalog_names, public_names)

    def test_catalog_entries_have_required_fields(self) -> None:
        required_fields = {
            "function_name",
            "signature",
            "category",
            "short_description",
            "core_semantic_tags",
            "what_it_does",
            "what_it_does_not_do",
            "required_parameters",
            "optional_parameters",
            "exact_effect_on_series",
            "behavior",
            "effect_type",
            "mutability",
            "required_arg_patterns",
            "mutually_exclusive_args",
            "recommended_usage",
            "forbidden_usage_patterns",
            "typical_caption_cues",
            "minimal_usage_example",
            "canonical_examples",
        }
        catalog = json.loads(CATALOG_PATH.read_text())
        for entry in catalog:
            self.assertTrue(required_fields.issubset(entry.keys()), entry["function_name"])
            self.assertTrue(entry["core_semantic_tags"], entry["function_name"])
            self.assertEqual(entry["mutability"], "returns_new_series")
            self.assertIn(entry["effect_type"], {"overwrite", "additive"})
            self.assertIn("initial_series", entry["minimal_usage_example"])
            self.assertIn("call", entry["minimal_usage_example"])
            self.assertIn("result_series", entry["minimal_usage_example"])
            self.assertIn("valid", entry["canonical_examples"])
            self.assertIn("invalid", entry["canonical_examples"])


if __name__ == "__main__":
    unittest.main()
