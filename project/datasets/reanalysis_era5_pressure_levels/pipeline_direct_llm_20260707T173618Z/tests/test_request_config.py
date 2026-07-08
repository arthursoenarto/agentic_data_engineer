import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PIPELINE_DIR / "config" / "request_config.json"
PIPELINE_PATH = PIPELINE_DIR / "pipeline.py"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("generated_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RequestConfigTests(unittest.TestCase):
    def test_exact_field_selector_combinations_are_preserved(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)

        combos = {
            (item["field"], item["selector"]["dimension"], item["selector"]["value"], item["selector"]["unit"])
            for item in config["selected_field_selector_combinations"]
        }
        self.assertEqual(
            combos,
            {
                ("temperature", "pressure_level", "500", "hPa"),
                ("temperature", "pressure_level", "850", "hPa"),
                ("geopotential", "pressure_level", "500", "hPa"),
            },
        )

    def test_retrieve_groups_do_not_widen_geopotential_to_850(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)

        requested_pairs = set()
        for group in config["retrieve_groups"]:
            for variable in group["request"]["variable"]:
                for pressure_level in group["request"]["pressure_level"]:
                    requested_pairs.add((variable, pressure_level))

        self.assertIn(("temperature", "500"), requested_pairs)
        self.assertIn(("temperature", "850"), requested_pairs)
        self.assertIn(("geopotential", "500"), requested_pairs)
        self.assertNotIn(("geopotential", "850"), requested_pairs)

    def test_access_context_credential_names_only(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)

        env_specs = config["access_context"]["credential_env_vars"]
        names = {spec["env_var"] for spec in env_specs}
        aliases = {alias for spec in env_specs for alias in spec.get("aliases", [])}
        self.assertEqual(names, {"CDSAPI_URL", "CDSAPI_KEY"})
        self.assertEqual(aliases, {"CDS_PERSONAL_ACCESS_TOKEN", "CDS_API_TOKEN"})

    def test_dotenv_loader_does_not_override_shell_environment(self):
        module = load_pipeline_module()
        old_value = os.environ.get("CDSAPI_URL")
        try:
            os.environ["CDSAPI_URL"] = "shell-value-must-win"
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                child = tmp_path / "child"
                child.mkdir()
                (tmp_path / ".env").write_text("CDSAPI_URL=dotenv-parent\n", encoding="utf-8")
                (child / ".env").write_text("CDSAPI_URL=dotenv-child\n", encoding="utf-8")
                module.load_dotenv_from_ancestors(child)
            self.assertEqual(os.environ["CDSAPI_URL"], "shell-value-must-win")
        finally:
            if old_value is None:
                os.environ.pop("CDSAPI_URL", None)
            else:
                os.environ["CDSAPI_URL"] = old_value


if __name__ == "__main__":
    unittest.main()
