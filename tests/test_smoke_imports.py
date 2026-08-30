from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = sorted((ROOT / "src").glob("*.py")) + sorted(
    (ROOT / "legacy").glob("**/*.py")
)


class ScriptSmokeTests(unittest.TestCase):
    def test_all_scripts_are_valid_python(self) -> None:
        for path in SCRIPT_PATHS:
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")

    def test_all_scripts_are_importable_without_network_calls(self) -> None:
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for path in SCRIPT_PATHS:
                with self.subTest(path=path.relative_to(ROOT)):
                    self._import_script(path)
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode

    def _import_script(self, path: Path) -> None:
        module_name = "smoke_" + "_".join(path.relative_to(ROOT).with_suffix("").parts)
        spec = importlib.util.spec_from_file_location(module_name, path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

        script_dir = str(path.parent)
        old_path = list(sys.path)
        old_queries = sys.modules.pop("queries", None)
        try:
            sys.path.insert(0, script_dir)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
            sys.modules.pop(module_name, None)
            sys.modules.pop("queries", None)
            if old_queries is not None:
                sys.modules["queries"] = old_queries


if __name__ == "__main__":
    unittest.main()
