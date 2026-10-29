"""Regression checks for safe demo generation and analytical outputs."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("audit_" + name, ROOT / "code" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoIsolationAuditTests(unittest.TestCase):
    def test_invalid_seed_preserves_earlier_demo(self):
        demo = load_script("run_demo.py")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / demo.MARKER).write_text("earlier run", encoding="utf-8")
            (workspace / "results").mkdir()
            saved = workspace / "results" / "results.json"
            saved.write_text("prior results", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "nonnegative integer"):
                demo.prepare(workspace, -1)
            self.assertEqual(saved.read_text(encoding="utf-8"), "prior results")
            self.assertEqual((workspace / demo.MARKER).read_text(encoding="utf-8"), "earlier run")

    def test_leaf_symlinks_are_rejected_before_cleanup_or_writes(self):
        demo = load_script("run_demo.py")
        for leaf in ("code/01_build_dataset.py", "data/raw/CPIAUCSL.csv"):
            with self.subTest(leaf=leaf), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                workspace = base / "demo"
                target = workspace / leaf
                target.parent.mkdir(parents=True)
                (workspace / demo.MARKER).write_text("marker", encoding="utf-8")
                (workspace / "results").mkdir()
                preserved = workspace / "results" / "prior.txt"
                preserved.write_text("prior run", encoding="utf-8")
                victim = base / "do-not-change.txt"
                victim.write_text("original", encoding="utf-8")
                try:
                    target.symlink_to(victim)
                except (OSError, NotImplementedError):
                    self.skipTest("symbolic links unavailable")
                with self.assertRaisesRegex(SystemExit, "points outside"):
                    demo.prepare(workspace, 42)
                self.assertEqual(victim.read_text(encoding="utf-8"), "original")
                self.assertEqual(preserved.read_text(encoding="utf-8"), "prior run")

    def test_hard_linked_marker_is_not_overwritten(self):
        demo = load_script("run_demo.py")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "demo"
            workspace.mkdir()
            victim = base / "do-not-change.txt"
            victim.write_text("original", encoding="utf-8")
            try:
                os.link(victim, workspace / demo.MARKER)
            except OSError:
                self.skipTest("hard links unavailable")
            with self.assertRaisesRegex(SystemExit, "hard-linked"):
                demo.prepare(workspace, 42)
            self.assertEqual(victim.read_text(encoding="utf-8"), "original")


class FigureAuditTests(unittest.TestCase):
    def test_terminal_recession_month_is_fully_shaded_and_missing_stops_span(self):
        # Load this pure plotting helper without producing the seven figures.
        tree = ast.parse((ROOT / "code" / "03_figures.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "shade_recessions")
        namespace = {"pd": pd, "rec": pd.Series([1.0, float("nan"), 1.0],
                     index=pd.date_range("2026-01-01", periods=3, freq="MS"))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "shade", "exec"), namespace)
        spans = []

        class Axis:
            def axvspan(self, start, end, **kwargs):
                spans.append((start, end))

        namespace["shade_recessions"](Axis(), "2026-01-01", "2026-03-01")
        self.assertEqual(spans, [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")),
                                 (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-01"))])


class TableAuditTests(unittest.TestCase):
    def test_synthetic_label_null_ratio_and_searched_threshold_aic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "code").mkdir()
            (root / "results").mkdir()
            (root / "SYNTHETIC_DEMO.txt").write_text("synthetic", encoding="utf-8")
            source = ROOT / "code" / "04_tables.py"
            script = root / "code" / source.name
            shutil.copyfile(source, script)
            result = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
            result["modern_kink"]["slope_ratio"] = None
            result["kink_test"]["aic_kink_conditional"] = 251.8
            (root / "results" / "results.json").write_text(json.dumps(result), encoding="utf-8")
            runpy.run_path(str(script))
            tables = list((root / "tables").glob("*.md"))
            self.assertEqual(len(tables), 5)
            for table in tables:
                self.assertTrue(table.read_text(encoding="utf-8").startswith("**SYNTHETIC DEMO:"))
            table3 = (root / "tables" / "table3_kink.md").read_text(encoding="utf-8")
            self.assertIn("counts the searched threshold", table3)
            self.assertNotIn("None", table3)




if __name__ == "__main__":
    unittest.main()
