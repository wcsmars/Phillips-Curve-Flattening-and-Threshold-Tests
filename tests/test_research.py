"""Offline checks using synthetic inputs and the included research outputs.

Run from the repository root: python -m unittest discover -s tests -v
No source-data download is needed. One test runs the full synthetic demo,
including both bootstraps, so it is the slowest check.
"""

import csv
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def markdown_cells(table):
    """Header cells and body rows of the pipe tables in a text (panels share a header)."""
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")]
             for line in table.splitlines()
             if line.startswith("|") and not set(line) <= set("|:- ")]
    return cells[0], [row for row in cells[1:] if row != cells[0]]


class DatasetBuildTests(unittest.TestCase):
    def test_transformations_and_relocated_execution(self):
        """Quarterly prices, ratios and lags use the documented definitions."""
        with tempfile.TemporaryDirectory(prefix="research fixture ") as temporary:
            base = Path(temporary)
            checkout = base / "relocated project with spaces"
            code = checkout / "code"
            raw = checkout / "data" / "raw"
            code.mkdir(parents=True)
            raw.mkdir(parents=True)
            script = code / "01_build_dataset.py"
            shutil.copyfile(ROOT / "code" / script.name, script)
            months = pd.date_range("2018-01-01", "2026-05-01", freq="MS")

            def write_series(name, values):
                pd.DataFrame({"observation_date": months, name: values}).to_csv(
                    raw / f"{name}.csv", index=False
                )

            # Quarterly means rise at known compound rates. Monthly deviations
            # cancel within a quarter, so averaging inflation instead of price
            # levels does not satisfy the expected values below.
            growth = {"CPIAUCSL": 1.01, "CPILFESL": 1.02,
                      "PCEPI": 1.015, "PCEPILFE": 1.025, "CPIENGSL": 1.03}
            # October 2025 is missing from CPIAUCSL, as in the real June 2026 vintage.
            gap = months.get_loc(pd.Timestamp("2025-10-01"))
            for name, factor in growth.items():
                write_series(name, [np.nan if name == "CPIAUCSL" and i == gap else
                                    100 * factor ** (i // 3) + (-1, 0, 1)[i % 3]
                                    for i in range(len(months))])
            monthly_patterns = {
                "UNRATE": (4, 5, 6),
                "UNEMPLOY": (100, 200, 400),
                "JTSJOL": (200, 300, 700),
                "MICH": (2, 4, 3),
                "EXPINF1YR": (1.8, 2.4, 3),
                "EXPINF10YR": (2, 2.2, 2.4),
            }
            for name, pattern in monthly_patterns.items():
                write_series(name, [pattern[i % 3] for i in range(len(months))])
            pd.DataFrame({"observation_date": pd.date_range(
                "2018-01-01", "2026-04-01", freq="QS"), "NROU": 4.25
            }).to_csv(raw / "NROU.csv", index=False)
            pd.DataFrame({"date": months + pd.offsets.MonthEnd(0),
                          "gscpi": [(-1, 0, 1)[i % 3] for i in range(len(months))]
                          }).to_csv(raw / "GSCPI.csv", index=False)

            elsewhere = base / "unrelated working directory"
            elsewhere.mkdir()
            completed = subprocess.run(
                [sys.executable, str(script)], cwd=elsewhere,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = checkout / "data" / "processed" / "quarterly.csv"
            self.assertTrue(output.is_file())
            self.assertFalse((elsewhere / "data").exists())
            self.assertIn("fewer than 3 months: CPIAUCSL 2025Q4 (2 of 3 months)\n",
                          completed.stdout)
            data = pd.read_csv(output, index_col="quarter", parse_dates=True)
            observed = data.loc["2019-07-01"]
            self.assertAlmostEqual(observed["pi_qa_cpi"], 400 * math.log(1.01))
            self.assertAlmostEqual(observed["pi_yoy_cpi"], 400 * math.log(1.01))
            self.assertAlmostEqual(observed["pi_adaptive"], 400 * math.log(1.01))
            self.assertAlmostEqual(observed["rel_energy"], 400 * math.log(1.03 / 1.02))
            self.assertAlmostEqual(observed["vu"], 1200 / 700)
            self.assertAlmostEqual(observed["u"], 5)
            self.assertAlmostEqual(observed["nrou"], 4.25)
            self.assertAlmostEqual(observed["ugap"], 0.75)
            self.assertAlmostEqual(observed["mich"], 3)
            self.assertAlmostEqual(observed["cle1y"], 2.4)
            self.assertAlmostEqual(observed["cle10y"], 2.2)
            self.assertAlmostEqual(observed["gscpi"], 0)
            self.assertTrue(pd.isna(data.loc["2018-01-01", "pi_qa_cpi"]))
            self.assertTrue(pd.isna(data.loc["2018-10-01", "pi_yoy_cpi"]))
            self.assertTrue(pd.isna(data.loc["2019-01-01", "pi_adaptive"]))
            # The snapshot deliberately ends at its last complete quarter.
            self.assertEqual(data.index[-1], pd.Timestamp("2026-01-01"))
            self.assertNotIn(pd.Timestamp("2026-04-01"), data.index)


class TableGenerationTests(unittest.TestCase):
    def test_null_descriptive_values_render_in_fresh_output_directory(self):
        with tempfile.TemporaryDirectory(prefix="research tables ") as temporary:
            base = Path(temporary)
            checkout = base / "relocated table project"
            code = checkout / "code"
            results = checkout / "results"
            code.mkdir(parents=True)
            results.mkdir()
            script = code / "04_tables.py"
            shutil.copyfile(ROOT / "code" / script.name, script)
            shutil.copyfile(ROOT / "results" / "results.json", results / "results.json")
            elsewhere = base / "unrelated working directory"
            elsewhere.mkdir()
            # Tables contain characters such as ĉ, κ and −, so file I/O that relies
            # on the platform default encoding fails on Windows (cp1252).
            # PYTHONIOENCODING keeps pandas from probing the locale under an ASCII console.
            completed = subprocess.run(
                [sys.executable, "-X", "warn_default_encoding",
                 "-W", "error::EncodingWarning", str(script)], cwd=elsewhere,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            tables = checkout / "tables"
            self.assertEqual({path.name for path in tables.glob("*.md")}, {
                "table1_descriptives.md", "table2_linear_pc.md", "table3_kink.md",
                "table4_robustness.md", "table5_decomposition.md",
            })
            self.assertFalse((elsewhere / "tables").exists())
            table = (tables / "table1_descriptives.md").read_text(encoding="utf-8")
            self.assertNotRegex(table, r"\b(?:None|NaN|nan|null)\b")
            header, body = markdown_cells(table)
            rows = {row[0]: row for row in body}
            saved = json.loads((results / "results.json").read_text(encoding="utf-8"))
            missing_cells = 0
            for key, label in {
                "vu": "Vacancy/unemployment ratio",
                "gscpi": "Global Supply Chain Pressure Index (sd)",
            }.items():
                for period, statistics in saved["descriptives"].items():
                    if statistics[key]["mean"] is None or statistics[key]["sd"] is None:
                        self.assertEqual(rows[label][header.index(period)], "—")
                        missing_cells += 1
            self.assertGreater(missing_cells, 0, "Fixture must exercise unavailable statistics")

            # Formatted cells are written verbatim: signs and trailing zeros survive.
            formats = {
                "table2_linear_pc.md": {"R²": r"\d\.\d{3}"},
                "table4_robustness.md": {"ĉ": r"\d\.\d{2}", "Slope below": r"-?\d+\.\d{2}",
                                         "R²": r"\d\.\d{2}"},
                "table5_decomposition.md": {None: r"[+-]\d+\.\d{2}"},
            }
            for name, columns in formats.items():
                header, body = markdown_cells((tables / name).read_text(encoding="utf-8"))
                for column, pattern in columns.items():
                    positions = [header.index(column)] if column else range(1, len(header))
                    for row in body:
                        for position in positions:
                            with self.subTest(table=name, row=row[0], column=header[position]):
                                self.assertRegex(row[position], rf"^{pattern}$")
            notes = (tables / "table2_linear_pc.md").read_text(encoding="utf-8")
            brk = pd.Timestamp(saved["quandt_andrews"]["break_date"])
            self.assertIn(f"estimated break {brk.year}Q{brk.quarter};", notes)


class SyntheticDemoTests(unittest.TestCase):
    def run_demo(self, script, workspace, *options):
        options = options or ("--prepare-only",)
        return subprocess.run(
            [sys.executable, str(script), *options, "--workspace", str(workspace)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
        )

    def test_prepared_inputs_use_source_formats_and_build_offline(self):
        spec = importlib.util.spec_from_file_location(
            "fetch_data_for_demo", ROOT / "code" / "00_fetch_data.py")
        fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch)
        with tempfile.TemporaryDirectory(prefix="research demo ") as temporary:
            workspace = Path(temporary) / "demo workspace"
            completed = self.run_demo(ROOT / "code" / "run_demo.py", workspace)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for series in (*fetch.FRED_SERIES, "GSCPI"):
                with self.subTest(series=series):
                    fetch.validate_csv((workspace / "data" / "raw" / f"{series}.csv").read_bytes(),
                                       series)
            built = subprocess.run(
                [sys.executable, str(workspace / "code" / "01_build_dataset.py")],
                cwd=temporary, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=120,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            # JOLTS starts in December 2000, so 2000Q4 v/u uses one month of vacancies.
            self.assertIn("JTSJOL 2000Q4 (1 of 3 months)", built.stdout)
            data = pd.read_csv(workspace / "data" / "processed" / "quarterly.csv",
                               index_col="quarter", parse_dates=True)
            self.assertEqual(data.index[-1], pd.Timestamp("2026-01-01"))
            self.assertEqual(data["vu"].first_valid_index(), pd.Timestamp("2000-10-01"))
            modern = data.loc["2001-01-01":, ["pi_qa_corecpi", "cle1y", "vu", "gscpi"]]
            self.assertTrue(modern.notna().all().all())
            # Both sides of the planted kink at v/u = 1 must be populated.
            self.assertGreaterEqual(int((modern["vu"] > 1).sum()), 10)
            self.assertGreaterEqual(int((modern["vu"] < 1).sum()), 10)

    def test_full_demo_recovers_planted_kink(self):
        """Runs estimation, figures and tables on synthetic data with a known kink."""
        with tempfile.TemporaryDirectory(prefix="research demo run ") as temporary:
            workspace = Path(temporary) / "demo workspace"
            completed = self.run_demo(ROOT / "code" / "run_demo.py", workspace, "--check")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            saved = json.loads((workspace / "results" / "results.json").read_text(encoding="utf-8"))
            low, high = saved["kink_test"]["c_ci95"]
            self.assertLessEqual(abs(saved["modern_kink"]["c_hat"] - 1.0), 0.25)
            self.assertTrue(low <= 1.0 <= high)
            self.assertLess(saved["kink_test"]["p_bootstrap"], 0.05)
            self.assertIn("Recovery check passed", completed.stdout)
            # Default-seed pin for the slope-break bootstrap: 0.135 when the pseudo-data
            # are built on the no-break fit, far higher if they are built on the data.
            qa = saved["quandt_andrews"]
            self.assertAlmostEqual(qa["p_bootstrap"], 0.135, delta=0.05)
            self.assertEqual(qa["B"], 1999)
            self.assertEqual(len(list((workspace / "figures").glob("*.png"))), 7)
            self.assertEqual(len(list((workspace / "tables").glob("*.md"))), 5)

    def test_new_inputs_clear_earlier_outputs(self):
        with tempfile.TemporaryDirectory(prefix="research demo reuse ") as temporary:
            workspace = Path(temporary) / "demo workspace"
            script = ROOT / "code" / "run_demo.py"
            self.assertEqual(self.run_demo(script, workspace).returncode, 0)
            stale = workspace / "results" / "results.json"
            stale.parent.mkdir()
            stale.write_text("{}", encoding="utf-8")
            completed = self.run_demo(script, workspace, "--prepare-only", "--seed", "7")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertFalse(stale.exists())
            self.assertIn("seed 7", (workspace / "SYNTHETIC_DEMO.txt").read_text(encoding="utf-8"))

    def test_check_requires_full_run(self):
        with tempfile.TemporaryDirectory(prefix="research demo flags ") as temporary:
            workspace = Path(temporary) / "demo workspace"
            completed = self.run_demo(ROOT / "code" / "run_demo.py", workspace,
                                      "--check", "--prepare-only")
            self.assertEqual(completed.returncode, 2)
            self.assertIn("--check needs the full run", completed.stderr)
            self.assertFalse(workspace.exists())

    def test_refuses_symlinks_out_of_the_workspace(self):
        with tempfile.TemporaryDirectory(prefix="research demo link ") as temporary:
            base = Path(temporary)
            workspace, outside = base / "demo workspace", base / "outside"
            (outside / "processed").mkdir(parents=True)
            (outside / "processed" / "keep.csv").write_text("keep", encoding="utf-8")
            workspace.mkdir()
            (workspace / "SYNTHETIC_DEMO.txt").write_text("marker", encoding="utf-8")
            try:
                (workspace / "data").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symbolic links are not available")
            completed = self.run_demo(ROOT / "code" / "run_demo.py", workspace)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("points outside the demo workspace", completed.stderr)
            self.assertTrue((outside / "processed" / "keep.csv").exists())
            self.assertFalse((outside / "raw").exists())

    def test_refuses_checkout_and_foreign_folders(self):
        with tempfile.TemporaryDirectory(prefix="research demo guard ") as temporary:
            checkout = Path(temporary) / "checkout copy"
            (checkout / "code").mkdir(parents=True)
            script = checkout / "code" / "run_demo.py"
            shutil.copyfile(ROOT / "code" / "run_demo.py", script)
            foreign = Path(temporary) / "existing folder"
            foreign.mkdir()
            (foreign / "notes.txt").write_text("keep", encoding="utf-8")
            for target in (checkout, foreign):
                with self.subTest(target=target.name):
                    completed = self.run_demo(script, target)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("run_demo:", completed.stderr)
            self.assertEqual(sorted(path.name for path in checkout.iterdir()), ["code"])
            self.assertEqual(sorted(path.name for path in foreign.iterdir()), ["notes.txt"])


class SavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "results" / "results.json").open(encoding="utf-8") as source:
            cls.results = json.load(source)
        cls.decomposition = pd.read_csv(
            ROOT / "results" / "decomposition.csv",
            index_col="quarter", parse_dates=True,
        )

    def test_decomposition_reconciles(self):
        data = self.decomposition
        self.assertFalse(data.empty)
        self.assertTrue(data.index.is_unique)
        self.assertTrue(data.index.is_monotonic_increasing)
        components = ["contrib_expectations", "contrib_tightness",
                      "contrib_supply", "contrib_const"]
        self.assertTrue(np.isfinite(data[components + ["actual_pi", "fitted_pi", "residual"]]).all().all())
        # Saved CSV columns are rounded independently to three decimals.
        np.testing.assert_allclose(data[components].sum(axis=1), data["fitted_pi"],
                                   atol=0.003, rtol=0)
        np.testing.assert_allclose(data["fitted_pi"] + data["residual"], data["actual_pi"],
                                   atol=0.0016, rtol=0)
        np.testing.assert_allclose(data["contrib_expectations"], data["cle1y"],
                                   atol=0.001, rtol=0)

    def test_decomposition_summary_matches_window_averages(self):
        windows = {
            "surge_vs_prepandemic": (("2021-04-01", "2022-10-01"),
                                     ("2019-10-01", "2020-01-01")),
            "disinflation_peak_to_2025": (("2025-01-01", "2026-01-01"),
                                         ("2022-04-01", "2022-10-01")),
        }
        for name, (later, earlier) in windows.items():
            with self.subTest(window=name):
                change = (self.decomposition.loc[slice(*later)].mean()
                          - self.decomposition.loc[slice(*earlier)].mean())
                for component, saved in self.results["decomposition_summary"][name].items():
                    # The summary is rounded to two decimals.
                    self.assertAlmostEqual(change[component], saved, delta=0.006)

    def test_holdout_rmse_matches_saved_projection_paths(self):
        for name in ("oos_pre2020", "oos_pre2023"):
            with self.subTest(evaluation=name):
                result = self.results[name]
                path = pd.DataFrame(result["path"])
                self.assertFalse(path.empty)
                dates = pd.to_datetime(path["quarter"])
                self.assertTrue(dates.is_unique)
                self.assertTrue(dates.is_monotonic_increasing)
                self.assertTrue((dates > pd.Timestamp(result["train_end"])).all())
                values = path[["actual_infl", "pred_linear", "pred_kink"]]
                self.assertTrue(np.isfinite(values).all().all())
                recomputed = {}
                for model in ("linear", "kink"):
                    errors = path["actual_infl"] - path[f"pred_{model}"]
                    recomputed[model] = float(np.sqrt(np.mean(errors ** 2)))
                    self.assertAlmostEqual(recomputed[model], result[f"rmse_{model}"],
                                           delta=0.0015)
                self.assertGreater(recomputed["linear"], 0)
                self.assertAlmostEqual(recomputed["kink"] / recomputed["linear"],
                                       result["rmse_ratio"], delta=0.002)

    def test_selected_threshold_minimizes_saved_grid(self):
        with (ROOT / "results" / "kink_grid_ssr.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))[1:]
        grid = {float(threshold): float(ssr) for threshold, ssr in rows}
        self.assertTrue(grid)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in grid.values()))
        selected = self.results["modern_kink"]["c_hat"]
        self.assertIn(selected, grid)
        self.assertAlmostEqual(grid[selected], min(grid.values()), places=8)
        self.assertGreaterEqual(self.results["kink_test"]["p_bootstrap"], 0)
        self.assertLessEqual(self.results["kink_test"]["p_bootstrap"], 1)

    def test_readme_results_table_matches_saved_results(self):
        """The README quotes saved estimates; regenerated results must be re-quoted."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        if "\n## Results" not in readme:
            self.skipTest("this README has no Results table")
        section = readme.split("\n## Results", 1)[1].split("\n## ", 1)[0]
        _, body = markdown_cells(section)
        prose = " ".join(section.split())
        r = self.results
        slopes = {x["name"]: x["params"]["ugap"] for x in r["panelA_accelerationist"]}
        qa, lin, kink, test = r["quandt_andrews"], r["modern_linear"], r["modern_kink"], r["kink_test"]
        brk = pd.Timestamp(qa["break_date"])
        expected = {  # row label prefix -> numbers quoted in that row, in order
            "Unemployment-gap slope": [slopes["1960Q1-1989Q4"], slopes["1990Q1-2019Q4"]],
            "Quandt–Andrews slope-break date": [brk.year, brk.quarter, qa["sup_wald"], qa["crit_1pct"],
                                                qa["p_bootstrap"]],
            "Linear vacancy/unemployment slope": [lin["params"]["vu"], lin["tvalues"]["vu"]],
            "Estimated vacancy/unemployment kink": [kink["c_hat"], *test["c_ci95"]],
            "Slope above / slope below": [kink["slope_ratio"]],
            "Bootstrap test of linearity": [test["p_bootstrap"]],
            "Holdout RMSE, trained through 2022Q4": [r["oos_pre2023"]["rmse_linear"],
                                                     r["oos_pre2023"]["rmse_kink"]],
            "Holdout RMSE, trained through 2019Q4": [r["oos_pre2020"]["rmse_linear"],
                                                     r["oos_pre2020"]["rmse_kink"],
                                                     r["oos_pre2020"]["c_trained"]],
        }
        for prefix, values in expected.items():
            with self.subTest(row=prefix):
                rows = [text for label, text in body if label.startswith(prefix)]
                self.assertEqual(len(rows), 1, "README results row missing or duplicated")
                # Numbers other than percentages, e.g. "−0.5554", "1983Q2", "0.40–1.40".
                quoted = re.findall(r"[−-]?\d+(?:\.\d+)?(?!\d|\.\d|%)", rows[0])
                self.assertEqual(len(quoted), len(values), quoted)
                for text, value in zip(quoted, values):
                    # Each quote must be the saved value rounded to the quoted precision.
                    decimals = len(text.partition(".")[2])
                    self.assertEqual(text.replace("−", "-"), f"{value:.{decimals}f}")
        # Numbers quoted in the prose around the table.
        pre2023, pre2020 = r["oos_pre2023"], r["oos_pre2020"]
        path = pd.DataFrame(pre2020["path"]).set_index("quarter").loc["2021-04-01":"2022-10-01"]
        shortfall = path["actual_infl"].to_numpy()[:, None] - path[["pred_linear", "pred_kink"]].to_numpy()
        share = r["decomposition_summary"]["disinflation_peak_to_2025"]
        for phrase in (
            f"use {lin['nobs']} quarters from 2001Q1 to 2026Q1",
            f"over the {len(pre2023['path'])} quarters from 2023Q1 to 2026Q1",
            f"roughly {round(100 * (1 - pre2023['rmse_ratio']))}% lower",
            f"inflation by roughly {shortfall.min():.1f} to {shortfall.max():.0f} percentage points",
            f"roughly {round(100 * share['contrib_tightness'] / share['actual_pi'])}% of the inflation decline",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prose)


if __name__ == "__main__":
    unittest.main()
