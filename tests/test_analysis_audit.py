"""Independent numerical and boundary checks for the estimation audit."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analysis_audit", ROOT / "code" / "02_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def sample(n=81):
    rng = np.random.default_rng(418)
    index = pd.date_range("2000-01-01", periods=n, freq="QS", name="quarter")
    x = rng.uniform(0.35, 1.65, n)
    z = rng.normal(size=n)
    # Deliberately use a threshold that is absent from a coarser every-other grid.
    y = 0.8 + 0.6 * np.minimum(x - 0.91, 0) + 3.1 * np.maximum(x - 0.91, 0)
    y += 0.25 * z + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"pi_gap": y, "vu": x, "gscpi": z}, index=index)


class EstimationValidationTests(unittest.TestCase):
    def test_import_does_not_read_inputs_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory) / "code"
            code.mkdir()
            target = code / "02_analysis.py"
            shutil.copyfile(ROOT / "code" / target.name, target)
            spec = importlib.util.spec_from_file_location("isolated_analysis", target)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"code"})

    def test_hac_matches_statsmodels_independently(self):
        data = sample()
        actual = analysis.ols_nw(data.pi_gap, data[["vu", "gscpi"]])
        expected = sm.OLS(data.pi_gap, sm.add_constant(data[["vu", "gscpi"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        np.testing.assert_allclose(actual.params, expected.params, rtol=1e-12)
        np.testing.assert_allclose(actual.cov_params(), expected.cov_params(), rtol=1e-12)

    def test_rejects_missing_internal_quarter_before_hac(self):
        data = sample()
        data.loc[data.index[30], "vu"] = np.nan
        with self.assertRaisesRegex(ValueError, "consecutive quarter"):
            analysis.ols_nw(data.pi_gap, data[["vu", "gscpi"]])

    def test_allows_missing_endpoints_before_hac(self):
        data = sample()
        data.loc[data.index[0], "vu"] = np.nan
        result = analysis.ols_nw(data.pi_gap, data[["vu", "gscpi"]])
        self.assertEqual(result.nobs, len(data) - 1)

    def test_rejects_duplicate_unsorted_and_nonquarter_dates(self):
        data = sample()
        bad_indices = [data.index[::-1], data.index[:-1].append(data.index[-2:-1]),
                       data.index + pd.Timedelta(days=1)]
        for index in bad_indices:
            with self.subTest(index=index[:2]):
                invalid = data.copy()
                invalid.index = index
                with self.assertRaises(ValueError):
                    analysis.validate_quarters(invalid)

    def test_rejects_infinite_values_and_unidentified_regression(self):
        data = sample()
        invalid = data.copy()
        invalid.loc[invalid.index[20], "pi_gap"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            analysis.ols_nw(invalid.pi_gap, invalid[["vu", "gscpi"]])
        data["gscpi"] = 1.0
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            analysis.ols_nw(data.pi_gap, data[["vu", "gscpi"]])

    def test_rejects_misaligned_outcome(self):
        data = sample()
        with self.assertRaisesRegex(ValueError, "identical quarter"):
            analysis.ols_nw(data.pi_gap.iloc[::-1], data[["vu", "gscpi"]])

    def test_search_excludes_unidentified_hinges(self):
        data = sample()
        scores = analysis.kink_search(data.pi_gap, data, [0.1, 0.91, 2.0])
        self.assertEqual(list(scores), [0.91])
        with self.assertRaisesRegex(ValueError, "no threshold"):
            analysis.kink_search(data.pi_gap, data, [0.1, 2.0])

    def test_qr_ssr_matches_independent_lstsq(self):
        data = sample()
        design = np.column_stack([np.ones(len(data)), data.vu, data.gscpi])
        outcomes = np.column_stack([data.pi_gap, data.pi_gap * 2 + 1])
        coefficients = np.linalg.lstsq(design, outcomes, rcond=None)[0]
        expected = ((outcomes - design @ coefficients) ** 2).sum(axis=0)
        np.testing.assert_allclose(analysis.residual_sums_of_squares(design, outcomes),
                                   expected, rtol=1e-12)


class BootstrapTests(unittest.TestCase):
    def test_full_grid_bootstrap_matches_independent_statsmodels_oracle(self):
        data = sample(61)
        grid = np.array([0.6, 0.91, 1.2])
        B, block, seed = 23, 8, 312
        observed, simulated = analysis.kink_bootstrap(
            data.pi_gap, data, grid, B=B, block=block, seed=seed)
        n = len(data)
        X = np.column_stack([np.ones(n), data.vu, data.gscpi])
        fit = sm.OLS(data.pi_gap.to_numpy(), X).fit()
        rng = np.random.default_rng(seed)
        outcomes = [data.pi_gap.to_numpy()]
        for _ in range(B):
            starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
            residuals = np.concatenate([fit.resid[s:s + block] for s in starts])[:n]
            outcomes.append(fit.fittedvalues + residuals)
        expected = []
        for outcome in outcomes:
            linear_ssr = sm.OLS(outcome, X).fit().ssr
            candidates = []
            for c in grid:
                kink = np.column_stack([np.ones(n), np.minimum(data.vu - c, 0),
                                        np.maximum(data.vu - c, 0), data.gscpi])
                candidates.append(sm.OLS(outcome, kink).fit().ssr)
            expected.append(n * np.log(linear_ssr / min(candidates)))
        np.testing.assert_allclose(np.r_[observed, simulated], expected, atol=1e-10)

    def test_add_one_pvalue_includes_ties_and_cannot_report_zero(self):
        self.assertEqual(analysis.bootstrap_pvalue(4, [1, 2, 3]), 0.25)
        self.assertEqual(analysis.bootstrap_pvalue(2, [1, 2, 3]), 0.75)
        self.assertEqual(analysis.bootstrap_pvalue(0, [1, 2, 3]), 1.0)
        for bad in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                analysis.bootstrap_pvalue(1, bad)

    def test_invalid_bootstrap_sizes_fail(self):
        data = sample()
        for options in ({"B": 0}, {"block": 0}, {"block": len(data) + 1}):
            with self.assertRaises(ValueError):
                analysis.kink_bootstrap(data.pi_gap, data, [0.91], **options)

    def test_exact_fit_does_not_produce_roundoff_driven_lr(self):
        data = sample()
        exact = 2 + 3 * data.vu + 0.5 * data.gscpi
        with self.assertRaisesRegex(ValueError, "numerical roundoff"):
            analysis.kink_bootstrap(exact, data, [0.6, 0.91, 1.2], B=9)


class HoldoutTests(unittest.TestCase):
    def test_one_quarter_holdout_keeps_intercept_and_matches_manual_fit(self):
        data = sample()
        mod = pd.DataFrame({"cle1y": 2.0}, index=data.index)
        grid = [0.6, 0.91, 1.2]
        result = analysis.oos(data, mod, grid, "2019-10-01", "2020-01-01", "2020-01-01")
        training, testing = data.iloc[:-1], data.iloc[-1:]
        X = np.column_stack([np.ones(len(training)), training.vu, training.gscpi])
        b = np.linalg.lstsq(X, training.pi_gap, rcond=None)[0]
        expected = np.array([1, testing.vu.iloc[0], testing.gscpi.iloc[0]]) @ b + 2
        self.assertAlmostEqual(result["path"][0]["pred_linear"], expected, delta=0.00051)
        self.assertEqual(len(result["path"]), 1)

    def test_holdout_outcomes_cannot_change_training_or_predictions(self):
        data = sample()
        mod = pd.DataFrame({"cle1y": 2.0}, index=data.index)
        options = (mod, [0.6, 0.91, 1.2], "2017-10-01", "2018-01-01", "2020-01-01")
        original = analysis.oos(data, *options)
        changed = data.copy()
        changed.loc["2018-01-01":, "pi_gap"] += 100
        repeated = analysis.oos(changed, *options)
        self.assertEqual(original["c_trained"], repeated["c_trained"])
        self.assertEqual(original["kink_slope_above_trained"], repeated["kink_slope_above_trained"])
        for first, second in zip(original["path"], repeated["path"]):
            self.assertEqual(first["pred_linear"], second["pred_linear"])
            self.assertEqual(first["pred_kink"], second["pred_kink"])
            self.assertAlmostEqual(second["actual_infl"] - first["actual_infl"], 100)

    def test_rejects_overlapping_or_empty_holdout(self):
        data = sample()
        mod = pd.DataFrame({"cle1y": 2.0}, index=data.index)
        for dates in (("2018-01-01", "2018-01-01", "2020-01-01"),
                      ("2020-01-01", "2021-01-01", "2021-01-01")):
            with self.assertRaises(ValueError):
                analysis.oos(data, mod, [0.91], *dates)


if __name__ == "__main__":
    unittest.main()
