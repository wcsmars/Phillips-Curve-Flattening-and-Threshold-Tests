"""Phillips-curve estimation: linear, rolling, structural-break, and threshold.

Reads  data/processed/quarterly.csv
Writes results/results.json and supporting CSVs

Sections
  A. Descriptive statistics
  B. Long-sample accelerationist Phillips curve (1960-2026), sub-samples
  C. Survey-expectations Phillips curve (1978-2026)
  D. Rolling 60-quarter slope + Quandt-Andrews sup-Wald break test
     (asymptotic critical values and a moving-block-bootstrap p-value)
  E. Modern sample (2001-2026): linear vs. kinked Phillips curve in v/u,
     grid-search threshold + moving-block-bootstrap sup-LR inference
  F. Historical decomposition of 2020-2026 inflation
  G. Out-of-sample tests (train pre-2020 / pre-2023, predict forward)
  H. Robustness grid (inflation measure x expectations proxy x supply control)
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

NW_LAGS = 4


def validate_quarters(frame, label="sample"):
    """HAC lags and resampling blocks require consecutive calendar quarters."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans:
        raise ValueError(f"{label} needs a nonempty quarterly DatetimeIndex")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{label} quarters must be unique and increasing")
    expected = pd.date_range(index[0], index[-1], freq="QS")
    if not index.equals(expected):
        raise ValueError(f"{label} must have consecutive quarter-start dates")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} contains infinite values")


def complete_sample(frame, label="sample"):
    """Allow missing endpoints, but never compress an internal time-series gap."""
    sample = frame.dropna()
    validate_quarters(sample, label)
    return sample


def with_constant(X):
    # The default 'skip' silently omits the intercept for a one-row holdout,
    # or whenever a regressor happens to be constant in the prediction window.
    return sm.add_constant(X, has_constant="add")


def require_identified(X):
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("regression design must be a finite matrix")
    if values.shape[0] <= values.shape[1]:
        raise ValueError("regression needs positive residual degrees of freedom")
    if np.linalg.matrix_rank(values) != values.shape[1]:
        raise ValueError("regression design is rank deficient")


def ols_nw(y, X, lags=NW_LAGS):
    if not y.index.equals(X.index):
        raise ValueError("outcome and regressors must have identical quarter indices")
    joint = complete_sample(pd.concat([y.rename("__outcome__"), X], axis=1))
    X = with_constant(joint.drop(columns="__outcome__"))
    require_identified(X)
    m = sm.OLS(joint["__outcome__"], X, missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def kink_design(xdat, c, supply="gscpi"):
    return with_constant(pd.DataFrame({
        "below": np.minimum(xdat["vu"] - c, 0),
        "above": np.maximum(xdat["vu"] - c, 0),
        "gscpi": xdat[supply]}, index=xdat.index))


def kink_search(y, xdat, grid, supply="gscpi"):
    """Search only estimable hinges; unpopulated regimes do not identify a kink."""
    scores = {}
    for c in grid:
        Xc = kink_design(xdat, c, supply)
        try:
            require_identified(Xc)
        except ValueError:
            continue
        scores[c] = sm.OLS(y, Xc, missing="raise").fit().ssr
    if not scores:
        raise ValueError("no threshold has an identified kink design")
    return scores


def bootstrap_pvalue(observed, simulated):
    """Add-one Monte Carlo convention; a finite simulation cannot establish p=0."""
    simulated = np.asarray(simulated, dtype=float)
    if simulated.ndim != 1 or simulated.size == 0 or not np.isfinite(simulated).all():
        raise ValueError("bootstrap statistics must be a nonempty finite vector")
    if not np.isfinite(observed):
        raise ValueError("observed bootstrap statistic must be finite")
    return float((1 + np.count_nonzero(simulated >= observed)) / (simulated.size + 1))


def residual_sums_of_squares(design, outcomes):
    """Exact least-squares SSR for one or many outcomes, using a stable QR basis."""
    require_identified(design)
    values = np.asarray(outcomes, dtype=float)
    Q, _ = np.linalg.qr(np.asarray(design, dtype=float), mode="reduced")
    residual = values - Q @ (Q.T @ values)
    return (residual ** 2).sum(axis=0)


def kink_bootstrap(y, xdat, grid, B=499, block=8, seed=42):
    """Recompute exactly the same threshold search for data and every null draw."""
    nobs = len(y)
    if not 1 <= block <= nobs or B < 1:
        raise ValueError("bootstrap needs positive replications and a block within the sample")
    X = with_constant(xdat[["vu", "gscpi"]])
    require_identified(X)
    fitted = sm.OLS(y, X, missing="raise").fit()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, nobs - block + 1, (B, int(np.ceil(nobs / block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :nobs]
    simulated = (fitted.fittedvalues.to_numpy()[:, None]
                 + fitted.resid.to_numpy()[indices.T])
    # First column is the observed outcome; all columns use the identical grid.
    outcomes = np.column_stack([np.asarray(y), simulated])
    linear_ssr = residual_sums_of_squares(X, outcomes)
    kink_ssr = np.full(B + 1, np.inf)
    scores = kink_search(y, xdat, grid)
    for c in scores:
        kink_ssr = np.minimum(kink_ssr, residual_sums_of_squares(
            kink_design(xdat, c), outcomes))
    # A mathematically exact fit usually leaves tiny floating-point residuals.
    # Taking their ratio produces a spurious LR statistic driven by roundoff.
    roundoff = (np.finfo(float).eps * nobs) ** 2 * (outcomes ** 2).sum(axis=0)
    if (linear_ssr <= roundoff).any() or (kink_ssr <= roundoff).any():
        raise ValueError("LR statistic requires residual variance above numerical roundoff")
    statistics = np.maximum(nobs * np.log(linear_ssr / kink_ssr), 0)
    return float(statistics[0]), statistics[1:]


def reg_record(m, name):
    return {
        "name": name,
        "params": {k: round(v, 4) for k, v in m.params.items()},
        "se": {k: round(v, 4) for k, v in m.bse.items()},
        "tvalues": {k: round(v, 4) for k, v in m.tvalues.items()},
        "pvalues": {k: round(v, 4) for k, v in m.pvalues.items()},
        "nobs": int(m.nobs), "r2": round(m.rsquared, 4),
        "r2_adj": round(m.rsquared_adj, 4),
    }


def oos(mdat, mod, grid, train_end, test_start, test_end):
    validate_quarters(mdat, "holdout source")
    if pd.Timestamp(test_start) <= pd.Timestamp(train_end):
        raise ValueError("holdout must begin strictly after training ends")
    tr = mdat.loc[:train_end]
    te = mdat.loc[test_start:test_end]
    validate_quarters(tr, "training sample")
    validate_quarters(te, "holdout sample")
    if tr.isna().any().any() or te.isna().any().any():
        raise ValueError("training and holdout inputs must be complete")
    ytr = tr["pi_gap"]
    Xl_tr = with_constant(tr[["vu", "gscpi"]])
    require_identified(Xl_tr)
    ml = sm.OLS(ytr, Xl_tr, missing="raise").fit()
    pl = ml.predict(with_constant(te[["vu", "gscpi"]]))
    # Re-estimate the threshold using only training observations.
    ssr_tr = kink_search(ytr, tr, grid)
    c_tr = min(ssr_tr, key=ssr_tr.get)
    Xk_tr = kink_design(tr, c_tr)
    mk = sm.OLS(ytr, Xk_tr, missing="raise").fit()
    Xk_te = kink_design(te, c_tr)
    pk = mk.predict(Xk_te)
    actual = te["pi_gap"]
    e_cle = mod.loc[te.index, "cle1y"]
    if not np.isfinite(e_cle).all():
        raise ValueError("holdout expectations must be finite")
    out = pd.DataFrame({"actual_infl": actual + e_cle,
                        "pred_linear": pl + e_cle,
                        "pred_kink": pk + e_cle})
    rmse_l = float(np.sqrt(((actual - pl) ** 2).mean()))
    rmse_k = float(np.sqrt(((actual - pk) ** 2).mean()))
    return {"train_end": train_end, "c_trained": float(c_tr),
            "rmse_linear": round(rmse_l, 3), "rmse_kink": round(rmse_k, 3),
            "rmse_ratio": round(rmse_k / rmse_l, 3) if rmse_l > 0 else None,
            "kink_slope_above_trained": round(mk.params["above"], 3),
            "path": out.round(3).reset_index().assign(
                quarter=lambda d: d["quarter"].astype(str)).to_dict("records")}


def main():
    ROOT = Path(__file__).resolve().parents[1]
    df = pd.read_csv(ROOT / "data/processed/quarterly.csv",
                     parse_dates=["quarter"]).set_index("quarter")
    RES = ROOT / "results"; RES.mkdir(exist_ok=True)
    TAB = ROOT / "tables"; TAB.mkdir(exist_ok=True)
    validate_quarters(df, "processed input")
    results = {}

    # =========================================================================
    # A. Descriptive statistics
    # =========================================================================
    desc_vars = ["pi_yoy_cpi", "pi_yoy_corecpi", "pi_yoy_corepce", "u", "ugap",
                 "vu", "mich", "cle1y", "gscpi"]
    periods = {"1960-1989": ("1960-01-01", "1989-10-01"),
               "1990-2019": ("1990-01-01", "2019-10-01"),
               "2020-2026Q1": ("2020-01-01", "2026-01-01")}
    desc = {}
    for pname, (a, b) in periods.items():
        sub = df.loc[a:b, desc_vars]
        desc[pname] = {v: {"mean": round(sub[v].mean(), 2),
                           "sd": round(sub[v].std(), 2),
                           "min": round(sub[v].min(), 2),
                           "max": round(sub[v].max(), 2),
                           "n": int(sub[v].notna().sum())} for v in desc_vars}
    results["descriptives"] = desc

    # =========================================================================
    # B. Long-sample accelerationist PC:  pi_qa_cpi - pi_adaptive = a + k*ugap
    # =========================================================================
    df["pi_gap_accel"] = df["pi_qa_cpi"] - df["pi_adaptive"]
    samples = {"1960Q1-1989Q4": ("1960-01-01", "1989-10-01"),
               "1990Q1-2019Q4": ("1990-01-01", "2019-10-01"),
               "2020Q1-2026Q1": ("2020-01-01", "2026-01-01"),
               "full 1960Q1-2026Q1": ("1960-01-01", "2026-01-01")}
    panelA = []
    for name, (a, b) in samples.items():
        sub = df.loc[a:b]
        m = ols_nw(sub["pi_gap_accel"], sub[["ugap"]])
        panelA.append(reg_record(m, name))
    results["panelA_accelerationist"] = panelA

    # with supply control (relative energy inflation), full + subsamples
    panelA_supply = []
    for name, (a, b) in samples.items():
        sub = df.loc[a:b]
        m = ols_nw(sub["pi_gap_accel"], sub[["ugap", "rel_energy"]])
        panelA_supply.append(reg_record(m, name))
    results["panelA_supply"] = panelA_supply

    # =========================================================================
    # C. Survey-expectations PC: pi_qa_corecpi - mich = a + k*ugap (+ supply)
    # =========================================================================
    df["pi_gap_mich"] = df["pi_qa_corecpi"] - df["mich"]
    samplesC = {"1978Q1-1989Q4": ("1978-01-01", "1989-10-01"),
                "1990Q1-2019Q4": ("1990-01-01", "2019-10-01"),
                "2020Q1-2026Q1": ("2020-01-01", "2026-01-01"),
                "full 1978Q1-2026Q1": ("1978-01-01", "2026-01-01")}
    panelC = []
    for name, (a, b) in samplesC.items():
        sub = df.loc[a:b]
        m = ols_nw(sub["pi_gap_mich"], sub[["ugap", "rel_energy"]])
        panelC.append(reg_record(m, name))
    results["panelC_survey"] = panelC

    # =========================================================================
    # D1. Rolling 60-quarter slope (accelerationist spec)
    # =========================================================================
    window = 60
    roll = []
    data_d = complete_sample(df.loc["1955-01-01":"2026-01-01", ["pi_gap_accel", "ugap"]], "rolling sample")
    for i in range(window, len(data_d) + 1):
        sub = data_d.iloc[i - window:i]
        m = ols_nw(sub["pi_gap_accel"], sub[["ugap"]])
        roll.append({"end": str(sub.index[-1].date()),
                     "kappa": round(m.params["ugap"], 4),
                     "se": round(m.bse["ugap"], 4)})
    results["rolling_slope"] = roll
    pd.DataFrame(roll).to_csv(RES / "rolling_slope.csv", index=False, lineterminator="\n")

    # =========================================================================
    # D2. Quandt-Andrews sup-Wald test for a break in the slope, 1960-2019
    # =========================================================================
    qa_data = complete_sample(df.loc["1960-01-01":"2019-10-01", ["pi_gap_accel", "ugap"]], "break sample")
    n = len(qa_data)
    trim = int(np.ceil(0.15 * n))
    breaks = range(trim, n - trim + 1)  # j = first post-break quarter
    if not breaks[0] == trim == n - breaks[-1]:
        raise RuntimeError("each regime must keep at least 15% of the quarters")
    sup_w, sup_date = -np.inf, None
    qa_stats = []
    for j in breaks:
        d = (np.arange(n) >= j).astype(float)
        X = pd.DataFrame({"ugap": qa_data["ugap"].values,
                          "D": d, "D_ugap": d * qa_data["ugap"].values},
                         index=qa_data.index)
        m = ols_nw(qa_data["pi_gap_accel"], X)
        w = float(m.wald_test("D_ugap = 0", scalar=True).statistic)
        qa_stats.append(w)
        if w > sup_w:
            sup_w, sup_date = w, str(qa_data.index[j].date())
    # Andrews (1993)/(2003) asymptotic critical values, p=1, 15% trimming
    results["quandt_andrews"] = {"sup_wald": round(sup_w, 2),
                                 "break_date": sup_date,
                                 "crit_10pct": 7.17, "crit_5pct": 8.85,
                                 "crit_1pct": 12.35,
                                 "note": "sup-Wald on slope-shift dummy, HAC(4); "
                                         "Andrews (1993, 2003) critical values"}

    # Calibrate the scan under a fixed-regressor no-break model with stable
    # intercept and slope. Noncircular moving residual blocks approximate serial
    # dependence; they do not establish exact finite-sample size or robustness
    # to changing conditional variance, endogeneity, or arbitrary intercept
    # breaks. An independent generator preserves section E's random draws.
    qa_y, qa_x = qa_data["pi_gap_accel"].values, qa_data["ugap"].values
    qa_null = sm.OLS(qa_y, with_constant(qa_x)).fit()
    qa_designs = []
    for j in breaks:
        d = (np.arange(n) >= j).astype(float)
        Xj = np.column_stack([np.ones(n), qa_x, d, d * qa_x])
        XtXi = np.linalg.inv(Xj.T @ Xj)
        qa_designs.append((Xj, XtXi @ Xj.T, Xj @ XtXi[-1]))


    def wald_by_date(Y, lags=NW_LAGS):
        """HAC Wald statistic for D_ugap = 0 at every break date (rows), one column
        per column of Y; the same Bartlett estimator as ols_nw (no df correction)."""
        stats = []
        for Xj, P, g in qa_designs:
            coef = P @ Y
            U = g[:, None] * (Y - Xj @ coef)
            v = (U * U).sum(0)
            for lag in range(1, lags + 1):
                v += 2 * (1 - lag / (lags + 1)) * (U[lag:] * U[:-lag]).sum(0)
            stats.append(coef[-1] ** 2 / v)
        return np.array(stats)


    if not np.allclose(wald_by_date(qa_y[:, None])[:, 0], qa_stats, rtol=1e-8):
        raise RuntimeError("bootstrap Wald statistic differs from the statsmodels one")
    qa_rng = np.random.default_rng(1983)
    QA_B, QA_BLOCK = 1999, 8
    qa_starts = qa_rng.integers(0, n - QA_BLOCK + 1, (QA_B, int(np.ceil(n / QA_BLOCK))))
    qa_index = (qa_starts[:, :, None] + np.arange(QA_BLOCK)).reshape(QA_B, -1)[:, :n]
    qa_boot = wald_by_date(qa_null.fittedvalues[:, None] + qa_null.resid[qa_index.T]).max(0)
    results["quandt_andrews"].update({
        "p_bootstrap": round(bootstrap_pvalue(sup_w, qa_boot), 4),
        "pvalue_method": "(1 + exceedances) / (B + 1)",
        "boot_crit_5pct": round(float(np.quantile(qa_boot, 0.95)), 2),
        "boot_crit_1pct": round(float(np.quantile(qa_boot, 0.99)), 2),
        "B": QA_B, "block_length": QA_BLOCK})

    # =========================================================================
    # E. Modern sample: linear vs kinked PC in v/u, 2001Q1-2026Q1
    # =========================================================================
    mod = df.loc["2001-01-01":"2026-01-01"].copy()
    mod["pi_gap"] = mod["pi_qa_corecpi"] - mod["cle1y"]
    mdat = complete_sample(mod[["pi_gap", "vu", "gscpi"]], "modern sample")
    y = mdat["pi_gap"]

    m_lin = ols_nw(y, mdat[["vu", "gscpi"]])
    results["modern_linear"] = reg_record(m_lin, "linear 2001-2026")

    # Horse race on the same quarters: the unemployment gap in place of v/u.
    # The post-2020 RMSEs are in-sample residuals of these full-sample fits.
    m_ugap = ols_nw(y, pd.concat([mod.loc[mdat.index, "ugap"], mdat["gscpi"]], axis=1))
    if int(m_ugap.nobs) != int(m_lin.nobs):
        raise ValueError("u-gap horse race must use the v/u quarters; ugap is missing in "
                         + ", ".join(f"{q.year}Q{q.quarter}" for q in
                                     mdat.index[mod.loc[mdat.index, "ugap"].isna()]))
    results["modern_ugap"] = reg_record(m_ugap, "linear u-gap 2001-2026")


    def rmse_post2020(resid):
        resid = resid[resid.index >= pd.Timestamp("2020-01-01")]
        return round(float(np.sqrt(np.mean(resid ** 2))), 3)


    results["modern_ugap"]["rmse_post2020_in_sample"] = {
        "vu": rmse_post2020(m_lin.resid), "ugap": rmse_post2020(m_ugap.resid)}


    def kink_fit(ydat, xdat, c):
        X = pd.DataFrame({"below": np.minimum(xdat["vu"] - c, 0),
                          "above": np.maximum(xdat["vu"] - c, 0),
                          "gscpi": xdat["gscpi"]}, index=xdat.index)
        return ols_nw(ydat, X)


    grid = np.round(np.arange(0.40, 1.41, 0.01), 2)
    ssrs = kink_search(y, mdat, grid)
    c_hat = min(ssrs, key=ssrs.get)
    m_kink = kink_fit(y, mdat, c_hat)
    results["modern_kink"] = reg_record(m_kink, f"kink at vu={c_hat}")
    results["modern_kink"]["c_hat"] = float(c_hat)
    results["modern_kink"]["slope_ratio"] = (
        round(m_kink.params["above"] / m_kink.params["below"], 2)
        if m_kink.params["below"] != 0 else None)
    results["modern_kink"]["quarters_above_kink"] = [
        f"{q.year}Q{q.quarter}" for q in mdat.index[mdat["vu"] > c_hat]]

    # sup-LR test of linear vs kink, moving-block bootstrap under the null
    X_lin = with_constant(mdat[["vu", "gscpi"]])
    fit_lin_plain = sm.OLS(y, X_lin).fit()
    ssr_kink = min(ssrs.values())
    nobs = len(y)
    B, block = 499, 8
    supLR, supLR_boot = kink_bootstrap(y, mdat, grid, B=B, block=block)
    p_boot = bootstrap_pvalue(supLR, supLR_boot)
    # The searched threshold is an additional fitted parameter. This AIC
    # remains a heuristic because standard regular-model theory is not assured.
    conditional_aic = sm.OLS(y, kink_design(mdat, c_hat)).fit().aic
    results["kink_test"] = {"supLR": round(float(supLR), 2),
                            "p_bootstrap": round(p_boot, 4), "B": B,
                            "pvalue_method": "(1 + exceedances) / (B + 1)",
                            "block_length": block,
                            "grid_points": len(ssrs), "grid_step": 0.01,
                            "aic_linear": round(fit_lin_plain.aic, 1),
                            "aic_kink": round(conditional_aic + 2, 1),
                            "aic_kink_conditional": round(conditional_aic, 1),
                            "aic_note": "kink AIC counts the searched threshold; "
                                        "heuristic under nonregular threshold inference"}

    # Approximate LR search set for the threshold (stored as c_ci95): grid points
    # whose LR distance from the minimum is below Hansen's (2000) 7.35 cutoff. That
    # cutoff is derived for discontinuous thresholds, so the set's coverage is not
    # established for this continuous kink.
    # Hansen's normalized SSR distance, not the log-LR used for linearity.
    lr_c = {c: nobs * (s - ssr_kink) / ssr_kink for c, s in ssrs.items()}
    ci_set = [c for c, v in lr_c.items() if v <= 7.35]
    results["kink_test"]["c_ci95"] = [min(ci_set), max(ci_set)]
    results["kink_test"]["search_set_note"] = (
        "normalized SSR distance with borrowed 7.35 cutoff; "
        "not a calibrated 95% confidence set for this continuous kink")
    pd.Series(ssrs, name="ssr").rename_axis("threshold").round(10).to_csv(
        RES / "kink_grid_ssr.csv", lineterminator="\n")

    # =========================================================================
    # F. Historical decomposition 2019Q4-2026Q1 using the kinked model
    # =========================================================================
    dec = mod.loc["2019-10-01":].copy()
    b = m_kink.params
    dec["below"] = np.minimum(dec["vu"] - c_hat, 0)
    dec["above"] = np.maximum(dec["vu"] - c_hat, 0)
    dec["contrib_expectations"] = dec["cle1y"]
    dec["contrib_tightness"] = b["below"] * dec["below"] + b["above"] * dec["above"]
    dec["contrib_supply"] = b["gscpi"] * dec["gscpi"]
    dec["contrib_const"] = b["const"]
    dec["fitted_pi"] = (dec["contrib_expectations"] + dec["contrib_tightness"]
                        + dec["contrib_supply"] + dec["contrib_const"])
    dec["actual_pi"] = dec["pi_qa_corecpi"]
    dec["residual"] = dec["actual_pi"] - dec["fitted_pi"]
    keep = ["actual_pi", "fitted_pi", "contrib_expectations", "contrib_tightness",
            "contrib_supply", "contrib_const", "residual", "vu", "gscpi", "cle1y"]
    dec[keep].round(3).to_csv(RES / "decomposition.csv", lineterminator="\n")

    surge = dec.loc["2021-04-01":"2022-10-01"]      # 2021Q2-2022Q4 surge window
    calm = dec.loc["2019-10-01":"2020-01-01"]       # pre-pandemic baseline
    disinf_from = dec.loc["2022-04-01":"2022-10-01"]  # peak window
    disinf_to = dec.loc["2025-01-01":"2026-01-01"]    # latest 5 quarters
    results["decomposition_summary"] = {
        "surge_vs_prepandemic": {
            k: round(surge[k].mean() - calm[k].mean(), 2)
            for k in ["actual_pi", "contrib_expectations", "contrib_tightness",
                      "contrib_supply", "residual"]},
        "disinflation_peak_to_2025": {
            k: round(disinf_to[k].mean() - disinf_from[k].mean(), 2)
            for k in ["actual_pi", "contrib_expectations", "contrib_tightness",
                      "contrib_supply", "residual"]},
    }

    # =========================================================================
    # G. Out-of-sample: train pre-2020 and pre-2023, predict forward
    # =========================================================================

    results["oos_pre2020"] = oos(mdat, mod, grid, "2019-10-01", "2020-01-01", "2026-01-01")
    results["oos_pre2023"] = oos(mdat, mod, grid, "2022-10-01", "2023-01-01", "2026-01-01")

    # =========================================================================
    # H. Robustness grid: inflation measure x expectations x supply control
    # =========================================================================
    rob = []
    for infl in ["pi_qa_corecpi", "pi_qa_corepce", "pi_qa_cpi"]:
        for exp_ in ["cle1y", "mich", "cle10y"]:
            for sup in ["gscpi", "rel_energy"]:
                sub = complete_sample(mod[[infl, exp_, "vu", sup]], "robustness sample")
                yy = sub[infl] - sub[exp_]
                ssr_r = kink_search(yy, sub, grid, supply=sup)
                c_r = min(ssr_r, key=ssr_r.get)
                Xr = pd.DataFrame({"below": np.minimum(sub["vu"] - c_r, 0),
                                   "above": np.maximum(sub["vu"] - c_r, 0),
                                   "sup": sub[sup]}, index=sub.index)
                mr = ols_nw(yy, Xr)
                rob.append({"inflation": infl, "expectations": exp_,
                            "supply": sup, "c_hat": float(c_r),
                            "slope_below": round(mr.params["below"], 3),
                            "slope_above": round(mr.params["above"], 3),
                            "t_above": round(mr.tvalues["above"], 2),
                            "se_above": round(mr.bse["above"], 3),
                            "nobs": int(mr.nobs), "r2": round(mr.rsquared, 3)})
    results["robustness"] = rob

    def json_safe(value):
        """Represent unavailable descriptive statistics as standard JSON null."""
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value


    with open(RES / "results.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(json_safe(results), f, indent=1, allow_nan=False)
        f.write("\n")
    print("written results.json")
    print("\n--- headline numbers ---")
    print("Panel A slopes:", [(r["name"], r["params"]["ugap"], r["pvalues"]["ugap"])
                              for r in panelA])
    print("QA break:", results["quandt_andrews"])
    print("Kink:", results["modern_kink"]["params"], "c=", c_hat,
          "ratio=", results["modern_kink"]["slope_ratio"])
    print("Kink test:", results["kink_test"])
    print("OOS pre2020:", {k: results["oos_pre2020"][k] for k in
                           ["c_trained", "rmse_linear", "rmse_kink", "rmse_ratio"]})
    print("OOS pre2023:", {k: results["oos_pre2023"][k] for k in
                           ["c_trained", "rmse_linear", "rmse_kink", "rmse_ratio"]})
    print("Decomp:", json.dumps(results["decomposition_summary"], indent=1))


if __name__ == "__main__":
    main()
