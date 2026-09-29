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

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/quarterly.csv",
                 parse_dates=["quarter"]).set_index("quarter")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
TAB = ROOT / "tables"; TAB.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
results = {}

NW_LAGS = 4


def ols_nw(y, X, lags=NW_LAGS):
    X = sm.add_constant(X)
    m = sm.OLS(y, X, missing="drop").fit(cov_type="HAC",
                                         cov_kwds={"maxlags": lags})
    return m


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
data_d = df.loc["1955-01-01":, ["pi_gap_accel", "ugap"]].dropna()
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
qa_data = df.loc["1960-01-01":"2019-10-01", ["pi_gap_accel", "ugap"]].dropna()
n = len(qa_data)
trim = int(0.15 * n)
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

# With a persistent u-gap and HAC variances from short edge regimes, the
# sup-Wald rejects too often at the asymptotic critical values in samples of
# this size. A fixed-regressor moving-block bootstrap of the no-break model
# gives a finite-sample p-value. It uses its own generator, so the section E
# bootstrap draws are unchanged.
qa_y, qa_x = qa_data["pi_gap_accel"].values, qa_data["ugap"].values
qa_null = sm.OLS(qa_y, sm.add_constant(qa_x)).fit()
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
    "p_bootstrap": round(float(np.mean(qa_boot >= sup_w)), 4),
    "boot_crit_5pct": round(float(np.quantile(qa_boot, 0.95)), 2),
    "boot_crit_1pct": round(float(np.quantile(qa_boot, 0.99)), 2),
    "B": QA_B, "block_length": QA_BLOCK})

# =========================================================================
# E. Modern sample: linear vs kinked PC in v/u, 2001Q1-2026Q1
# =========================================================================
mod = df.loc["2001-01-01":"2026-01-01"].copy()
mod["pi_gap"] = mod["pi_qa_corecpi"] - mod["cle1y"]
mdat = mod[["pi_gap", "vu", "gscpi"]].dropna()
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
ssrs = {}
for c in grid:
    Xc = sm.add_constant(pd.DataFrame(
        {"below": np.minimum(mdat["vu"] - c, 0),
         "above": np.maximum(mdat["vu"] - c, 0),
         "gscpi": mdat["gscpi"]}, index=mdat.index))
    ssrs[c] = sm.OLS(y, Xc).fit().ssr
c_hat = min(ssrs, key=ssrs.get)
m_kink = kink_fit(y, mdat, c_hat)
results["modern_kink"] = reg_record(m_kink, f"kink at vu={c_hat}")
results["modern_kink"]["c_hat"] = float(c_hat)
results["modern_kink"]["slope_ratio"] = round(
    m_kink.params["above"] / m_kink.params["below"], 2)
results["modern_kink"]["quarters_above_kink"] = [
    f"{q.year}Q{q.quarter}" for q in mdat.index[mdat["vu"] > c_hat]]

# sup-LR test of linear vs kink, moving-block bootstrap under the null
X_lin = sm.add_constant(mdat[["vu", "gscpi"]])
fit_lin_plain = sm.OLS(y, X_lin).fit()
ssr_lin = fit_lin_plain.ssr
ssr_kink = min(ssrs.values())
nobs = len(y)
supLR = nobs * np.log(ssr_lin / ssr_kink)

B, block = 499, 8
resid = fit_lin_plain.resid.values
fitted = fit_lin_plain.fittedvalues.values
supLR_boot = []
nblocks = int(np.ceil(nobs / block))
for b in range(B):
    starts = rng.integers(0, nobs - block + 1, nblocks)
    eb = np.concatenate([resid[s:s + block] for s in starts])[:nobs]
    yb = fitted + eb
    ssr_lin_b = sm.OLS(yb, X_lin).fit().ssr
    ssr_k_b = np.inf
    for c in grid[::2]:                      # coarser grid inside bootstrap
        Xc = sm.add_constant(pd.DataFrame(
            {"below": np.minimum(mdat["vu"] - c, 0),
             "above": np.maximum(mdat["vu"] - c, 0),
             "gscpi": mdat["gscpi"]}, index=mdat.index))
        ssr_k_b = min(ssr_k_b, sm.OLS(yb, Xc).fit().ssr)
    supLR_boot.append(nobs * np.log(ssr_lin_b / ssr_k_b))
p_boot = float(np.mean(np.array(supLR_boot) >= supLR))
results["kink_test"] = {"supLR": round(float(supLR), 2),
                        "p_bootstrap": round(p_boot, 4), "B": B,
                        "block_length": block,
                        "aic_linear": round(fit_lin_plain.aic, 1),
                        "aic_kink": round(sm.OLS(y, sm.add_constant(
                            pd.DataFrame({"below": np.minimum(mdat["vu"] - c_hat, 0),
                                          "above": np.maximum(mdat["vu"] - c_hat, 0),
                                          "gscpi": mdat["gscpi"]},
                                         index=mdat.index))).fit().aic, 1)}

# Approximate LR search set for the threshold (stored as c_ci95): grid points
# whose LR distance from the minimum is below Hansen's (2000) 7.35 cutoff. That
# cutoff is derived for discontinuous thresholds, so the set's coverage is not
# established for this continuous kink.
lr_c = {c: nobs * np.log(s / ssr_kink) for c, s in ssrs.items()}
ci_set = [c for c, v in lr_c.items() if v <= 7.35]
results["kink_test"]["c_ci95"] = [min(ci_set), max(ci_set)]
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
def oos(train_end, test_start, test_end):
    tr = mdat.loc[:train_end]
    te = mdat.loc[test_start:test_end]
    ytr = tr["pi_gap"]
    # linear
    ml = sm.OLS(ytr, sm.add_constant(tr[["vu", "gscpi"]])).fit()
    pl = ml.predict(sm.add_constant(te[["vu", "gscpi"]]))
    # kink: re-estimate threshold on training data only
    ssr_tr = {}
    for c in grid:
        Xc = sm.add_constant(pd.DataFrame(
            {"below": np.minimum(tr["vu"] - c, 0),
             "above": np.maximum(tr["vu"] - c, 0),
             "gscpi": tr["gscpi"]}, index=tr.index))
        ssr_tr[c] = sm.OLS(ytr, Xc).fit().ssr
    c_tr = min(ssr_tr, key=ssr_tr.get)
    Xk_tr = sm.add_constant(pd.DataFrame(
        {"below": np.minimum(tr["vu"] - c_tr, 0),
         "above": np.maximum(tr["vu"] - c_tr, 0),
         "gscpi": tr["gscpi"]}, index=tr.index))
    mk = sm.OLS(ytr, Xk_tr).fit()
    Xk_te = sm.add_constant(pd.DataFrame(
        {"below": np.minimum(te["vu"] - c_tr, 0),
         "above": np.maximum(te["vu"] - c_tr, 0),
         "gscpi": te["gscpi"]}, index=te.index))
    Xk_te = Xk_te.reindex(columns=Xk_tr.columns, fill_value=1.0)
    pk = mk.predict(Xk_te)
    actual = te["pi_gap"]
    e_cle = mod.loc[te.index, "cle1y"]
    out = pd.DataFrame({"actual_infl": actual + e_cle,
                        "pred_linear": pl + e_cle,
                        "pred_kink": pk + e_cle})
    rmse_l = float(np.sqrt(((actual - pl) ** 2).mean()))
    rmse_k = float(np.sqrt(((actual - pk) ** 2).mean()))
    return {"train_end": train_end, "c_trained": float(c_tr),
            "rmse_linear": round(rmse_l, 3), "rmse_kink": round(rmse_k, 3),
            "rmse_ratio": round(rmse_k / rmse_l, 3),
            "kink_slope_above_trained": round(mk.params["above"], 3),
            "path": out.round(3).reset_index().assign(
                quarter=lambda d: d["quarter"].astype(str)).to_dict("records")}

results["oos_pre2020"] = oos("2019-10-01", "2020-01-01", "2026-01-01")
results["oos_pre2023"] = oos("2022-10-01", "2023-01-01", "2026-01-01")

# =========================================================================
# H. Robustness grid: inflation measure x expectations x supply control
# =========================================================================
rob = []
for infl in ["pi_qa_corecpi", "pi_qa_corepce", "pi_qa_cpi"]:
    for exp_ in ["cle1y", "mich", "cle10y"]:
        for sup in ["gscpi", "rel_energy"]:
            sub = mod[[infl, exp_, "vu", sup]].dropna()
            yy = sub[infl] - sub[exp_]
            ssr_r = {}
            for c in grid:
                Xc = sm.add_constant(pd.DataFrame(
                    {"below": np.minimum(sub["vu"] - c, 0),
                     "above": np.maximum(sub["vu"] - c, 0),
                     "sup": sub[sup]}, index=sub.index))
                ssr_r[c] = sm.OLS(yy, Xc).fit().ssr
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
