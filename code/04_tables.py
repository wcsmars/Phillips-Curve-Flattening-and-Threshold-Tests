"""Render paper tables (markdown) from results.json into tables/."""
import json
import math
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
TAB = ROOT / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def stars(p):
    if p is None or not math.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(b, se, p):
    return f"{number(b, '.3f')}{stars(p)} ({number(se, '.3f')})"


def number(value, spec):
    return "—" if value is None or not math.isfinite(value) else format(value, spec)


def write_table(name, content):
    if (ROOT / "SYNTHETIC_DEMO.txt").is_file():
        content = "**SYNTHETIC DEMO: simulated inputs; not U.S. empirical results.**\n\n" + content
    (TAB / name).write_text(content + "\n", encoding="utf-8", newline="\n")


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def markdown(frame):
    """Write formatted cells verbatim; tabulate would turn "+3.50" into "3.5".

    Columns made only of numbers stay right-aligned, as before.
    """
    align = ["right" if all(is_number(v) for v in frame[c]) else "left" for c in frame.columns]
    return frame.to_markdown(index=False, disable_numparse=True, colalign=align)


def quarter(iso_date):
    stamp = pd.Timestamp(iso_date)
    return f"{stamp.year}Q{stamp.quarter}"


# Table 1: descriptives -------------------------------------------------------
labels = {"pi_yoy_cpi": "Headline CPI inflation (y/y, %)",
          "pi_yoy_corecpi": "Core CPI inflation (y/y, %)",
          "pi_yoy_corepce": "Core PCE inflation (y/y, %)",
          "u": "Unemployment rate (%)",
          "ugap": "Unemployment gap (pp)",
          "vu": "Vacancy/unemployment ratio",
          "mich": "Michigan 1y expected inflation (%)",
          "cle1y": "Cleveland Fed 1y expected inflation (%)",
          "gscpi": "Global Supply Chain Pressure Index (sd)"}
rows = []
for v, lab in labels.items():
    row = {"Variable": lab}
    for per, d in res["descriptives"].items():
        s = d[v]
        row[per] = ("—" if s["mean"] is None or s["sd"] is None
                    else f"{s['mean']:.2f} ({s['sd']:.2f})")
    rows.append(row)
t1 = pd.DataFrame(rows)
t1_md = ("**Table 1. Descriptive statistics by period — mean (standard deviation), quarterly data**\n\n"
         + markdown(t1)
         + "\n\n*Notes:* v/u available from 2000Q4 (JOLTS); Michigan from 1978; Cleveland Fed from 1982; GSCPI from 1998. 2020–2026Q1 column ends with the last complete quarter (2026Q1).")
write_table("table1_descriptives.md", t1_md)

# Table 2: linear Phillips curves --------------------------------------------
rows = []
for r in res["panelA_accelerationist"]:
    rows.append({"Sample": r["name"],
                 "Slope on u-gap": cell(r["params"]["ugap"], r["se"]["ugap"], r["pvalues"]["ugap"]),
                 "Supply control": "—", "N": r["nobs"], "R²": f"{r['r2']:.3f}"})
for r in res["panelA_supply"]:
    rows.append({"Sample": r["name"] + " (w/ supply)",
                 "Slope on u-gap": cell(r["params"]["ugap"], r["se"]["ugap"], r["pvalues"]["ugap"]),
                 "Supply control": cell(r["params"]["rel_energy"], r["se"]["rel_energy"], r["pvalues"]["rel_energy"]),
                 "N": r["nobs"], "R²": f"{r['r2']:.3f}"})
t2a = pd.DataFrame(rows)
rows = []
for r in res["panelC_survey"]:
    rows.append({"Sample": r["name"],
                 "Slope on u-gap": cell(r["params"]["ugap"], r["se"]["ugap"], r["pvalues"]["ugap"]),
                 "Supply control": cell(r["params"]["rel_energy"], r["se"]["rel_energy"], r["pvalues"]["rel_energy"]),
                 "N": r["nobs"], "R²": f"{r['r2']:.3f}"})
t2c = pd.DataFrame(rows)
qa = res["quandt_andrews"]
t2_md = ("**Table 2. Linear Phillips-curve estimates across eras**\n\n"
         "*Panel A — accelerationist specification: π(headline CPI, q/q ann.) − mean of past 4 quarters = α + κ·(u − u\\*) [+ γ·relative energy inflation]*\n\n"
         + markdown(t2a) + "\n\n"
         "*Panel B — survey-expectations specification: π(core CPI, q/q ann.) − Michigan 1y expectation = α + κ·(u − u\\*) + γ·relative energy inflation*\n\n"
         + markdown(t2c) + "\n\n"
         f"*Notes:* Newey–West (HAC) standard errors with 4 lags in parentheses; \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10. "
         f"Quandt–Andrews sup-Wald test for an unknown break in κ (1960–2019, 15% trimming): sup-W = {qa['sup_wald']:.2f}, "
         f"estimated break {quarter(qa['break_date'])}; asymptotic 1% critical value {qa['crit_1pct']:.2f} (Andrews 1993, 2003). "
         f"An approximate fixed-regressor moving-block bootstrap imposes a stable intercept and slope under the null: "
         f"p = {qa['p_bootstrap']:.3f} (B = {qa['B']}, block length {qa['block_length']}). "
         f"Its size is not established under arbitrary intercept breaks or changing conditional volatility.")
write_table("table2_linear_pc.md", t2_md)

# Table 3: linear vs kinked v/u curve -----------------------------------------
lin, kink, kt = res["modern_linear"], res["modern_kink"], res["kink_test"]
aic_note = ("AIC counts the searched threshold as one fitted parameter; this is a heuristic "
            "comparison under weak threshold identification. " if "aic_kink_conditional" in kt else
            "AIC treats ĉ as fixed and does not penalize the threshold search. ")
t3 = pd.DataFrame([
    {"": "v/u (linear)", "(1) Linear": cell(lin["params"]["vu"], lin["se"]["vu"], lin["pvalues"]["vu"]), "(2) Kinked": "—"},
    {"": "v/u below kink", "(1) Linear": "—", "(2) Kinked": cell(kink["params"]["below"], kink["se"]["below"], kink["pvalues"]["below"])},
    {"": "v/u above kink", "(1) Linear": "—", "(2) Kinked": cell(kink["params"]["above"], kink["se"]["above"], kink["pvalues"]["above"])},
    {"": "GSCPI", "(1) Linear": cell(lin["params"]["gscpi"], lin["se"]["gscpi"], lin["pvalues"]["gscpi"]),
     "(2) Kinked": cell(kink["params"]["gscpi"], kink["se"]["gscpi"], kink["pvalues"]["gscpi"])},
    {"": "Constant", "(1) Linear": cell(lin["params"]["const"], lin["se"]["const"], lin["pvalues"]["const"]),
     "(2) Kinked": cell(kink["params"]["const"], kink["se"]["const"], kink["pvalues"]["const"])},
    {"": "Kink location ĉ", "(1) Linear": "—", "(2) Kinked": f"{kink['c_hat']:.2f} [approx. LR set {kt['c_ci95'][0]:.2f}–{kt['c_ci95'][1]:.2f}]"},
    {"": "Slope ratio (above/below)", "(1) Linear": "—", "(2) Kinked": number(kink['slope_ratio'], '.2f')},
    {"": "R²", "(1) Linear": f"{lin['r2']:.3f}", "(2) Kinked": f"{kink['r2']:.3f}"},
    {"": "AIC", "(1) Linear": f"{kt['aic_linear']:.1f}", "(2) Kinked": f"{kt['aic_kink']:.1f}"},
    {"": "N", "(1) Linear": str(lin["nobs"]), "(2) Kinked": str(kink["nobs"])},
])
t3_md = ("**Table 3. Linear vs. kinked Phillips curve in labor-market tightness, 2001Q1–2026Q1**\n\n"
         "*Dependent variable: core CPI inflation (q/q annualized) − Cleveland Fed 1-year expected inflation*\n\n"
         + markdown(t3) + "\n\n"
         f"*Notes:* Newey–West (4 lags) standard errors in parentheses; \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10. "
         f"Kink located by grid search over v/u ∈ [0.40, 1.40]; approximate LR search set uses the Hansen (2000) cutoff, "
         f"whose coverage is not established here for a continuous kink. Slope standard errors condition on the selected threshold. "
         + aic_note
         + f"sup-LR test of linearity = {kt['supLR']:.2f}, moving-block bootstrap p-value = {kt['p_bootstrap']:.2f} "
         f"(B = {kt['B']}, block length {kt['block_length']}).")
write_table("table3_kink.md", t3_md)

# Table 4: robustness grid ------------------------------------------------------
def kink_pattern(label, specs):
    """Summarise where one expectations group places the kink (from the saved results)."""
    if not specs:
        return ""
    c = sorted(x["c_hat"] for x in specs)
    centre = c[(len(c) - 1) // 2]                      # lower median, an estimated grid value
    near = [v for v in c if abs(v - centre) <= 0.02 + 1e-9]
    where = f"{near[0]:.2f}" if near[0] == near[-1] else f"{near[0]:.2f}–{near[-1]:.2f}"
    steeper = sum(x["slope_above"] > x["slope_below"] for x in specs)
    return (f"{len(near)} of the {len(specs)} specifications using {label} expectations place ĉ at "
            f"{where}; in {steeper} of {len(specs)} the slope above the kink exceeds the slope below.")


infl_lab = {"pi_qa_corecpi": "Core CPI", "pi_qa_corepce": "Core PCE", "pi_qa_cpi": "Headline CPI"}
exp_lab = {"cle1y": "Cleveland 1y", "mich": "Michigan 1y", "cle10y": "Cleveland 10y"}
sup_lab = {"gscpi": "GSCPI", "rel_energy": "Rel. energy"}
rows = [{"Inflation": infl_lab[x["inflation"]], "Expectations": exp_lab[x["expectations"]],
         "Supply": sup_lab[x["supply"]], "ĉ": f"{x['c_hat']:.2f}",
         "Slope below": f"{x['slope_below']:.2f}",
         "Slope above": f"{x['slope_above']:.2f} ({x['se_above']:.2f})",
         "R²": f"{x['r2']:.2f}"} for x in res["robustness"]]
t4_md = ("**Table 4. Robustness of the kinked specification, 2001Q1–2026Q1**\n\n"
         + markdown(pd.DataFrame(rows)) + "\n\n"
         "*Notes:* Each row re-estimates the kinked model with the stated inflation measure (q/q annualized, minus the stated "
         "expectation measure) and supply control; ĉ re-estimated by grid search per specification. Newey–West (4 lags) "
         "standard errors in parentheses; the search grid is v/u ∈ [0.40, 1.40]. "
         + kink_pattern("Cleveland Fed", [x for x in res["robustness"] if x["expectations"] != "mich"]) + " "
         + kink_pattern("Michigan household", [x for x in res["robustness"] if x["expectations"] == "mich"]) + "")
write_table("table4_robustness.md", t4_md)

# Table 5: decomposition summary -------------------------------------------------
d = res["decomposition_summary"]
lab = {"actual_pi": "Actual core CPI inflation (q/q ann.)",
       "contrib_expectations": "Expected inflation (Cleveland 1y)",
       "contrib_tightness": "Labor-market tightness (v/u terms)",
       "contrib_supply": "Supply-chain pressure (GSCPI)",
       "residual": "Residual"}
rows = [{"Component": v,
         "Surge: 2021Q2–2022Q4 vs. 2019Q4–2020Q1 (pp)": f"{d['surge_vs_prepandemic'][k]:+.2f}",
         "Disinflation: 2025Q1–2026Q1 vs. 2022Q2–2022Q4 (pp)": f"{d['disinflation_peak_to_2025'][k]:+.2f}"}
        for k, v in lab.items()]
t5_md = ("**Table 5. Accounting for the surge and the disinflation (kinked model, Table 3 column 2)**\n\n"
         + markdown(pd.DataFrame(rows)) + "\n\n"
         "*Notes:* Entries are changes in period-average fitted contributions between the stated windows; components sum to the "
         "actual change up to rounding. The constant nets out of changes.")
write_table("table5_decomposition.md", t5_md)

print("tables written:", sorted(p.name for p in TAB.glob("*.md")))
