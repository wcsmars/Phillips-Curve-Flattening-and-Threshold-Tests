"""Produce all paper figures (PNG, 200 dpi) into figures/."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data/processed/quarterly.csv",
                 parse_dates=["quarter"]).set_index("quarter")
res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
rec = pd.read_csv(ROOT / "data/raw/USREC.csv", na_values=["."])
rec.columns = ["date", "usrec"]
rec["date"] = pd.to_datetime(rec["date"])
rec = rec.set_index("date")["usrec"]
rec = pd.to_numeric(rec, errors="raise").sort_index()
if rec.index.has_duplicates or not rec.dropna().isin([0, 1]).all():
    raise ValueError("USREC must have unique dates and only 0, 1 or missing observations")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})
BLUE, RED, GREY, GOLD, GREEN = "#1f4e79", "#b03a2e", "#7f8c8d", "#b7950b", "#1e8449"


def save_figure(fig, path):
    if (ROOT / "SYNTHETIC_DEMO.txt").is_file():
        for ax in fig.axes:
            ax.set_title("SYNTHETIC DEMO\n" + ax.get_title())
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def shade_recessions(ax, start, end):
    r = rec.loc[start:end]
    in_rec = False
    for d, v in r.items():
        if v == 1 and not in_rec:
            rs, in_rec = d, True
        elif v != 1 and in_rec:
            ax.axvspan(rs, d, color="grey", alpha=0.18, lw=0)
            in_rec = False
    if in_rec:
        # Monthly indicators describe the full month, including the last row.
        ax.axvspan(rs, r.index[-1] + pd.offsets.MonthBegin(1),
                   color="grey", alpha=0.18, lw=0)


# Figure 1: inflation history -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
sub = df.loc["1960-01-01":]
ax.plot(sub.index, sub["pi_yoy_cpi"], color=GREY, lw=1.1, label="Headline CPI (y/y)")
ax.plot(sub.index, sub["pi_yoy_corecpi"], color=BLUE, lw=1.6, label="Core CPI (y/y)")
shade_recessions(ax, "1960-01-01", "2026-03-01")
ax.axhline(2, color="black", lw=0.6, ls=":")
ax.set_ylabel("Percent"); ax.legend(frameon=False, loc="upper right")
ax.set_title("U.S. consumer price inflation, 1960–2026")
save_figure(fig, FIG / "fig1_inflation_history.png")

# Figure 2: Phillips scatter by era ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.6))
eras = [("1960Q1–1989Q4", "1960-01-01", "1989-10-01", RED),
        ("1990Q1–2019Q4", "1990-01-01", "2019-10-01", BLUE),
        ("2020Q1–2026Q1", "2020-01-01", "2026-01-01", GOLD)]
for label, a, b, c in eras:
    s = df.loc[a:b].copy()
    s["pi_gap_accel"] = s["pi_qa_cpi"] - s["pi_adaptive"]
    s = s[["ugap", "pi_gap_accel"]].dropna()
    ax.scatter(s["ugap"], s["pi_gap_accel"], s=14, alpha=0.55, color=c, label=label)
    m = sm.OLS(s["pi_gap_accel"], sm.add_constant(s["ugap"])).fit()
    xs = np.linspace(s["ugap"].min(), s["ugap"].max(), 50)
    ax.plot(xs, m.params.iloc[0] + m.params.iloc[1] * xs, color=c, lw=2)
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("Unemployment gap (u − u*, pp)")
ax.set_ylabel("Inflation surprise (π − πᵉ, pp, annualized)")
ax.legend(frameon=False)
ax.set_title("The flattening Phillips curve: inflation vs. slack by era")
save_figure(fig, FIG / "fig2_pc_scatter_eras.png")

# Figure 3: rolling slope ------------------------------------------------------
roll = pd.DataFrame(res["rolling_slope"])
roll["end"] = pd.to_datetime(roll["end"])
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(roll["end"], roll["kappa"], color=BLUE, lw=1.8)
ax.fill_between(roll["end"], roll["kappa"] - 1.645 * roll["se"],
                roll["kappa"] + 1.645 * roll["se"], color=BLUE, alpha=0.15, lw=0)
ax.axhline(0, color="black", lw=0.7)
shade_recessions(ax, str(roll["end"].iloc[0].date()), "2026-03-01")
ax.set_ylabel("Slope $\\kappa$ on unemployment gap")
ax.set_title("Rolling 60-quarter Phillips-curve slope (90% confidence band)")
save_figure(fig, FIG / "fig3_rolling_slope.png")

# Figure 4: v/u ratio ----------------------------------------------------------
c_hat = res["modern_kink"]["c_hat"]
fig, ax = plt.subplots(figsize=(8, 3.8))
s = df.loc["2001-01-01":, "vu"].dropna()
ax.plot(s.index, s.values, color=BLUE, lw=1.8)
ax.axhline(c_hat, color=RED, lw=1.2, ls="--",
           label=f"Estimated kink ($\\hat{{c}}$ = {c_hat:.2f})")
ax.axhline(1.0, color=GREY, lw=0.8, ls=":", label="v/u = 1 (one vacancy per unemployed person)")
shade_recessions(ax, "2001-01-01", "2026-03-01")
ax.set_ylabel("Vacancies / unemployed (v/u)"); ax.legend(frameon=False, loc="upper left")
ax.set_title("Labor-market tightness, 2001–2026")
save_figure(fig, FIG / "fig4_vu_timeseries.png")

# Figure 5: kinked Phillips curve in v/u space --------------------------------
mod = df.loc["2001-01-01":"2026-01-01"].copy()
mod["pi_gap"] = mod["pi_qa_corecpi"] - mod["cle1y"]
d = mod[["pi_gap", "vu", "gscpi"]].dropna()
b = res["modern_kink"]["params"]
d["partial"] = d["pi_gap"] - b["gscpi"] * d["gscpi"]   # net of supply effect
pre = d.index < "2020-01-01"
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(d["vu"][pre], d["partial"][pre], s=16, color=BLUE, alpha=0.55,
           label="2001Q1–2019Q4")
ax.scatter(d["vu"][~pre], d["partial"][~pre], s=22, color=RED, alpha=0.75,
           label="2020Q1–2026Q1")
xs = np.linspace(d["vu"].min(), d["vu"].max(), 200)
yfit = (b["const"] + b["below"] * np.minimum(xs - c_hat, 0)
        + b["above"] * np.maximum(xs - c_hat, 0))
ax.plot(xs, yfit, color="black", lw=2, label="Kinked fit")
ax.axvline(c_hat, color=GREY, lw=0.8, ls="--")
ax.set_xlabel("Vacancy-to-unemployment ratio (v/u)")
ax.set_ylabel("Core CPI inflation − expected (net of supply), pp")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Inflation gap vs. labor-market tightness with fitted kink, 2001–2026")
save_figure(fig, FIG / "fig5_kink_scatter.png")

# Figure 6: decomposition ------------------------------------------------------
dec = pd.read_csv(ROOT / "results/decomposition.csv",
                  parse_dates=["quarter"]).set_index("quarter")
dec = dec.loc["2020-01-01":]
comp = pd.DataFrame({
    "Expectations": dec["contrib_expectations"],
    "Labor-market tightness": dec["contrib_tightness"],
    "Supply-chain pressure": dec["contrib_supply"],
    "Constant + residual": dec["contrib_const"] + dec["residual"]})
fig, ax = plt.subplots(figsize=(8.4, 4.4))
bottom_pos = np.zeros(len(comp)); bottom_neg = np.zeros(len(comp))
colors = {"Expectations": GREY, "Labor-market tightness": RED,
          "Supply-chain pressure": GOLD, "Constant + residual": "#d5d8dc"}
x = np.arange(len(comp))
for col in comp.columns:
    v = comp[col].values
    pos = np.clip(v, 0, None); neg = np.clip(v, None, 0)
    ax.bar(x, pos, bottom=bottom_pos, color=colors[col], label=col, width=0.8)
    ax.bar(x, neg, bottom=bottom_neg, color=colors[col], width=0.8)
    bottom_pos += pos; bottom_neg += neg
ax.plot(x, dec["actual_pi"].values, color="black", lw=2, marker="o", ms=3.5,
        label="Actual core CPI inflation (q/q ann.)")
ax.set_xticks(x[::2])
ax.set_xticklabels([f"{q.year}Q{q.quarter}" for q in comp.index][::2],
                   rotation=45, ha="right")
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("Percentage points"); ax.legend(frameon=False, fontsize=8.5, ncol=2)
ax.set_title("Decomposing the surge and the disinflation, 2020–2026 (kinked model)")
save_figure(fig, FIG / "fig6_decomposition.png")

# Figure 7: out-of-sample disinflation test -----------------------------------
oos = pd.DataFrame(res["oos_pre2023"]["path"])
oos["quarter"] = pd.to_datetime(oos["quarter"])
hist = df.loc["2021-01-01":"2026-01-01", "pi_qa_corecpi"]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(hist.index, hist.values, color="black", lw=2, label="Actual core CPI (q/q ann.)")
ax.plot(oos["quarter"], oos["pred_kink"], color=RED, lw=1.8, ls="--",
        marker="s", ms=3.5, label="Kinked projection (trained ≤ 2022Q4)")
ax.plot(oos["quarter"], oos["pred_linear"], color=BLUE, lw=1.8, ls=":",
        marker="^", ms=3.5, label="Linear projection (trained ≤ 2022Q4)")
ax.axvline(pd.Timestamp("2023-01-01"), color=GREY, lw=1, ls="--")
ax.text(pd.Timestamp("2023-02-15"), ax.get_ylim()[1] * 0.95, "out of sample →",
        fontsize=8.5, color=GREY)
ax.set_ylabel("Percent (annualized)"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Holdout projections conditional on realized inputs, 2023–2026")
save_figure(fig, FIG / "fig7_oos.png")

print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
