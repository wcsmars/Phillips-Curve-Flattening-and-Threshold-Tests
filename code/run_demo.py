"""Run the full pipeline offline on synthetic inputs with a planted kink.

The source inputs are not bundled, so this script lets a reader run the
estimation, figures and tables without network access. It writes synthetic
series in the CSV layout that ``code/00_fetch_data.py`` produces, copies the
four pipeline scripts into a separate workspace (``demo/`` by default) and runs
them there. The saved results, figures and downloaded inputs of this checkout
are never read or modified.

The synthetic inflation gap (core inflation minus 1-year expectations) follows
a kinked curve in the vacancy/unemployment ratio with its threshold at
PLANTED_KINK, so the run also shows whether the grid search recovers a known
break. Every number in the workspace is synthetic, even though the figures and
tables keep the titles of the U.S. study.

Usage (from any working directory):
    python code/run_demo.py                  # inputs + full pipeline
    python code/run_demo.py --prepare-only   # inputs and scripts only
    python code/run_demo.py --workspace DIR --seed 7
    python code/run_demo.py --check          # fail unless the planted kink is recovered
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ("01_build_dataset.py", "02_analysis.py", "03_figures.py", "04_tables.py")
MARKER = "SYNTHETIC_DEMO.txt"
OUTPUTS = ("data/processed", "results", "figures", "tables")  # written by the pipeline

# Planted model for the quarterly inflation gap:
#   gap = CONSTANT + SLOPE_BELOW * min(v/u - K, 0) + SLOPE_ABOVE * max(v/u - K, 0)
#         + GSCPI_EFFECT * gscpi + noise
PLANTED_KINK = 1.0
SLOPE_BELOW, SLOPE_ABOVE = 0.5, 3.5
CONSTANT, GSCPI_EFFECT, NOISE_SD = 0.2, 0.35, 0.5

# First observation of each synthetic series, mirroring the real sources.
MONTHLY_STARTS = {
    "CPIAUCSL": "1948-01-01", "CPILFESL": "1957-01-01", "PCEPILFE": "1959-01-01",
    "PCEPI": "1959-01-01", "CPIENGSL": "1957-01-01", "UNRATE": "1948-01-01",
    "UNEMPLOY": "1948-01-01", "JTSJOL": "2000-12-01", "MICH": "1978-01-01",
    "EXPINF1YR": "1982-01-01", "EXPINF10YR": "1982-01-01", "USREC": "1948-01-01",
}
FIRST_QUARTER, LAST_MONTH = "1948-01-01", "2026-05-01"  # 2026Q2 is incomplete


def ar1(rng, n, rho, sd):
    """Zero-mean AR(1) path started from its stationary distribution."""
    path = np.empty(n)
    path[0] = rng.normal(0, sd / np.sqrt(1 - rho ** 2))
    for i in range(1, n):
        path[i] = rho * path[i - 1] + rng.normal(0, sd)
    return path


def simulate_quarters(seed):
    """Quarterly latent economy; returns a DataFrame indexed by quarter start."""
    rng = np.random.default_rng(seed)
    quarters = pd.date_range(FIRST_QUARTER, LAST_MONTH, freq="QS")
    n = len(quarters)
    t = np.arange(n) / (n - 1)
    year = np.asarray(quarters.year + (quarters.quarter - 1) / 4, dtype=float)

    nrou = 5.4 - 1.0 * t
    u = np.clip(nrou + ar1(rng, n, 0.9, 0.4), 2.5, 12.0)
    # Beveridge-style link: tighter markets have more vacancies per unemployed.
    vu = np.exp(2.01 - 1.43 * np.log(u) + rng.normal(0, 0.05, n))
    labor_force = 60_000 * np.exp(1.04 * t)                     # thousands
    expected = 2.6 + 3.0 * np.exp(-((year - 1979) / 5) ** 2) + ar1(rng, n, 0.95, 0.2)
    gscpi = ar1(rng, n, 0.85, 0.45)
    rel_energy = rng.normal(0, 12, n)

    gap = (CONSTANT + SLOPE_BELOW * np.minimum(vu - PLANTED_KINK, 0)
           + SLOPE_ABOVE * np.maximum(vu - PLANTED_KINK, 0)
           + GSCPI_EFFECT * gscpi + 0.02 * rel_energy + rng.normal(0, NOISE_SD, n))
    core_cpi = expected + gap
    rates = {                                                  # annualized, percent
        "CPILFESL": core_cpi,
        "CPIAUCSL": core_cpi + 0.08 * rel_energy + rng.normal(0, 0.3, n),
        "CPIENGSL": core_cpi + rel_energy,
        "PCEPILFE": core_cpi - 0.35 + rng.normal(0, 0.35, n),
    }
    rates["PCEPI"] = rates["PCEPILFE"] + 0.06 * rel_energy + rng.normal(0, 0.3, n)

    frame = pd.DataFrame({
        "u": u, "nrou": nrou, "vu": vu, "labor_force": labor_force,
        "cle1y": expected + rng.normal(0, 0.1, n),
        "cle10y": 1.3 + 0.5 * expected + rng.normal(0, 0.05, n),
        "mich": expected + 0.3 + rng.normal(0, 0.25, n),
        "gscpi": gscpi,
    }, index=quarters)
    for code, rate in rates.items():
        # Levels are held flat within a quarter, so quarterly averages of the
        # monthly levels reproduce the planted quarterly rates exactly.
        frame[code] = 100 * np.exp(np.cumsum(rate) / 400)
    rising = frame["u"] - frame["u"].shift(2)
    frame["recession"] = (rising > 0.5).astype(int)
    return frame


def to_months(quarterly, wiggle=0.0):
    """Repeat quarterly values monthly; the wiggle averages to zero in a quarter."""
    months = pd.date_range(quarterly.index[0], LAST_MONTH, freq="MS")
    values = quarterly.reindex(months, method="ffill").to_numpy()
    return pd.Series(values + wiggle * np.resize([-1.0, 0.0, 1.0], len(months)), months)


def write_fred(raw, code, series, digits):
    series = series.loc[MONTHLY_STARTS.get(code, series.index[0]):]
    frame = pd.DataFrame({"observation_date": series.index.strftime("%Y-%m-%d"),
                          code: series.round(digits).to_numpy()})
    frame.to_csv(raw / f"{code}.csv", index=False, lineterminator="\n")


def write_inputs(raw, seed):
    """Write all fourteen inputs in the downloaded CSV formats."""
    q = simulate_quarters(seed)
    unrate = to_months(q["u"], wiggle=0.1)
    unemployed = unrate / 100 * to_months(q["labor_force"])
    for code in ("CPIAUCSL", "CPILFESL", "PCEPILFE", "PCEPI", "CPIENGSL"):
        write_fred(raw, code, to_months(q[code]), 4)
    write_fred(raw, "UNRATE", unrate, 4)
    write_fred(raw, "UNEMPLOY", unemployed, 1)
    # Vacancies scale with each month's unemployment, so the quarterly ratio of
    # averages equals the planted quarterly v/u.
    write_fred(raw, "JTSJOL", to_months(q["vu"]) * unemployed, 3)
    write_fred(raw, "MICH", to_months(q["mich"], wiggle=0.05), 4)
    write_fred(raw, "EXPINF1YR", to_months(q["cle1y"], wiggle=0.05), 4)
    write_fred(raw, "EXPINF10YR", to_months(q["cle10y"], wiggle=0.02), 4)
    write_fred(raw, "USREC", to_months(q["recession"]).astype(int), 0)
    write_fred(raw, "NROU", q["nrou"].loc["1949-01-01":"2026-04-01"], 4)
    gscpi = to_months(q["gscpi"], wiggle=0.1).loc["1998-01-01":]
    pd.DataFrame({"date": (gscpi.index + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
                  "gscpi": gscpi.round(4).to_numpy()}).to_csv(raw / "GSCPI.csv", index=False,
                                                              lineterminator="\n")


def prepare(workspace, seed):
    """Create the workspace, refusing the checkout itself or a foreign folder."""
    workspace = workspace.resolve()
    if workspace == ROOT or workspace in ROOT.parents:
        raise SystemExit("run_demo: the workspace must not be the checkout or contain it")
    if workspace.exists() and (not workspace.is_dir() or (
            any(workspace.iterdir()) and not (workspace / MARKER).is_file())):
        raise SystemExit(f"run_demo: {workspace} is not an empty folder or an earlier "
                         "demo workspace; choose another --workspace")
    # A symlink here would send the cleanup and the writes below outside the workspace.
    for name in (*OUTPUTS, "code", "data", "data/raw", MARKER):
        target = workspace / name
        if target.is_symlink() or not target.resolve().is_relative_to(workspace):
            raise SystemExit(f"run_demo: {target} points outside the demo workspace; "
                             "remove it or choose another --workspace")
    # Outputs of an earlier run would otherwise sit next to inputs from a new seed.
    for name in OUTPUTS:
        shutil.rmtree(workspace / name, ignore_errors=True)
    code, raw = workspace / "code", workspace / "data" / "raw"
    code.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    (workspace / MARKER).write_text(
        "Synthetic demo workspace written by code/run_demo.py. Every input, "
        f"estimate, figure and table here is synthetic (seed {seed}, planted kink "
        f"at v/u = {PLANTED_KINK}); none of it describes the U.S. economy.\n",
        encoding="utf-8", newline="\n")
    for name in PIPELINE:
        shutil.copyfile(ROOT / "code" / name, code / name)
    write_inputs(raw, seed)
    return workspace


def run_pipeline(workspace):
    for name in PIPELINE:
        print(f"== {name}", flush=True)
        completed = subprocess.run([sys.executable, str(workspace / "code" / name)],
                                   cwd=workspace)
        if completed.returncode != 0:
            raise SystemExit(f"run_demo: {name} failed with exit code {completed.returncode}")


def summarize(workspace):
    res = json.loads((workspace / "results" / "results.json").read_text(encoding="utf-8"))
    kink, test = res["modern_kink"], res["kink_test"]
    print(f"\nPlanted kink at v/u = {PLANTED_KINK:.2f}; estimated {kink['c_hat']:.2f}.")
    print(f"Planted slopes below/above: {SLOPE_BELOW} / {SLOPE_ABOVE}; "
          f"estimated {kink['params']['below']} / {kink['params']['above']}.")
    low, high = test["c_ci95"]
    verdict = "contains" if low <= PLANTED_KINK <= high else "excludes"
    print(f"Approximate LR search set {low:.2f}-{high:.2f} {verdict} the planted kink.")
    print(f"sup-LR = {test['supLR']}, moving-block bootstrap p = {test['p_bootstrap']}.")
    print(f"Synthetic outputs are in {workspace}")
    return res


# Recovery checks for --check, which CI runs with the default seed. A wrong
# threshold search, kink regressor or slope estimate moves the default run far
# outside these tolerances. Other seeds can miss them by chance, because the
# threshold is weakly identified in some synthetic samples.
KINK_TOLERANCE, SLOPE_ABOVE_TOLERANCE = 0.25, 1.5


def recovery_problems(res, workspace):
    """Failed recovery checks for a finished demo run; empty when all pass."""
    kink, (low, high) = res["modern_kink"], res["kink_test"]["c_ci95"]
    problems = []
    if abs(kink["c_hat"] - PLANTED_KINK) > KINK_TOLERANCE:
        problems.append(f"estimated kink {kink['c_hat']} is far from {PLANTED_KINK}")
    if not low <= PLANTED_KINK <= high:
        problems.append(f"LR search set {low}-{high} excludes the planted kink")
    if abs(kink["params"]["above"] - SLOPE_ABOVE) > SLOPE_ABOVE_TOLERANCE:
        problems.append(f"upper slope {kink['params']['above']} is far from {SLOPE_ABOVE}")
    for folder, pattern, count in (("figures", "*.png", 7), ("tables", "*.md", 5)):
        if len(list((workspace / folder).glob(pattern))) != count:
            problems.append(f"expected {count} files in {folder}/")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, default=ROOT / "demo",
                        help="output folder (default: demo/ in this checkout)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true",
                        help="write synthetic inputs and scripts without estimating")
    parser.add_argument("--check", action="store_true",
                        help="exit with an error unless the planted kink is recovered")
    args = parser.parse_args()
    if args.check and args.prepare_only:
        parser.error("--check needs the full run; drop --prepare-only")
    workspace = prepare(args.workspace, args.seed)
    print(f"Synthetic inputs written to {workspace / 'data' / 'raw'}")
    if not args.prepare_only:
        run_pipeline(workspace)
        problems = recovery_problems(summarize(workspace), workspace)
        if args.check and problems:
            raise SystemExit("run_demo: planted model not recovered: " + "; ".join(problems))
        if args.check:
            print("Recovery check passed: the planted kink was recovered.")


if __name__ == "__main__":
    main()
