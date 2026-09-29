"""Build the quarterly analysis dataset from raw FRED / NY Fed downloads.

Inputs:  data/raw/*.csv  (FRED fredgraph.csv format; GSCPI.csv from NY Fed xls)
Output:  data/processed/quarterly.csv

All series are converted to quarterly frequency by within-quarter averaging.
Inflation is measured two ways:
  pi_qa_*  : 400 * log(P_q / P_{q-1})   (quarterly annualized)
  pi_yoy_* : 100 * log(P_q / P_{q-4})   (year over year)

A quarter with a missing month is averaged over the months available, and the
build lists every such quarter. In the June 2026 vintage, BLS published no
October 2025 value for CPIAUCSL, CPILFESL, CPIENGSL, UNRATE or UNEMPLOY
(federal shutdown), so those 2025Q4 averages use November and December only.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
LAST_QUARTER = "2026-01-01"  # last complete quarter of the snapshot
partial = []  # quarters averaged from fewer than three monthly observations


def fred(series: str) -> pd.Series:
    df = pd.read_csv(RAW / f"{series}.csv")
    df.columns = ["date", series]
    df["date"] = pd.to_datetime(df["date"])
    df[series] = pd.to_numeric(df[series], errors="coerce")
    return df.set_index("date")[series]


def to_q(s: pd.Series) -> pd.Series:
    # A missing month (a source gap, or a series starting mid-quarter) leaves the
    # quarter averaged over fewer months; report it rather than pass it silently.
    months = s.resample("QS").count().loc[:LAST_QUARTER]
    partial.extend(f"{s.name} {q.year}Q{q.quarter} ({n} of 3 months)"
                   for q, n in months[(months > 0) & (months < 3)].items())
    return s.resample("QS").mean()


# --- price indices -> inflation -------------------------------------------
prices = {"cpi": "CPIAUCSL", "corecpi": "CPILFESL", "corepce": "PCEPILFE",
          "pce": "PCEPI", "cpienergy": "CPIENGSL"}
cols = {}
for name, code in prices.items():
    p = to_q(fred(code))
    cols[f"pi_qa_{name}"] = 400 * np.log(p / p.shift(1))
    cols[f"pi_yoy_{name}"] = 100 * np.log(p / p.shift(4))

# --- labor market ----------------------------------------------------------
u = to_q(fred("UNRATE"))
nrou = fred("NROU").resample("QS").mean()  # already quarterly
cols["u"] = u
cols["nrou"] = nrou
cols["ugap"] = u - nrou
vac = to_q(fred("JTSJOL"))
unlevel = to_q(fred("UNEMPLOY"))
cols["vu"] = vac / unlevel

# --- expectations ----------------------------------------------------------
cols["mich"] = to_q(fred("MICH"))            # Michigan 1y ahead, from 1978
cols["cle1y"] = to_q(fred("EXPINF1YR"))      # Cleveland Fed 1y, from 1982
cols["cle10y"] = to_q(fred("EXPINF10YR"))    # Cleveland Fed 10y, from 1982

# --- supply-side measures ---------------------------------------------------
g = pd.read_csv(RAW / "GSCPI.csv", parse_dates=["date"]).set_index("date")["gscpi"]
cols["gscpi"] = to_q(g)

df = pd.DataFrame(cols)

# relative energy inflation (supply proxy available before 1998)
df["rel_energy"] = df["pi_qa_cpienergy"] - df["pi_qa_corecpi"]

# adaptive (backward-looking) expectations: mean of past 4 quarters' annualized
# headline CPI inflation -- used for the accelerationist long-sample spec
df["pi_adaptive"] = df["pi_qa_cpi"].shift(1).rolling(4).mean()

# trim to complete quarters: drop the current partial quarter (2026Q2 has
# only April, or April and May, for most monthly series)
df = df.loc["1948-01-01":LAST_QUARTER]
df.index.name = "quarter"
df.to_csv(OUT / "quarterly.csv", lineterminator="\n")

print(df.loc["1959-10-01":].head(3).round(2).to_string())
print(df.tail(6).round(2).to_string())
print("\nrows:", len(df), "| v/u available:", df["vu"].first_valid_index(),
      "->", df["vu"].last_valid_index())
print("quarters averaged from fewer than 3 months:", "; ".join(partial) or "none")
