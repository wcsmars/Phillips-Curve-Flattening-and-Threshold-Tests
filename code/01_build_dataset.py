"""Build quarterly data from validated raw FRED / NY Fed CSVs.

Inputs: data/raw/*.csv. Output: data/processed/quarterly.csv.
Monthly levels are averaged within calendar quarters. Inflation is
400 * log(P_q / P_{q-1}) annualized, or 100 * log(P_q / P_{q-4}) year over year.
Missing months are retained as missing and incomplete quarters are reported.
In the June 2026 vintage, five BLS series have no October 2025 observation;
their 2025Q4 means therefore use November and December only.
"""
from datetime import date
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
LAST_QUARTER = "2026-01-01"  # fixed study end, not the current calendar quarter


def fred(series: str, raw_dir=RAW) -> pd.Series:
    """Read one source without disguising wrong columns or corrupt values as NaN.

    This reader is self-contained because the offline demo relocates the build
    script without the downloader. Only the sources' empty and dot missing-value
    markers are accepted. A source supplies one observation per month (NROU:
    one per quarter), including when its labels are at month/quarter end.
    """
    df = pd.read_csv(Path(raw_dir) / f"{series}.csv", dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    value_column = "gscpi" if series == "GSCPI" else series
    if (len(df.columns) != 2 or df.columns[0].lower() not in {"date", "observation_date"}
            or df.columns[1] != value_column):
        raise ValueError(f"{series}: unexpected CSV columns: {list(df.columns)!r}")
    if df.empty:
        raise ValueError(f"{series}: no observations")
    # date.fromisoformat also rejects timestamps and ambiguous regional dates.
    parsed = [date.fromisoformat(value) for value in df.iloc[:, 0]]
    if any(value != parsed_date.isoformat() for value, parsed_date in zip(df.iloc[:, 0], parsed)):
        raise ValueError(f"{series}: dates must use YYYY-MM-DD")
    dates = pd.DatetimeIndex(parsed)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError(f"{series}: dates must be unique and increasing")
    periods = dates.to_period("Q" if series == "NROU" else "M")
    if not periods.is_unique:
        raise ValueError(f"{series}: multiple observations in one source period")
    text = df.iloc[:, 1].str.strip()
    missing = text.isin(["", "."])
    numeric = pd.to_numeric(text.where(~missing), errors="raise")
    if not numeric.notna().any():
        raise ValueError(f"{series}: no numeric observations")
    if not np.isfinite(numeric[~missing]).all():
        raise ValueError(f"{series}: nonfinite observation")
    if series in {"CPIAUCSL", "CPILFESL", "PCEPILFE", "PCEPI", "CPIENGSL",
                  "UNEMPLOY"} and (numeric <= 0).any():
        raise ValueError(f"{series}: observations must be positive")
    if series in {"UNRATE", "NROU"} and ((numeric < 0) | (numeric > 100)).any():
        raise ValueError(f"{series}: percentage outside [0, 100]")
    if series == "JTSJOL" and (numeric < 0).any():
        raise ValueError(f"{series}: job openings cannot be negative")
    return pd.Series(numeric.to_numpy(dtype=float), index=dates,
                     name=value_column)


def to_q(s: pd.Series, partial=None) -> pd.Series:
    """Quarterly level mean, with no interpolation or implicit filling."""
    months = s.resample("QS").count().loc[:LAST_QUARTER]
    if partial is not None:
        partial.extend(f"{s.name} {q.year}Q{q.quarter} ({n} of 3 months)"
                       for q, n in months[months < 3].items())
    return s.resample("QS").mean()


def build_dataset(raw_dir=RAW):
    """Return the analysis frame and incomplete-quarter notes without writing."""
    partial = []

    def monthly(code):
        return to_q(fred(code, raw_dir), partial)

    prices = {"cpi": "CPIAUCSL", "corecpi": "CPILFESL", "corepce": "PCEPILFE",
              "pce": "PCEPI", "cpienergy": "CPIENGSL"}
    cols = {}
    for name, code in prices.items():
        p = monthly(code)
        cols[f"pi_qa_{name}"] = 400 * np.log(p / p.shift(1))
        cols[f"pi_yoy_{name}"] = 100 * np.log(p / p.shift(4))

    u = monthly("UNRATE")
    nrou = fred("NROU", raw_dir).resample("QS").mean()
    cols["u"], cols["nrou"], cols["ugap"] = u, nrou, u - nrou
    cols["vu"] = monthly("JTSJOL") / monthly("UNEMPLOY")
    cols["mich"] = monthly("MICH")
    cols["cle1y"] = monthly("EXPINF1YR")
    cols["cle10y"] = monthly("EXPINF10YR")
    cols["gscpi"] = monthly("GSCPI")
    df = pd.DataFrame(cols)
    df["rel_energy"] = df["pi_qa_cpienergy"] - df["pi_qa_corecpi"]
    # A missing quarter stays on the calendar: shifts never bridge a gap.
    df["pi_adaptive"] = df["pi_qa_cpi"].shift(1).rolling(4).mean()
    df = df.loc["1948-01-01":LAST_QUARTER]
    if df.empty:
        raise ValueError("No observations within the study period")
    if np.isinf(df.to_numpy()).any():
        raise ValueError("Dataset contains infinite derived values")
    df.index.name = "quarter"
    return df, partial


def write_dataset(df, destination):
    """Replace the processed CSV only once its complete contents are ready."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False,
                                         mode="w", encoding="utf-8", newline="") as handle:
            temporary = Path(handle.name)
            df.to_csv(handle, lineterminator="\n")
        if destination.exists():
            os.chmod(temporary, destination.stat().st_mode & 0o777)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(temporary, 0o666 & ~umask)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    df, partial = build_dataset()
    write_dataset(df, OUT / "quarterly.csv")
    print(df.loc["1959-10-01":].head(3).round(2).to_string())
    print(df.tail(6).round(2).to_string())
    print("\nrows:", len(df), "| v/u available:", df["vu"].first_valid_index(),
          "->", df["vu"].last_valid_index())
    print("quarters averaged from fewer than 3 months:", "; ".join(partial) or "none")


if __name__ == "__main__":
    main()
