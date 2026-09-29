"""Fetch the public source inputs; existing valid inputs are reused unless --force.

Run ``python code/00_fetch_data.py`` from any working directory. Downloads use
urllib; GSCPI workbook parsing requires pandas, xlrd (legacy XLS), and openpyxl
(XLSX). The New York Fed sometimes serves XLS bytes under an .xlsx filename.
Acquisition retrieves current source vintages, not a historical data snapshot.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from datetime import date, datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
FRED_SERIES = (
    "CPIAUCSL", "CPILFESL", "PCEPILFE", "PCEPI", "CPIENGSL", "UNRATE",
    "NROU", "JTSJOL", "UNEMPLOY", "MICH", "EXPINF1YR", "EXPINF10YR", "USREC",
)
GSCPI_URL = (
    "https://www.newyorkfed.org/medialibrary/research/interactives/"
    "gscpi/downloads/gscpi_data.xlsx"
)
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


def source_url(series):
    return GSCPI_URL if series == "GSCPI" else (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + series
    )


def validate_csv(payload, series):
    """Reject error pages, malformed observations, and empty source responses."""
    rows = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
    header = next(rows, None)
    expected_value = "gscpi" if series == "GSCPI" else series
    if (not header or len(header) != 2 or header[1] != expected_value
            or header[0].lower() not in {"date", "observation_date"}):
        raise ValueError(f"{series}: unexpected CSV columns: {header!r}")
    dates = []
    numeric_count = 0
    for row in rows:
        if not row:
            continue
        if len(row) != 2:
            raise ValueError(f"{series}: expected two columns per observation")
        observation_date = date.fromisoformat(row[0])
        if dates and observation_date <= dates[-1]:
            raise ValueError(f"{series}: dates must be unique and increasing")
        value = row[1].strip()
        if value not in {"", "."}:
            if not math.isfinite(float(value)):
                raise ValueError(f"{series}: nonfinite observation")
            numeric_count += 1
        dates.append(observation_date)
    if not dates or not numeric_count:
        raise ValueError(f"{series}: no numeric observations")
    return {
        "rows": len(dates), "numeric_observations": numeric_count,
        "first_date": dates[0].isoformat(), "last_date": dates[-1].isoformat(),
    }


def parse_gscpi_workbook(payload):
    """Extract Date/GSCPI and ignore blank/footer/copyright workbook cells."""
    import pandas as pd

    engine = "xlrd" if payload.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") else "openpyxl"
    with pd.ExcelFile(io.BytesIO(payload), engine=engine) as workbook:
        sheet = next((name for name in workbook.sheet_names
                      if name.strip().casefold() in {"gscpi monthly data", "monthly data"}), None)
        if sheet is None:
            raise ValueError("GSCPI: workbook has no monthly data sheet")
        frame = pd.read_excel(workbook, sheet_name=sheet)
    frame.columns = [str(column).strip().casefold() for column in frame.columns]
    if not {"date", "gscpi"}.issubset(frame.columns):
        raise ValueError("GSCPI: monthly sheet must contain Date and GSCPI columns")
    dates = frame["date"].map(lambda value: pd.to_datetime(value, errors="coerce"))
    values = pd.to_numeric(frame["gscpi"], errors="coerce")
    valid = dates.notna() & values.notna() & values.map(
        lambda value: math.isfinite(value)
    )
    output = pd.DataFrame({"date": dates[valid], "gscpi": values[valid]})
    output = output.sort_values("date")
    # LF on every platform, so the file and its recorded checksum do not depend on the OS.
    payload = output.to_csv(index=False, date_format="%Y-%m-%d",
                            lineterminator="\n").encode("utf-8")
    validate_csv(payload, "GSCPI")
    return payload


def download(url, timeout=30, attempts=3):
    """Use a finite per-request timeout and at most three attempts by default."""
    request = Request(url, headers={"User-Agent": "EconomicResearchPublicData/1.0"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(payload) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Source response exceeds the 20 MiB download limit")
            return payload
        except (URLError, TimeoutError, OSError) as error:
            # A client error such as 404 will not change on retry; 408 and 429 may.
            permanent = (isinstance(error, HTTPError) and 400 <= error.code < 500
                         and error.code not in {408, 429})
            if permanent or attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise ValueError("attempts must be positive")


def atomic_write(path, payload):
    """Only replace an existing source file after its new contents validate."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        # Temporary files are owner-only; keep the permissions a normal write would give.
        if path.exists():
            mode = path.stat().st_mode & 0o777
        else:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def existing_coverage(destination, series):
    """Validated span of the current input file, or None if it is absent or invalid."""
    if not destination.exists():
        return None
    try:
        return validate_csv(destination.read_bytes(), series)
    except (ValueError, UnicodeError, csv.Error):
        return None


def prepare(series, raw_dir, force=False, timeout=30, attempts=3):
    """Download and validate one input without writing anything."""
    raw_dir = Path(raw_dir)
    destination = raw_dir / f"{series}.csv"
    current = existing_coverage(destination, series)
    plan = {"series": series, "destination": destination,
            "provenance": raw_dir / f"{series}.provenance.json",
            "retrieved_at": None, "source_sha256": None}
    if current is not None and not force:
        plan.update(payload=destination.read_bytes(), metadata=current, action="reused")
        return plan
    source = download(source_url(series), timeout=timeout, attempts=attempts)
    payload = parse_gscpi_workbook(source) if series == "GSCPI" else source
    metadata = validate_csv(payload, series)
    # A refresh may revise values but must not silently shorten or thin the history.
    # Months are compared, so a switch between month-start and month-end dates passes.
    if current is not None and (metadata["first_date"][:7] > current["first_date"][:7]
                                or metadata["last_date"][:7] < current["last_date"][:7]
                                or metadata["numeric_observations"]
                                < current["numeric_observations"]):
        raise ValueError(
            f"{series}: new download covers {metadata['first_date']} to "
            f"{metadata['last_date']} with {metadata['numeric_observations']} values, "
            f"less than the existing {current['first_date']} to {current['last_date']} "
            f"with {current['numeric_observations']}; delete {destination.name} "
            "to accept the shorter series")
    plan.update(payload=payload, metadata=metadata, action="downloaded",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(source).hexdigest())
    return plan


def commit(plan):
    """Write a prepared input and its provenance record."""
    series, payload, metadata = plan["series"], plan["payload"], plan["metadata"]
    destination, provenance, action = plan["destination"], plan["provenance"], plan["action"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if action == "downloaded":
        atomic_write(destination, payload)
    retrieved_at, source_sha256 = plan["retrieved_at"], plan["source_sha256"]
    digest = hashlib.sha256(payload).hexdigest()
    # Preserve the actual retrieval time on reuse; never invent an old vintage.
    if action == "reused" and provenance.exists():
        try:
            previous = json.loads(provenance.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and previous.get("sha256") == digest:
                print(f"{series}: reused existing valid input")
                return
        except (ValueError, OSError):
            pass
    metadata.update({
        "series": series,
        "source_url": source_url(series),
        "sha256": digest,
        "source_sha256": source_sha256,
        "retrieved_at_utc": retrieved_at,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": action,
        "vintage_note": "Current source vintage when downloaded; historical revisions may differ.",
    })
    atomic_write(provenance, (json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    print(f"{series}: {action} {metadata['rows']} rows")


def acquire(series, raw_dir, force=False, timeout=30, attempts=3):
    commit(prepare(series, raw_dir, force, timeout, attempts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--series", nargs="+", choices=(*FRED_SERIES, "GSCPI"),
                        default=(*FRED_SERIES, "GSCPI"))
    parser.add_argument("--force", action="store_true", help="Explicitly refresh existing inputs")
    parser.add_argument("--timeout", type=float, default=30, help="Per-request timeout in seconds")
    parser.add_argument("--attempts", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    # Validate every requested series before replacing any file, so a failed
    # download or validation leaves every input unchanged.
    plans = []
    for series in args.series:
        try:
            plans.append(prepare(series, args.raw_dir, args.force, args.timeout, args.attempts))
        except Exception as error:
            parser.exit(1, f"{series}: acquisition failed: {error}; no input files were changed\n")
    written = []
    for plan in plans:
        try:
            commit(plan)
        except OSError as error:
            parser.exit(1, f"{plan['series']}: writing the input failed: {error}; "
                           f"already written: {', '.join(written) or 'none'}\n")
        written.append(plan["series"])


if __name__ == "__main__":
    main()
