"""Regression checks for source integrity and calendar-preserving transforms."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


fetch = module("fetch_audit", "00_fetch_data.py")
build = module("build_audit", "01_build_dataset.py")


class InputIntegrityTests(unittest.TestCase):
    def test_corrupt_sources_rejected_by_download_and_build(self):
        cases = [
            ("UNRATE", "date,OTHER\n2020-01-01,1\n"),
            ("UNRATE", "date,UNRATE\n20200101,4\n"),
            ("UNRATE", "date,UNRATE\n2020-01-01,4\n2020-01-31,5\n"),
            ("NROU", "date,NROU\n2020-01-01,4\n2020-02-01,5\n"),
            ("UNRATE", "date,UNRATE\n2020-01-01,broken\n"),
            ("UNRATE", "date,UNRATE\n2020-01-01,inf\n"),
            ("UNRATE", "date,UNRATE\n2020-01-01,NaN\n"),
            ("UNRATE", "date,UNRATE\n2020-01-01,101\n"),
            ("UNEMPLOY", "date,UNEMPLOY\n2020-01-01,0\n"),
            ("JTSJOL", "date,JTSJOL\n2020-01-01,-1\n"),
            ("CPIAUCSL", "date,CPIAUCSL\n2020-01-01,-1\n"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for series, content in cases:
                with self.subTest(series=series, content=content):
                    with self.assertRaises(ValueError):
                        fetch.validate_csv(content.encode(), series)
                    (Path(directory) / f"{series}.csv").write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        build.fred(series, directory)

    def test_missing_markers_are_allowed_but_not_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            content = "date,UNRATE\n2020-01-01,4\n2020-02-01,.\n2020-03-01,\n"
            path = Path(directory) / "UNRATE.csv"
            path.write_text(content, encoding="utf-8")
            fetched = fetch.validate_csv(content.encode(), "UNRATE")
            series = build.fred("UNRATE", directory)
            self.assertEqual(fetched["numeric_observations"], 1)
            np.testing.assert_allclose(series.to_numpy(), [4, np.nan, np.nan])
            notes = []
            self.assertEqual(build.to_q(series, notes).iloc[0], 4)
            self.assertEqual(notes, ["UNRATE 2020Q1 (1 of 3 months)"])

    def test_wholly_missing_quarter_stays_on_calendar_and_is_reported(self):
        s = pd.Series([100., 121.], index=pd.to_datetime(["2020-01-01", "2020-07-01"]),
                      name="CPIAUCSL")
        notes = []
        quarterly = build.to_q(s, notes)
        self.assertEqual(len(quarterly), 3)
        self.assertTrue(pd.isna(quarterly.loc["2020-04-01"]))
        self.assertTrue(pd.isna((quarterly / quarterly.shift(1)).loc["2020-07-01"]))
        self.assertIn("CPIAUCSL 2020Q2 (0 of 3 months)", notes)

    def test_refresh_cannot_replace_lost_history_with_a_newer_observation(self):
        previous = b"date,UNRATE\n2020-01-01,4\n2020-02-01,5\n2020-03-01,6\n"
        # Same numeric count and a wider span used to pass the coverage guard.
        replacement = b"date,UNRATE\n2020-01-01,4\n2020-02-01,.\n2020-03-01,6\n2020-04-01,7\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "UNRATE.csv"
            path.write_bytes(previous)
            with patch.object(fetch, "download", return_value=replacement):
                with self.assertRaisesRegex(ValueError, "historically incomplete"):
                    fetch.acquire("UNRATE", directory, force=True)
            self.assertEqual(path.read_bytes(), previous)

    def test_gscpi_dated_corruption_is_not_treated_as_a_footer(self):
        for row in (["2020-02-29", "broken"], ["2020-02-29", None],
                    ["not-a-date", 2], [44000, 2]):
            with self.subTest(row=row):
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "Monthly Data"
                sheet.append(["Date", "GSCPI"])
                sheet.append(["2020-01-31", 1])
                sheet.append(row)
                payload = io.BytesIO()
                workbook.save(payload)
                with self.assertRaisesRegex(ValueError, "invalid"):
                    fetch.parse_gscpi_workbook(payload.getvalue())

    def test_quarter_start_and_end_labels_have_same_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "NROU.csv"
            path.write_bytes(b"date,NROU\n2020-01-01,4\n2020-04-01,5\n")
            replacement = b"date,NROU\n2020-03-31,4.1\n2020-06-30,5.1\n"
            with patch.object(fetch, "download", return_value=replacement):
                fetch.acquire("NROU", directory, force=True)
            self.assertEqual(path.read_bytes(), replacement)

    def test_processed_file_survives_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "quarterly.csv"
            destination.write_text("preserve previous dataset", encoding="utf-8")
            with patch.object(build.os, "replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    build.write_dataset(pd.DataFrame({"x": [1]}), destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "preserve previous dataset")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_provenance_write_failure_rolls_back_new_input(self):
        previous = b"date,UNRATE\n2020-01-01,4\n"
        replacement = b"date,UNRATE\n2020-01-01,5\n"
        for has_previous in (False, True):
            with self.subTest(has_previous=has_previous), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "UNRATE.csv"
                provenance = Path(directory) / "UNRATE.provenance.json"
                if has_previous:
                    path.write_bytes(previous)
                    provenance.write_text('{"previous": true}', encoding="utf-8")
                real_write = fetch.atomic_write

                def failing_write(destination, payload):
                    if destination == provenance:
                        raise PermissionError("cannot write provenance")
                    return real_write(destination, payload)

                with patch.object(fetch, "download", return_value=replacement), \
                        patch.object(fetch, "atomic_write", side_effect=failing_write):
                    with self.assertRaises(PermissionError):
                        fetch.acquire("UNRATE", directory, force=True)
                if has_previous:
                    self.assertEqual(path.read_bytes(), previous)
                    self.assertEqual(provenance.read_text(encoding="utf-8"), '{"previous": true}')
                else:
                    self.assertFalse(path.exists())
                    self.assertFalse(provenance.exists())


if __name__ == "__main__":
    unittest.main()
