"""Offline acquisition tests: python -m unittest discover -s tests."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from openpyxl import Workbook


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "00_fetch_data.py"
SPEC = importlib.util.spec_from_file_location("fetch_data", MODULE_PATH)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class DownloadTests(unittest.TestCase):
    fred = b"observation_date,UNRATE\n2020-01-01,3.6\n2020-02-01,.\n2020-03-01,4.4\n"

    def workbook(self, sheet="GSCPI Monthly Data", rows=None):
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = sheet
        worksheet.append(["Date", "GSCPI", None, "Copyright"])
        for row in rows or [
            [None, None, None, "New York Fed Economic Research"],
            ["29-Feb-2020", "1.25", None, None],
            ["31-Jan-2020", -0.5, None, None],
            ["Footer", "not an observation", None, "terms"],
            [None, None, None, None],
        ]:
            worksheet.append(row)
        output = io.BytesIO()
        workbook.save(output)
        return output.getvalue()

    def test_fred_missing_observations_are_preserved(self):
        self.assertEqual(fetch.validate_csv(self.fred, "UNRATE"), {
            "rows": 3, "numeric_observations": 2,
            "first_date": "2020-01-01", "last_date": "2020-03-01",
        })

    def test_invalid_source_responses_rejected(self):
        invalid = [
            b"<html>Service unavailable</html>",
            b"observation_date,UNRATE\n2020-01-01,.\n",
            b"observation_date,UNRATE\n2020-01-01,inf\n",
            b"observation_date,UNRATE\n2020-01-01,1\n2020-01-01,2\n",
            b"observation_date,UNRATE\nnot-a-date,1\n",
            b"observation_date,OTHER\n2020-01-01,1\n",
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                fetch.validate_csv(payload, "UNRATE")

    def test_gscpi_filters_blank_and_copyright_rows_and_sorts_dates(self):
        for sheet in ("GSCPI Monthly Data", "Monthly Data"):
            with self.subTest(sheet=sheet):
                result = fetch.parse_gscpi_workbook(self.workbook(sheet=sheet))
                self.assertEqual(result.decode(), "date,gscpi\n2020-01-31,-0.5\n2020-02-29,1.25\n")

    def test_gscpi_rejects_duplicates_and_missing_sheet(self):
        with self.assertRaises(ValueError):
            fetch.parse_gscpi_workbook(self.workbook(sheet="Unrelated"))
        with self.assertRaises(ValueError):
            fetch.parse_gscpi_workbook(self.workbook(rows=[
                ["31-Jan-2020", 1], ["31-Jan-2020", 2],
            ]))

    def test_existing_valid_input_reused_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "UNRATE.csv").write_bytes(self.fred)
            with patch.object(fetch, "download", side_effect=AssertionError("network used")):
                fetch.acquire("UNRATE", raw)
                first_metadata = (raw / "UNRATE.provenance.json").read_bytes()
                fetch.acquire("UNRATE", raw)
                self.assertEqual(first_metadata, (raw / "UNRATE.provenance.json").read_bytes())
            metadata = json.loads(first_metadata)
            self.assertIsNone(metadata["retrieved_at_utc"])
            self.assertEqual(metadata["sha256"], hashlib.sha256(self.fred).hexdigest())

    def test_force_download_records_provenance_and_rejects_invalid_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            path = raw / "UNRATE.csv"
            path.write_bytes(self.fred)
            replacement = b"observation_date,UNRATE\n2020-01-01,3.7\n2020-02-01,3.9\n2020-03-01,4.4\n"
            with patch.object(fetch, "download", return_value=replacement) as download:
                fetch.acquire("UNRATE", raw, force=True)
                download.assert_called_once()
            metadata = json.loads((raw / "UNRATE.provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(path.read_bytes(), replacement)
            self.assertIsNotNone(metadata["retrieved_at_utc"])
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(replacement).hexdigest())
            provenance = (raw / "UNRATE.provenance.json").read_bytes()
            with patch.object(fetch, "download", return_value=b"<html>Error</html>"):
                with self.assertRaises(ValueError):
                    fetch.acquire("UNRATE", raw, force=True)
            self.assertEqual(path.read_bytes(), replacement)
            self.assertEqual(provenance, (raw / "UNRATE.provenance.json").read_bytes())

    def test_refresh_that_shortens_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            path = raw / "UNRATE.csv"
            path.write_bytes(self.fred)
            truncated = b"observation_date,UNRATE\n2020-01-01,3.7\n"
            with patch.object(fetch, "download", return_value=truncated):
                with self.assertRaisesRegex(ValueError, "less than the existing"):
                    fetch.acquire("UNRATE", raw, force=True)
            self.assertEqual(path.read_bytes(), self.fred)
            self.assertFalse((raw / "UNRATE.provenance.json").exists())

    def test_refresh_with_fewer_values_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            path = raw / "UNRATE.csv"
            path.write_bytes(self.fred)
            thinned = b"observation_date,UNRATE\n2020-01-01,.\n2020-02-01,.\n2020-03-01,4.4\n"
            with patch.object(fetch, "download", return_value=thinned):
                with self.assertRaisesRegex(ValueError, "less than the existing"):
                    fetch.acquire("UNRATE", raw, force=True)
            self.assertEqual(path.read_bytes(), self.fred)

    def test_refresh_with_month_start_dates_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "GSCPI.csv").write_bytes(b"date,gscpi\n2020-01-31,1\n2020-02-29,2\n")
            month_start = b"date,gscpi\n2020-01-01,1.5\n2020-02-01,2.5\n"
            with patch.object(fetch, "download", return_value=b"workbook"), \
                    patch.object(fetch, "parse_gscpi_workbook", return_value=month_start):
                fetch.acquire("GSCPI", raw, force=True)
            self.assertEqual((raw / "GSCPI.csv").read_bytes(), month_start)

    def test_write_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            revised = b"observation_date,UNRATE\n2020-01-01,3.5\n2020-02-01,3.8\n2020-03-01,4.4\n"
            real_write = fetch.atomic_write

            def failing_write(path, payload):
                if path.name == "UNEMPLOY.csv":
                    raise PermissionError("locked")
                real_write(path, payload)

            argv = ["00_fetch_data.py", "--raw-dir", str(raw), "--series", "UNRATE", "UNEMPLOY"]
            with patch.object(fetch, "download", return_value=revised), \
                    patch.object(fetch, "validate_csv", return_value={
                        "rows": 3, "numeric_observations": 3,
                        "first_date": "2020-01-01", "last_date": "2020-03-01"}), \
                    patch.object(fetch, "atomic_write", side_effect=failing_write), \
                    patch.object(sys, "argv", argv), patch("sys.stderr") as stderr:
                with self.assertRaises(SystemExit) as stopped:
                    fetch.main()
            self.assertEqual(stopped.exception.code, 1)
            message = "".join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn("UNEMPLOY: writing the input failed", message)
            self.assertIn("already written: UNRATE", message)

    def test_failed_refresh_changes_no_input(self):
        """Every requested series validates before any file is replaced."""
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            unrate = raw / "UNRATE.csv"
            unrate.write_bytes(self.fred)
            revised = b"observation_date,UNRATE\n2020-01-01,3.5\n2020-02-01,3.8\n2020-03-01,4.4\n"

            def respond(url, **_):
                if url.endswith("UNRATE"):
                    return revised
                raise URLError("offline")

            argv = ["00_fetch_data.py", "--raw-dir", str(raw), "--force",
                    "--series", "UNRATE", "UNEMPLOY"]
            with patch.object(fetch, "download", side_effect=respond), \
                    patch.object(sys, "argv", argv), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as stopped:
                    fetch.main()
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(unrate.read_bytes(), self.fred)
            self.assertEqual(sorted(path.name for path in raw.iterdir()), ["UNRATE.csv"])

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_replaced_input_keeps_its_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "UNRATE.csv"
            path.write_bytes(self.fred)
            path.chmod(0o644)
            fetch.atomic_write(path, b"replacement")
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            fresh = Path(directory) / "NEW.csv"
            fetch.atomic_write(fresh, b"new")
            umask = os.umask(0)
            os.umask(umask)
            self.assertEqual(fresh.stat().st_mode & 0o777, 0o666 & ~umask)

    def test_invalid_existing_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            path = raw / "UNRATE.csv"
            path.write_bytes(b"Service unavailable")
            with patch.object(fetch, "download", return_value=self.fred):
                fetch.acquire("UNRATE", raw)
            self.assertEqual(path.read_bytes(), self.fred)

    def test_network_retries_are_bounded(self):
        with patch.object(fetch, "urlopen", side_effect=URLError("offline")) as request:
            with patch.object(fetch.time, "sleep") as sleep:
                with self.assertRaises(URLError):
                    fetch.download(fetch.source_url("UNRATE"), timeout=7, attempts=3)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == 7 for call in request.call_args_list))

    def test_permanent_http_errors_are_not_retried(self):
        url = fetch.source_url("UNRATE")
        for code, calls in ((404, 1), (429, 3), (503, 3)):
            error = HTTPError(url, code, "status", {}, None)
            with self.subTest(code=code):
                with patch.object(fetch, "urlopen", side_effect=error) as request:
                    with patch.object(fetch.time, "sleep"):
                        with self.assertRaises(HTTPError):
                            fetch.download(url, attempts=3)
                self.assertEqual(request.call_count, calls)


if __name__ == "__main__":
    unittest.main()
