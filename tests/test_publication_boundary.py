"""Keep the public checkout within its reviewed publication scope.

These standard-library checks run before dependency installation in CI.
Local release tooling also checks file contents and staged blobs before push.
"""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Add a file here only after reviewing it for publication. Do not infer this
# list from the directory contents: that would approve accidental additions.
REVIEWED_FILES = {
    ".github/workflows/tests.yml", ".gitignore", "README.md", "requirements.txt",
    "code/00_fetch_data.py", "code/01_build_dataset.py", "code/02_analysis.py",
    "code/03_figures.py", "code/04_tables.py", "code/run_all.sh", "code/run_demo.py",
    "figures/fig1_inflation_history.png", "figures/fig2_pc_scatter_eras.png",
    "figures/fig3_rolling_slope.png", "figures/fig4_vu_timeseries.png",
    "figures/fig5_kink_scatter.png", "figures/fig6_decomposition.png",
    "figures/fig7_oos.png", "results/results.json", "results/decomposition.csv",
    "results/kink_grid_ssr.csv", "results/rolling_slope.csv",
    "tests/test_analysis_audit.py", "tests/test_data_audit.py", "tests/test_downloads.py",
    "tests/test_outputs_audit.py", "tests/test_research.py",
    "tests/test_publication_boundary.py",
}


def git(*args, input=None):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], input=input,
                                   text=True, encoding="utf-8", timeout=60)


class PublicationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / ".git").exists() or not shutil.which("git"):
            raise unittest.SkipTest("publication checks require a Git checkout")

    def test_tracked_inventory_contains_only_reviewed_regular_files(self):
        entries = git("ls-files", "--stage", "-z").split("\0")
        tracked = set()
        for entry in filter(None, entries):
            metadata, name = entry.split("\t", 1)
            mode, _, stage = metadata.split()
            self.assertIn(mode, {"100644", "100755"}, f"Linked or nested Git content: {name}")
            self.assertEqual(stage, "0", f"Unmerged entry: {name}")
            tracked.add(name)
        self.assertEqual(tracked, REVIEWED_FILES)

    def test_local_sensitive_paths_are_ignored(self):
        candidates = {
            ".env", ".env.local", "config.yaml", "config.yml",
            "credentials.yaml", "credentials.yml", "credentials.json",
            "credentials/token.json", "secrets.txt", "secrets/token.txt",
            "private/local.py", "ibkr_credentials", "brokerage/account.csv",
            "statements/month.csv", "execution_logs/orders.csv", "id_rsa",
            "id_ed25519", "client.key", "client.pem", "client.p12", "client.pfx",
            "data/raw/source.csv", "data/processed/quarterly.csv",
            "output/pdf/guide.json", "output/pdf/guide.pdf",
            "code/05_assemble.py", "code/06_fix_docx_xml.py", "code/07_build_guide.py",
            "requirements-docs.txt", "tests/test_private_outputs.py",
            ".public-maintenance/validation.json",
        }
        ignored = set(git("check-ignore", "--no-index", "--stdin",
                          input="\n".join(sorted(candidates)) + "\n").splitlines())
        self.assertEqual(ignored, candidates)

    def test_reachable_history_contains_only_reviewed_paths(self):
        paths = set(filter(None, git(
            "log", "--branches", "--tags", "--remotes=origin", "--format=",
            "--name-only", "--no-renames", "--diff-merges=first-parent",
        ).splitlines()))
        self.assertFalse(paths - REVIEWED_FILES, sorted(paths - REVIEWED_FILES))


if __name__ == "__main__":
    unittest.main()
