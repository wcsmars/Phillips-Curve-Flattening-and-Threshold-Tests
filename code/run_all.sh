#!/bin/sh
# Replication: dataset -> estimation -> figures -> tables
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" code/01_build_dataset.py
"$PYTHON" code/02_analysis.py
"$PYTHON" code/03_figures.py
"$PYTHON" code/04_tables.py
echo "Done. See results/, figures/ and tables/."
