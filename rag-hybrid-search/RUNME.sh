#!/usr/bin/env bash
# Runs everything that doesn't need an API key: setup, index build, and
# every retrieval/eval script. Run from the project root: bash RUNME.sh
set -e

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python3 -m spacy download en_core_web_md

export PYTHONPATH=src

echo "=== building index ===" && python3 src/build_index.py
echo "=== dense vs sparse vs hybrid (5 sample queries) ===" && python3 scripts/compare_retrieval.py
echo "=== full 49-question golden eval ===" && python3 scripts/run_retrieval_eval.py
echo "=== RRF weight sweep ===" && python3 scripts/tune_rrf_weight.py
echo "=== fixed vs structure-aware chunking ===" && python3 scripts/compare_chunking.py

echo ""
echo "Done. Generation is separate -- edit .env and put your real GROQ_API_KEY in it, then:"
echo "  python3 scripts/test_generation.py"
