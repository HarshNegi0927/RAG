# Run from the project root in PowerShell: .\RUNME.ps1
# If PowerShell blocks it ("running scripts is disabled on this system"), run:
#   powershell -ExecutionPolicy Bypass -File .\RUNME.ps1
$ErrorActionPreference = "Stop"

python -m venv venv

.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m spacy download en_core_web_md

$env:PYTHONPATH = "src"

Write-Host "=== building index ==="
.\venv\Scripts\python.exe src\build_index.py

Write-Host "=== dense vs sparse vs hybrid (5 sample queries) ==="
.\venv\Scripts\python.exe scripts\compare_retrieval.py

Write-Host "=== full 49-question golden eval ==="
.\venv\Scripts\python.exe scripts\run_retrieval_eval.py

Write-Host "=== RRF weight sweep ==="
.\venv\Scripts\python.exe scripts\tune_rrf_weight.py

Write-Host "=== fixed vs structure-aware chunking ==="
.\venv\Scripts\python.exe scripts\compare_chunking.py

Write-Host ""
Write-Host "Done. Generation is separate -- edit .env and put your real GROQ_API_KEY in it, then:"
Write-Host '  .\venv\Scripts\python.exe scripts\test_generation.py'
