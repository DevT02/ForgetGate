$ErrorActionPreference = "Stop"

Write-Host "ForgetGate: Full audit (analysis + manifest)"

python scripts/analyze_kshot_experiments.py --seeds 42 123 456
python scripts/summarize_kshot_results.py --seeds 42 123 456
python scripts/summarize_clean_baselines.py --seeds 42 123 456
python scripts/audit_results_manifest.py --seeds 42 123 456 --prompt-seeds 42 123
python scripts/audit_results_integrity.py --results-dir results/logs --analysis-dir results/analysis --config configs/experiment_suites.yaml
python scripts/audit_variance_report.py --results-dir results/logs --analysis-dir results/analysis --seeds 42 123 456 --std-threshold 5.0

Write-Host "Done."
