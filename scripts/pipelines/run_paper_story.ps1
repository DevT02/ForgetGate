param(
    [switch]$UseActivePython,
    [string]$PythonExe = "python",
    [string]$Config = "configs/experiment_suites.yaml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($UseActivePython) {
    $PythonExe = "python"
}

Write-Host "[RUN ] feature-guided audit"
Write-Host "       $PythonExe scripts/audit_feature_guided_vpt.py --log-dir results/logs --analysis-dir results/analysis --checkpoint-dir checkpoints/vpt_resurrector --seeds 42 123 --sweep-seeds 42"
& $PythonExe scripts/audit_feature_guided_vpt.py --log-dir results/logs --analysis-dir results/analysis --checkpoint-dir checkpoints/vpt_resurrector --seeds 42 123 --sweep-seeds 42

Write-Host ""
Write-Host "[RUN ] paper story summary"
Write-Host "       $PythonExe scripts/build_paper_story.py --analysis-dir results/analysis"
& $PythonExe scripts/build_paper_story.py --analysis-dir results/analysis

Write-Host ""
Write-Host "Paper story pipeline complete."
Write-Host "Key outputs:"
Write-Host "  results/analysis/feature_guided_vpt_audit.md"
Write-Host "  results/analysis/paper_story_summary.md"
Write-Host "  results/analysis/paper_story_tables.tex"
