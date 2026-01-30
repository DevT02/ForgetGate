$ErrorActionPreference = "Stop"

Write-Host "ForgetGate: High-value additions (resume-safe)"

function RunIfMissing($logPath, $cmdArgs) {
    if (Test-Path $logPath) {
        Write-Host "[SKIP] $logPath"
        return
    }
    $cmdLine = ($cmdArgs -join " ")
    Write-Host "[RUN] $cmdLine"
    & $cmdArgs[0] $cmdArgs[1..($cmdArgs.Length-1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $cmdLine"
    }
}

# 1) Dependency-aware eval (seed 456)
$depLog = "results/logs/eval_dependency_vit_cifar10_forget0_seed_456_evaluation.json"
RunIfMissing $depLog @("python","scripts/4_adv_evaluate.py","--config","configs/experiment_suites.yaml","--suite","eval_dependency_vit_cifar10_forget0","--seed","456")

# 2) Backdoor/trigger eval (seed 42)
$bdLog = "results/logs/eval_backdoor_vit_cifar10_forget0_seed_42_evaluation.json"
RunIfMissing $bdLog @("python","scripts/4_adv_evaluate.py","--config","configs/experiment_suites.yaml","--suite","eval_backdoor_vit_cifar10_forget0","--seed","42")

# 3) Prune-then-unlearn baseline eval (seed 42)
$pruneLog = "results/logs/eval_prune_kl_vit_cifar10_forget0_seed_42_evaluation.json"
RunIfMissing $pruneLog @("python","scripts/4_adv_evaluate.py","--config","configs/experiment_suites.yaml","--suite","eval_prune_kl_vit_cifar10_forget0","--seed","42")

Write-Host "Done."
