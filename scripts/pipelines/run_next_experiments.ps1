$ErrorActionPreference = "Stop"

$python = "C:\Users\devli\miniconda3\envs\forgetgate\python.exe"
$config = "configs\experiment_suites.yaml"

function Run-Step {
    param(
        [string]$Label,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    & $python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label"
    }
}

Set-Location "C:\Users\devli\Desktop\Development\ForgetGate"

Run-Step "SCRUB strong audit seed 42" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_scrub_distill_vit_cifar10_forget0", "unlearn_scrub_feature_subspace_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "40",
    "--restarts", "4",
    "--alpha-ratio", "0.2"
)

Run-Step "SCRUB strong audit seed 123" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_scrub_distill_vit_cifar10_forget0", "unlearn_scrub_feature_subspace_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "40",
    "--restarts", "4",
    "--alpha-ratio", "0.2"
)

Run-Step "Robust-base patch audit high seed 42" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_robust_base_forget0_high",
    "--patch-size", "32"
)

Run-Step "Robust-base patch audit ultra seed 42" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_robust_base_forget0_ultra",
    "--patch-size", "32"
)

Run-Step "Robust-base patch audit high seed 123" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_salun_robust_base_forget0_high",
    "--patch-size", "32",
    "--train-batch-size", "16",
    "--max-train-forget-samples", "256",
    "--max-train-retain-samples", "256"
)

Run-Step "Robust-base patch audit ultra seed 123" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_salun_robust_base_forget0_ultra",
    "--patch-size", "32",
    "--train-batch-size", "16",
    "--max-train-forget-samples", "256",
    "--max-train-retain-samples", "256"
)

Run-Step "Faithful full-ft SalUn patch 48x48 seed 123" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_salun_fullft_vit_cifar10_forget0_smoke", "unlearn_salun_fullft_patch_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "48"
)

Write-Host ""
Write-Host "All queued experiments finished." -ForegroundColor Green
