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

Run-Step "ORBIT frame surface seed 42" -CommandArgs @(
    "scripts\23_conditional_frame_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_patch_stage2_vit_cifar10_forget0_proto",
    "--frame-width", "16"
)

Run-Step "CE-U frame surface seed 42" -CommandArgs @(
    "scripts\23_conditional_frame_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_combined_stage2_vit_cifar10_forget0_proto",
    "--frame-width", "16"
)

Run-Step "Faithful SalUn full-ft frame surface seed 42" -CommandArgs @(
    "scripts\23_conditional_frame_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_fullft_vit_cifar10_forget0_smoke", "unlearn_salun_fullft_patch_stage2_vit_cifar10_forget0_proto",
    "--frame-width", "16"
)

Write-Host ""
Write-Host "All frame-surface experiments finished." -ForegroundColor Green
