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

Run-Step "ORBIT multi-patch surface seed 42" -CommandArgs @(
    "scripts\24_conditional_multi_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_patch_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32",
    "--num-patches", "2"
)

Run-Step "CE-U multi-patch surface seed 42" -CommandArgs @(
    "scripts\24_conditional_multi_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_combined_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32",
    "--num-patches", "2"
)

Run-Step "Faithful SalUn full-ft multi-patch surface seed 42" -CommandArgs @(
    "scripts\24_conditional_multi_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_fullft_vit_cifar10_forget0_smoke", "unlearn_salun_fullft_patch_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32",
    "--num-patches", "2"
)

Write-Host ""
Write-Host "All multi-patch surface experiments finished." -ForegroundColor Green
