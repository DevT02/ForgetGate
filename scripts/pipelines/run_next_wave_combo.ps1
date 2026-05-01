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

Run-Step "ORBIT combined stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_orbit_combined_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "ORBIT combined stage2 audit seed 42 adam" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_combined_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "ORBIT combined stage2 audit seed 42 apgd" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_combined_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "apgd_margin",
    "--steps", "40",
    "--restarts", "1",
    "--alpha-ratio", "0.2"
)

Run-Step "ORBIT combined stage2 patch audit seed 42" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_combined_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32"
)

Run-Step "ORBIT stochastic CVaR teacher stage2 train seed 123" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_orbit_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--seed", "123"
)

Run-Step "ORBIT stochastic CVaR teacher audit seed 123 adam" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "ORBIT stochastic CVaR teacher audit seed 123 apgd" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "apgd_margin",
    "--steps", "40",
    "--restarts", "1",
    "--alpha-ratio", "0.2"
)

Run-Step "CE-U combined stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_ceu_combined_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "CE-U combined stage2 audit seed 42 adam" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_combined_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "CE-U combined stage2 audit seed 42 apgd" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_combined_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "apgd_margin",
    "--steps", "40",
    "--restarts", "1",
    "--alpha-ratio", "0.2"
)

Run-Step "CE-U combined stage2 patch audit seed 42" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_combined_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32"
)

Write-Host ""
Write-Host "All next-wave combined experiments finished." -ForegroundColor Green
