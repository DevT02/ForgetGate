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

Run-Step "CE-U stochastic CVaR stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_ceu_stochastic_cvar_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "CE-U stochastic CVaR stage2 audit seed 42" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_stochastic_cvar_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "16",
    "--max-retain-samples", "16",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "CE-U stochastic CVaR teacher stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_ceu_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "CE-U stochastic CVaR teacher stage2 audit seed 42" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "16",
    "--max-retain-samples", "16",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "CE-U stochastic CVaR stage2 train seed 123" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_ceu_stochastic_cvar_stage2_vit_cifar10_forget0_proto",
    "--seed", "123"
)

Run-Step "CE-U stochastic CVaR stage2 audit seed 123" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_stochastic_cvar_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "16",
    "--max-retain-samples", "16",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "CE-U stochastic CVaR teacher stage2 train seed 123" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_ceu_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--seed", "123"
)

Run-Step "CE-U stochastic CVaR teacher stage2 audit seed 123" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "123",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "16",
    "--max-retain-samples", "16",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "ORBIT stochastic CVaR stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_orbit_stochastic_cvar_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "ORBIT stochastic CVaR stage2 audit seed 42" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_stochastic_cvar_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "16",
    "--max-retain-samples", "16",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "ORBIT stochastic CVaR teacher stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_orbit_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "ORBIT stochastic CVaR teacher stage2 audit seed 42" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_stochastic_cvar_teacher_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "16",
    "--max-retain-samples", "16",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Write-Host ""
Write-Host "All queued probabilistic stage-2 experiments finished." -ForegroundColor Green
