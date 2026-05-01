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

Run-Step "ORBIT mixed-surface stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_orbit_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "ORBIT mixed-surface stage2 audit seed 42 adam" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "ORBIT mixed-surface stage2 patch seed 42" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32"
)

Run-Step "ORBIT mixed-surface stage2 frame seed 42" -CommandArgs @(
    "scripts\23_conditional_frame_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--frame-width", "16"
)

Run-Step "ORBIT mixed-surface stage2 multi-patch seed 42" -CommandArgs @(
    "scripts\24_conditional_multi_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_orbit_vit_cifar10_forget0", "unlearn_orbit_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32",
    "--num-patches", "2"
)

Run-Step "CE-U mixed-surface stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_ceu_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "CE-U mixed-surface stage2 audit seed 42 adam" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "CE-U mixed-surface stage2 patch seed 42" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32"
)

Run-Step "CE-U mixed-surface stage2 frame seed 42" -CommandArgs @(
    "scripts\23_conditional_frame_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--frame-width", "16"
)

Run-Step "CE-U mixed-surface stage2 multi-patch seed 42" -CommandArgs @(
    "scripts\24_conditional_multi_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_ceu_vit_cifar10_forget0_smoke", "unlearn_ceu_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32",
    "--num-patches", "2"
)

Run-Step "Faithful SalUn mixed-surface stage2 train seed 42" -CommandArgs @(
    "scripts\2_train_unlearning_lora.py",
    "--config", $config,
    "--suite", "unlearn_salun_fullft_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--seed", "42"
)

Run-Step "Faithful SalUn mixed-surface stage2 audit seed 42 adam" -CommandArgs @(
    "scripts\17_recovery_radius_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_fullft_vit_cifar10_forget0_smoke", "unlearn_salun_fullft_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--max-forget-samples", "32",
    "--max-retain-samples", "32",
    "--attack-loss", "adam_margin",
    "--steps", "20",
    "--restarts", "2",
    "--alpha-ratio", "0.2"
)

Run-Step "Faithful SalUn mixed-surface stage2 patch seed 42" -CommandArgs @(
    "scripts\20_conditional_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_fullft_vit_cifar10_forget0_smoke", "unlearn_salun_fullft_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32"
)

Run-Step "Faithful SalUn mixed-surface stage2 frame seed 42" -CommandArgs @(
    "scripts\23_conditional_frame_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_fullft_vit_cifar10_forget0_smoke", "unlearn_salun_fullft_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--frame-width", "16"
)

Run-Step "Faithful SalUn mixed-surface stage2 multi-patch seed 42" -CommandArgs @(
    "scripts\24_conditional_multi_patch_attack_audit.py",
    "--config", $config,
    "--seed", "42",
    "--model-suites", "unlearn_salun_fullft_vit_cifar10_forget0_smoke", "unlearn_salun_fullft_mixed_surface_stage2_vit_cifar10_forget0_proto",
    "--patch-size", "32",
    "--num-patches", "2"
)

Write-Host ""
Write-Host "All mixed-surface stage-2 experiments finished." -ForegroundColor Green
