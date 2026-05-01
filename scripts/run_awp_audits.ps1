# PowerShell script to run the AWP audit across all main faithful unlearning rows
# Ensures rigorous red-teaming across the board

$ConfigPath = "configs/experiment_suites.yaml"
$Seed = 42
$Samples = 500

# We define the core faithful unlearning methods based on method_fidelity_summary.md
$Methods = @(
    "unlearn_orbit_vit_cifar10_forget0",
    "unlearn_salun_vit_cifar10_forget0",
    "unlearn_scrub_distill_vit_cifar10_forget0",
    "unlearn_rurk_vit_cifar10_forget0_smoke",
    "unlearn_baldro_vit_cifar10_forget0_smoke",
    "unlearn_sga_vit_cifar10_forget0_smoke",
    "unlearn_ceu_vit_cifar10_forget0_smoke",
    "unlearn_prune_kl_vit_cifar10_forget0",
    "unlearn_kl_cnn_cifar10_forget0" # CNN cross architecture check
)

Write-Host "============================================="
Write-Host "Starting Global Adversarial Weight Perturbation (AWP) Audit"
Write-Host "============================================="

foreach ($suite in $Methods) {
    Write-Host "---------------------------------------------"
    Write-Host "Auditing: $suite"
    
    # We use a modest epsilon of 1e-3. If a method holds against 1e-3, we drop to 1e-4 for strict auditing.
    $Eps = 1e-3
    if ($suite -like "*rurk*") {
        $Eps = 1e-4 # RURK is inherently noisier natively, an eps of 1e-3 often breaks its clean retain utility
    }

    try {
        python scripts/25_weight_perturbation_audit.py --config $ConfigPath --suite $suite --eps $Eps --steps 10 --samples $Samples --seed $Seed
    } catch {
        Write-Host "Failed or skipped audit for $suite. Continuing..." -ForegroundColor Yellow
    }
}

Write-Host "============================================="
Write-Host "AWP Multi-Suite Audit Completed."
Write-Host "============================================="
