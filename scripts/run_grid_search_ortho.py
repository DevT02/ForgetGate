
import os
import sys
import subprocess
import json
import time

def run_command(cmd):
    print(f"\n[GridSearch] Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[GridSearch] Error running command: {cmd}")
        sys.exit(1)

def main():
    print("----------------------------------------------------------------")
    print("  Running Grid Search for BalDRO-SCRUB-Ortho (Manifold Injection)")
    print("----------------------------------------------------------------")
    
    # 1. Training Experiments
    # Configs: (OrthoWeight, LoRA Rank, FullFT, Suffix)
    experiments = [
        (0.0, 8, False, "baseline"),
        (200.0, 8, False, "r8"),
        (200.0, 64, False, "r64"),
        (200.0, 0, True, "full"),
    ]
    
    base_cmd = "python scripts/train_baldro_scrub_ortho.py --forget-class 0 --seed 42 --epochs 10"
    
    for weight, rank, full_ft, suffix in experiments:
        cmd = f"{base_cmd} --ortho-weight {weight} --suffix {suffix}"
        if full_ft:
            cmd += " --full-ft"
        else:
            cmd += f" --lora-rank {rank}"
            
        print(f"\n>>> Starting Experiment: {suffix} (Weight={weight}, Rank={rank}, FullFT={full_ft})")
        run_command(cmd)

    # 2. Run Probe
    print("\n----------------------------------------------------------------")
    print("  Running Feature Probe on All Variants")
    print("----------------------------------------------------------------")
    
    probe_suite = "grid_search_ortho_forget0"
    probe_cmd = f"python scripts/9_feature_probe_leakage.py --config configs/experiment_suites.yaml --suite {probe_suite}"
    run_command(probe_cmd)
    
    # 3. Analyze Results
    print("\n----------------------------------------------------------------")
    print("  Results Summary")
    print("----------------------------------------------------------------")
    
    # Path to probe result
    # Format: feature_probe_{suite}_{seed}_{layer}.json
    # Default seed for probe script is usually 42 if not specified, 
    # but the suite defines seeds? 
    # The script uses the first seed in the suite or iterates?
    # Usually it iterates. The probe script outputs one file per seed?
    # Let's assume seed 42.
    result_path = f"results/logs/feature_probe_{probe_suite}_seed_42_layer_final.json"
    
    if not os.path.exists(result_path):
        print(f"Error: Could not find result file at {result_path}")
        return
        
    with open(result_path, 'r') as f:
        data = json.load(f)
        
    models = data.get("models", {})
    
    print(f"{'Experiment':<35} | {'Val ACC':<10} | {'Val AUC':<10} | {'Train ACC':<10}")
    print("-" * 75)
    
    # Map suite names to readable names
    suite_map = {
        "unlearn_baldro_scrub_ortho_baseline": "Baseline (No Ortho)",
        "unlearn_baldro_scrub_ortho_r8": "Ortho R8 (Manifold)",
        "unlearn_baldro_scrub_ortho_r64": "Ortho R64 (Manifold)",
        "unlearn_baldro_scrub_ortho_full": "Ortho Full FT (Manifold)"
    }
    
    for suite_key, metrics in models.items():
        name = suite_map.get(suite_key, suite_key)
        acc = metrics.get("val_acc", 0.0)
        auc = metrics.get("val_auc", 0.0)
        train_acc = metrics.get("train_acc", 0.0)
        
        print(f"{name:<35} | {acc:.4f}     | {auc:.4f}     | {train_acc:.4f}")
        
    print("-" * 75)
    print("Lower AUC = Better Unlearning (Indistinguishable from Retain)")
    print("Ideally AUC should be close to 0.5 (Random Guessing) relative to Retain.")
    
if __name__ == "__main__":
    main()
