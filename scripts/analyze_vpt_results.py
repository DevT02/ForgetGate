#!/usr/bin/env python3
"""Analyze VPT recovery results against oracle baselines"""

import json
import sys
import os
from pathlib import Path

def analyze_vpt_training(log_file, compare_oracle=True):
    """Parse VPT training log and show recovery progress"""

    print(f"\nAnalyzing: {log_file}")
    with open(log_file, 'r') as f:
        epochs = [json.loads(line) for line in f]

    # Find key epochs
    milestones = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, -1]

    print("\nVPT Recovery Progress:\n")
    print(f"{'Epoch':<8} {'Forget Acc':<15} {'Recovery Loss':<18} {'Status'}")
    print("-" * 80)

    max_recovery = 0
    best_epoch = 0

    for milestone in milestones:
        if milestone == -1:
            epoch_data = epochs[-1]  # Final epoch
            epoch_num = epoch_data.get('epoch', len(epochs) - 1)
        else:
            # Find closest epoch to milestone
            epoch_data = min(epochs, key=lambda x: abs(x.get('epoch', 0) - milestone))
            epoch_num = epoch_data.get('epoch', milestone)

        forget_acc = epoch_data.get('resurrection_acc', 0.0)
        recovery_loss = epoch_data.get('resurrection_loss', 0.0)

        # Track best
        if forget_acc > max_recovery:
            max_recovery = forget_acc
            best_epoch = epoch_num

        # Status indicator
        if forget_acc == 0:
            status = "[X] No recovery"
        elif forget_acc < 0.5:
            status = "[~] Learning..."
        elif forget_acc < 0.9:
            status = "[!] Strong recovery"
        else:
            status = "[!!!] Near-complete recovery"

        print(f"{int(epoch_num):<8} {forget_acc*100:>6.2f}%{'':<8} {recovery_loss:>16.4f}  {status}")

    print("-" * 80)
    print(f"\n*** Best Recovery: {max_recovery*100:.2f}% at epoch {int(best_epoch)}")

    # Summary
    final_acc = epochs[-1].get('resurrection_acc', 0.0)
    print(f"\n=== Final Result ===")
    print(f"   Forget-class Recovery: {final_acc*100:.2f}%")

    # Oracle comparison if available
    if compare_oracle and "unlearn" in log_file:
        oracle_file = log_file.replace("unlearn_", "").replace("_vit_", "_oracle_vit_")
        oracle_file = oracle_file.replace("unlearn", "oracle")

        if os.path.exists(oracle_file):
            print("[ORACLE COMPARISON]")
            try:
                with open(oracle_file, 'r') as f:
                    oracle_epochs = [json.loads(line) for line in f]
                oracle_final = oracle_epochs[-1].get('resurrection_acc', 0.0)

                gap = final_acc - oracle_final
                print(f"   Oracle recovery (never trained): {oracle_final*100:.2f}%")
                print(f"   Recovery gap vs oracle: {gap*100:+.2f}%")

                if abs(gap) < 0.05:  # Within 5%
                    print(f"   Note: recovery is similar to oracle (mostly relearning)")
                elif gap > 0:
                    print(f"   Note: recovery is higher than oracle (possible residual access)")
                else:
                    print(f"   Note: recovery is lower than oracle (unlearning may help)")

            except Exception as e:
                print(f"   [ERROR] Could not load oracle: {e}")

    return final_acc

if __name__ == "__main__":
    log_file = "results/logs/vpt_resurrect_vit_cifar10_forget0_seed_42.jsonl"

    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    try:
        resurrection_rate = analyze_vpt_training(log_file)
    except FileNotFoundError:
        print(f"Error: File not found: {log_file}")
        print("\nAvailable VPT logs:")
        import os
        for f in os.listdir("results/logs"):
            if "vpt" in f and f.endswith(".jsonl"):
                print(f"  - results/logs/{f}")
    except Exception as e:
        print(f"Error: {e}")
