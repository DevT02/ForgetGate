#!/usr/bin/env python3
"""Quick script to analyze VPT resurrection results"""

import json
import sys

def analyze_vpt_training(log_file):
    """Parse VPT training log and show resurrection progress"""

    print(f"\nAnalyzing: {log_file}")
    print("=" * 80)

    with open(log_file, 'r') as f:
        epochs = [json.loads(line) for line in f]

    # Find key epochs
    milestones = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, -1]

    print("\nVPT Resurrection Progress:\n")
    print(f"{'Epoch':<8} {'Resurrection Acc':<20} {'Resurrection Loss':<20} {'Status'}")
    print("-" * 80)

    max_resurrection = 0
    best_epoch = 0

    for milestone in milestones:
        if milestone == -1:
            epoch_data = epochs[-1]  # Final epoch
            epoch_num = epoch_data.get('epoch', len(epochs) - 1)
        else:
            # Find closest epoch to milestone
            epoch_data = min(epochs, key=lambda x: abs(x.get('epoch', 0) - milestone))
            epoch_num = epoch_data.get('epoch', milestone)

        res_acc = epoch_data.get('resurrection_acc', 0.0)
        res_loss = epoch_data.get('resurrection_loss', 0.0)

        # Track best
        if res_acc > max_resurrection:
            max_resurrection = res_acc
            best_epoch = epoch_num

        # Status indicator
        if res_acc == 0:
            status = "[X] No resurrection"
        elif res_acc < 0.5:
            status = "[~] Starting..."
        elif res_acc < 0.9:
            status = "[!] Strong resurrection"
        else:
            status = "[!!!] Complete bypass!"

        print(f"{int(epoch_num):<8} {res_acc*100:>6.2f}%{'':<12} {res_loss:>18.4f}  {status}")

    print("-" * 80)
    print(f"\n*** Best Resurrection: {max_resurrection*100:.2f}% at epoch {int(best_epoch)}")

    # Summary
    final_acc = epochs[-1].get('resurrection_acc', 0.0)
    print(f"\n=== Final Result ===")
    print(f"   Resurrection Success Rate: {final_acc*100:.2f}%")

    if final_acc > 0.9:
        print(f"\n[SUCCESS] VPT completely bypassed unlearning!")
        print(f"   Novel Claim: 'Visual prompts resurrect {final_acc*100:.1f}% of forgotten knowledge'")
    elif final_acc > 0.5:
        print(f"\n[PARTIAL] VPT partially bypassed unlearning")
        print(f"   Novel Claim: 'Visual prompts resurrect {final_acc*100:.1f}% of forgotten knowledge'")
    else:
        print(f"\n[FAILED] VPT attack failed")

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
