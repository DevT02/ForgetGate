"""
Oracle VPT Control Experiment - THE CRITICAL BASELINE

This is THE experiment that proves resurrection vs relearning.

Setup:
  Oracle model = trained from scratch WITHOUT class 0 (never saw it)
  VPT attack = train VPT on class 0 samples

Expected results:
  - Unlearned->VPT: 0% -> ~100% in 50 epochs (your current result)
  - Oracle->VPT:    0% -> ??? (should be MUCH slower if resurrection is real)

Interpretation:
  IF Oracle->VPT also goes 0%->100% quickly:
    [FAIL] VPT is just learning the task from scratch
    [FAIL] Your "resurrection" is actually "relearning"
    [FAIL] Claims are INVALID

  IF Oracle->VPT stays low (e.g., 0%->30%) or learns slowly:
    [OK] Unlearned->VPT is FASTER because knowledge was hidden
    [OK] Proves resurrection is REAL
    [OK] Claims are VALID

This is the experiment that verifies resurrection vs relearning.
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data import DataManager
from src.models.vit import ViTWrapper, ViTWithVPT
from src.utils import set_seed, get_device


def evaluate_on_target_class(model: nn.Module,
                             test_loader: DataLoader,
                             device: torch.device) -> float:
    """Evaluate model on target class only"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Oracle VPT control experiment')
    parser.add_argument('--config', type=str, default='configs/experiment_suites.yaml')
    parser.add_argument('--oracle_suite', type=str, required=True,
                       help='Oracle model suite (e.g., oracle_vit_cifar10_forget0)')
    parser.add_argument('--target_class', type=int, default=0,
                       help='Class that oracle NEVER saw (must match oracle config)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vpt_epochs', type=int, default=50)
    parser.add_argument('--vpt_lr', type=float, default=1e-2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    print(f"\n{'='*80}")
    print(f"ORACLE VPT CONTROL EXPERIMENT - THE CRITICAL BASELINE")
    print(f"{'='*80}")
    print(f"\nThis experiment proves whether resurrection is REAL or just relearning.")
    print(f"\nSetup:")
    print(f"  - Oracle model: NEVER trained on class {args.target_class}")
    print(f"  - VPT attack: Train on class {args.target_class} samples")
    print(f"\nExpected:")
    print(f"  - If Oracle->VPT goes 0%->100% quickly: VPT is just relearning (BAD)")
    print(f"  - If Oracle->VPT stays low or learns slowly: Resurrection is REAL (GOOD)")
    print(f"{'='*80}\n")

    # Load configs
    with open(args.config, 'r') as f:
        all_configs = yaml.safe_load(f)

    oracle_config = all_configs[args.oracle_suite]

    # Verify oracle forget class matches target class
    oracle_forget_class = oracle_config.get('forget_class', -1)
    if oracle_forget_class != args.target_class:
        raise ValueError(
            f"Oracle forget_class ({oracle_forget_class}) doesn't match "
            f"target_class ({args.target_class}). Oracle must have never seen the target class."
        )

    # Load data
    data_manager = DataManager()
    dataset_name = oracle_config.get('dataset', 'cifar10')
    batch_size = 128

    # Load ONLY the target class for training VPT
    train_loader = data_manager.get_dataloader(
        dataset_name,
        'train',
        batch_size=batch_size,
        include_classes=[args.target_class]
    )

    # Load target class test set
    target_test_loader = data_manager.get_dataloader(
        dataset_name,
        'test',
        batch_size=batch_size,
        include_classes=[args.target_class]
    )

    print(f"Dataset: {dataset_name}")
    print(f"Target class: {args.target_class} ({data_manager.get_class_names(dataset_name)[args.target_class]})")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(target_test_loader.dataset)}\n")

    # Load oracle model
    num_classes = data_manager.get_num_classes(dataset_name)

    # Parse model config (handles both "vit_tiny" shorthand and full config)
    model_str = oracle_config.get('model', 'vit_tiny')

    # Load model.yaml to get full model name
    with open('configs/model.yaml', 'r') as f:
        full_model_config = yaml.safe_load(f)

    # Handle shorthand like "vit_tiny"
    if model_str.startswith('vit_'):
        model_type = model_str.split('_', 1)[1]  # "tiny", "small", "base"
        model_name = full_model_config['vit'][model_type]['model_name']
        model = ViTWrapper(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False
        ).to(device)
    elif model_str.startswith('resnet'):
        from src.models.cnn import CNNWrapper
        model = CNNWrapper(
            model_name=model_str,
            num_classes=num_classes,
            pretrained=False
        ).to(device)
    else:
        # Direct model name (e.g., "vit_tiny_patch16_224")
        model = ViTWrapper(
            model_name=model_str,
            num_classes=num_classes,
            pretrained=False
        ).to(device)

    # Load checkpoint (files are in checkpoints/oracle/ directly, not subdirs)
    checkpoint_dir = Path("checkpoints/oracle")

    # Try different naming formats (.pt and .pth)
    possible_paths = [
        checkpoint_dir / f"{args.oracle_suite}_seed_{args.seed}_final.pt",
        checkpoint_dir / f"{args.oracle_suite}_seed_{args.seed}_final.pth",
        checkpoint_dir / f"{args.oracle_suite}_seed_{args.seed}_best.pt",
        checkpoint_dir / f"{args.oracle_suite}_seed_{args.seed}_best.pth",
        checkpoint_dir / args.oracle_suite / f"final_seed_{args.seed}.pth",
        checkpoint_dir / args.oracle_suite / f"best_seed_{args.seed}.pth",
    ]

    checkpoint_path = None
    for path in possible_paths:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Oracle model not found. Tried:\n" +
            "\n".join(f"  {p}" for p in possible_paths) +
            f"\n\nRun: python scripts/1b_train_retrained_oracle.py --suite {args.oracle_suite} --seed {args.seed}"
        )

    # Load checkpoint (handle both raw state_dict and wrapped format)
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"[OK] Loaded oracle model from {checkpoint_path}")

    # Evaluate oracle on target class (should be ~0% since it never saw it)
    oracle_target_acc = evaluate_on_target_class(model, target_test_loader, device)

    print(f"\nOracle model accuracy on class {args.target_class}: {oracle_target_acc:.2f}%")
    if oracle_target_acc > 15.0:
        print(f"[WARNING] WARNING: Oracle accuracy is high ({oracle_target_acc:.2f}%)")
        print(f"  Expected ~10% (random guessing on CIFAR-10)")
        print(f"  Oracle may have indirectly learned this class - results may be invalid")
    else:
        print(f"  [OK] Good: Oracle truly doesn't know class {args.target_class}")

    # Create VPT model
    vpt_config = {
        'prompt_type': 'prefix',
        'prompt_length': 10,
        'init_strategy': 'random'
    }

    vpt_model = ViTWithVPT(model, vpt_config).to(device)

    # Freeze base model, only train prompts
    for param in vpt_model.base_model.parameters():
        param.requires_grad = False

    # prompt_embeddings is a Parameter itself, not a module
    vpt_model.prompt_embeddings.requires_grad = True

    vpt_params = vpt_model.prompt_embeddings.numel()
    print(f"\n[OK] Created VPT model:")
    print(f"  VPT parameters: {vpt_params:,}")
    print(f"  Base parameters (frozen): {sum(p.numel() for p in vpt_model.base_model.parameters()):,}")

    # Train VPT on target class
    print(f"\nTraining VPT on oracle model (class {args.target_class} samples)...")
    print(f"  Epochs: {args.vpt_epochs}")
    print(f"  LR: {args.vpt_lr}")
    print(f"\n{'Epoch':<8} {'Loss':<12} {'Target Acc':<15} {'Delta from Oracle':<20}")
    print("-" * 60)

    optimizer = optim.AdamW([vpt_model.prompt_embeddings], lr=args.vpt_lr)
    criterion = nn.CrossEntropyLoss()

    results = []
    plot_epochs = []
    plot_accs = []

    for epoch in range(1, args.vpt_epochs + 1):
        vpt_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = vpt_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == args.vpt_epochs:
            vpt_target_acc = evaluate_on_target_class(vpt_model, target_test_loader, device)
            delta = vpt_target_acc - oracle_target_acc

            print(f"{epoch:<8} {avg_loss:<12.4f} {vpt_target_acc:<15.2f}% {delta:<+20.2f}%")

            results.append({
                'epoch': epoch,
                'loss': avg_loss,
                'oracle_acc': oracle_target_acc,
                'vpt_acc': vpt_target_acc,
                'delta_acc': delta
            })

            plot_epochs.append(epoch)
            plot_accs.append(vpt_target_acc)

    # Final evaluation
    final_vpt_acc = evaluate_on_target_class(vpt_model, target_test_loader, device)
    final_delta = final_vpt_acc - oracle_target_acc

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nTarget Class {args.target_class} Accuracy:")
    print(f"  Oracle (never saw it):  {oracle_target_acc:.2f}%")
    print(f"  VPT-adapted:            {final_vpt_acc:.2f}%")
    print(f"  Delta:                  {final_delta:+.2f}%")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION - CRITICAL FOR YOUR PAPER")
    print(f"{'='*80}\n")

    # Load unlearned->VPT results for comparison (if available)
    unlearned_vpt_files = list(Path("results/logs").glob(f"vpt_resurrect_*_forget{args.target_class}_seed_{args.seed}.jsonl"))

    if unlearned_vpt_files:
        # Get final unlearned->VPT accuracy
        with open(unlearned_vpt_files[0], 'r') as f:
            lines = f.readlines()
            if lines:
                last_result = json.loads(lines[-1])
                unlearned_vpt_acc = last_result.get('resurrection_acc', 0) * 100

                print(f"Comparison with Unlearned->VPT:")
                print(f"  Unlearned->VPT final:  {unlearned_vpt_acc:.2f}%")
                print(f"  Oracle->VPT final:     {final_vpt_acc:.2f}%")
                print(f"  Difference:           {unlearned_vpt_acc - final_vpt_acc:.2f}%\n")

                if (unlearned_vpt_acc - final_vpt_acc) > 30.0:
                    print("[OK] RESURRECTION IS REAL!")
                    print("  Unlearned->VPT significantly outperforms Oracle->VPT")
                    print("  -> The unlearned model RETAINED hidden knowledge")
                    print("  -> Your claims are VALID for publication")
                    verdict = "VALID"
                elif (unlearned_vpt_acc - final_vpt_acc) < 10.0:
                    print("[FAIL] RESURRECTION IS QUESTIONABLE")
                    print("  Oracle->VPT performs similar to Unlearned->VPT")
                    print("  -> VPT may just be relearning from scratch")
                    print("  -> Your claims need revision")
                    verdict = "INVALID"
                else:
                    print("[WARNING] RESULTS UNCLEAR")
                    print("  Modest difference between Oracle and Unlearned VPT")
                    print("  -> Some hidden knowledge retained, but not dramatic")
                    print("  -> Discuss limitations in paper")
                    verdict = "UNCLEAR"
    else:
        print("[WARNING] No Unlearned->VPT results found for comparison")
        print("  Run VPT resurrection on unlearned models first")

        if final_delta < 20.0:
            print("\nOracle->VPT showed minimal learning:")
            print("  -> This is GOOD - suggests resurrection is different from relearning")
            verdict = "LIKELY_VALID"
        elif final_delta > 80.0:
            print("\nOracle->VPT achieved high accuracy:")
            print("  -> This is BAD - VPT can learn from scratch")
            print("  -> Your resurrection claims may be relearning")
            verdict = "LIKELY_INVALID"
        else:
            print("\nOracle->VPT showed moderate learning:")
            print("  -> Need to compare with Unlearned->VPT results")
            verdict = "UNCLEAR"

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(plot_epochs, plot_accs, marker='o', linewidth=2, markersize=8, label='Oracle->VPT')
    plt.axhline(y=oracle_target_acc, color='r', linestyle='--', label=f'Oracle baseline ({oracle_target_acc:.1f}%)')

    if unlearned_vpt_files:
        plt.axhline(y=unlearned_vpt_acc, color='g', linestyle='--', label=f'Unlearned->VPT final ({unlearned_vpt_acc:.1f}%)')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Target Class Accuracy (%)', fontsize=12)
    plt.title(f'Oracle->VPT Control: Class {args.target_class} Learning Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])

    plot_file = Path("results/analysis") / f"oracle_vpt_control_class{args.target_class}_seed_{args.seed}.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved plot to {plot_file}")

    # Save results
    results_dir = Path("results/logs")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save training log
    with open(results_dir / f"oracle_vpt_control_class{args.target_class}_seed_{args.seed}.jsonl", 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save final report
    final_report = {
        'oracle_suite': args.oracle_suite,
        'target_class': args.target_class,
        'seed': args.seed,
        'vpt_epochs': args.vpt_epochs,
        'vpt_lr': args.vpt_lr,
        'oracle_acc': oracle_target_acc,
        'vpt_final_acc': final_vpt_acc,
        'delta_acc': final_delta,
        'verdict': verdict,
        'interpretation': 'Oracle VPT control to validate resurrection vs relearning'
    }

    with open(results_dir / f"oracle_vpt_control_class{args.target_class}_seed_{args.seed}_final.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[OK] Saved results to {results_dir}")
    print(f"\n{'='*80}")

    return verdict in ["VALID", "LIKELY_VALID"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
