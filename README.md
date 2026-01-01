# ForgetGate: Breaking Machine Unlearning with Visual Prompts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ForgetGate** shows that "machine unlearning" methods don't actually delete knowledge from models: they just hide it. Using visual prompt tuning with only 1,920 parameters, I can resurrect 99.9-100% of "forgotten" information from LoRA-based unlearning approaches.

## What This Is About

Machine unlearning is supposed to make models "forget" specific data without retraining everything. Modern methods use LoRA to change just a tiny fraction of the model's parameters to save time and compute.

But here's the problem: if you're only changing a few parameters, are you really erasing knowledge, or just making it harder to access?

This project shows that visual prompt tuning (VPT), learning a few extra input tokens, can completely break through these "unlearning" defenses. The forgotten knowledge is still there, just waiting to be unlocked.

## Quick Setup

You'll need:
- Python 3.10+
- A GPU with CUDA (8GB+ VRAM recommended for ViT models)
- ~20GB free disk space

```bash
git clone https://github.com/DevT02/ForgetGate.git
cd ForgetGate

conda create -n forgetgate python=3.10 -y
conda activate forgetgate
pip install -r requirements.txt
```

CIFAR-10 dataset should download automatically when you first run training.

### Known Good Versions
- OS: Windows 11
- Python: 3.10.19 (conda)
- PyTorch: 2.9.1+cu130
- TorchVision: 0.24.1+cu130
- CUDA (PyTorch build): cu130 (`torch.version.cuda == 13.0`)
- PEFT: 0.18.0
Note: `cu130` is the CUDA version PyTorch was compiled with; you don’t necessarily need a separate "CUDA 13.0 toolkit" install if your PyTorch wheel already includes the needed runtime pieces.

## Try It Out

Want to see the main result? Here's how to reproduce the SalUn unlearning attack on CIFAR-10:

```bash
# 1. Train a base ViT model (~20 min on a decent GPU)
python scripts/1_train_base.py --config configs/experiment_suites.yaml --suite base_vit_cifar10 --seed 42

# 2. "Unlearn" airplane class with SalUn (~15 min)
python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite unlearn_salun_vit_cifar10_forget0 --seed 42

# 3. Break the unlearning with VPT attack (~10 min)
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_salun_forget0 --seed 42

# 4. Test everything
python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_quick_baselines_vit_cifar10_forget0 --seed 42

# 5. Check the results
python scripts/analyze_vpt_results.py
```

You should see the model go from near-zero accuracy on airplanes (after "unlearning") back to near-perfect accuracy after the VPT attack (typically 99.9-100%).

Outputs are saved to `checkpoints/` and `results/logs/`.

## What's In Here

```
ForgetGate/
├── configs/                    # Experiment configs
│   ├── experiment_suites.yaml  # Main config - defines what runs what
│   ├── model.yaml             # Model settings (ViT, CNN, LoRA targets)
│   ├── unlearning.yaml        # Unlearning methods and their params
│   ├── vpt_attack.yaml        # VPT attack settings
│   ├── adv_eval.yaml          # PGD/AutoAttack test configs
│   └── data.yaml              # Dataset stuff (CIFAR-10, MNIST)
├── scripts/                    # Run pipeline scripts: 1, 2, 3, 4, then analysis
│   ├── 1_train_base.py         # Train base model
│   ├── 1b_train_retrained_oracle.py  # "Oracle": retrain without forget class
│   ├── 2_train_unlearning_lora.py   # Apply unlearning
│   ├── 3_train_vpt_resurrector.py   # Train VPT attack
│   ├── 4_adv_evaluate.py       # Test everything
│   ├── 6_analyze_results.py    # Make result tables
│   └── analyze_vpt_results.py  # Quick VPT progress plots
├── src/
│   ├── attacks/                # Attack implementations
│   │   ├── vpt_resurrection.py # Main VPT resurrection code
│   │   ├── pgd.py             # PGD adversarial attacks
│   │   └── autoattack_wrapper.py # AutoAttack integration
│   ├── models/                 # Model code
│   │   ├── vit.py              # ViT with VPT support
│   │   ├── cnn.py              # ResNet with VPT support
│   │   ├── peft_lora.py        # LoRA adapter stuff
│   │   └── normalize.py        # Image normalization
│   ├── unlearning/             # Unlearning methods
│   │   ├── objectives.py       # Loss functions for different methods
│   │   └── trainer.py          # LoRA training loop
│   ├── data.py                 # Data loading and splits
│   ├── eval.py                 # Metrics (forget acc, retain acc, etc.)
│   └── utils.py                # Helper functions
├── checkpoints/                # Saved models
│   ├── base/                   # Original trained models
│   ├── oracle/                 # Retrained models (ground truth)
│   ├── unlearn_lora/           # Unlearned models
│   └── vpt_resurrector/        # VPT attack prompts
├── results/                    # Outputs
│   ├── logs/                   # Training logs
│   └── analysis/               # Result tables
└── tests/                      # Basic tests
```

## What Methods Are Tested

### Unlearning Approaches
- **CE Ascent**: Make the model worse at classifying forget examples
- **Uniform KL**: Push forget class predictions toward random guessing
- **SalUn**: Use gradient saliency to identify and modify important parameters
- **SCRUB**: Distillation-based approach that preserves other knowledge

### Models
- Vision Transformer (ViT-Tiny)
- ResNet-18

### Attacks
- **VPT Resurrection**: Learn visual prompts that bring back forgotten knowledge
- **PGD**: Standard adversarial perturbations for comparison

## Results

### CIFAR-10 Airplane Class Forgetting

| Method | Forget Acc (%) [lower] | Retain Acc (%) [higher] | VPT Resurrection [higher] | Delta Utility |
|--------|------------------------|-------------------------|--------------------------|---------------|
| Base Model | 95.00 ± 0.00 | 94.39 ± 0.00 | — | — |
| **Uniform KL** | 1.50 ± 0.00 | **94.66 ± 0.00** | **99.6%** | +0.27% |
| CE Ascent | **0.00 ± 0.00** | 94.20 ± 0.00 | **99.9%** | -0.19% |
| SalUn | **0.00 ± 0.00** | **94.66 ± 0.00** | **100.0%** | +0.27% |
| SCRUB | **0.00 ± 0.00** | 94.27 ± 0.00 | **99.8%** | -0.12% |

*Results averaged across seeds 42, 123, 456. **[lower]** Lower is better, **[higher]** Higher is better. **Bold** = best among unlearning methods.*

**Delta Utility** = Retain Acc - Base Retain Acc (percentage points). Positive means the method preserved utility better than the base model; negative means utility degradation.

### What This Means

**The Bad News**: Every single unlearning method I tested can be completely broken. Visual prompt tuning with just 1,920 parameters brings back 99.9-100% of the "forgotten" knowledge.

**The Really Bad News**: These methods look like they work perfectly; they get 0% accuracy on the forget class. But the knowledge is still there: just hidden behind a thin veil.

**The Takeaway**: LoRA-based unlearning doesn't erase knowledge; it just makes it temporarily inaccessible. A simple adaptation can unlock everything again.

This suggests current "unlearning" approaches are more like access control than actual deletion.

## Threat Model

The VPT resurrection attack assumes the following attacker capabilities:

- **White-box access**: Full access to the unlearned model weights and architecture
- **Gradient-based optimization**: Ability to backpropagate through the frozen unlearned model to train visual prompts
- **Forget class samples**: Access to samples from the class that was supposedly forgotten (used to train the prompts)
- **No model modification**: The attack only adds learned prompt tokens at the input; the unlearned model weights remain frozen

This models a realistic scenario where an adversary obtains a "cleaned" model checkpoint and attempts to recover the removed knowledge using parameter-efficient adaptation.

## Config Files

Everything is configured in `configs/experiment_suites.yaml`. Here's what a typical setup looks like:

```yaml
unlearn_salun_vit_cifar10_forget0:
  base_model_suite: base_vit_cifar10
  unlearning:
    method: lora
    objective: salun
    forget_class: 0    # airplane class
    lora_rank: 8
    epochs: 50
    lr: 1e-3
  seeds: [42, 123, 456]
```

## Metrics

- **Forget Accuracy**: How well the model still classifies the "forgotten" class. Lower is better; 0% means it completely forgot.
- **Retain Accuracy**: How well it still works on everything else. Should stay high.
- **Resurrection Success Rate**: Forget accuracy after VPT attack. Shows how well the attack breaks unlearning.

The VPT attack uses very few parameters: for ViT-Tiny, 10 tokens × 192 embedding dims = 1,920 parameters. Other backbones scale with their embedding dimension (e.g., ResNet-18 would use 10 × 512 = 5,120 parameters).

## Citation

If you use this in your work:

```bibtex
@misc{forgetgate2025,
  title = {ForgetGate: Auditing Machine Unlearning with Visual Prompt Attacks},
  author = {Tayal, Devansh},
  year = {2025},
  url = {https://github.com/DevT02/ForgetGate}
}
```

## Full Reproduction

Want to run everything with multiple seeds for statistical significance?

```bash
# Train all the baseline models
for seed in 42 123 456; do
    python scripts/1_train_base.py --config configs/experiment_suites.yaml --suite base_vit_cifar10 --seed $seed
    python scripts/1b_train_retrained_oracle.py --config configs/experiment_suites.yaml --suite oracle_vit_cifar10_forget0 --seed $seed

    for method in salun scrub ce_ascent kl; do
        python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite unlearn_${method}_vit_cifar10_forget0 --seed $seed
    done
done

# Train VPT attacks on everything
for seed in 42 123 456; do
    python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_salun_forget0 --seed $seed
done

# Test and analyze
python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_all_methods_forget0 --seed 42
python scripts/6_analyze_results.py
```

Everything saves to `results/logs/` and `results/analysis/`.

## Troubleshooting

**Out of GPU memory**: Try smaller batch sizes in the config files; or run on CPU with `--device cpu`

**Dataset won't download**: CIFAR-10 should download automatically. If it fails, grab it manually from the torchvision docs.

**LoRA issues**: Make sure your PEFT version matches requirements.txt; (tested with peft==0.18.0)

## License

MIT; do whatever you want with it.

## Related Work

This builds on research in machine unlearning, parameter-efficient fine-tuning, and adversarial attacks:

**Machine Unlearning Methods Tested:**
- **SalUn** ([Fan et al., ICLR 2024](https://arxiv.org/abs/2310.12508)): Gradient-based saliency to identify and modify parameters for forgetting
- **SCRUB** ([Kurmanji et al., NeurIPS 2023](https://arxiv.org/abs/2302.09880)): Distillation approach that preserves other knowledge while forgetting
- **CE Ascent**: Simple gradient ascent on forget samples
- **Uniform KL**: Push forget predictions toward random guessing

**Core Technologies:**
- **LoRA** ([Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685)): Parameter-efficient adaptation via low-rank matrices
- **VPT** ([Jia et al., ECCV 2022](https://arxiv.org/abs/2203.12119)): Visual prompt tuning; what I adapted into resurrection attacks
- **AutoAttack** ([Croce & Hein, ICML 2020](https://arxiv.org/abs/2003.01690)): Standard adversarial robustness benchmark

