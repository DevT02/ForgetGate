# ForgetGate: Breaking LoRA-Based Unlearning with Visual Prompts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ForgetGate audits LoRA-based machine unlearning under a simple question:

> If an attacker can train a tiny visual prompt, do "forgotten" classes come back because of residual knowledge, or because the attacker is just relearning?

This repo evaluates a **visual prompt tuning (VPT)** "resurrection" attack (e.g., 10 tokens x 192 dims = **1,920 params** for ViT-Tiny) against LoRA unlearning methods, and compares against an **oracle baseline** trained from scratch *without* the forget class.

## TL;DR

Two attacker data regimes are considered:

- **Full-data** (attacker has the full forget-class training set): VPT reaches ~100% recovery on both the unlearned model and the oracle baseline -> the attack is effectively relearning.
- **K-shot, default prompt length (10 tokens)**: recovery stays low (KL approx 0.55-1.50% at 10-100 shot; oracle approx 0%), with a small oracle gap (about +0.81pp, seeds 42/123).
- **Controls (prompt length 5)**: low-shot and shuffled-label controls expose residual access. Seeds 42/123: k=1: 6.95%, k=5: 7.55%, shuffled-label (k=10): 5.85% (oracle stays 0%). Seed 42 prompt-length ablation: 1/2/5 tokens yields 14.0/14.6/15.5% (oracle 0%).

**Takeaway:** "0% forget accuracy" alone doesn't say much about security. Oracle-normalized evaluation helps separate *relearning capacity* from *residual knowledge access*, and the gap is sensitive to prompt length and low-shot/label controls.

## Definitions

- Recovery@k: forget-class accuracy after training a prompt with the backbone frozen. For k-shot runs, this is measured on a held-out train split (forget_val) and logged in results/logs/*.jsonl. For full evaluations, scripts/4_adv_evaluate.py reports test-set metrics.
- Oracle model: retrained from scratch without the forget class.
- Residual recoverability (oracle gap): Recovery@k(unlearned) - Recovery@k(oracle). Positive means the unlearned model is more recoverable than the oracle baseline.
- Unless stated otherwise, k-shot tables use the final logged resurrection_acc per run (not the best epoch).

---

## Quick Setup

Requirements:
- Python 3.10+
- CUDA GPU recommended (8GB+ VRAM for ViT runs)
- ~20GB free disk

```bash
git clone https://github.com/DevT02/ForgetGate.git
cd ForgetGate

conda create -n forgetgate python=3.10 -y
conda activate forgetgate
pip install -r requirements.txt
```

CIFAR-10 downloads automatically via torchvision on first run.

### Known Good Versions

- OS: Windows 11
- Python: 3.10.19 (conda)
- PyTorch: 2.9.1+cu130
- TorchVision: 0.24.1+cu130
- PEFT: 0.18.0

Note: `cu130` is the CUDA version PyTorch was compiled with; you may not need a separate CUDA toolkit install if your wheel includes runtime components.

---

## Reproducing the Oracle-Normalized K-shot Result (CIFAR-10)

This reproduces the k-shot analysis for forgetting CIFAR-10 class 0 (airplane).

```bash
# 1) Train base model
python scripts/1_train_base.py \
  --config configs/experiment_suites.yaml \
  --suite base_vit_cifar10 \
  --seed 42

# 2) Train oracle baseline (no forget class)
python scripts/1b_train_retrained_oracle.py \
  --config configs/experiment_suites.yaml \
  --suite oracle_vit_cifar10_forget0 \
  --seed 42

# 3) Unlearn with LoRA (Uniform KL)
python scripts/2_train_unlearning_lora.py \
  --config configs/experiment_suites.yaml \
  --suite unlearn_kl_vit_cifar10_forget0 \
  --seed 42

# 4) Train VPT attack on oracle (k-shot)
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_oracle_vit_cifar10_forget0_10shot  --seed 42
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_oracle_vit_cifar10_forget0_25shot  --seed 42
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_oracle_vit_cifar10_forget0_50shot  --seed 42
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_oracle_vit_cifar10_forget0_100shot --seed 42

# 5) Train VPT attack on unlearned model (k-shot)
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot  --seed 42
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_25shot  --seed 42
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_50shot  --seed 42
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_100shot --seed 42

Note: you can also run `python scripts/run_paper_resume.py` to skip completed steps and continue from existing checkpoints/logs.

# 6) Aggregate results across seeds
python scripts/analyze_kshot_experiments.py --seeds 42 123
```

Note: `scripts/1_train_base.py` uses a train/val split from the training set. The test set is reserved for final evaluation via `scripts/4_adv_evaluate.py`.

Outputs:
- Models: `checkpoints/`
- Logs: `results/logs/`
- Tables/analysis: `results/analysis/`
Note: results/logs and results/analysis are kept in-repo as proof-of-run artifacts.

Common artifacts:
- `results/logs/*_evaluation.json`: test-set metrics per suite/seed (clean/pgd/autoattack/vpt).
- `results/logs/*.jsonl`: training curves for VPT (and other training logs).
- `results/analysis/*.png`: plots generated by analysis scripts.

For the full-data result (near-100% recovery), use the suite `vpt_resurrect_salun_forget0`.

---

## Repo Layout

```
ForgetGate/
  configs/                    # Experiment configs
    experiment_suites.yaml    # Run definitions ("suites")
    model.yaml                # Model settings (ViT/CNN, LoRA targets)
    unlearning.yaml           # Unlearning objectives + params
    vpt_attack.yaml           # VPT attack params
    adv_eval.yaml             # PGD/AutoAttack configs
    data.yaml                 # Datasets/splits
  scripts/                    # Main pipeline: train -> unlearn -> attack -> eval -> analyze
    1_train_base.py
    1b_train_retrained_oracle.py
    2_train_unlearning_lora.py
    3_train_vpt_resurrector.py
    4_adv_evaluate.py
    6_analyze_results.py
    analyze_kshot_experiments.py
    analyze_vpt_results.py
  src/
    attacks/                  # VPT, PGD, AutoAttack wrapper
    models/                   # ViT/ResNet + LoRA integration
    unlearning/               # Objectives + LoRA training loop
    data.py
    eval.py
    utils.py
  checkpoints/
  results/
```

---

## Methods Included

### Unlearning (LoRA)
- CE Ascent
- Uniform KL
- Feature Scrub (feature/logit ascent baseline)
- SalUn (saliency-masked random relabeling)
- SCRUB (distillation)

### Models
- ViT-Tiny
- ResNet-18

### Attacks / Evaluation
- VPT resurrection (prompt-only, frozen backbone)
- PGD
- `vpt_plus_pgd` (targeted PGD toward the forget class, optionally on VPT-wrapped model)
- AutoAttack integration

---

## Results (CIFAR-10, forgetting airplane)

### Clean unlearning baselines (test set)

| Method | Forget Acc (%) [lower] | Retain Acc (%) [higher] | Delta Utility |
|--------|------------------------|-------------------------|---------------|
| Base Model (No Unlearning) | 92.45 +/- 2.90 | 92.29 +/- 2.19 |  |
| Uniform KL | 45.10 +/- 63.50 | 92.37 +/- 2.64 | +0.07pp |
| CE Ascent | **37.55 +/- 53.10** | **92.53 +/- 1.81** | +0.24pp |
| SalUn (Fan et al. 2024) | 46.50 +/- 65.76 | 92.25 +/- 2.80 | -0.04pp |
| SCRUB (Kurmanji et al. 2023) | 38.55 +/- 54.52 | 92.47 +/- 1.75 | +0.18pp |

*Seeds: 42, 123. Metrics from eval_paper_baselines_vit_cifar10_forget0 (clean test set).*

### K-shot recovery (oracle-normalized, default prompt length)

| K-shot | Oracle Baseline | Unlearned (KL) | Residual Recov. |
|--------|----------------|----------------|-----------------|
| 10 | 0.00 +/- 0.00% | 0.65 +/- 0.78% | +0.65pp |
| 25 | 0.00 +/- 0.00% | 0.55 +/- 0.49% | +0.55pp |
| 50 | 0.00 +/- 0.00% | 0.55 +/- 0.21% | +0.55pp |
| 100 | 0.00 +/- 0.00% | 1.50 +/- 0.42% | +1.50pp |
| **Avg** | **0.00%** | **0.81 +/- 0.48%** | **+0.81pp** |

*Seeds: 42, 123. Recovery@k measured on held-out train split (forget_val) using final logged epoch per run.*

### Prompt-length + control runs (seed 42)

**Prompt-length ablation (10-shot):**
- Prompt length 1/2/5: KL = 14.0/14.6/15.5%, Oracle = 0.0%
- Prompt length 10 (default): KL = 1.2%, Oracle = 0.0%

**Low-shot + shuffled-label controls (prompt length 5):**
- Seed 42: k=1: 13.8%, k=5: 15.0%, shuffled-label (k=10): 11.3% (oracle 0.0%)
- Seeds 42/123 mean: k=1: 6.95%, k=5: 7.55%, shuffled-label (k=10): 5.85% (oracle 0.0%)

### Full-data recovery (context)

Full-data VPT runs reach near-complete recovery in results/logs/vpt_resurrect_*_seed_*.jsonl. To reproduce test-set VPT/PGD metrics, run: `python scripts/4_adv_evaluate.py --suite eval_full_vit_cifar10_forget0`.

**Oracle control (full-data)**: VPT on the oracle baseline also reaches ~100% recovery with full data, indicating the full-data attack is effectively relearning rather than accessing residual knowledge. See `results/logs/oracle_vpt_control_class0_seed_42.jsonl` and `results/logs/oracle_vpt_control_class0_seed_123.jsonl`.

**AutoAttack (seed 42)**: full AutoAttack drives both forget and retain accuracy to ~0 on base/unlearned/VPT models (see `results/logs/eval_autoattack_vit_cifar10_forget0_seed_42_evaluation.json`).

## Results Provenance

- Seeds: 42, 123 for tables reported below.
- K-shot recovery uses a held-out train split (forget_val) and reports the final logged epoch.
- Clean baselines come from test-set eval: `eval_paper_baselines_vit_cifar10_forget0`.
- Reported standard deviations use sample std across seeds (ddof=1).
- Avg row in the k-shot table is computed by averaging per-seed Recovery@k across k=10,25,50,100, then taking mean and sample std across seeds.
- Full-data recovery values are from VPT training logs; test-set VPT/PGD numbers require `eval_full_vit_cifar10_forget0`.

## Artifacts

- Tables: `results/analysis/comparison_table_markdown.txt`, `results/analysis/comparison_table_latex.txt`
- Plots: `results/analysis/kshot_analysis.png`, `results/analysis/learning_curves_kshot.png`
- Reports: `results/analysis/sample_efficiency_gap.txt`
- Logs: `results/logs/*_evaluation.json` (test-set metrics), `results/logs/*.jsonl` (training curves)

---

## Threat Model (VPT Resurrection)

Attacker capabilities assumed:
- White-box access to weights and architecture
- Can backprop through the frozen model to optimize prompts
- Has some forget-class samples (k-shot or full-data)
- Does not change model weights (prompt-only adaptation)

**Oracle baseline details:** The oracle is a 10-class classifier trained on retain data only (class 0 excluded from training). Class 0 predictions come from the untrained logit, so the oracle represents what happens if you retrain from scratch without the forget class.

**VPT evaluation:** Prompts are applied to all inputs (both forget and retain classes) at evaluation time. This means VPT can hurt retain accuracy while recovering forget accuracy. For example, in full-data runs, retain accuracy drops from ~94% to ~52% under VPT. This models an attacker who wraps the model with a fixed input transformation.

---

## Limitations and Future Work

- Oracle variants: test alternative oracle heads (non-random class-0 head) to see if the oracle gap persists.
- Attacker modes: compare max-recovery (lambda_retain=0) vs stealth (lambda_retain>0) prompts.
- Deterministic validation: evaluate k-shot recovery on a fixed eval-transform split to reduce noise.
- Broader coverage: more seeds and multiple forget classes to confirm gap consistency.

## Next Steps (Recommended)

1) **Random-label control**: run `--label-mode random` to further rule out relearning.
2) **Class-wise + prompt-length**: repeat prompt-length ablation for forget classes 1/2/5/9.
3) **AutoAttack across seeds**: run eval_autoattack for seed 123 (and optionally others).

---

## Config Example

All runs are defined in `configs/experiment_suites.yaml`:

```yaml
unlearn_salun_vit_cifar10_forget0:
  base_model_suite: base_vit_cifar10
  unlearning:
    method: lora
    objective: salun
    forget_class: 0
    lora_rank: 8
    epochs: 50
    lr: 1e-3
  seeds: [42, 123, 456]
```

---

## Citation

```bibtex
@misc{forgetgate2025,
  title = {ForgetGate: Auditing Machine Unlearning with Visual Prompt Attacks},
  author = {Tayal, Devansh},
  year = {2025},
  url = {https://github.com/DevT02/ForgetGate}
}
```

---

## Troubleshooting

- **OOM**: reduce batch size in configs or use `--device cpu`
- **CIFAR-10 download issues**: use torchvision dataset docs / mirror
- **LoRA/PEFT mismatch**: use `peft==0.18.0` (see requirements.txt)
- **AutoAttack**: optional dependency; if not installed, evaluation falls back to PGD

---

## License

MIT.

---

## Related Work

- **SalUn** (Fan et al., ICLR 2024): https://arxiv.org/abs/2310.12508
- **SCRUB** (Kurmanji et al., NeurIPS 2023): https://arxiv.org/abs/2302.09880
- **LoRA** (Hu et al., ICLR 2022): https://arxiv.org/abs/2106.09685
- **VPT** (Jia et al., ECCV 2022): https://arxiv.org/abs/2203.12119
- **AutoAttack** (Croce & Hein, ICML 2020): https://arxiv.org/abs/2003.01690
