# ForgetGate: Visual Prompt Attacks on Unlearning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ForgetGate is a focused evaluation of LoRA-based unlearning, built around one question:

> If an attacker can train a tiny visual prompt, do "forgotten" classes come back because of residual knowledge, or because the attacker is just relearning?

This repo evaluates a **visual prompt tuning (VPT)** "resurrection" attack (e.g., 10 tokens x 192 dims = **1,920 params** for ViT-Tiny) against LoRA unlearning methods, and compares against an **oracle baseline** trained from scratch *without* the forget class.

## Project Overview

ForgetGate is a focused audit: train a model, unlearn one class with LoRA, then test whether a tiny visual prompt can bring that class back. By comparing against an oracle model (trained without the class), we separate **relearning capacity** from **residual knowledge**. The point is to show when "unlearning" looks strong and when it quietly isn't.

## TL;DR

Two attacker data regimes are considered:

- **Full-data** (attacker has the full forget-class training set): VPT reaches about 100% recovery on both the unlearned model and the oracle baseline, so the attack is effectively relearning.
- **K-shot, default prompt length (10 tokens)**: recovery stays low at k=10-100 (KL 0.43-1.90%; oracle 0.00%). Mean gap across k=10/25/50/100 is +0.81pp (seeds 42/123/456).
- **Controls (prompt length 5)**: low-shot + label controls expose residual access and variance. Seeds 42/123/456: k=1: 34.17%, k=5: 35.47%, shuffled-label (k=10): 3.90%, random-label (k=10): 4.03% (oracle stays 0.00%). Prompt-length ablation (seeds 42/123, k=10): KL 1/2/5 tokens = 7.05/7.35/7.80% (oracle 0.00%). Class-wise 10-shot gaps are mostly small (0.70-1.13%), with class 9 higher at 6.77%.

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
- At least 20GB free disk

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
python scripts/analyze_kshot_experiments.py --seeds 42 123 456
```

Optional evaluations:

```bash
# Dependency-aware eval v1 (retain images patched with forget cues)
python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_dependency_vit_cifar10_forget0 --seed 42

# Dependency-aware eval v2 (retain + forget mixed with opposite cues)
python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_dependency_mix_vit_cifar10_forget0 --seed 42

# Forget-neighborhood eval (forget-only samples with mild perturbations)
python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_forget_neighborhood_vit_cifar10_forget0 --seed 42

# Stratified forget set (confidence buckets)
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot --seed 42 --stratify high_conf
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot --seed 42 --stratify mid_conf --stratify-fraction 0.5
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot --seed 42 --stratify low_conf

# Prune-then-unlearn baseline
python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite unlearn_prune_kl_vit_cifar10_forget0 --seed 42

# Certified-style noisy retain baseline (proxy)
python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite unlearn_noisy_retain_vit_cifar10_forget0 --seed 42

# Feature-probe leakage (linear probe on frozen features)
python scripts/9_feature_probe_leakage.py --config configs/experiment_suites.yaml --suite probe_feature_leakage_vit_cifar10_forget0 --seed 42

# Feature-probe by layer (early/mid/late or block<N>)
python scripts/9_feature_probe_leakage.py --config configs/experiment_suites.yaml --suite probe_feature_leakage_vit_cifar10_forget0 --seed 42 --probe-layer early
python scripts/9_feature_probe_leakage.py --config configs/experiment_suites.yaml --suite probe_feature_leakage_vit_cifar10_forget0 --seed 42 --probe-layer mid
python scripts/9_feature_probe_leakage.py --config configs/experiment_suites.yaml --suite probe_feature_leakage_vit_cifar10_forget0 --seed 42 --probe-layer late

# Forget-neighborhood sweep (noise_std curve)
python scripts/10_neighborhood_sweep.py --config configs/experiment_suites.yaml --suite eval_forget_neighborhood_vit_cifar10_forget0 --seed 42

# VPT stealth-recovery tradeoff sweep (lambda_retain)
python scripts/11_vpt_tradeoff_sweep.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot --seed 42 --lambdas 0.0,0.1,0.5,1.0,2.0,5.0
```

Note: `scripts/1_train_base.py` uses a train/val split from the training set. The test set is reserved for final evaluation via `scripts/4_adv_evaluate.py`.

Outputs:
- Models: `checkpoints/`
- Logs: `results/logs/`
- Tables/analysis: `results/analysis/`
Note: the main branch omits checkpoints/logs to keep clones lightweight. Full artifacts live on `prompt-ablation-classgap`.

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
  scripts/                    # Main pipeline: train, unlearn, attack, eval, analyze
    1_train_base.py
    1b_train_retrained_oracle.py
    2_train_unlearning_lora.py
    3_train_vpt_resurrector.py
    4_adv_evaluate.py
    6_analyze_results.py
    analyze_kshot_experiments.py
    analyze_vpt_results.py
    audit_results_manifest.py
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
- Smoothed Gradient Ascent (SGA; supports optional normal-data loader)
- Feature Scrub (feature/logit ascent baseline)
- SalUn (saliency-masked random relabeling; faithful under full‑finetune, LoRA is proxy)
- SCRUB (distillation; KL‑based forget loss + retain KL/CE, full max/min schedule + optional rewind)
- Prune-then-unlearn (optional magnitude pruning before LoRA)
- Noisy retain-only finetune (certified-style proxy; optional)

### Unlearning (Full/Partial Finetune Baselines)
- Full finetune (no LoRA)
- Head-only finetune (classifier head)
- Last-block finetune (final transformer block)

### Models
- ViT-Tiny
- ResNet-18

### Attacks / Evaluation
- VPT resurrection (prompt-only, frozen backbone)
- PGD
- `vpt_plus_pgd` (targeted PGD toward the forget class, optionally on VPT-wrapped model)
- AutoAttack integration
- Dependency-aware eval v1 (retain images patched with forget cues)
- Dependency-aware eval v2 (retain + forget images with opposite cues)
- Forget-neighborhood eval (mild perturbations on forget samples)
- Backdoor patch eval (poisoning-style, attack success rate)
- Feature-probe leakage (linear probe on frozen features)
- Feature-probe by layer (early/mid/late or block index)
- Forget-neighborhood sweep (noise_std curve)
- VPT stealth-recovery tradeoff (lambda_retain sweep)

---

## 2025-2026 Research Signals (Notes)

Recent work points out evaluation pitfalls and suggests stronger checks. Here is how we used those ideas:

- **Dependency-aware evaluation**: independent forget/retain splits can be misleading. We added patch-based dependency eval v1 (retain+forget cue) and v2 (retain+forget mixed cues).
- **Multi-criteria evaluation**: MUSE (ICLR 2025) argues against narrow metrics. We use this as a reminder to report more than accuracy.
- **Certified unlearning via noisy retain finetune**: ICML 2025 proposes guarantees via noisy fine-tuning. We added a noisy retain-only proxy baseline (`objective: noisy_retain` + `grad_noise_std`).
- **System-aware unlearning**: ICML 2025 formalizes realistic attacker models. Our oracle baseline and dependency evals are closer to that setting than worst-case retraining.
- **Poisoning removal**: ICLR 2025 shows practical unlearning can fail to remove poisoning effects. This motivates backdoor/trigger evals.
- **Distillation and self-distillation**: 2025 work suggests distillation can improve robustness. We list student- and self-distillation as future baselines.
- **Adversarial mixup unlearning**: ICLR 2025 proposes generator-assisted mixup. We list it as a future baseline.

We focus on vision, but these ideas still map cleanly to evaluation choices here.

**References (external)**
- CMU ML Blog (2025): LLM unlearning benchmarks are weak measures of progress (https://blog.ml.cmu.edu/2025/04/18/llm-unlearning-benchmarks-are-weak-measures-of-progress/)
- CMU ML Blog (2025): Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning (https://blog.ml.cmu.edu/2025/05/22/unlearning-or-obfuscating-jogging-the-memory-of-unlearned-llms-via-benign-relearning/)
- MUSE (2024/2025): Machine Unlearning Six-Way Evaluation for Language Models (https://muse-bench.github.io/ ; https://arxiv.org/abs/2407.06460)
- Certified Unlearning for Neural Networks (ICML 2025) (https://proceedings.mlr.press/v267/koloskova25a.html)
- System-Aware Unlearning Algorithms: Use Lesser, Forget Faster (ICML 2025) (https://proceedings.mlr.press/v267/lu25h.html)
- Machine Unlearning Fails to Remove Data Poisoning Attacks (ICLR 2025) (https://proceedings.iclr.cc/paper_files/paper/2025/hash/7e810b2c75d69be186cadd2fe3febeab-Abstract-Conference.html)
- Distillation Robustifies Unlearning (arXiv 2025) (https://arxiv.org/abs/2506.06278)
- UnDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning (NAACL 2025) (https://aclanthology.org/2025.naacl-long.444/)
- Adversarial Mixup Unlearning (ICLR 2025) (https://proceedings.iclr.cc/paper_files/paper/2025/hash/bd5508e429849637d177603c32169aba-Abstract-Conference.html)
- Eight Methods to Evaluate Robust Unlearning in LLMs (arXiv 2024) (https://arxiv.org/abs/2402.16835)

---

## Novelty and Positioning

ForgetGate is not just a set of runs. It is a simple diagnostic setup that separates:

1) **Relearning capacity** (what any model can recover with limited data), and
2) **Residual access** (what remains after unlearning beyond that baseline).

The oracle baseline is the counterfactual model trained without the forget class. Recovery can be interpreted as residual access only when it exceeds oracle recovery. This keeps us honest about the common failure mode: splits that make unlearning look strong without testing dependencies or post-unlearning attacks.

We also include **feature-probe leakage** as a lightweight, model-agnostic check: a linear probe trained on frozen representations to predict forget vs retain. If a shallow probe separates forget samples after "unlearning," residual class information is still present even when output accuracy is low.

---

## Benign Relearning Probe (Vision)

Recent work on benign relearning shows that small, loosely related public data can "jog" unlearned LLMs back to forgotten information. We can translate this idea to vision by fine-tuning the unlearned model on a small, public, non-forget subset (or a nearby class) and measuring whether it recovers forget-class accuracy without direct forget-class data.

Suggested implementation (new baseline):
1) Start from the unlearned model checkpoint.
2) Fine-tune on a small retain-only slice or "neighbor" class for N epochs.
3) Re-run `eval_paper_baselines_vit_cifar10_forget0` and compare forget recovery to oracle.

This directly tests the "obfuscation vs forgetting" critique with a vision analog and strengthens the narrative beyond prompt-only attacks.

### Benign relearning probe (run)

Train:
```
python scripts/7_benign_relearn.py --config configs/experiment_suites.yaml --suite benign_relearn_vit_cifar10_forget0 --seed 42
```

Evaluate:
```
python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_benign_relearn_vit_cifar10_forget0 --seed 42
```

## Results (CIFAR-10, forgetting airplane)

### Results at a glance

- Oracle-normalized k-shot recovery is small for k=10/25/50/100 (KL 0.43-1.90%, oracle 0.00%).
- Low-shot controls (prompt length 5) show large recovery and high variance, so they are treated as stress tests.
- Class-wise gaps are mostly small, with class 9 higher than the rest.

### Clean unlearning baselines (test set)

| Method | Forget Acc (%) [lower] | Retain Acc (%) [higher] | Delta Utility |
|--------|------------------------|-------------------------|---------------|
| Base Model (No Unlearning) | 91.83 +/- 4.36 | 91.80 +/- 3.77 | 0.00pp |
| Uniform KL | 33.73 +/- 46.33 | 92.04 +/- 3.90 | +0.25pp |
| CE Ascent | 53.47 +/- 46.56 | 91.83 +/- 3.71 | +0.03pp |
| SalUn (Fan et al. 2024) | 58.27 +/- 50.47 | 91.97 +/- 4.01 | +0.17pp |
| SCRUB (Kurmanji et al. 2023) | 53.87 +/- 46.87 | 91.86 +/- 3.71 | +0.06pp |
| Noisy Retain-Only (Proxy) | 52.87 +/- 33.75 | 92.31 +/- 3.59 | +0.51pp |

*Seeds: 42, 123, 456. Metrics from eval_paper_baselines_vit_cifar10_forget0 (clean test set).*
Delta utility is defined as retain accuracy minus the base retain accuracy for the same seed group, so the base row is 0.00pp by definition.

Interpretation: clean baselines show that forgetting on the target class is possible (low forget accuracy) while retain remains high, but the variance is large across seeds.

### Prune-then-unlearn baseline (seed 42)

| Method | Forget Acc (%) | Retain Acc (%) |
|--------|----------------|----------------|
| Prune + KL (seed 42) | 17.00 | 94.44 |

*From `results/logs/eval_prune_kl_vit_cifar10_forget0_seed_42_evaluation.json` (clean test set).*

### Benign relearning probe (seed 42)

| Method | Forget Acc (%) | Retain Acc (%) |
|--------|----------------|----------------|
| Base | 94.20 | 94.10 |
| Unlearn KL | 14.40 | 94.36 |
| Benign relearn | 16.70 | 94.10 |

*From `results/logs/eval_benign_relearn_vit_cifar10_forget0_seed_42_evaluation.json` (clean test set).*

### Backdoor patch eval (seed 42)

Backdoor eval reports attack success rate (ASR) as `forget_acc`. Retain accuracy is not meaningful here because labels are forced to the target class during the trigger test.

| Method | Backdoor ASR (Forget Acc) | Retain Acc |
|--------|---------------------------|------------|
| Base | 9.99 | 0.00 |
| Unlearn LoRA | 7.57 | 0.00 |
| Unlearn KL | 1.31 | 0.00 |

*From `results/logs/eval_backdoor_vit_cifar10_forget0_seed_42_evaluation.json`.*

### Oracle-normalized k-shot recovery (default prompt length)

| K-shot | Oracle Baseline | Unlearned (KL) | Residual Recov. |
|--------|----------------|----------------|-----------------|
| 10 | 0.00 +/- 0.00% | 0.47 +/- 0.64% | +0.47pp |
| 25 | 0.00 +/- 0.00% | 0.43 +/- 0.40% | +0.43pp |
| 50 | 0.00 +/- 0.00% | 0.43 +/- 0.25% | +0.43pp |
| 100 | 0.00 +/- 0.00% | 1.90 +/- 0.75% | +1.90pp |

*Seeds: 42, 123, 456. Recovery@k measured on held-out train split (forget_val) using final logged epoch per run.*

Average Oracle-to-VPT gap across k=10/25/50/100 (default prompt length) is +0.81pp (seeds 42/123/456).

### Class-wise oracle gap (10-shot, default prompt length)

| Forget Class | Oracle Baseline | Unlearned (KL) | Residual Recov. |
|-------------|----------------|----------------|-----------------|
| 1 | 0.00% | 0.73% | +0.73pp |
| 2 | 0.00% | 0.70% | +0.70pp |
| 5 | 0.00% | 1.13% | +1.13pp |
| 9 | 0.00% | 6.77% | +6.77pp |

*Seeds: 42, 123, 456. Recovery@k measured on held-out train split (forget_val) using final logged epoch per run.*

### Stress tests (controls and ablations)

**Prompt-length ablation (10-shot, seeds 42/123):**
- Prompt length 1/2/5: KL = 7.05/7.35/7.80%, Oracle = 0.00%
- Prompt length 10 (default): KL = 0.47%, Oracle = 0.00%

**VPT stealth-recovery tradeoff (lambda_retain sweep, 10-shot):**

| lambda_retain | Resurrection Acc (%) |
|---:|---:|
| 0.0 | 35.27 +/- 49.29 |
| 0.1 | 35.27 +/- 49.29 |
| 0.5 | 35.30 +/- 49.34 |
| 1.0 | 35.33 +/- 49.32 |
| 2.0 | 35.33 +/- 49.32 |
| 5.0 | 35.27 +/- 49.29 |

*Seeds: 42, 123, 456. Variance is large because seed 456 recovers strongly while seed 123 is near zero.*

**Low-shot + label controls (prompt length 5):**
- Seeds 42/123/456 mean: k=1: 34.17%, k=5: 35.47%, shuffled-label (k=10): 3.90%, random-label (k=10): 4.03% (oracle 0.00%).
- These have high variance across seeds; see `results/analysis/kshot_summary.md` for per-seed values.

**Stratified forget set (pilot, seed 42):**
- High-confidence subset: 12.70% (KL, k=10)
- Low-confidence subset: 13.60% (KL, k=10)
- Mid-confidence subset: 14.00% (KL, k=10)

**Dependency-aware eval v1 (retain images patched with forget cue):**
- Base + unlearned retain stays about 0.87-0.94 across seeds 42/123/456; forget stays 0.00.
- VPT retain drops sharply on seed 123 (0.361) but stays 0.929 on seed 42. VPT dependency eval is not available for seed 456 (no VPT checkpoint).

**Dependency-aware eval v2 (retain+forget mixed cues):**
- Seed 42 results (dependency_mix_patch):
  - Base: Forget 0.9420, Retain 0.9404
  - Unlearn LoRA: Forget 0.7480, Retain 0.9417
  - VPT: Forget 0.7830, Retain 0.9293

**Forget-neighborhood eval (mild noise on forget-only samples):**

| Method | Forget Acc (%) |
|--------|----------------|
| Base | 91.83 +/- 4.36 |
| Unlearn LoRA | 53.47 +/- 46.55 |
| VPT | 92.70 +/- 11.78 |

*Seeds: 42, 123, 456. Retain accuracy is not reported because this eval uses forget-only samples.*

**Forget-neighborhood sweep (noise_std vs forget accuracy):**

| noise_std | Base (Forget %) | Unlearn LoRA (Forget %) | VPT (Forget %) |
|---:|---:|---:|---:|
| 0.0 | 91.83 +/- 4.36 | 53.47 +/- 46.56 | 92.70 +/- 11.78 |
| 0.01 | 91.83 +/- 4.36 | 53.47 +/- 46.55 | 92.70 +/- 11.87 |
| 0.02 | 91.77 +/- 3.96 | 53.30 +/- 46.41 | 92.37 +/- 12.53 |
| 0.05 | 89.20 +/- 4.51 | 49.67 +/- 43.75 | 88.87 +/- 18.59 |

*Seeds: 42, 123, 456. Forget-only eval; retain accuracy is not reported.*

**Feature-probe leakage (linear probe on frozen features, final layer):**

| Method | Val Acc (%) | AUC |
|--------|-------------|-----|
| Base | 98.55 +/- 0.43 | 0.9949 +/- 0.0047 |
| Unlearn KL | 98.35 +/- 0.37 | 0.9943 +/- 0.0040 |
| Benign relearn | 98.29 +/- 0.51 | 0.9950 +/- 0.0039 |
| Oracle | 93.98 +/- 0.78 | 0.9516 +/- 0.0097 |

*Seeds: 42, 123, 456. Probe trained on frozen features to predict forget vs retain.*

**Feature-probe by layer (val acc and AUC):**

| Layer | Model | Val Acc (%) | AUC |
|---|---|---:|---:|
| early | base_vit_cifar10 | 93.15 +/- 0.29 | 0.9142 +/- 0.0040 |
| early | unlearn_kl_vit_cifar10_forget0 | 92.30 +/- 0.75 | 0.9004 +/- 0.0059 |
| early | benign_relearn_vit_cifar10_forget0 | 93.08 +/- 0.14 | 0.9162 +/- 0.0117 |
| early | oracle_vit_cifar10_forget0 | 92.80 +/- 1.04 | 0.9014 +/- 0.0175 |
| mid | base_vit_cifar10 | 95.33 +/- 0.65 | 0.9650 +/- 0.0032 |
| mid | unlearn_kl_vit_cifar10_forget0 | 95.00 +/- 0.78 | 0.9644 +/- 0.0109 |
| mid | benign_relearn_vit_cifar10_forget0 | 95.35 +/- 0.15 | 0.9691 +/- 0.0031 |
| mid | oracle_vit_cifar10_forget0 | 93.38 +/- 1.20 | 0.9229 +/- 0.0127 |
| late | base_vit_cifar10 | 98.21 +/- 0.70 | 0.9954 +/- 0.0040 |
| late | unlearn_kl_vit_cifar10_forget0 | 98.25 +/- 0.47 | 0.9950 +/- 0.0031 |
| late | benign_relearn_vit_cifar10_forget0 | 98.11 +/- 0.66 | 0.9950 +/- 0.0030 |
| late | oracle_vit_cifar10_forget0 | 93.15 +/- 0.63 | 0.9470 +/- 0.0113 |
| final | base_vit_cifar10 | 98.55 +/- 0.43 | 0.9949 +/- 0.0047 |
| final | unlearn_kl_vit_cifar10_forget0 | 98.35 +/- 0.37 | 0.9943 +/- 0.0040 |
| final | benign_relearn_vit_cifar10_forget0 | 98.29 +/- 0.51 | 0.9950 +/- 0.0039 |
| final | oracle_vit_cifar10_forget0 | 93.98 +/- 0.78 | 0.9516 +/- 0.0097 |

### Full-data recovery (context)

Full-data VPT runs reach near-complete recovery in results/logs/vpt_resurrect_*_seed_*.jsonl. To reproduce test-set VPT/PGD metrics, run: `python scripts/4_adv_evaluate.py --suite eval_full_vit_cifar10_forget0`.

**Oracle control (full-data)**: VPT on the oracle baseline also reaches about 100% recovery with full data, indicating the full-data attack is effectively relearning rather than accessing residual knowledge. See `results/logs/oracle_vpt_control_class0_seed_42.jsonl` and `results/logs/oracle_vpt_control_class0_seed_123.jsonl`.

**AutoAttack (seeds 42/123/456)**: full AutoAttack drives both forget and retain accuracy to about 0 on base/unlearned models for all seeds. VPT AutoAttack is available for seeds 42 and 123 (see `results/logs/eval_autoattack_vit_cifar10_forget0_seed_{42,123,456}_evaluation.json`).

---

## Feature-Probe Defense Analysis: LoRA's Structural Limitation

The feature-probe results above show that a linear probe on frozen features detects the forgotten class at 98%+ accuracy even after LoRA unlearning -- barely different from the base model (99%). This section explains *why* and shows that the gap can be closed with the right approach.

### The core finding

LoRA unlearning changes the model's **outputs** (forget accuracy drops to 0%) but barely touches the **features**. A linear probe on frozen 192-dim CLS token features separates forget from retain samples at 96-99% accuracy across all LoRA-based methods. The oracle model (retrained from scratch without the forget class) sits at ~94%. The gap is structural: LoRA modifies a low-rank subspace of the weights that can rotate the classification head away from the forget class, but the backbone's feature representations remain largely intact. The class information is still encoded in the features -- it is just no longer connected to the correct output.

### Expanded feature-probe results (seed 42, final layer)

| Method | Adaptation | Probe Acc (%) | Probe AUC |
|--------|-----------|:---:|:---:|
| Base (no unlearning) | -- | 99.1 | 0.997 |
| Uniform KL | LoRA | 98.4 | 0.996 |
| SalUn | LoRA | 98.4 | 0.996 |
| OrthoReg | LoRA | 98.5 | 0.992 |
| RURK | LoRA | 98.3 | 0.993 |
| SGA | LoRA | 97.3 | 0.982 |
| BalDRO | LoRA | 96.5 | 0.981 |
| FaLW | LoRA | 96.0 | 0.976 |
| BalDRO-SCRUB-Ortho (Baseline, LoRA) | LoRA | 97.2 | 0.995 |
| BalDRO-SCRUB-Ortho (r64) | LoRA | 96.4 | 0.984 |
| SCRUB | LoRA | 94.4 | 0.958 |
| BalDRO-SCRUB-Ortho (Full FT) | Full FT | 91.2 | 0.958 |
| Pure Manifold (Full FT) | Full FT | 90.2 | 0.955 |
| Imposter Defense (Full FT) | Full FT | 91.4 | 0.943 |
| **Oracle** | **--** | **93.8** | **0.947** |

*Probe: linear classifier (BCEWithLogits) trained on frozen features. AUC computed via rank-based method. All results from seed 42 on CIFAR-10 test set (10,000 samples). Oracle retrained from scratch without class 0.*

### What the table shows

1. **LoRA methods leave massive feature remnants.** Every LoRA-based method sits at 96-99% probe accuracy and 0.976-0.996 AUC. The probe can trivially separate forget from retain samples because LoRA's low-rank update cannot overwrite the deep feature representations that encode class identity.

2. **SCRUB is the exception among LoRA methods.** SCRUB achieves 94.4% / 0.958 AUC, close to oracle (93.8% / 0.947). This is because SCRUB uses teacher-student distillation which explicitly aligns the student's feature distribution with the teacher's, disrupting the original class-specific feature structure.

3. **Full fine-tuning is necessary for feature-level privacy.** All Full FT defenses (91-91.4% accuracy, 0.943-0.958 AUC) approach or match oracle levels. Full FT has enough capacity to actually overwrite the feature representations, unlike LoRA which operates in a constrained subspace.

4. **Imposter defense achieves oracle-level probe resistance.** The imposter approach (mapping forget features to random retain samples) reaches AUC 0.943 -- at or below the oracle's 0.947. This means the probe finds it no easier to detect the forgotten class than in a model that never saw it.

### Defense methods tested

- **BalDRO-SCRUB-Ortho (Baseline)**: BalDRO hard-sample mining + SCRUB distillation + feature orthogonalization, LoRA rank 4. Feature defense has limited effect due to LoRA's expressivity bottleneck.
- **BalDRO-SCRUB-Ortho (Full FT)**: Same objectives but with full fine-tuning. Crosses below oracle probe accuracy (91.2% vs 93.8%) and reaches AUC 0.958.
- **Pure Manifold (Full FT)**: Feature-only defense (no BalDRO/SCRUB loss terms). Maps forget features toward retain centroid. AUC 0.955.
- **Imposter Defense (Full FT)**: Instead of mapping to a static centroid, maps each forget sample to a random retain sample from the training set. This preserves distributional spread and achieves AUC 0.943.

### t-SNE visualization (supplementary)

t-SNE projections of the 192-dim features into 2D show centroid distance and variance metrics for each defense. The imposter defense achieves the lowest centroid distance (8.2 vs 44.7 for baseline) but retains low variance ratio (0.08 vs retain 1.0). However, the t-SNE variance ratio is a 2D projection artifact and does not correspond to probe separability in the original 192-dim space. The probe AUC is the metric that matters, and the imposter defense achieves oracle-level AUC despite low t-SNE variance ratio. See `results/plots/tsne_comparison_grid_search_vis.png`.

### Conclusion

LoRA-based unlearning creates a surface-level illusion of forgetting: outputs change but features persist. A simple linear probe exposes this at 96-99% accuracy. Closing the gap to oracle requires either (a) distillation-based objectives that explicitly align features (SCRUB), or (b) full fine-tuning with feature-level objectives (manifold injection, imposter defense). The oracle itself sits at ~94% / 0.947 AUC because even a model that never saw a class retains some inherent distributional structure -- this is the theoretical floor. The imposter defense reaches this floor (AUC 0.943), leaving no further room for improvement.

---

## Results Provenance

- Seeds by table:
  - Clean baselines: 42, 123, 456
  - K-shot recovery + class-wise gap: 42, 123, 456
  - Prompt-length ablation: 42, 123
  - Low-shot + label controls: 42, 123, 456
- K-shot recovery uses a held-out train split (forget_val) and reports the final logged epoch.
- Clean baselines come from test-set eval: `eval_paper_baselines_vit_cifar10_forget0`.
- Reported standard deviations use sample std across seeds (ddof=1).
- Low-shot controls (k=1/5) use prompt length 5; k=10+ uses default prompt length 10.
- Prompt-length ablation (1/2/5 tokens) aggregates seeds 42/123.
- Legacy default-prompt k=1/5 logs are archived in `results/logs/legacy_prompt10` to avoid confusion with the prompt-length-5 controls.
- Full-data recovery values are from VPT training logs; test-set VPT/PGD numbers require `eval_full_vit_cifar10_forget0`.
- Clean baselines table is generated from `results/logs/eval_paper_baselines_vit_cifar10_forget0_seed_*_evaluation.json` (see `results/analysis/clean_baselines_summary.md`).
- Low-shot per-seed breakdowns are in `results/analysis/kshot_summary.md`.
- Dependency eval logs: `results/logs/eval_dependency_vit_cifar10_forget0_seed_*_evaluation.json` and `results/logs/eval_dependency_mix_vit_cifar10_forget0_seed_*_evaluation.json`.
- Forget-neighborhood logs: `results/logs/eval_forget_neighborhood_vit_cifar10_forget0_seed_*_evaluation.json`.
- Feature-probe logs: `results/logs/feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_*.json` and `results/logs/feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_*_layer_*.json`.
- Feature-probe defense analysis logs: `results/logs/feature_probe_grid_search_ortho_forget0_seed_42_layer_final.json`, `results/logs/feature_probe_probe_pure_manifold_seed_42_layer_final.json`, `results/logs/feature_probe_probe_imposter_seed_42_layer_final.json`, `results/logs/feature_probe_probe_feature_leakage_2026_methods_vit_cifar10_forget0_seed_42_layer_final.json`.

## Audit / Manifest

To verify coverage and detect missing runs, generate a manifest:

```bash
python scripts/audit_results_manifest.py --seeds 42 123 456 --prompt-seeds 42 123
```

Outputs:
- `results/analysis/manifest.json`
- `results/analysis/manifest_report.md`

## Artifacts

- Tables: `results/analysis/comparison_table_markdown.txt`, `results/analysis/comparison_table_latex.txt`
- Plots: `results/analysis/kshot_analysis.png`, `results/analysis/learning_curves_kshot.png`
- Reports: `results/analysis/sample_efficiency_gap.txt`, `results/analysis/kshot_summary.md`
- Logs: `results/logs/*_evaluation.json` (test-set metrics), `results/logs/*.jsonl` (training curves)

---

## Threat Model (VPT Resurrection)

Attacker capabilities assumed:
- White-box access to weights and architecture
- Can backprop through the frozen model to optimize prompts
- Has some forget-class samples (k-shot or full-data)
- Does not change model weights (prompt-only adaptation)

**Oracle baseline details:** The oracle is a 10-class classifier trained on retain data only (class 0 excluded from training). Class 0 predictions come from the untrained logit, so the oracle represents what happens if you retrain from scratch without the forget class.

**VPT evaluation:** Prompts are applied to all inputs (both forget and retain classes) at evaluation time. This means VPT can hurt retain accuracy while recovering forget accuracy. For example, in full-data runs, retain accuracy drops from about 94% to about 52% under VPT. This models an attacker who wraps the model with a fixed input transformation.

---

## Limitations and Future Work

- Oracle variants: test alternative oracle heads (non-random class-0 head) to see if the oracle gap persists.
- Attacker modes: compare max-recovery (lambda_retain=0) vs stealth (lambda_retain>0) prompts.
- Deterministic validation: evaluate k-shot recovery on a fixed eval-transform split to reduce noise.
- Broader coverage: more seeds and multiple forget classes to confirm gap consistency.
- Stratify forget sets by difficulty (RUM-style) and report recovery gaps per subset.
- Dependency-aware evaluations: add mixed-cue or patched-retain tests to reduce benchmark fragility.

## Next Steps (Recommended)

1) **Dependency-aware eval v2**: run `eval_dependency_mix_vit_cifar10_forget0` to test mixed-cue leakage.
2) **Prune-then-unlearn baseline**: run `unlearn_prune_kl_vit_cifar10_forget0` and compare recovery vs standard KL.
3) **Class-wise + prompt-length**: repeat prompt-length ablation for forget classes 1/2/5/9.
4) **Certified-style proxy**: run `unlearn_noisy_retain_vit_cifar10_forget0` with a nonzero `grad_noise_std` and compare to KL.

## Follow-ups (If You Want More Rigor)

- Add additional seeds beyond 456 for tighter confidence intervals.
- Run AutoAttack for seed 123/456 to reduce variance in robustness claims.
- Report k-shot recovery on a fixed, deterministic split for even tighter comparisons.
- Add RUM-style stratified forget subsets (high/low confidence) and compare recovery curves.

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

unlearn_prune_kl_vit_cifar10_forget0:
  base_model_suite: base_vit_cifar10
  unlearning:
    method: lora
    objective: uniform_kl
    forget_class: 0
    lora_rank: 8
    epochs: 50
    lr: 1e-3
    pruning:
      enabled: true
      amount: 0.2
      strategy: global_unstructured
      module_types: ["Linear"]
      make_permanent: true
      apply_to_teacher: true
  seeds: [42, 123, 456]

unlearn_noisy_retain_vit_cifar10_forget0:
  base_model_suite: base_vit_cifar10
  unlearning:
    method: lora
    objective: noisy_retain
    forget_class: 0
    lora_rank: 8
    epochs: 50
    lr: 1e-3
    grad_noise_std: 0.001
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
- **AutoAttack**: included via requirements; remove the auto-attack line if you want a lighter install

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
- **Certified MU via Noisy SGD** (NeurIPS 2024): https://papers.nips.cc/paper/2024/hash/448abd486677165ceedfa790e9a61802-Abstract-Conference.html
- **Langevin Unlearning** (NeurIPS 2024): https://arxiv.org/pdf/2401.10371
- **RUM Meta-Algorithm / Forget-Set Factors** (Zhao et al., 2024): https://arxiv.org/abs/2406.01257
- **Prune-First Then Unlearn** (NeurIPS 2023): https://arxiv.org/abs/2304.04934
- **Residual Knowledge under Perturbed Samples (RURK)** (Hsu et al., 2026): https://arxiv.org/abs/2601.22359
- **Statistical MIA for Unlearning Auditing** (Sun et al., 2026): https://arxiv.org/abs/2602.01150
- **Forgetting Similar Samples / Neighborhood Effect** (Xu et al., 2026): https://arxiv.org/abs/2601.06938
- **BalDRO (Hard-Sample Reweighting for Unlearning)** (Shao et al., 2026): https://arxiv.org/abs/2601.09172
- **Behemoth (Synthetic Unlearning Benchmark)** (Iofinova & Alistarh, 2026): https://arxiv.org/abs/2601.23153
- **FaLW (Long-Tailed Unlearning)** (Yu et al., 2026): https://arxiv.org/abs/2601.18650
- **Regularized Divergence Kernel Tests (Audit)** (Ribero et al., 2026): https://arxiv.org/abs/2601.19755
- **EvoMU (Evolutionary Machine Unlearning)** (Batorski & Swoboda, 2026): https://arxiv.org/abs/2602.02139
- **AGT (Adversarial Gating + Adaptive Orthogonality)** (Li et al., 2026): https://arxiv.org/abs/2602.01703
- **Sparsity-Aware Unlearning (LLMs)** (Wang et al., 2026): https://arxiv.org/abs/2602.00577
- **Orthogonal Entropy Unlearning (QNNs)** (Zhang et al., 2026): https://arxiv.org/abs/2602.00567
- **ReLAPSe (Prompt Search for Diffusion Unlearning)** (Kolton et al., 2026): https://arxiv.org/abs/2602.00350
- **Illusion of Forgetting (Latent Optimization Attack)** (Li et al., 2026): https://arxiv.org/abs/2602.00175
- **GUDA (Group Attribution via Unlearning)** (Murata et al., 2026): https://arxiv.org/abs/2601.22651
- **Lethe (Persistent Federated Unlearning)** (Tan et al., 2026): https://arxiv.org/abs/2601.22601
- **FedCARE (Conflict-Aware FU)** (Li et al., 2026): https://arxiv.org/abs/2601.22589
- **LOFT (Low-Dimensional Subspace Unlearning)** (Fang et al., 2026): https://arxiv.org/abs/2601.22456
- **PerTA (Per-Parameter Task Arithmetic)** (Cai et al., 2026): https://arxiv.org/abs/2601.22030
- **CLReg (Contrastive Latent Unlearning)** (Tang & Khanna, 2026): https://arxiv.org/abs/2601.22028
- **ViKeR (Visual-Guided Key-Token Regularization)** (Cai et al., 2026): https://arxiv.org/abs/2601.22020
- **UnlearnShield (Inversion Defense)** (Xue et al., 2026): https://arxiv.org/abs/2601.20325
- **Representation Unlearning (Info Compression)** (Almudevar & Ortega, 2026): https://arxiv.org/abs/2601.21564
- **FIT (Continual LLM Unlearning)** (Xu et al., 2026): https://arxiv.org/abs/2601.21682
- **KIF (Activation-Signature Erasure)** (Mahmood et al., 2026): https://arxiv.org/abs/2601.10566
- **Circuit-Guided Difficulty Metric** (Cheng et al., 2026): https://arxiv.org/abs/2601.09624
- **Certified Unlearning under Distribution Shift** (Guo & Cao, 2026): https://arxiv.org/abs/2601.06967
- **Sequential Subspace Noise (Certified)** (Dolgova & Stich, 2026): https://arxiv.org/abs/2601.05134
- **Erasure Illusion / PSG Stress-Test** (Jia et al., 2025): https://arxiv.org/abs/2512.19025
- **Selective Forgetting Privacy Benchmark** (Qian et al., 2025): https://arxiv.org/abs/2512.18035
- **Dual-View Inference Attack (DVIA)** (Xue et al., 2025): https://arxiv.org/abs/2512.16126
- **Teleportation Defenses (WARP)** (Maheri et al., 2025): https://arxiv.org/abs/2512.00272
- **Leak@ (Probabilistic Decoding Leakage)** (Reisizadeh et al., 2025): https://arxiv.org/abs/2511.04934
- **REMIND (Residual Memorization Auditor)** (Cohen et al., 2025): https://arxiv.org/abs/2511.04228
- **Knowledge Holes in Unlearned LLMs** (Ko et al., 2025): https://arxiv.org/abs/2511.00030
- **Single-Seed Evaluation Limits** (Lanyon et al., 2025): https://arxiv.org/abs/2510.26714
- **Geometric-Disentanglement Unlearning (GU)** (Zhou et al., 2025): https://arxiv.org/abs/2511.17100
- **Smoothed Gradient Ascent (SGA)** (Pang et al., 2025): https://arxiv.org/abs/2510.22376
- **Implicit Gradient Surgery (EUPMU)** (Zhou et al., 2025): https://arxiv.org/abs/2510.22124
- **LLM Unlearning with LLM Beliefs** (Li et al., 2025): https://arxiv.org/abs/2510.19422
- **Forgetting-MarI (Marginal Info Regularization)** (Xu et al., 2025): https://arxiv.org/abs/2511.11914
- **KUnBR (Knowledge Density + Block Reinsertion)** (Guo et al., 2025): https://arxiv.org/abs/2511.11667
- **SGTM (Knowledge Localization)** (Shilov et al., 2025): https://arxiv.org/abs/2512.05648
- **ModHiFi-U (Data-Free Class Unlearning)** (Kashyap et al., 2025): https://arxiv.org/abs/2511.19566
- **POUR (Representation-Level Unlearning)** (Le et al., 2025): https://arxiv.org/abs/2511.19339
- **GFOES (Few-shot Zero-Glance Unlearning)** (Song et al., 2025): https://arxiv.org/abs/2511.13116
- **Hubble (Memorization Testbed)** (Wei et al., 2025): https://arxiv.org/abs/2510.19811
