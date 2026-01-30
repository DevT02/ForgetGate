# ForgetGate: Breaking LoRA-Based Unlearning with Visual Prompts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ForgetGate audits LoRA-based machine unlearning under a simple question:

> If an attacker can train a tiny visual prompt, do "forgotten" classes come back because of residual knowledge, or because the attacker is just relearning?

This repo evaluates a **visual prompt tuning (VPT)** "resurrection" attack (e.g., 10 tokens x 192 dims = **1,920 params** for ViT-Tiny) against LoRA unlearning methods, and compares against an **oracle baseline** trained from scratch *without* the forget class.

## Project Overview

ForgetGate is a small, focused audit: we train a model, unlearn one class with LoRA, then test whether a tiny visual prompt can bring that class back. By comparing against an oracle model (trained without the class), we separate **relearning capacity** from **residual knowledge**. The result is a practical checklist and set of plots that show when "unlearning" looks secure and when it doesn't.

## TL;DR

Two attacker data regimes are considered:

- **Full-data** (attacker has the full forget-class training set): VPT reaches ~100% recovery on both the unlearned model and the oracle baseline -> the attack is effectively relearning.
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
python scripts/analyze_kshot_experiments.py --seeds 42 123 456
```

Optional evaluations:

```bash
# Dependency-aware eval (retain images patched with forget cues)
python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_dependency_vit_cifar10_forget0 --seed 42

# Stratified forget set (high-confidence samples only)
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot --seed 42 --stratify high_conf

# Prune-then-unlearn baseline
python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite unlearn_prune_kl_vit_cifar10_forget0 --seed 42

# Certified-style noisy retain baseline (proxy)
python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite unlearn_noisy_retain_vit_cifar10_forget0 --seed 42
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
  scripts/                    # Main pipeline: train -> unlearn -> attack -> eval -> analyze
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
- Feature Scrub (feature/logit ascent baseline)
- SalUn (saliency-masked random relabeling)
- SCRUB (distillation)
- Prune-then-unlearn (optional magnitude pruning before LoRA)
- Noisy retain-only finetune (certified-style proxy; optional)

### Models
- ViT-Tiny
- ResNet-18

### Attacks / Evaluation
- VPT resurrection (prompt-only, frozen backbone)
- PGD
- `vpt_plus_pgd` (targeted PGD toward the forget class, optionally on VPT-wrapped model)
- AutoAttack integration
- Dependency-aware eval (retain images patched with forget cues)

---

## 2025-2026 Research Signals (How We Map Them)

Recent work highlights evaluation pitfalls and stronger baselines:

- **Dependency-aware evaluation**: CMU’s 2025 critique shows that independent forget/retain splits can be misleading; dependency-style tests often reveal failures. We implemented patch-based dependency eval to address this. [R1]
- **Multi-criteria evaluation**: MUSE (ICLR 2025) argues unlearning must be assessed across privacy, utility, sustainability, and sequential requests, not just forget/retain. We treat this as a caution on narrow metrics. [R2]
- **Certified unlearning via noisy retain finetune**: ICML 2025 proposes guarantees using noisy finetuning on retain data. We added a noisy retain-only baseline (`objective: noisy_retain` + `grad_noise_std`). [R3]
- **System-aware unlearning**: ICML 2025 formalizes weaker but realistic attacker models; our oracle baseline and dependency eval are closer to “system-aware” stress tests than worst-case retraining guarantees. [R4]
- **Poisoning removal remains hard**: ICLR 2025 shows many practical unlearning methods fail to remove poisoning effects, reinforcing the need for attack-driven audits like ours. [R5]
- **Distillation can robustify unlearning**: ArXiv 2025 (UNDO) reports distillation improves robustness of unlearning against finetuning reversal. This motivates future student-distillation baselines. [R6]

We focus on vision but treat these as general design signals for unlearning evaluation and baselines.

**References**
- [R1] CMU ML Blog (2025): LLM unlearning benchmarks are weak measures of progress
- [R2] MUSE (ICLR 2025): Machine unlearning at scale
- [R3] Certified Unlearning via Noisy Finetuning (ICML 2025)
- [R4] System-Aware Unlearning (ICML 2025)
- [R5] Machine Unlearning Fails to Remove Data Poisoning (ICLR 2025)
- [R6] UNDO: Distillation Robustifies Unlearning Against Finetuning (arXiv 2025)

---

## Results (CIFAR-10, forgetting airplane)

### Clean unlearning baselines (test set)

| Method | Forget Acc (%) [lower] | Retain Acc (%) [higher] | Delta Utility |
|--------|------------------------|-------------------------|---------------|
| Base Model (No Unlearning) | 90.63 +/- 3.76 | 90.67 +/- 3.21 | 0.00pp |
| Uniform KL | 58.90 +/- 50.87 | 90.76 +/- 3.36 | +0.09pp |
| CE Ascent | **53.37 +/- 46.48** | **90.89 +/- 3.12** | +0.22pp |
| SalUn (Fan et al. 2024) | 59.70 +/- 51.82 | 90.61 +/- 3.46 | -0.06pp |
| SCRUB (Kurmanji et al. 2023) | 54.03 +/- 46.96 | 90.83 +/- 3.10 | +0.16pp |

*Seeds: 42, 123, 456. Metrics from eval_paper_baselines_vit_cifar10_forget0 (clean test set).*

### K-shot recovery (oracle-normalized, default prompt length)

| K-shot | Oracle Baseline | Unlearned (KL) | Residual Recov. |
|--------|----------------|----------------|-----------------|
| 10 | 0.00 +/- 0.00% | 0.47 +/- 0.64% | +0.47pp |
| 25 | 0.00 +/- 0.00% | 0.43 +/- 0.40% | +0.43pp |
| 50 | 0.00 +/- 0.00% | 0.43 +/- 0.25% | +0.43pp |
| 100 | 0.00 +/- 0.00% | 1.90 +/- 0.75% | +1.90pp |

*Seeds: 42, 123, 456. Recovery@k measured on held-out train split (forget_val) using final logged epoch per run.*

Average Oracle->VPT gap across k=10/25/50/100 (default prompt length) is +0.81pp (seeds 42/123/456).

### Class-wise oracle gap (10-shot, default prompt length)

| Forget Class | Oracle Baseline | Unlearned (KL) | Residual Recov. |
|-------------|----------------|----------------|-----------------|
| 1 | 0.00% | 0.73% | +0.73pp |
| 2 | 0.00% | 0.70% | +0.70pp |
| 5 | 0.00% | 1.13% | +1.13pp |
| 9 | 0.00% | 6.77% | +6.77pp |

*Seeds: 42, 123, 456. Recovery@k measured on held-out train split (forget_val) using final logged epoch per run.*

### Prompt-length + control runs

**Prompt-length ablation (10-shot, seeds 42/123):**
- Prompt length 1/2/5: KL = 7.05/7.35/7.80%, Oracle = 0.00%
- Prompt length 10 (default): KL = 0.47%, Oracle = 0.00%

**Low-shot + label controls (prompt length 5):**
- Seeds 42/123/456 mean: k=1: 34.17%, k=5: 35.47%, shuffled-label (k=10): 3.90%, random-label (k=10): 4.03% (oracle 0.00%). These have high variance across seeds; see `results/analysis/kshot_summary.md` for per-seed values.

**Stratified forget set (pilot, seed 42):**
- High-confidence subset: 12.70% (KL, k=10)
- Low-confidence subset: 13.60% (KL, k=10)

**Dependency-aware eval (retain images patched with forget cue):**
- Base + unlearned retain stays ~0.91-0.94 across seeds 42/123/456; forget stays 0.00.
- VPT retain drops sharply on seed 123 (0.361) but stays 0.929 on seed 42. VPT dependency eval is not available for seed 456 (no full-data VPT checkpoint).

### Full-data recovery (context)

Full-data VPT runs reach near-complete recovery in results/logs/vpt_resurrect_*_seed_*.jsonl. To reproduce test-set VPT/PGD metrics, run: `python scripts/4_adv_evaluate.py --suite eval_full_vit_cifar10_forget0`.

**Oracle control (full-data)**: VPT on the oracle baseline also reaches ~100% recovery with full data, indicating the full-data attack is effectively relearning rather than accessing residual knowledge. See `results/logs/oracle_vpt_control_class0_seed_42.jsonl` and `results/logs/oracle_vpt_control_class0_seed_123.jsonl`.

**AutoAttack (seeds 42/123/456)**: full AutoAttack drives both forget and retain accuracy to ~0 on base/unlearned/VPT models (see `results/logs/eval_autoattack_vit_cifar10_forget0_seed_{42,123,456}_evaluation.json`).

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
- Dependency eval logs: `results/logs/eval_dependency_vit_cifar10_forget0_seed_*_evaluation.json`.

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

**VPT evaluation:** Prompts are applied to all inputs (both forget and retain classes) at evaluation time. This means VPT can hurt retain accuracy while recovering forget accuracy. For example, in full-data runs, retain accuracy drops from ~94% to ~52% under VPT. This models an attacker who wraps the model with a fixed input transformation.

---

## Limitations and Future Work

- Oracle variants: test alternative oracle heads (non-random class-0 head) to see if the oracle gap persists.
- Attacker modes: compare max-recovery (lambda_retain=0) vs stealth (lambda_retain>0) prompts.
- Deterministic validation: evaluate k-shot recovery on a fixed eval-transform split to reduce noise.
- Broader coverage: more seeds and multiple forget classes to confirm gap consistency.
- Stratify forget sets by difficulty (RUM-style) and report recovery gaps per subset.
- Dependency-aware evaluations: add mixed-cue or patched-retain tests to reduce benchmark fragility.

## Next Steps (Recommended)

1) **Dependency-aware eval**: run `eval_dependency_vit_cifar10_forget0` to test retain accuracy under forget-cue patches.
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
- **Langevin Unlearning** (NeurIPS 2024): https://papers.nips.cc/paper/2024/hash/b75010ff30742a1c131d221015c63e11-Abstract-Conference.html
- **RUM Meta-Algorithm / Forget-Set Factors** (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/04d29b82a4e9d5512d00526d42c907a9-Abstract-Conference.html
- **Prune-First Then Unlearn** (NeurIPS 2023): https://papers.nips.cc/paper_files/paper/2023/hash/b06f05af4d976ce00d1677ebba1c8466-Abstract-Conference.html
