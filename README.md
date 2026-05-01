# ForgetGate: Multi-Regime Red Teaming of Machine Unlearning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]()

**Paper:** *Multi-Regime Red Teaming of Machine Unlearning* — Devansh Tayal, Michigan State University

---

ForgetGate is a controlled benchmark and empirical study of **machine unlearning robustness** under multiple attack regimes. The central question:

> When a model appears to have forgotten a class on clean evaluation, what kinds of adaptive attacks can still recover that behavior, and which defenses actually survive across attack regimes?

## Key Findings

**1. Per-example conditional recovery is the dominant broad threat.**
Non-robust methods remain conditionally recoverable even when universal perturbations largely fail. This holds across SalUn, SCRUB, CE-U, SGA, and BalDRO.

**2. Local-delivery robustness is not a single property.**
Patch-aware stage-2 defenses reduce 32×32 conditional patch recovery on ORBIT (78.1%→31.2%) and CE-U (68.8%→43.8%), but fail to transfer to border-frame and multi-patch delivery surfaces. Local-delivery transfer is an open problem.

**3. Defenses are partial and method-specific.**
Feature-subspace stage-2 provides the first consistent improvement to recovery radii on BalDRO, SalUn, and ORBIT without collapsing clean accuracy. RURK resists both attack families. SCRUB exhibits generic adversarial fragility rather than selective forgetting leakage.

## Methods Evaluated

| Method | Status |
|---|---|
| SalUn | Literature anchor |
| SCRUB | Literature anchor |
| RURK | Newer literature method |
| CE-U | Newer literature method |
| SGA | Adapted recent method |
| BalDRO | Adapted recent method |
| ORBIT | Repo-defined benchmark method |
| Robust-base SalUn | Defense frontier |

**ORBIT** (oracle-aligned, representation-dispersive unlearning) is defined in this repository. It combines margin-based forget-logit suppression, masked dark-knowledge distillation from a frozen oracle, feature alignment on forget inputs, and a dispersion penalty to weaken linear and prototype recovery channels. See `src/unlearning/objectives.py`.

## Attack Regimes

| Regime | Description |
|---|---|
| Per-example conditional recovery | `adam_margin` / `apgd_margin` per-example targeted attack |
| Conditional patch | Small-area (32×32) input-conditional patch, ~2% image area |
| Border-frame | Conditional 16px border frame delivery |
| Multi-patch | Two simultaneous conditional patches |
| Amortized short-budget | Learned generator from short Adam teacher trajectories |
| Universal perturbations | Fourier, tile, warp, codebook baselines |

## Setup

```bash
git clone https://github.com/DevT02/ForgetGate.git
cd ForgetGate

conda create -n forgetgate python=3.10 -y
conda activate forgetgate
pip install -r requirements.txt
```

CIFAR-10 downloads automatically via torchvision on first run.

**Known good environment:** Python 3.10, PyTorch 2.0+, PEFT 0.18.0, CUDA GPU recommended.

## Repo Structure

```
ForgetGate/
  src/
    unlearning/
      objectives.py      # All unlearning objectives including ORBIT
      trainer.py
    models/
      vit.py             # ViT-Tiny + LoRA, VPT variants
    attacks/
  scripts/
    audits/              # AWP, MIA, recovery radius audits
    attacks/             # Conditional patch / frame / multi-patch
    train/
  configs/
    unlearning.yaml
    experiment_suites.yaml
  results/
    analysis/
      figures/
        paper_draft.tex  # Paper source
        paper_draft.pdf  # Paper PDF
      metrics/           # Per-method result JSONs
      reports/           # Defense summary reports
```

## Key Scripts

```bash
# Train an unlearning checkpoint
python scripts/2_train_unlearning_lora.py

# Recovery radius audit (main conditional attack)
python scripts/17_recovery_radius_audit.py

# Conditional patch attack
python scripts/20_conditional_patch_attack_audit.py

# Border-frame attack
python scripts/23_conditional_frame_attack_audit.py

# Multi-patch attack
python scripts/24_conditional_multi_patch_attack_audit.py

# AWP audit
python scripts/audits/25_weight_perturbation_audit.py

# MIA audit
python scripts/audits/26_mia_audit.py
```

## Paper

Source and compiled PDF are in `results/analysis/figures/`. An arXiv link will be added here upon submission.

## Legacy Note

This repository originated as a LoRA + VPT resurrection study. That track is preserved in [`docs/legacy_lora_vpt.md`](docs/legacy_lora_vpt.md) as historical context but is no longer the top-level story.
