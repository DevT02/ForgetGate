# ForgetGate: The Recovery-Certification Gap

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]()

**Paper:** *The Recovery-Certification Gap: A Theory of Adversarial Unlearning*, Devansh Tayal. **[Read the working paper (PDF)](results/analysis/figures/paper_draft.pdf)**

---

ForgetGate argues that machine unlearning should be audited by one measurable quantity, the per-example *recovery radius*: the smallest input change that re-elicits a forgotten class. If a small perturbation recovers the forgotten class, the information was never really erased, so clean forget accuracy is the wrong test. The paper builds this into a small theory (four theorems) and validates each on an eight-method CIFAR-10 benchmark.

> A method can pass clean-accuracy and even rate-based audits and still leak under the recovery radius, and that radius is what an unlearning *certificate* must answer to.

## The theory (four theorems)

1. **Recovery-certification gap.** A method's measured recovery radius implies a floor that any certified-unlearning budget must respect, so an oracle-free audit can already challenge an over-tight certificate. It is a necessary condition rather than a strict certificate, since the Lipschitz term uses a local proxy.
2. **Selective-leakage test.** Calibrated against a zero-knowledge oracle null (a model retrained without the forgotten class), it separates genuine residual knowledge from generic adversarial fragility. The null has no intrinsic forget-vs-retain asymmetry, so a gap over it is real leakage.
3. **Covering-family impossibility.** Worst-case robustness to all patch and border-frame delivery surfaces at once forces a constant classifier, which is why mixed-surface patch defenses cannot transfer. The constructive target is stochastic (non-covering) delivery priors.
4. **Smoothed certified-recovery objective.** A Gaussian-smoothed objective attaches a randomized-smoothing certificate to unlearning. Trained end-to-end over 3 seeds it drives forget accuracy to 9-12% with retain preserved, and certifies an L2 radius of 0.193 that, in its own norm, sits below where the class re-emerges. A sigma sweep exposes a certification-utility tradeoff.

## Empirical findings

**1. Per-example conditional recovery is the dominant broad threat.**
Non-robust methods remain conditionally recoverable even when universal perturbations largely fail. This holds across SalUn, SCRUB, CE-U, SGA, and BalDRO.

**2. Local-delivery robustness is not a single property.**
Patch-aware stage-2 defenses reduce 32x32 conditional patch recovery on ORBIT (78.1% to 31.2%) and CE-U (68.8% to 43.8%), but fail to transfer to border-frame and multi-patch delivery surfaces. This is the empirical witness for Theorem 3.

**3. Defenses are partial and method-specific.**
Feature-subspace stage-2 gives the first consistent improvement to recovery radii on BalDRO, SalUn, and ORBIT without collapsing clean accuracy. RURK resists both attack families. SCRUB looks like generic adversarial fragility under most seeds; a strong attack on one seed exposes a transient selective gap, so its selectivity is not robust.

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

**[The Recovery-Certification Gap: A Theory of Adversarial Unlearning (PDF)](results/analysis/figures/paper_draft.pdf)**, a working paper. Source (`paper_draft.tex`) and all result JSONs and figures are in `results/analysis/`. Code and full results are in this repo; an arXiv link will be added on submission.

## Legacy Note

This repository originated as a LoRA + VPT resurrection study. That track is preserved in [`docs/legacy_lora_vpt.md`](docs/legacy_lora_vpt.md) as historical context but is no longer the top-level story.
