# ForgetGate: The Recovery-Certification Gap

*Auditing machine unlearning with the recovery radius: a fragility confound in selective-leakage metrics, plus a certified input-space recovery floor.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]()

**Paper:** *The Recovery-Certification Gap: Auditing Machine Unlearning Beyond Clean Forget Accuracy*, Devansh Tayal. **[Read the working paper (PDF)](results/analysis/figures/paper_draft.pdf)**

---

ForgetGate argues that machine unlearning should be audited by one measurable quantity, the per-example *recovery radius*: the smallest input change that re-elicits a forgotten class. If a small perturbation recovers the forgotten class, clean forget accuracy is the wrong test.

**Headline finding.** The forget-vs-retain *selectivity* metric used across the unlearning-audit literature to tell residual knowledge from generic adversarial fragility is **confounded by a checkpoint's global brittleness**. Across 132 audits over 57 checkpoints, apparent selectivity correlates *negatively* with how fragile a model is overall, because the forget radius is *super-linearly elastic* to fragility (observational slope b=1.31; a controlled base-smoothness experiment gives a causal b=1.36, CI [1.31, 1.84]). The correction is a fragility-matched taxonomy that separates genuine leakers (SalUn, RURK) from brittleness artifacts (SCRUB). The eight-method CIFAR-10 benchmark and four supporting theorems are the evidence and formal footing for this finding, not the end in themselves.

> A method can pass clean-accuracy and even rate-based audits and still leak under the recovery radius, and that radius is what an unlearning *certificate* must answer to.

## Supporting results (four theorems)

1. **Recovery-certification gap.** A method's measured recovery radius gives a necessary diagnostic for certified-unlearning budget claims. The plotted evidence is a heuristic ranking because the Lipschitz term uses a local proxy rather than a certified global upper bound.
2. **Selective-leakage test.** Calibrated against a retain-only oracle null (a model retrained without the forgotten class), it separates genuine residual knowledge from generic adversarial fragility. The null is an operational baseline for target-label fragility; a gap over it is evidence of residual leakage.
3. **Covering-family impossibility.** Exact worst-case robustness to all patch and border-frame delivery surfaces at once forces a constant classifier. The constructive target is distributional local-delivery robustness, not global worst-case invariance.
4. **Smoothed certified-recovery objective.** A Gaussian-smoothed objective is paired with a post-training randomized-smoothing certificate for a binary forgotten-class detector. Trained end-to-end over 3 seeds it drives forget accuracy to 9-12% with retain preserved, and certifies an L2 detector-radius scale of 0.193 that, in its own norm, sits below where the class re-emerges. A sigma sweep exposes a certification-utility tradeoff.

## Empirical findings

**1. The selectivity metric is confounded by global fragility (central finding).**
The standard forget-vs-retain selectivity gap partly measures how brittle a checkpoint is, not how much it leaks. Across 132 audits over 57 checkpoints, selectivity correlates negatively with global fragility, because the forget radius is super-linearly elastic to it (b=1.31; controlled causal b=1.36). Reporting fragility alongside selectivity, and using the fragility-matched taxonomy, separates genuine leakers (SalUn, RURK) from brittleness artifacts (SCRUB, which a retain-only oracle null places at the bottom).

**2. Per-example conditional recovery is the dominant broad threat.**
Non-robust methods remain conditionally recoverable even when universal perturbations largely fail. This holds across SalUn, SCRUB, CE-U, SGA, and BalDRO.

**3. Local-delivery robustness is not a single property.**
Patch-aware stage-2 defenses reduce 32x32 conditional patch recovery on ORBIT (78.1% to 31.2%) and CE-U (68.8% to 43.8%), but fail to transfer to border-frame and multi-patch delivery surfaces. This is empirical context for Theorem 3's worst-case obstruction.

**4. Defenses are partial and method-specific.**
Feature-subspace stage-2 gives positive point-estimate shifts on BalDRO, SalUn, and ORBIT without collapsing clean accuracy, clearest on ORBIT and SalUn and tiny on BalDRO. RURK resists both attack families. SCRUB looks like generic adversarial fragility under most seeds; a strong attack on one seed exposes a transient selective gap, so its selectivity is not robust.

**5. The constructive certificate now has a limited cross-architecture replication.**
The main smoothed-margin result is still CIFAR-10 / ViT-Tiny, but the constructive objective has a two-seed ResNet-18 / CIFAR-100 breadth check. It drives class-0 forget accuracy to 3-7% while preserving retain accuracy at about 88%, certifies an L2 detector-radius scale of 0.193 on 92-94% of audited forget points, and keeps a selective recovery gap of about 5.2x. Older single-seed ViT-Small/CIFAR-100 rows remain feasibility checks, not breadth claims.

## Methods Evaluated

| Method | Status |
|---|---|
| SalUn | Literature anchor |
| SCRUB | Literature anchor |
| RURK-style | Objective-level adaptation; random-noise robustness rather than the original adversarial term |
| CE-ascent (labeled CE-U) | Gradient-ascent baseline; not corrected-target CE-U |
| SGA | Adapted recent method |
| BalDRO | Adapted recent method |
| ORBIT | Diagnostic stress-test method introduced here |
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

**Known good environment:** Python 3.10, PyTorch 2.1.x, torchvision 0.16.x, PEFT 0.13.x, Accelerate 0.34.x, Transformers 4.41-4.44, CUDA GPU recommended.

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
# Run lightweight verification
python -m pytest -q tests/test_basic.py tests/test_frs_leaderboard.py tests/test_cross_method_transfer.py

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

**[The Recovery-Certification Gap: Auditing Machine Unlearning Beyond Clean Forget Accuracy (PDF)](results/analysis/figures/paper_draft.pdf)**, a working paper. Source (`paper_draft.tex`) and all result JSONs and figures are in `results/analysis/`. Code and full results are in this repo; an arXiv link will be added on submission.

## Legacy Note

This repository originated as a LoRA + VPT resurrection study. That track is preserved in [`docs/legacy_lora_vpt.md`](docs/legacy_lora_vpt.md) as historical context but is no longer the top-level story.
