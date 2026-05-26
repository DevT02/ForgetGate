# Theory artifacts runbook

This runbook tracks the code, data, and runs that back the four theorems in
`results/analysis/figures/theory_appendix.tex`. The corresponding paper plan
is in `results/analysis/reports/paper_overhaul_plan.md`.

Updated: 2026-05-26. Branch: `main-no-checkpoints`.

## Headline results (Theorem 4, full end-to-end)

Trained 100 epochs of base ViT-Tiny on CIFAR-10 seed 42 → val_acc 94.2%.
Trained 10 epochs of smoothed-margin LoRA (σ=0.10, n_noise=4, γ=1.0,
lora_rank=8) on top → **forget 95.7% → 12.0%** while
**retain 93.9% → 93.8% (preserved)**.

**Smoothed certified-radius audit** (n_noise=256, α=1e-3, 64 forget + 64
retain): median certified L₂ radius **R = 0.1933** on both splits. This
is the saturation point of σ·Φ⁻¹(p_lower) when 256/256 noise draws
predict non-forget at α=10⁻³.

**Recovery-radius audit** (adam_margin, eps_max=8/255, same 64+64): forget
median ε = **0.001762**, retain median ε = **0.006127**. A clean
**3.5× selective gap** confirming Theorem 2's Pinsker test
(p_F ≈ 1, p_R ≈ 1 at eps_max, but separated by a real radius gap).

**Predicted-vs-measured plot** (`results/analysis/figures/predicted_vs_measured_radius.png`):
128 samples joined; 0 retain violations; 10 forget violations all of which
are clean-success cases where the un-smoothed model leaks the forgotten
class on clean input while the smoothed classifier remains robust
(positive signal for smoothing, not a Theorem 4 violation).

---

## What lives where

| Theorem | Code | Analysis | Output artifact |
|---|---|---|---|
| Thm 1 (Lipschitz → certified ε) | `src/theory/recovery_certification.py::lipschitz_proxy`, `cert_budget_upper_bound` | `scripts/analysis/31_lipschitz_radius_scatter.py` | `results/analysis/figures/lipschitz_radius_scatter.png`, `results/analysis/metrics/lipschitz_radius_scatter.json` |
| Thm 2 (Pinsker selective leakage) | `src/theory/recovery_certification.py::pinsker_kl_lb`, `bretagnolle_huber_kl_lb`, `JsonAnalyzer` | `scripts/analysis/30_pinsker_kl_table.py` | `results/analysis/reports/pinsker_kl_table.md`, `results/analysis/metrics/pinsker_kl_table.json` |
| Thm 3 (Local-delivery impossibility) | (proof only — no new code; the existing patch+frame+multi-patch audits are its empirical witness) | — | The covering visualization noted in `paper_overhaul_plan.md` step 4 |
| Thm 4 (Smoothed certified recovery) | `src/unlearning/objectives.py::SmoothedMarginUnlearning`, `configs/unlearning.yaml::smoothed_margin`, `src/theory/recovery_certification.py::smoothed_certified_radius` | `scripts/audits/27_smoothed_radius_audit.py` | `results/analysis/metrics/smoothed_radius_seed_*_sigma_*_n*_*.json` (per run) |

## What's already run on existing data

```
# Pinsker KL table  -- 226 rows from existing recovery_radius_*.json
conda activate forgetgate
python scripts/analysis/30_pinsker_kl_table.py
```

```
# Lipschitz-radius scatter  -- 40 joined rows from existing jacobian + radius JSONs
conda activate forgetgate
python scripts/analysis/31_lipschitz_radius_scatter.py
```

Outputs already on disk:

- `results/analysis/reports/pinsker_kl_table.md`
- `results/analysis/metrics/pinsker_kl_table.json`
- `results/analysis/figures/lipschitz_radius_scatter.png`
- `results/analysis/metrics/lipschitz_radius_scatter.json`

Sample Cor 1.1 readout from the scatter run (γ=2, B_logits=10):

| method | L proxy | median r̂ | max claimable ε_c |
|---|---|---|---|
| Uniform-KL | 16.06 | 0.0013 | 0.0495 |
| SalUn | 14.74 | 0.0044 | 0.0484 |
| ORBIT | 26.42 | 0.0250 | 0.0335 |
| SCRUB | 47.06 | 0.0314 | **0.0131** |

SCRUB's L is 3× ORBIT's and 3× SalUn's. That quantitatively explains why
SCRUB looks generically fragile (Thm 2's KL gap is near zero on SCRUB
rows of the Pinsker table): its Lipschitz is large enough that small
adversarial perturbations swing the margin both selectively and
non-selectively. Thm 1 × Thm 2 working together on real data.

## What needs GPU and what's blocking it

The two remaining runs:

1. **Smoothed certified-radius audit** — `scripts/audits/27_smoothed_radius_audit.py`
2. **Smoothed-margin unlearning training** — `scripts/train/2_train_unlearning_lora.py` with objective name `smoothed_margin`

Both load a base ViT-Tiny checkpoint named
`checkpoints/base/base_vit_cifar10_seed_42_final.pt`. That file is not in
this clone — only `base_vit_cifar10_pgdtrain_*` variants are.

### Unblock paths (in order of effort)

**A. Restore the original base checkpoint (preferred if you have it).**

Drop the file at `checkpoints/base/base_vit_cifar10_seed_42_final.pt` then:

```
conda activate forgetgate

# Theorem 4 audit -- predicted vs measured radius on already-unlearned SalUn
python scripts/audits/27_smoothed_radius_audit.py \
    --model-suites unlearn_salun_vit_cifar10_forget0 \
    --seed 42 \
    --sigma 0.10 --n-noise 256 --alpha 1e-3 \
    --max-forget 64 --max-retain 64 --batch-chunk 64

# Theorem 4 *training* -- smoothed-margin unlearner from the same base
python scripts/train/2_train_unlearning_lora.py \
    --suite unlearn_salun_vit_cifar10_forget0 \
    --objective smoothed_margin \
    --seed 42
# (adjust the suite-driver CLI to whatever your latest pipeline expects;
#  the objective name "smoothed_margin" is now wired through the factory)

# Then re-run the audit with --model-suites pointing at the new smoothed
# adapter to get the *predicted* certified radius from Thm 4.
```

**B. Train a fresh base from scratch (~2-4 GPU hours).**

```
conda activate forgetgate
python scripts/1_train_base.py --config configs/experiment_suites.yaml \
    --suite base_vit_cifar10 --seed 42
```

The suite definition trains 100 epochs of ViT-Tiny on CIFAR-10 at lr=1e-3,
batch=128 (see `configs/experiment_suites.yaml::base_vit_cifar10`). After
this completes you have to *also* re-train all the unlearn adapters that
sat on top of the old base (`unlearn_kl_*`, `unlearn_salun_*`,
`unlearn_scrub_distill_*`), since the existing adapters were trained against
a different base.

**C. Wire the existing pgdtrain base as a stand-in (~30 seconds, not
recommended for the paper).**

The simplest possible patch is to add a base-checkpoint override flag to
the audit script. The wiring change is tiny:

```python
# in scripts/audits/27_smoothed_radius_audit.py::_load_model, replace
#     base_ckpt = _checkpoint_or_best("checkpoints/base", base, seed)
# with a CLI override:
#     base_ckpt = args.base_ckpt_override or _checkpoint_or_best(...)
```

This makes the pipeline runnable for a *smoke validation*, but the
resulting numbers are not interpretable for the paper because the existing
SalUn LoRA adapter was trained against the original (missing) base — it
will misalign with `pgdtrain_mid`.

## Sanity checks that already passed

- `python src/theory/recovery_certification.py` — self-tests pass.
- `python tests/smoke_smoothed_margin.py` — smoothed objective forward +
  backward works on three batch configurations (mixed, no-forget,
  all-forget).
- `python scripts/audits/27_smoothed_radius_audit.py --help` — CLI loads
  with no missing imports.
- `python scripts/analysis/30_pinsker_kl_table.py` — 226 rows.
- `python scripts/analysis/31_lipschitz_radius_scatter.py` — 40 joined
  rows, plot saved.

## Recommendation

Restore the original `base_vit_cifar10_seed_42_final.pt` (option A) — it is
the only path that leaves the existing unlearn adapters interpretable. The
audit run on SalUn alone is ~5 GPU-minutes; the smoothed-margin training
run is ~10-30 GPU-minutes for a 10-epoch LoRA fine-tune. The two together
generate the Theorem 4 headline plot (predicted vs measured certified
radius) without any further engineering.
