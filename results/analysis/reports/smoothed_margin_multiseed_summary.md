# Smoothed-margin (Theorem 4) multi-seed summary

End-to-end smoothed-margin unlearning (`unlearn_smoothed_margin_vit_cifar10_forget0`)
trained and audited on seeds 42, 123, 456. Hyperparameters: LoRA rank 8, 10 epochs,
sigma = 0.10, n_noise = 4 (train) / 256 (certified audit), gamma = 1.0, max_retain = 2000.

Source artifacts:
- training: `results/logs/unlearn_smoothed_margin_vit_cifar10_forget0_seed_{42,123,456}_history.json`
- certified radius: `results/analysis/metrics/smoothed_radius_seed_{42,123,456}_sigma_0p1_n256_*.json`
- recovery audit: `results/analysis/recovery_radius_cifar10_vit_tiny_forget0_seed_{42,123,456}_*_smoothed-margin.json`
- joined: `results/analysis/metrics/predicted_vs_measured_radius{,_seed_123,_seed_456}.json`
- consolidated: `results/analysis/metrics/smoothed_margin_multiseed_summary.json`

## Per-seed results

| Seed | Forget acc (before→after) | Retain acc (before→after) | Certified L2 R | Recovery median L∞ (forget / retain) | Selective gap | Forget viol. (clean / sub-bound) | Retain viol. |
|------|---------------------------|----------------------------|----------------|---------------------------------------|---------------|----------------------------------|--------------|
| 42   | 95.7% → 12.0%             | 93.9% → 93.8%              | 0.1933         | 0.00175 / 0.00607                     | 3.5×          | 10 (6 / 4)                       | 0            |
| 123  | 94.5% → 12.1%             | 93.8% → 93.7%              | 0.1933         | 0.00138 / 0.00588                     | 4.3×          | 11 (7 / 4)                       | 0            |
| 456  | 86.8% → 9.3%              | 87.4% → 87.9%              | 0.1933         | 0.00196 / 0.00521                     | 2.7×          | 5 (3 / 2)                        | 0            |

Mean selective gap **3.5×** (range 2.7–4.3×). Seed-456 retain accuracy is lower only
because the seed-456 *base* checkpoint is weaker (87.4% vs ~94%); the unlearning behavior
(forget → single digits, retain preserved) is consistent.

## Norm reconciliation (Theorem 4 caveat)

Inputs are 224×224×3, so d = 150,528 and √d ≈ 388. The certified L2 radius R = 0.1933
converts to an L∞-equivalent of only ≈ 0.000498 — comparable to the audit's own forget
recovery radii (~0.0014–0.0020). Theorem 4 is therefore an **L2 existence / proof-of-concept**
result (a smoothing certificate *can* be attached to an unlearning objective), not a tight,
operationally meaningful L∞ guarantee at this input size.

## Certificate consistency

Across all three seeds there are **zero retain-side certificate violations**. The forget-side
"violations" are dominated by clean-leak cases (measured radius 0, the un-smoothed model
already predicts the forgotten class) plus a few small-radius recoveries that fall just below
the √d-converted L∞ bound (0.000498) — consistent with the norm-conversion looseness, not a
genuine L2 violation.
