# Smoothed-margin LoRA adapter (Theorem 4, seed 42)

Trained by `scripts/train/train_smoothed_margin.py` against
`checkpoints/base/base_vit_cifar10_thm4run_seed_42_final.pt` (the fresh
100-epoch base co-committed in this branch under the `thm4run` lineage
tag, not the pre-existing `base_vit_cifar10_seed_42_final.pt` that other
adapters here were trained against).

Hyperparameters: `sigma=0.10`, `n_noise=4`, `gamma=1.0`, `lora_rank=8`,
`epochs=10`, `lr=1e-3`, `max_retain=2000`, `forget_class=0`.

Result: forget acc 95.7% -> 12.0%, retain acc 93.9% -> 93.8% (preserved).

Matched audits:
- `results/analysis/metrics/smoothed_radius_seed_42_sigma_0p1_n256_smoothed_margin_vit_cifar10_forget0.json`
  -- Cohen-Rosenfeld-Kolter certified L_2 radius (median = 0.193).
- `results/analysis/recovery_radius_cifar10_vit_tiny_forget0_seed_42_test_with_retain_adam_margin_smoothed-margin.json`
  -- measured L_inf recovery radius (forget median 0.00176, retain 0.00613).

Reproduce:
```
python scripts/train/train_smoothed_margin.py --base-suite base_vit_cifar10_thm4run \
    --seed 42 --epochs 10 --sigma 0.10 --n-noise 4 --gamma 1.0 \
    --max-retain 2000 --suite-tag unlearn_smoothed_margin_vit_cifar10_forget0
```
(Loads `checkpoints/base/base_vit_cifar10_thm4run_seed_42_final.pt`.)
