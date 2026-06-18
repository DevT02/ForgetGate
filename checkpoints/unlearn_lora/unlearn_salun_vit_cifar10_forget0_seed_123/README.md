# SalUn baseline (forget class 0, seed 123) — REGENERATED

This adapter replaces the earlier checkpoint, which did not reproduce the
paper's clean-forget rate (audited to ~0.3-0.5). Retrained from the committed
recipe `configs/suites/repro_salun.yaml` (objective: salun, rank 8, 50 epochs,
lr 1e-3). Audits to clean-forget 0.0%. Identical content to
`unlearn_salun_repro_vit_cifar10_forget0_seed_123`.
