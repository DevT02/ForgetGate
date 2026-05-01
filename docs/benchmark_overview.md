# Benchmark Overview

ForgetGate is currently organized as a **multi-regime benchmark for machine
unlearning robustness**.

## Scope

The benchmark currently emphasizes:

- CIFAR-10
- ViT-Tiny checkpoints
- mostly LoRA-based unlearning implementations
- selected full-finetune and cross-architecture checks

This is a controlled benchmark, not yet a broad large-scale benchmark.

## Main Benchmark Rows

Main-text rows:

- literature anchors: `SalUn`, `SCRUB`
- newer literature methods: `RURK`, `CE-U`
- adapted recent methods: `SGA`, `BalDRO`
- repo-defined benchmark method: `ORBIT`
- robust-base `SalUn`

Appendix/support:

- `Uniform KL`
- `Prune+KL`
- CNN checks
- legacy manifold/imposter branch

Proxy/partial rows:

- LoRA `SalUn`
- `FaLW`
- `Langevin`
- `OrthoReg`
- `noisy_retain`

## Attack Hierarchy

Primary:

- full-image conditional recovery
- one-patch conditional recovery
- frame delivery
- multi-patch delivery

Secondary:

- amortized short-budget attacks
- older prompt/probe attacks
- universal perturbation baselines

## Defense Acceptance Rule

A defense should not be treated as a main positive result unless:

1. clean forgetting remains valid
2. retain utility remains acceptable
3. the defense helps under more than one relevant attack family, or clearly
   improves the primary threat model without causing obvious regressions

## Best-Supported Current Defense Story

- feature-subspace stage-2 helps `BalDRO`, `SalUn`, and the repo-defined `ORBIT` row
- teacher-guided stochastic CVaR helps the repo-defined `ORBIT` row slightly
- `CE-U` combined stage-2 is a qualified partial row
- robust-base `SalUn` defines a frontier, not a fixed robust point

## Main Negative Story

- `RURK` remains hard
- `SCRUB` is better interpreted as generic adversarial fragility
- patch-aware defenses often fail to transfer to frame and multi-patch delivery
- mixed-surface stage-2 does not solve transfer reliably

## Key Files

- `results/analysis/method_fidelity_summary.md`
- `results/analysis/feature_subspace_tradeoff.md`
- `results/analysis/next_wave_experiments.md`
- `results/analysis/delivery_surface_plan.md`
- `results/analysis/multi_patch_surface_plan.md`
- `results/analysis/mixed_surface_stage2_plan.md`
- `results/analysis/figures/paper_draft.tex`
