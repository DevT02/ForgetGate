# Legacy LoRA/VPT Track

This document preserves the original project framing.

## Original Repo Question

The repository originally asked:

> If an attacker can train a tiny visual prompt, do forgotten classes come back
> because of residual knowledge, or because the attacker is just relearning?

That older track emphasized:

- LoRA-based unlearning
- visual prompt tuning (VPT) resurrection
- oracle-normalized recovery
- low-shot and prompt-length controls

## What That Legacy Track Contributed

- oracle-normalized evaluation rather than raw recovery alone
- prompt-length and low-shot controls
- feature-guided prompt attacks such as:
  - `linear_probe`
  - `prototype_margin`
  - `oracle_contrastive_probe`
- evidence that some LoRA-unlearned checkpoints could remain vulnerable even
  when clean forget accuracy looked good

## Why It Is No Longer the Whole Repo Story

The repository later accumulated stronger and broader results:

- full-image conditional recovery became the main threat model
- multiple modern unlearning methods were added
- frame and multi-patch delivery surfaces were added
- defense evaluation became broader than prompt-only recovery

So the old LoRA/VPT story is still valid history, but it is now one layer of
the project rather than the full project identity.

## Where To Look

Relevant legacy scripts:

- `scripts/3_train_vpt_resurrector.py`
- `scripts/analyze_kshot_experiments.py`
- `scripts/analyze_vpt_results.py`
- `scripts/9_feature_probe_leakage.py`

Relevant legacy concepts:

- oracle-normalized recovery
- low-shot prompt controls
- prompt-length ablations
- feature-guided prompt attacks

If you want the current benchmark story instead, go to:

- [Benchmark Overview](benchmark_overview.md)

