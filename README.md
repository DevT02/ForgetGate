# ForgetGate: Visual Prompt Attacks on Unlearning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ForgetGate studies one narrow question:

> If an attacker can train a tiny visual prompt, do "forgotten" classes come back because of residual knowledge, or because the attacker is just relearning?

The project uses visual prompt tuning (VPT) to attack unlearned vision models and compares recovery against an oracle model retrained from scratch without the forget class.

## Branches

- `main`: lightweight code and docs branch. It keeps the training and evaluation pipeline, but does not ship the large checkpoint/log bundle.
- `prompt-ablation-classgap`: full artifact branch with tracked logs, checkpoints, extended analyses, and the complete paper-facing result tables.

If you want the exact tracked results, use `prompt-ablation-classgap`. If you want the core codebase without large artifacts, stay on `main`.

## Headline Result

The tracked artifact branch reports three simple takeaways:

- Full-data VPT recovery reaches about 100 percent on both the unlearned model and the oracle, so that regime is mostly relearning.
- Default k-shot recovery is small but nonzero when normalized against the oracle.
- Stronger controls and follow-up methods show the problem is unstable rather than solved: low-shot recovery, prompt-length changes, and higher-shot SCRUB runs can still expose residual access or seed sensitivity.

This repo is intentionally about auditing that gap, not about claiming a universal unlearning benchmark.

## What Ships On Main

- Core training pipeline: base model, oracle retraining, LoRA unlearning, VPT resurrection, adversarial evaluation.
- Lightweight analysis helpers: `scripts/summarize_kshot_results.py` and `scripts/audit_results_manifest.py`.
- Resume helpers: `scripts/run_paper_resume.py` and `scripts/run_rigor_resume.ps1`.
- Reliability fixes:
  - `src/data.py` can fall back to the local CIFAR-10 tar archive if the extracted dataset tree is missing or broken.
  - `scripts/3_train_vpt_resurrector.py` defaults to `num_workers=0` on Windows unless you override it, which avoids CUDA worker spawn failures.

Extended feature-probe analyses, neighborhood sweeps, benign relearning runs, and the large tracked result bundle remain on `prompt-ablation-classgap`.

## Quick Setup

Requirements:

- Python 3.10+
- CUDA GPU recommended for ViT runs
- about 20 GB free disk for a fresh experiment setup

```bash
git clone https://github.com/DevT02/ForgetGate.git
cd ForgetGate

conda create -n forgetgate python=3.10 -y
conda activate forgetgate
pip install -r requirements.txt
```

## Minimal Reproduction

This is the smallest core path for the oracle-normalized k-shot setup on CIFAR-10 class 0.

```bash
# 1) Train the base model
python scripts/1_train_base.py \
  --config configs/experiment_suites.yaml \
  --suite base_vit_cifar10 \
  --seed 42

# 2) Train the oracle model without the forget class
python scripts/1b_train_retrained_oracle.py \
  --config configs/experiment_suites.yaml \
  --suite oracle_vit_cifar10_forget0 \
  --seed 42

# 3) Unlearn with Uniform KL LoRA
python scripts/2_train_unlearning_lora.py \
  --config configs/experiment_suites.yaml \
  --suite unlearn_kl_vit_cifar10_forget0 \
  --seed 42

# 4) Train VPT attacks
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_oracle_vit_cifar10_forget0_10shot --seed 42
python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot --seed 42

# 5) Aggregate any k-shot logs that exist in this worktree
python scripts/summarize_kshot_results.py --seeds 42 123 456
```

For longer resume-safe runs:

- Cross-platform: `python scripts/run_paper_resume.py`
- Windows helper for the targeted rigor pass: `powershell -ExecutionPolicy Bypass -File .\scripts\run_rigor_resume.ps1`

## Main Files

```text
ForgetGate/
  configs/
    experiment_suites.yaml
    model.yaml
    unlearning.yaml
    vpt_attack.yaml
    adv_eval.yaml
    data.yaml
  scripts/
    1_train_base.py
    1b_train_retrained_oracle.py
    2_train_unlearning_lora.py
    3_train_vpt_resurrector.py
    4_adv_evaluate.py
    6_analyze_results.py
    analyze_kshot_experiments.py
    analyze_vpt_results.py
    audit_results_manifest.py
    run_paper_resume.py
    run_rigor_resume.ps1
    summarize_kshot_results.py
  src/
    attacks/
    models/
    unlearning/
    data.py
    eval.py
    utils.py
  results/
    logs/
    analysis/
```

## Notes

- `main` keeps `results/logs/.gitkeep` and `results/analysis/.gitkeep`, so summary scripts can run even before artifacts exist.
- `scripts/summarize_kshot_results.py` reports whatever matching logs are actually present in the current worktree; it does not assume the full artifact bundle is available.
- `scripts/audit_results_manifest.py` is useful for checking which expected result files are missing before you trust a table.

## Related Work

Only the closest anchors are listed here:

- SalUn: https://arxiv.org/abs/2310.12508
- SCRUB: https://arxiv.org/abs/2302.09880
- LoRA: https://arxiv.org/abs/2106.09685
- VPT: https://arxiv.org/abs/2203.12119
- AutoAttack: https://arxiv.org/abs/2003.01690
- RURK: https://arxiv.org/abs/2601.22359
- MUSE: https://arxiv.org/abs/2407.06460

## Citation

```bibtex
@misc{forgetgate2025,
  title = {ForgetGate: Auditing Machine Unlearning with Visual Prompt Attacks},
  author = {Tayal, Devansh},
  year = {2025},
  url = {https://github.com/DevT02/ForgetGate}
}
```

## License

MIT.
