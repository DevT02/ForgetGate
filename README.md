# ForgetGate: Visual Prompt Attacks on Unlearning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ForgetGate is a machine learning research project about a simple question:

> When a model says it has "forgotten" a class, is that information really gone, or is it still recoverable?

I study that question in vision models by unlearning a class, attacking the model with a tiny learned visual prompt, and comparing the result against an oracle model retrained from scratch without the forgotten class.

## Why This Matters

Many unlearning results focus on one surface metric: the model stops predicting the forgotten class. That can look convincing while still leaving recoverable information inside the model. This project is about testing whether the forgetting is real or just hidden.

## Project Snapshot

- End-to-end PyTorch pipeline for training, unlearning, attacking, and auditing vision models.
- Main setup: ViT-Tiny and ResNet-18, LoRA-based unlearning, oracle retraining, VPT resurrection attacks, and adversarial evaluation.
- Main finding: low forget-class accuracy alone is not enough to claim secure unlearning. Small prompt attacks can still recover class information beyond what an oracle model can relearn.

## My Contribution

- Designed the evaluation setup around an oracle baseline, so recovery can be separated into ordinary relearning versus residual information left behind after unlearning.
- Built the experiment pipeline, attack code, result summarizers, and resume-safe scripts used to run and audit the project.
- Ran a broader set of controls than a single headline metric: k-shot recovery, prompt-length stress tests, label controls, stronger baselines such as SCRUB, and artifact-auditing utilities.

## Branches

- `main`: lightweight code and docs branch. It keeps the training and evaluation pipeline, but does not ship the large checkpoint/log bundle.
- `prompt-ablation-classgap`: full artifact branch with tracked logs, checkpoints, extended analyses, and the complete paper-facing result tables.

The exact tracked results live on `prompt-ablation-classgap`. The lightweight codebase remains on `main`.

## Core Result

The tracked artifact branch reports three simple takeaways:

- Full-data VPT recovery reaches about 100 percent on both the unlearned model and the oracle, so that regime is mostly relearning.
- Default k-shot recovery is small but nonzero when normalized against the oracle.
- Stronger controls and follow-up methods show the problem is unstable rather than solved: low-shot recovery, prompt-length changes, and higher-shot SCRUB runs can still expose residual access or seed sensitivity.

This repo is intentionally an audit of post-unlearning recoverability, not a claim that one new method solves unlearning in general.

## Full Artifact Highlights

The richer project story is documented on [`prompt-ablation-classgap`](https://github.com/DevT02/ForgetGate/tree/prompt-ablation-classgap). The main tracked results from that branch are:

### Selected Results

- Default oracle-normalized KL recovery stays small at k=10/25/50/100: 0.47, 0.43, 0.43, and 1.90 percent, while the oracle stays at 0.00 percent.
- Prompt-length-5 stress tests are much harsher than the default k-shot table: k=1 is 4.27 percent, k=5 is 4.53 percent, shuffled-label is 3.90 percent, and random-label is 4.03 percent.
- The SCRUB follow-up is mixed rather than cleanly stronger: 0.00 percent at 10-shot and 25-shot across seeds 42/123/456, but 50-shot and 100-shot become highly seed-sensitive because seed 42 recovers strongly while seeds 123 and 456 stay at 0.00 percent.
- The artifact branch also includes the broader diagnostic side of the project: dependency-aware evaluation, feature-probe leakage, neighborhood sweeps, class-wise gaps, prompt-length ablations, and the larger checkpoint/log bundle behind those tables.

| Setting | Result |
|---|---|
| KL, default prompt, 10-shot | 0.47% recovery vs 0.00% oracle |
| KL, default prompt, 100-shot | 1.90% recovery vs 0.00% oracle |
| KL, prompt-length-5 stress test, k=1 | 4.27% recovery vs 0.00% oracle |
| KL, prompt-length-5 stress test, k=5 | 4.53% recovery vs 0.00% oracle |
| SCRUB, 10-shot / 25-shot | 0.00% across tracked seeds |
| SCRUB, 50-shot / 100-shot | high seed sensitivity rather than stable robustness |

So the short version is not just "KL is weak." It is closer to: low clean forget accuracy does not reliably imply resistance to recovery, and that weakness depends heavily on the attacker setup and the seed.

## What Ships On Main

- Core training pipeline: base model, oracle retraining, LoRA unlearning, VPT resurrection, adversarial evaluation.
- Lightweight analysis helpers: `scripts/summarize_kshot_results.py` and `scripts/audit_results_manifest.py`.
- Resume helpers: `scripts/run_paper_resume.py` and `scripts/run_rigor_resume.ps1`.
- Reliability fixes:
  - `src/data.py` can fall back to the local CIFAR-10 tar archive if the extracted dataset tree is missing or broken.
  - `scripts/3_train_vpt_resurrector.py` defaults to `num_workers=0` on Windows unless you override it, which avoids CUDA worker spawn failures.

Extended feature-probe analyses, neighborhood sweeps, benign relearning runs, and the large tracked result bundle remain on `prompt-ablation-classgap`.

## Engineering Highlights

- Suite-driven experiment pipeline for base training, oracle retraining, LoRA unlearning, prompt attacks, and evaluation.
- Resume-safe helpers for longer experiment batches on Windows and conda-based setups.
- Lightweight result-auditing scripts to summarize k-shot runs and check which expected artifacts are missing.
- Reliability fixes that make the repo easier to run on real machines, especially Windows CUDA setups and CIFAR-10 environments with broken extracted data trees.

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
