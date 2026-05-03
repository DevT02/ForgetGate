"""
Cross-method conditional-attack transfer harness.

For each ordered pair (attacker, target) of unlearning methods, train a
conditional perturbation generator on the *attacker* checkpoint and evaluate
the resulting perturbations against the *target* checkpoint. This produces an
NxN transfer matrix per seed: high transfer means the residual-knowledge
subspaces overlap across methods; near-zero transfer means each method
carves out a distinct vulnerability surface. Either outcome is publishable.

This script is split deliberately into two layers:

  1. Pure orchestration (matrix construction, JSON/LaTeX output, seed
     aggregation). This is fully unit-testable and is what
     tests/test_cross_method_transfer.py exercises.

  2. The attack itself, factored as two callables behind a single Protocol:

         AttackBackend.train(suite_A, seed)   -> generator artifact
         AttackBackend.evaluate(generator, suite_B, seed) -> {forget_recovery, retain_recovery}

     The default backend re-uses the conditional-attack training loop in
     scripts/attacks/19_conditional_attack_audit.py via importlib (the same
     pattern 19 itself uses for 17). A `MockBackend` is provided for tests.

By keeping orchestration and attack separate, the matrix layout, NaN handling,
and serialization can be validated without GPU, while the actual attack call
is delegated to code that has already produced the paper's published numbers.
"""

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Attack backend protocol
# ---------------------------------------------------------------------------


class AttackBackend(Protocol):
    """A pluggable attack backend.

    train(): produce a generator artifact (file path or in-memory object) by
    training the conditional attacker against the attacker model.

    evaluate(): apply the generator's perturbations to a target model and
    return success rates on forget samples and retain controls.
    """

    def train(self, attacker_suite: str, seed: int) -> object: ...

    def evaluate(
        self, generator_artifact: object, target_suite: str, seed: int
    ) -> Dict[str, float]: ...


# ---------------------------------------------------------------------------
# Default backend (delegates to 19_conditional_attack_audit.py)
# ---------------------------------------------------------------------------


def _load_audit_module():
    path = os.path.join(os.path.dirname(__file__), "19_conditional_attack_audit.py")
    spec = importlib.util.spec_from_file_location("conditional_attack_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class ConditionalAttackBackend:
    """Default backend that delegates to 19_conditional_attack_audit.py's
    `train_generator` and `eval_generator` programmatic API.

    Pass a `config_overrides` dict (or None) to override any field of the
    underlying ConditionalAttackConfig (eps_max, norm, epochs, sample budgets,
    etc.). The defaults match the paper's main conditional-attack settings.
    """

    config_overrides: Optional[Dict[str, object]] = None
    _config_obj: object = None

    def _config(self):
        if self._config_obj is not None:
            return self._config_obj
        mod = _load_audit_module()
        cfg = mod.ConditionalAttackConfig()
        if self.config_overrides:
            for k, v in self.config_overrides.items():
                if not hasattr(cfg, k):
                    raise ValueError(f"Unknown ConditionalAttackConfig field: {k!r}")
                setattr(cfg, k, v)
        self._config_obj = cfg
        return cfg

    def train(self, attacker_suite: str, seed: int) -> object:
        mod = _load_audit_module()
        return mod.train_generator(attacker_suite, seed, config=self._config())

    def evaluate(
        self, generator_artifact: object, target_suite: str, seed: int
    ) -> Dict[str, float]:
        mod = _load_audit_module()
        result = mod.eval_generator(
            generator_artifact, target_suite, seed, config=self._config()
        )
        return {
            "forget_recovery": float(result["forget_recovery"]),
            "retain_recovery": float(result["retain_recovery"]),
        }


@dataclass
class MockBackend:
    """A deterministic mock backend used by tests and CI smoke checks.

    train() returns the attacker suite name as the artifact. evaluate()
    produces a fully-deterministic 'transfer' rate that is highest when
    attacker == target and decays based on a simple string-hash distance.
    Importantly: evaluate() never reads the filesystem or imports torch.
    """

    seed_offset: int = 0

    def train(self, attacker_suite: str, seed: int) -> object:
        return f"mock-generator::{attacker_suite}::seed{seed}"

    def evaluate(self, generator_artifact: object, target_suite: str, seed: int) -> Dict[str, float]:
        attacker = str(generator_artifact).split("::")[1]
        same = attacker == target_suite
        # Deterministic "similarity" stand-in
        a = sum(ord(c) for c in attacker)
        b = sum(ord(c) for c in target_suite)
        sim = 1.0 - min(abs(a - b), 50) / 50.0
        forget_rate = 1.0 if same else max(0.0, 0.4 * sim)
        retain_rate = 0.0 if same else max(0.0, 0.05 * sim)
        return {"forget_recovery": forget_rate, "retain_recovery": retain_rate}


# ---------------------------------------------------------------------------
# Orchestration (pure)
# ---------------------------------------------------------------------------


def build_transfer_matrix(
    methods: Sequence[Tuple[str, str]],
    seed: int,
    backend: AttackBackend,
) -> Dict:
    """Build a per-seed N x N transfer matrix.

    methods: ordered list of (suite_name, display_label) pairs.
    """
    n = len(methods)
    forget = [[float("nan")] * n for _ in range(n)]
    retain = [[float("nan")] * n for _ in range(n)]
    errors: List[Dict] = []

    for i, (a_suite, _) in enumerate(methods):
        try:
            generator = backend.train(a_suite, seed)
        except Exception as e:
            errors.append({"phase": "train", "attacker": a_suite, "seed": seed, "error": str(e)})
            continue
        for j, (b_suite, _) in enumerate(methods):
            try:
                ev = backend.evaluate(generator, b_suite, seed)
            except Exception as e:
                errors.append({"phase": "eval", "attacker": a_suite,
                               "target": b_suite, "seed": seed, "error": str(e)})
                continue
            forget[i][j] = float(ev.get("forget_recovery", float("nan")))
            retain[i][j] = float(ev.get("retain_recovery", float("nan")))

    return {
        "seed": seed,
        "rows_attackers": [m[1] for m in methods],
        "cols_targets": [m[1] for m in methods],
        "forget_recovery": forget,
        "retain_recovery": retain,
        "errors": errors,
    }


def aggregate_matrices(matrices: Sequence[Dict]) -> Dict:
    """Average forget/retain matrices element-wise across seeds."""
    if not matrices:
        return {"forget_mean": [], "retain_mean": [], "labels": []}
    labels = matrices[0]["rows_attackers"]
    for m in matrices:
        if m["rows_attackers"] != labels or m["cols_targets"] != labels:
            raise ValueError("All matrices must share the same row/col label layout")
    n = len(labels)
    forget_sum = [[0.0] * n for _ in range(n)]
    retain_sum = [[0.0] * n for _ in range(n)]
    forget_cnt = [[0] * n for _ in range(n)]
    retain_cnt = [[0] * n for _ in range(n)]
    for m in matrices:
        for i in range(n):
            for j in range(n):
                fv = m["forget_recovery"][i][j]
                rv = m["retain_recovery"][i][j]
                if fv == fv:  # not NaN
                    forget_sum[i][j] += fv
                    forget_cnt[i][j] += 1
                if rv == rv:
                    retain_sum[i][j] += rv
                    retain_cnt[i][j] += 1
    forget_mean = [
        [forget_sum[i][j] / forget_cnt[i][j] if forget_cnt[i][j] else float("nan")
         for j in range(n)] for i in range(n)
    ]
    retain_mean = [
        [retain_sum[i][j] / retain_cnt[i][j] if retain_cnt[i][j] else float("nan")
         for j in range(n)] for i in range(n)
    ]
    return {
        "labels": labels,
        "n_seeds": len(matrices),
        "seeds": [m["seed"] for m in matrices],
        "forget_mean": forget_mean,
        "retain_mean": retain_mean,
    }


def render_matrix_table(matrix: Dict, kind: str = "forget_mean") -> str:
    """Render an aggregated matrix as a LaTeX table.

    kind: "forget_mean" or "retain_mean".
    """
    if kind not in {"forget_mean", "retain_mean"}:
        raise ValueError(f"unknown kind: {kind}")
    pretty = {"forget_mean": "Forget recovery",
              "retain_mean": "Retain-to-forget"}[kind]
    labels = matrix["labels"]
    n = len(labels)
    cols_spec = "l" + "r" * n
    head = " & ".join([r"Attacker $\downarrow$ / Target $\rightarrow$"] +
                      [lbl.replace("_", r"\_") for lbl in labels])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Cross-method conditional-attack transfer ({pretty}). "
        r"Cell $(i,j)$ is the success rate of an attacker trained on method $i$ "
        r"evaluated against method $j$. Diagonals are within-method results "
        r"(matched to the regime audit) and serve as sanity checks.}}",
        rf"\label{{tab:cross_transfer_{kind}}}",
        rf"\begin{{tabular}}{{{cols_spec}}}",
        r"\toprule",
        head + r" \\",
        r"\midrule",
    ]
    grid = matrix[kind]
    for i, row_label in enumerate(labels):
        cells = []
        for j in range(n):
            v = grid[i][j]
            cells.append("--" if v != v else f"{v:.2f}")
        lines.append(row_label.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_method(arg: str) -> Tuple[str, str]:
    if "=" in arg:
        suite, label = arg.split("=", 1)
        return suite.strip(), label.strip()
    return arg, arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True,
                        type=parse_method,
                        help="SUITE or SUITE=LABEL. Repeat for each method.")
    parser.add_argument("--seed", action="append", type=int, required=True,
                        help="Repeat for multiple seeds.")
    parser.add_argument("--backend", choices=["conditional", "mock"], default="conditional")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join("results", "analysis"))
    args = parser.parse_args()

    backend: AttackBackend
    if args.backend == "mock":
        backend = MockBackend()
    else:
        backend = ConditionalAttackBackend()

    matrices = [build_transfer_matrix(args.method, seed, backend) for seed in args.seed]
    aggregated = aggregate_matrices(matrices)

    metrics_dir = os.path.join(args.output_dir, "metrics")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    json_path = os.path.join(metrics_dir, "cross_method_transfer.json")
    with open(json_path, "w") as f:
        json.dump({"per_seed": matrices, "aggregated": aggregated}, f, indent=2)

    forget_tex = os.path.join(figures_dir, "cross_transfer_forget.tex")
    retain_tex = os.path.join(figures_dir, "cross_transfer_retain.tex")
    with open(forget_tex, "w") as f:
        f.write(render_matrix_table(aggregated, "forget_mean"))
    with open(retain_tex, "w") as f:
        f.write(render_matrix_table(aggregated, "retain_mean"))

    print(f"Wrote {json_path}")
    print(f"Wrote {forget_tex}")
    print(f"Wrote {retain_tex}")


if __name__ == "__main__":
    main()
