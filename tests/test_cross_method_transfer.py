"""
Tests for scripts/attacks/cross_method_transfer.py.

Run with: pytest tests/test_cross_method_transfer.py
"""

import importlib.util
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

_PATH = os.path.join(ROOT, "scripts", "attacks", "cross_method_transfer.py")
_spec = importlib.util.spec_from_file_location("xtransfer", _PATH)
xt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xt)


METHODS = [
    ("unlearn_orbit_vit_cifar10_forget0", "ORBIT"),
    ("unlearn_salun_vit_cifar10_forget0", "SalUn"),
    ("unlearn_rurk_vit_cifar10_forget0", "RURK"),
]


def test_mock_backend_diagonal_is_one():
    backend = xt.MockBackend()
    matrix = xt.build_transfer_matrix(METHODS, seed=42, backend=backend)
    forget = matrix["forget_recovery"]
    n = len(METHODS)
    for i in range(n):
        assert forget[i][i] == pytest.approx(1.0)


def test_matrix_has_correct_shape_and_labels():
    backend = xt.MockBackend()
    matrix = xt.build_transfer_matrix(METHODS, seed=42, backend=backend)
    n = len(METHODS)
    assert matrix["rows_attackers"] == [m[1] for m in METHODS]
    assert matrix["cols_targets"] == [m[1] for m in METHODS]
    assert len(matrix["forget_recovery"]) == n
    assert all(len(row) == n for row in matrix["forget_recovery"])
    assert len(matrix["retain_recovery"]) == n


def test_aggregate_averages_across_seeds():
    backend = xt.MockBackend()
    seeds = [42, 123, 456]
    mats = [xt.build_transfer_matrix(METHODS, seed=s, backend=backend) for s in seeds]
    agg = xt.aggregate_matrices(mats)
    assert agg["n_seeds"] == 3
    assert agg["seeds"] == seeds
    assert agg["labels"] == [m[1] for m in METHODS]
    n = len(METHODS)
    for i in range(n):
        # Mock is deterministic given (attacker, target), so means equal singletons
        for j in range(n):
            assert agg["forget_mean"][i][j] == pytest.approx(mats[0]["forget_recovery"][i][j])


def test_aggregate_handles_nan_cells():
    # Attacker 0 fails to train on seed 0, ok on seed 1.
    backend = xt.MockBackend()
    m1 = xt.build_transfer_matrix(METHODS, seed=1, backend=backend)
    m0 = xt.build_transfer_matrix(METHODS, seed=0, backend=backend)
    # Manually inject a NaN row to simulate a training failure on seed 0.
    n = len(METHODS)
    for j in range(n):
        m0["forget_recovery"][0][j] = float("nan")
    agg = xt.aggregate_matrices([m0, m1])
    # The row 0 mean should now equal m1's row 0 (only that contributed).
    for j in range(n):
        assert agg["forget_mean"][0][j] == pytest.approx(m1["forget_recovery"][0][j])


def test_aggregate_rejects_label_mismatch():
    backend = xt.MockBackend()
    m_a = xt.build_transfer_matrix(METHODS, seed=42, backend=backend)
    m_b = xt.build_transfer_matrix(METHODS[:2], seed=42, backend=backend)
    with pytest.raises(ValueError):
        xt.aggregate_matrices([m_a, m_b])


def test_render_table_emits_booktabs_and_labels():
    backend = xt.MockBackend()
    mats = [xt.build_transfer_matrix(METHODS, seed=s, backend=backend) for s in [42, 123]]
    agg = xt.aggregate_matrices(mats)
    forget_tex = xt.render_matrix_table(agg, "forget_mean")
    retain_tex = xt.render_matrix_table(agg, "retain_mean")
    for tex in (forget_tex, retain_tex):
        assert r"\toprule" in tex
        assert r"\midrule" in tex
        assert r"\bottomrule" in tex
    for label in ("ORBIT", "SalUn", "RURK"):
        assert label in forget_tex
        assert label in retain_tex


def test_render_table_uses_dash_for_nan():
    backend = xt.MockBackend()
    mats = [xt.build_transfer_matrix(METHODS, seed=42, backend=backend)]
    agg = xt.aggregate_matrices(mats)
    # Force a NaN
    agg["forget_mean"][0][1] = float("nan")
    tex = xt.render_matrix_table(agg, "forget_mean")
    assert "--" in tex


def test_render_rejects_unknown_kind():
    backend = xt.MockBackend()
    mats = [xt.build_transfer_matrix(METHODS, seed=42, backend=backend)]
    agg = xt.aggregate_matrices(mats)
    with pytest.raises(ValueError):
        xt.render_matrix_table(agg, "totally_unknown")


def test_train_failures_are_recorded():
    class FlakyBackend:
        def train(self, attacker_suite, seed):
            if "rurk" in attacker_suite:
                raise RuntimeError("synthetic train failure")
            return f"gen::{attacker_suite}"

        def evaluate(self, gen, target, seed):
            return {"forget_recovery": 0.5, "retain_recovery": 0.0}

    matrix = xt.build_transfer_matrix(METHODS, seed=42, backend=FlakyBackend())
    rurk_idx = [m[1] for m in METHODS].index("RURK")
    # Entire RURK row is NaN
    for j in range(len(METHODS)):
        v = matrix["forget_recovery"][rurk_idx][j]
        assert v != v
    # And the failure is in errors
    assert any(e["phase"] == "train" and "rurk" in e["attacker"] for e in matrix["errors"])


def test_eval_failures_are_recorded():
    class FlakyBackend:
        def train(self, attacker_suite, seed):
            return f"gen::{attacker_suite}"

        def evaluate(self, gen, target, seed):
            if "salun" in target:
                raise RuntimeError("synthetic eval failure")
            return {"forget_recovery": 0.5, "retain_recovery": 0.0}

    matrix = xt.build_transfer_matrix(METHODS, seed=42, backend=FlakyBackend())
    salun_col = [m[1] for m in METHODS].index("SalUn")
    for i in range(len(METHODS)):
        v = matrix["forget_recovery"][i][salun_col]
        assert v != v
    assert any(e["phase"] == "eval" and "salun" in e["target"] for e in matrix["errors"])


def test_parse_method_supports_label_override():
    suite, label = xt.parse_method("unlearn_orbit_vit=ORBIT")
    assert suite == "unlearn_orbit_vit"
    assert label == "ORBIT"
    suite, label = xt.parse_method("unlearn_salun_vit_cifar10_forget0")
    assert suite == "unlearn_salun_vit_cifar10_forget0"
    assert label == "unlearn_salun_vit_cifar10_forget0"
