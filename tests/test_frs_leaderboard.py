"""
Tests for scripts/analysis/build_frs_leaderboard.py.

Run with: pytest tests/test_frs_leaderboard.py
"""

import importlib.util
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

_FRS_PATH = os.path.join(ROOT, "scripts", "analysis", "build_frs_leaderboard.py")
_spec = importlib.util.spec_from_file_location("frs_module", _FRS_PATH)
frs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frs)


def _row(method, seed=42, **fields):
    return {"method": method, "seed": seed, **fields}


def test_perfect_method_scores_one():
    row = _row(
        "ideal",
        cond_recovery=0.0,
        patch_recovery=0.0,
        frame_recovery=0.0,
        multi_recovery=0.0,
        mia_auc_max=0.5,
        retain_accuracy=0.95,
        retain_baseline=0.95,
    )
    out = frs.compute_row_frs(row)
    assert out["frs"] == pytest.approx(1.0, abs=1e-9)
    assert out["n_regimes"] == 5
    assert out["retain_penalty"] == 0.0


def test_worst_method_scores_zero():
    row = _row(
        "worst",
        cond_recovery=1.0,
        patch_recovery=1.0,
        frame_recovery=1.0,
        multi_recovery=1.0,
        mia_auc_max=1.0,
        retain_accuracy=0.95,
        retain_baseline=0.95,
    )
    out = frs.compute_row_frs(row)
    assert out["frs"] == pytest.approx(0.0, abs=1e-9)


def test_weights_renormalize_when_regimes_missing():
    row_full = _row(
        "full",
        cond_recovery=0.5,
        patch_recovery=0.5,
        frame_recovery=0.5,
        multi_recovery=0.5,
        mia_auc_max=0.75,
    )
    row_partial = _row(
        "partial",
        cond_recovery=0.5,
        mia_auc_max=0.75,
    )
    out_full = frs.compute_row_frs(row_full)
    out_partial = frs.compute_row_frs(row_partial)
    # All terms produce identical robustness inputs for both rows, so
    # renormalization should yield the same FRS regardless of regime count.
    assert out_full["frs"] == pytest.approx(out_partial["frs"], abs=1e-9)
    assert out_partial["n_regimes"] == 2
    assert sum(out_partial["renormalized_weights"].values()) == pytest.approx(1.0)


def test_retain_penalty_applies_per_pp():
    base_row = dict(
        method="x",
        seed=42,
        cond_recovery=0.0,
        patch_recovery=0.0,
        frame_recovery=0.0,
        multi_recovery=0.0,
        mia_auc_max=0.5,
        retain_baseline=0.95,
    )
    no_drop = frs.compute_row_frs({**base_row, "retain_accuracy": 0.95})
    five_pp = frs.compute_row_frs({**base_row, "retain_accuracy": 0.90})
    # default penalty is 0.05 per pp -> 5pp drop = 0.25
    assert no_drop["frs"] == pytest.approx(1.0)
    assert five_pp["frs"] == pytest.approx(0.75, abs=1e-9)
    # negative drops (retain better than baseline) should not give a bonus
    bonus_attempt = frs.compute_row_frs({**base_row, "retain_accuracy": 1.0})
    assert bonus_attempt["frs"] == pytest.approx(1.0)


def test_mia_auc_clamping():
    # AUC below 0.5 is clamped to 0.5 (treated as "ideal")
    row = _row("clamp_low",
               cond_recovery=0.0, mia_auc_max=0.3)
    out = frs.compute_row_frs(row)
    assert out["frs"] == pytest.approx(1.0, abs=1e-9)


def test_missing_all_regimes_returns_warning():
    row = _row("empty", retain_accuracy=0.9, retain_baseline=0.9)
    out = frs.compute_row_frs(row)
    assert out["frs"] is None
    assert "warning" in out


def test_aggregate_averages_across_seeds():
    rows = [
        _row("M", seed=42, cond_recovery=0.0, mia_auc_max=0.5),   # FRS = 1.0
        _row("M", seed=123, cond_recovery=0.5, mia_auc_max=0.5),  # cond term -> 0.5
    ]
    out = frs.build_leaderboard(rows)
    by_method = {r["method"]: r for r in out["by_method"]}
    assert "M" in by_method
    seed_42 = next(r for r in out["per_row"] if r["seed"] == 42)
    seed_123 = next(r for r in out["per_row"] if r["seed"] == 123)
    expected_mean = (seed_42["frs"] + seed_123["frs"]) / 2.0
    assert by_method["M"]["frs_mean"] == pytest.approx(expected_mean, abs=1e-9)
    assert by_method["M"]["n_seeds"] == 2
    assert by_method["M"]["seeds"] == [42, 123]


def test_leaderboard_sorted_descending():
    rows = [
        _row("low",  cond_recovery=0.9, mia_auc_max=0.9),
        _row("high", cond_recovery=0.0, mia_auc_max=0.5),
        _row("mid",  cond_recovery=0.5, mia_auc_max=0.7),
    ]
    out = frs.build_leaderboard(rows)
    methods = [r["method"] for r in out["by_method"]]
    scores = [r["frs_mean"] for r in out["by_method"]]
    assert methods == ["high", "mid", "low"]
    assert scores[0] >= scores[1] >= scores[2]


def test_render_leaderboard_emits_booktabs():
    rows = [_row("ORBIT", cond_recovery=0.5, mia_auc_max=0.7)]
    out = frs.build_leaderboard(rows)
    tex = frs.render_leaderboard(out["by_method"])
    assert r"\begin{table}" in tex
    assert r"\toprule" in tex
    assert r"\midrule" in tex
    assert r"\bottomrule" in tex
    assert "ORBIT" in tex


def test_validate_input_rejects_missing_method_field():
    with pytest.raises(ValueError):
        frs._validate_input([{"seed": 42}])


def test_validate_input_rejects_non_list():
    with pytest.raises(ValueError):
        frs._validate_input({"method": "x"})


def test_custom_weights_renormalize_correctly():
    weights = {"cond_recovery": 0.6, "mia_auc_max": 0.4}
    row = _row("x", cond_recovery=0.5, mia_auc_max=0.5)
    out = frs.compute_row_frs(row, weights=weights)
    # cond term: 1 - 0.5 = 0.5; mia term: 1 - 2*(0.5 - 0.5) = 1.0
    # FRS = 0.6*0.5 + 0.4*1.0 = 0.7
    assert out["frs"] == pytest.approx(0.7, abs=1e-9)
