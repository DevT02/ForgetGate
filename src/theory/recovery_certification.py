"""
Recovery-certification utilities (Theorems 1-4 of paper_draft.tex theory appendix).

This module replaces the heuristic resurrection_bounds.py with four functions
whose mathematics matches the four theorems and a JsonAnalyzer that consumes
the JSON output of scripts/audits/17_recovery_radius_audit.py.

  - lipschitz_proxy(model, x_batch, target_class, ...)
        Spectral-norm estimate of the margin function's input Jacobian at audit
        points. Used as L in Theorem 1's recovery--certification gap. For L_inf
        recovery the relevant constant is the L_1 norm of the margin gradient;
        we expose both L_inf and L_2 conversions.

  - cert_budget_upper_bound(median_r, gamma_oracle, L_proxy, B_logits)
        Inverts Corollary 1.1 to bound the maximum certified-unlearning epsilon
        a method can honestly claim from its observed median recovery radius
        and the oracle's expected margin.

  - pinsker_kl_lb(p_F, p_R)
        Pinsker's KL lower bound 2 (p_F - p_R)^2; the variational-information
        floor on selective leakage at the audit scale (Theorem 2).

  - smoothed_certified_radius(p_top_smooth, sigma)
        Cohen-Rosenfeld-Kolter L_2 certified radius for the smoothed margin
        indicator (Theorem 4).

Plus:

  - JsonAnalyzer(path)
        Loads a recovery_radius JSON and exposes p_F / p_R / median_r / KL.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Theorem 1 -- Lipschitz of the margin
# ---------------------------------------------------------------------------

def lipschitz_proxy(
    model,
    x_batch,
    target_class: int,
    norm: str = "linf",
    estimator: str = "input_jacobian",
    n_iters: int = 1,
):
    """Estimate the input-space Lipschitz constant of the margin

        m_t(x) = f_t(x) - max_{j != t} f_j(x)

    around each example in ``x_batch``. Two estimators:

      - "input_jacobian" (default): the Jacobian of the margin w.r.t. the
        input. The L_inf operator norm of this 1xD Jacobian equals its L_1
        vector norm; the L_2 operator norm equals its L_2 vector norm. Both
        are returned. This is an *exact* local Lipschitz at the point of
        evaluation; the global constant is the sup of this over the audit
        set.

      - "power_iteration": power iteration on the Hessian-vector product to
        get a tighter spectral norm (slower; useful when the input Jacobian
        is too small).

    Args:
        model: nn.Module returning logits of shape [B, K].
        x_batch: torch.Tensor [B, ...]; lives on the same device as model.
        target_class: int t in [K].
        norm: "linf" -> return L_1 norm of the gradient (L_inf Lipschitz).
              "l2"   -> return L_2 norm of the gradient (L_2 Lipschitz).
        estimator: see above.
        n_iters: power-iteration steps when estimator="power_iteration".

    Returns:
        dict with per-example "L" array (shape [B]) and "L_max" scalar
        (the sup, the global Lipschitz proxy that goes into Theorem 1).
    """
    # Local import keeps the module importable without torch in pure-analysis
    # contexts (e.g. scripts/analysis/30_pinsker_kl_table.py runs without GPU).
    import torch

    if estimator == "input_jacobian":
        x = x_batch.clone().detach().requires_grad_(True)
        logits = model(x)
        target_logit = logits[:, target_class]
        # second-best non-target logit per row
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, target_class] = False
        runnerup_logit = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
        margin = target_logit - runnerup_logit  # shape [B]
        grad = torch.autograd.grad(margin.sum(), x, create_graph=False)[0]
        flat = grad.flatten(start_dim=1)  # [B, D]
        if norm == "linf":
            L = flat.abs().sum(dim=1)  # L_1 norm of the gradient
        elif norm == "l2":
            L = flat.norm(p=2, dim=1)
        else:
            raise ValueError(f"unknown norm: {norm}")
        L = L.detach().cpu().numpy()
        return {"L": L, "L_max": float(L.max()), "L_median": float(np.median(L))}

    if estimator == "power_iteration":
        x = x_batch.clone().detach()
        D = x.numel() // x.shape[0]
        v = torch.randn_like(x.view(x.shape[0], -1))
        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12)
        Ls = []
        for _ in range(n_iters):
            x_req = x.clone().detach().requires_grad_(True)
            logits = model(x_req)
            tgt = logits[:, target_class]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, target_class] = False
            ru = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
            margin = tgt - ru
            gv = torch.autograd.grad(margin.sum(), x_req, create_graph=True)[0]
            gv = gv.view(x_req.shape[0], -1)
            jv = (gv * v).sum(dim=1)  # margin gradient dotted with v -- a scalar per row
            # second-order: differentiate jv w.r.t. x_req to get J^T (J v)
            hv = torch.autograd.grad(jv.sum(), x_req, create_graph=False)[0]
            hv = hv.view(x_req.shape[0], -1)
            sigma = hv.norm(p=2, dim=1)
            v = hv / (hv.norm(p=2, dim=1, keepdim=True) + 1e-12)
            Ls = sigma.detach().cpu().numpy()
        L = np.sqrt(np.maximum(Ls, 0.0))
        return {"L": L, "L_max": float(L.max()), "L_median": float(np.median(L))}

    raise ValueError(f"unknown estimator: {estimator}")


# ---------------------------------------------------------------------------
# Theorem 1 / Corollary 1.1 -- recovery -> certification gap
# ---------------------------------------------------------------------------

def cert_budget_upper_bound(
    median_r: float,
    gamma_oracle: float,
    L_proxy: float,
    B_logits: float,
) -> float:
    """Upper bound on the (epsilon_c, delta) certification budget a method can
    honestly claim, given an empirically measured median recovery radius.

    Corollary 1.1 rearranged:

        E_{x ~ mu_t}[r_t(x; theta_u)] >= (gamma - 4 B epsilon_c) / L
        =>  epsilon_c <= (gamma - L * median_r) / (4 B)

    Args:
        median_r:     median recovery radius (the audit median, in L_inf).
        gamma_oracle: oracle expected margin separation gamma > 0.
        L_proxy:      Lipschitz proxy of the margin (lipschitz_proxy['L_max']).
        B_logits:     logit-magnitude bound ||f||_inf.

    Returns:
        max claimable epsilon_c. Negative values mean *the method cannot be
        certified at any positive level* given its measured radius — i.e. the
        empirical bound is incompatible with any non-trivial unlearning
        certificate.
    """
    if L_proxy <= 0:
        return float("inf")
    if B_logits <= 0:
        raise ValueError("B_logits must be > 0")
    return (gamma_oracle - L_proxy * median_r) / (4.0 * B_logits)


# ---------------------------------------------------------------------------
# Theorem 2 -- Pinsker selective leakage
# ---------------------------------------------------------------------------

def pinsker_kl_lb(p_F: float, p_R: float) -> float:
    """Pinsker lower bound on KL between Ber(p_F) and Ber(p_R):

            |p_F - p_R| <= sqrt(KL/2)   =>   KL >= 2 (p_F - p_R)^2.

    Returns the lower bound 2 (p_F - p_R)^2. This is in nats.
    """
    return 2.0 * (p_F - p_R) ** 2


def bretagnolle_huber_kl_lb(p_F: float, p_R: float) -> float:
    """Bretagnolle-Huber, tighter than Pinsker near the boundary.

        |p_F - p_R| <= sqrt(1 - exp(-KL))   =>   KL >= -log(1 - (p_F - p_R)^2).
    """
    delta2 = (p_F - p_R) ** 2
    if delta2 >= 1.0:
        return float("inf")
    return -math.log(1.0 - delta2)


def kl_bernoulli_exact(p_F: float, p_R: float) -> float:
    """Exact KL(Ber(p_F) || Ber(p_R)) for reference. Returns nats."""
    eps = 1e-12
    pF = min(max(p_F, eps), 1 - eps)
    pR = min(max(p_R, eps), 1 - eps)
    return pF * math.log(pF / pR) + (1 - pF) * math.log((1 - pF) / (1 - pR))


# ---------------------------------------------------------------------------
# Theorem 4 -- smoothed certified radius
# ---------------------------------------------------------------------------

def smoothed_certified_radius(p_top_smooth: float, sigma: float) -> float:
    """Cohen-Rosenfeld-Kolter L_2 certified radius for a smoothed classifier.

    If the top class of the Gaussian-smoothed classifier has probability
    p_A = p_top_smooth and the runner-up has probability at most 1 - p_A,
    then the smoothed classifier's prediction is constant on the L_2 ball
    of radius

        R = sigma * Phi^{-1}(p_A).

    For our setting we use this on the *margin indicator*: p_top_smooth =
    Pr[m_t(x + delta; theta) <= 0] for delta ~ N(0, sigma^2 I). When that
    probability is above 1/2 we get a positive certified radius for "model
    does NOT predict the forgotten class on the ball of radius R".

    Args:
        p_top_smooth: probability the smoothed classifier predicts non-forget
                      under Gaussian noise. Must be in (0.5, 1) for a positive
                      certificate.
        sigma:        Gaussian noise std.

    Returns:
        Certified L_2 radius. 0.0 if p_top_smooth <= 0.5.
    """
    if p_top_smooth <= 0.5:
        return 0.0
    if p_top_smooth >= 1.0:
        return float("inf")
    # Inverse standard normal CDF
    from math import sqrt
    # use scipy if available; fall back to NumPy
    try:
        from scipy.stats import norm  # type: ignore
        return float(sigma * norm.ppf(p_top_smooth))
    except Exception:
        # crude rational approximation good to ~1e-4
        # (Beasley-Springer-Moro). Sufficient for our scatter plots.
        return float(sigma * _ndtri(p_top_smooth))


def _ndtri(p: float) -> float:
    """Standard normal inverse CDF (Beasley-Springer-Moro fallback)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


# ---------------------------------------------------------------------------
# JSON ingestion -- consume scripts/audits/17_recovery_radius_audit.py output
# ---------------------------------------------------------------------------

@dataclass
class AuditSummary:
    """One row's worth of audit data for the Pinsker / Lipschitz analyses."""
    suite: str
    seed: int
    forget_p_at_eps: float       # p_F = Pr[r_t <= eps_max] empirical
    retain_p_at_eps: float       # p_R = same on retain control
    forget_median_r: float       # median r_t on successful attacks (np.nan if none)
    forget_clean_success: float
    retain_clean_success: float
    eps_max: float
    n_forget: int
    n_retain: int

    @property
    def selective_gap(self) -> float:
        return self.forget_p_at_eps - self.retain_p_at_eps

    @property
    def pinsker_kl_lb(self) -> float:
        return pinsker_kl_lb(self.forget_p_at_eps, self.retain_p_at_eps)


class JsonAnalyzer:
    """Loads one recovery_radius_*.json and exposes per-method AuditSummary
    rows usable by scripts/analysis/30_pinsker_kl_table.py and 31_*."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "r") as f:
            self.data = json.load(f)
        self.meta = self.data["meta"]
        # support both new (forget_recovery/retain_control) and older formats
        if "forget_recovery" in self.data:
            self.forget = self.data["forget_recovery"]
            self.retain = self.data.get("retain_control", {})
        elif "splits" in self.data:
            self.forget = self.data["splits"].get("forget", {})
            self.retain = self.data["splits"].get("retain", {})
        else:
            raise ValueError(f"unrecognized recovery_radius JSON shape: {path}")

    def suites(self) -> List[str]:
        return sorted(self.forget.keys())

    def summarize(self) -> List[AuditSummary]:
        out = []
        eps_max = float(self.meta.get("eps_max", float("nan")))
        seed = int(self.meta.get("seed", -1))
        for suite in self.suites():
            f = self.forget[suite]
            r = self.retain.get(suite, {})

            fsum = f.get("summary", {})
            rsum = r.get("summary", {})

            forget_p = float(fsum.get("success_rate", float("nan")))
            retain_p = float(rsum.get("success_rate", float("nan")))
            forget_med = float(fsum.get("median_radius", float("nan")))
            f_clean = float(fsum.get("clean_success_rate", float("nan")))
            r_clean = float(rsum.get("clean_success_rate", float("nan")))
            n_f = int(fsum.get("n_samples", 0))
            n_r = int(rsum.get("n_samples", 0))

            out.append(AuditSummary(
                suite=suite,
                seed=seed,
                forget_p_at_eps=forget_p,
                retain_p_at_eps=retain_p,
                forget_median_r=forget_med,
                forget_clean_success=f_clean,
                retain_clean_success=r_clean,
                eps_max=eps_max,
                n_forget=n_f,
                n_retain=n_r,
            ))
        return out


# ---------------------------------------------------------------------------
# tiny self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pinsker
    assert abs(pinsker_kl_lb(0.5, 0.5) - 0.0) < 1e-12
    assert abs(pinsker_kl_lb(1.0, 0.0) - 2.0) < 1e-12

    # smoothed radius
    assert smoothed_certified_radius(0.5, 1.0) == 0.0
    r = smoothed_certified_radius(0.975, 1.0)
    # Phi^{-1}(0.975) ~ 1.96
    assert 1.9 < r < 2.0, r

    # cert budget
    eps = cert_budget_upper_bound(median_r=0.01, gamma_oracle=5.0, L_proxy=100.0, B_logits=10.0)
    # (5 - 100*0.01) / (4*10) = (5 - 1) / 40 = 0.1
    assert abs(eps - 0.1) < 1e-9, eps

    print("recovery_certification self-tests passed")
