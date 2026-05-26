# Paper Overhaul Plan: From Benchmark to Theory + Validation

**Date:** 2026-05-24
**Author:** D. Tayal (with theory drafted to `results/analysis/figures/theory_appendix.tex`)

---

## Diagnosis: why the current draft underperforms

The current `paper_draft.tex` is a clean, careful multi-regime benchmark. Its
weakness is structural, not stylistic:

1. **Every claim is an empirical observation.** Nothing in the paper is
   provable. The eq. (2) "proposed defense objective" appears as a paragraph,
   not as a result.
2. **The forget-vs-retain-control "selective leakage" idea is never
   formalized.** It is invoked as a qualitative reading of SCRUB and ORBIT
   but has no statistical or information-theoretic anchor.
3. **The recovery radius is reported but never bridged to certified-unlearning
   literature.** As a result, the paper sits next to RURK, Langevin
   Unlearning, and Yu et al. (2025) without ever joining their conversation.
4. **The "local-delivery transfer is an open problem" framing is too weak.**
   Empirically we already know mixed-surface stage-2 fails. The paper should
   say *why*, structurally — not just "future work."
5. **The existing `src/theory/resurrection_bounds.py` is heuristic.** Its
   "PAC sample-complexity" computation uses a hand-tuned constant
   ($K = 0.1$) and conflates frozen-parameter fraction with retained
   information. It cannot anchor the paper.

## What the new theory gives the paper

Four theorems (in `theory_appendix.tex`) reorganize the entire narrative:

| Theorem | What it formalizes | Empirical hook that becomes its test |
|---|---|---|
| **Thm 1** Recovery–Certification Gap | A pointwise Lipschitz bound: $r_t(x;\theta_u) \geq -m_t(x;\theta_u)/L$ — and a corollary that every claimed $(\epsilon_c, \delta)$-certificate forces a minimum expected radius. | Per-method `(Lipschitz proxy, median radius)` scatter against the bound's line. Already computable from `16_jacobian_access_audit.py` + `17_recovery_radius_audit.py`. |
| **Thm 2** Pinsker Selective-Leakage | $|p_F - p_R| \leq \sqrt{\tfrac12 \mathrm{KL}}$ on the binary recoverability outcome. | The existing forget-vs-retain table per method becomes a *KL test*. SCRUB row formally certifies as vacuous selective leakage. |
| **Thm 3** Local-Delivery Transfer Impossibility | Worst-case invariance over a covering family of delivery masks forces the classifier to be constant — hence mixed-surface stage-2 cannot work for non-trivial models. | The mixed-surface failure stops being "future work" and becomes a *negative result* with the right framing. |
| **Thm 4** Smoothed Certified-Recovery Objective | A Gaussian-smoothing replacement for the inner-attack term in eq. (2) of the main paper, with a Cohen–Rosenfeld–Kolter certified radius and a PAC generalization bound. | New $\sigma$-sweep experiment on ORBIT/CE-U: *predicted* vs *measured* radius. This is the new headline figure. |

## Concrete restructure

### New section order

```
1. Introduction
2. Setup, recovery radius, and the residual forget-channel capacity R_eps
   ← absorbs the current "Threat Models and Attack Regimes" section
3. Theory                                          ← from theory_appendix.tex
   3.1 Definitions (R_eps, selective leakage)
   3.2 Thm 1 (Recovery–certification gap)
   3.3 Thm 2 (Pinsker selective separation)
   3.4 Thm 3 (Local-delivery transfer impossibility)
   3.5 Thm 4 (Smoothed certified objective)
4. Methods and benchmark substrate
   ← shortened: literature anchors and ORBIT defined briefly
5. Empirical validation
   5.1 Thm 1 check: Lipschitz–radius scatter across 8 methods
   5.2 Thm 2 check: KL gap table; SCRUB falsified as selective-leakage case
   5.3 Patch/frame/multi-patch: Thm 3 corollary as the headline
   5.4 Thm 4 implementation: predicted-vs-realized smoothed radius curve
6. Discussion and limitations
7. Conclusion
Appendices: seed variance, k-shot controls, etc. (current appendices kept)
```

### Sections to delete or compress

- `Short-Budget and Auxiliary Regimes` § — fold the Uniform-KL and SGA
  paragraphs into a single "appendix: regime-defining auxiliary methods."
- `Earlier prompt/probe regime` paragraph — move entirely to appendix; it
  predates the recovery-radius framing.
- `LoRA as benchmark substrate` paragraph — compress to one sentence in
  Methods.
- The "Proposed Defense Objective" subsection — replace with a forward
  reference to Theorem 4 and a one-paragraph design note.

### New figures the theory unlocks

1. **Lipschitz–radius scatter (Thm 1):** x-axis $\hat L$, y-axis $\hat r$,
   one point per (method, seed). Line: $\hat r = -\bar m / \hat L$.
   *Reading:* methods on the line are tight (their radii are explained
   purely by their Lipschitz constant); methods strictly above the line have
   margin headroom (forgetting can be sharpened); methods below the line
   reveal a Lipschitz estimate too loose to be useful.
2. **Selective-leakage KL table (Thm 2):** columns
   `(p_F, p_R, |p_F-p_R|, KL_LB)`; methods sorted by `KL_LB`. ORBIT/CE-U at
   top, SCRUB at bottom near zero. A one-line bound: "$\mathrm{KL} \geq
   2|p_F - p_R|^2$ certifies selective leakage at the audit scale."
3. **Covering visualization (Thm 3):** show a CIFAR image with a 32×32 patch
   ∪ 16-px frame; caption: "covering implies invariance implies
   constancy — mixed-surface stage-2 cannot deliver worst-case
   robustness without collapsing the classifier."
4. **Predicted-vs-realized smoothed radius (Thm 4):** x-axis $\sigma$;
   y-axis radius; two curves — predicted from Cohen-Rosenfeld-Kolter and
   measured from `17_recovery_radius_audit.py` on $\theta_\sigma^*$.

## New experiments needed (minimal additions)

All four are 1-2 days of GPU each, reuse existing audit code.

| Exp | Cost | Code change |
|---|---|---|
| Per-method Jacobian Lipschitz estimate at audit points | 1 hr GPU per method × 8 methods × 3 seeds | already in `16_jacobian_access_audit.py` — add a wrapper that returns the operator norm of the margin Jacobian; aggregate. |
| Pinsker / KL table from existing forget+retain `p_F, p_R` | trivial | post-processing of existing JSONs in `results/analysis/metrics/`. |
| Smoothed-objective training on ORBIT seed 42 | 4–8 hr GPU per σ × 4 σ values | new file `scripts/train/train_smoothed_orbit.py`: drop-in modification of `2_train_unlearning_lora.py` replacing the inner PGD with $m$ Gaussian noise samples. |
| Predicted-radius certification | 1 hr GPU per σ | new file `scripts/audits/27_smoothed_radius_audit.py`: implement Cohen et al. certification on the smoothed margin indicator; reuse audit harness. |

## What to drop from the existing codebase

- `src/theory/resurrection_bounds.py` — replace with `src/theory/recovery_certification.py` (new file: implements the four theorem-aligned utilities listed below). The current file's `K = 0.1` heuristic and information-retained estimator do not survive contact with Thm 1.

## New utility module to add

`src/theory/recovery_certification.py` with four functions matching the four theorems:

```python
def lipschitz_proxy(model, x_batch, target_class) -> float:
    """Jacobian-of-margin operator-norm estimate at audit points."""

def cert_budget_upper_bound(median_r, gamma_oracle, L_proxy, B_logits) -> float:
    """Eq. from Cor. 1.1 inverted: max epsilon_c the method can claim."""

def pinsker_kl_lb(p_F: float, p_R: float) -> float:
    """Returns 2 * (p_F - p_R)^2; the KL lower bound from Thm 2."""

def smoothed_certified_radius(p_top_smooth: float, sigma: float) -> float:
    """Cohen et al. radius for the smoothed margin classifier (Thm 4)."""
```

These four functions are the smallest possible code-level commitment to the
theory: every empirical figure in the new manuscript calls exactly one of
them.

## Title and framing changes

**Current title:** *Multi-Regime Red Teaming of Machine Unlearning*

**Suggested new title (one of):**

1. *The Recovery–Certification Gap: A Theory of Adversarial Unlearning*
2. *Unlearning You Can Audit: Selective Leakage, Local-Delivery Impossibility, and a Smoothed Certificate*
3. *Forget-Channel Capacity: Bridging Recovery Audits and Certified Unlearning*

I prefer (1). It is direct about the contribution, places the paper in the
certified-unlearning conversation, and signals theory + matched empirical.

**One-sentence abstract opener (replacement for current one):**

> We show that the per-example recovery radius — the smallest input
> perturbation that re-elicits a forgotten class — is a quantitative
> proxy for certified unlearning, with a Lipschitz lower bound, a Pinsker
> selective-leakage test, a topological transfer-impossibility, and a
> matched smoothed-certificate objective.

## Risk audit on the new theorems

| Theorem | Sharpest objection | Defense |
|---|---|---|
| Thm 1 | Lipschitz constant of a ViT margin is hard to estimate tightly; loose $\hat L$ gives a loose bound. | The bound is one-sided: a loose $\hat L$ still gives a *valid* lower bound, just an unhelpfully large one. We need the bound to be useful for at least the methods where the audit is most informative — empirical scatter will say. |
| Thm 2 | Pinsker is loose; tighter KL bounds exist (Bretagnolle–Huber, Le Cam). | Acknowledge; either swap Pinsker for Bretagnolle–Huber (also one inequality) or keep Pinsker for transparency and note the slack. |
| Thm 3 | The "covering at scale ε=1" assumption is the unphysical part — real audits use ε ≪ 1. | The theorem still says: *as the audit budget grows toward unit scale, transfer cost grows toward constancy*. Sharpen by giving a *budget-graded* impossibility: a Lipschitz-controlled bound on how close a non-constant classifier can come to invariance over near-covering families. (One-paragraph extension; quick to add.) |
| Thm 4 | Smoothing certifies $L_2$, not $L_\infty$; audits in the paper use $L_\infty$. | Cite the $L_\infty \leftrightarrow L_2$ conversion penalty $\sqrt{d}$, or run the smoothed audit in $L_2$. The latter is one config flag in `17_recovery_radius_audit.py`. |

## Next concrete actions

1. Add `theory_appendix.tex` to the main paper via `\input{theory_appendix}`
   in `paper_draft.tex` immediately after the threat-models section.
2. Implement the four utilities in `src/theory/recovery_certification.py`
   (replace `resurrection_bounds.py`).
3. Run the four new experiments above on the existing checkpoints; produce
   the four new figures.
4. Rewrite the abstract and intro around the four theorems.
5. Either retitle or keep the current title with a subtitle that signals
   the theory.

The paper still has every empirical artifact it has now. What changes is
that each artifact now points at a theorem instead of standing alone — which
is the difference between a benchmark and a result.
