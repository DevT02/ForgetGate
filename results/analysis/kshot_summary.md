# K-shot Summary

Seeds: 42, 123, 456

## Default prompt length (10 tokens)

| K-shot | Oracle | KL | Missing (Oracle/KL) |
|---|---|---|---|
| 10 | 0.00% +/- 0.00% | 0.47% +/- 0.64% | []/[] |
| 25 | 0.00% +/- 0.00% | 0.43% +/- 0.40% | []/[] |
| 50 | 0.00% +/- 0.00% | 0.43% +/- 0.25% | []/[] |
| 100 | 0.00% +/- 0.00% | 1.90% +/- 0.75% | []/[] |

## SCRUB follow-up (default prompt length)

| K-shot | Oracle | SCRUB | Missing (Oracle/SCRUB) |
|---|---|---|---|
| 10 | 0.00% +/- 0.00% | 0.00% +/- 0.00% | []/[] |
| 25 | 0.00% +/- 0.00% | 0.00% +/- 0.00% | []/[] |
| 50 | 0.00% +/- 0.00% | 26.93% +/- 46.65% | []/[] |
| 100 | 0.00% +/- 0.00% | 28.20% +/- 48.84% | []/[] |

Coverage note: SCRUB follow-up uses tracked logs for seeds 42/123/456 at k=10/25/50/100.
For k=25, seed 42 comes from the canonical `vpt_resurrect_scrub_forget0_25shot` log and
seeds 123/456 come from `vpt_resurrect_scrub_forget0_10shot_kshot25` override logs.

### SCRUB follow-up per-seed

| Seed | SCRUB k=10 | SCRUB k=25 | SCRUB k=50 | SCRUB k=100 |
|---|---|---|---|---|
| 42 | 0.00% | 0.00% | 80.80% | 84.60% |
| 123 | 0.00% | 0.00% | 0.00% | 0.00% |
| 456 | 0.00% | 0.00% | 0.00% | 0.00% |

## Low-shot controls (prompt length 5)

| K-shot | Oracle | KL | Missing (Oracle/KL) |
|---|---|---|---|
| 1 | 0.00% +/- 0.00% | 4.27% +/- 7.04% | []/[] |
| 5 | 0.00% +/- 0.00% | 4.53% +/- 7.68% | []/[] |

Source note: prompt-length-5 low-shot controls prefer `*_10shot_prompt5_kshot{k}` logs and
fall back to legacy `*_{k}shot_prompt5` logs when the newer filenames are incomplete.

### Low-shot controls (prompt length 5) per-seed

| Seed | Oracle k=1 | KL k=1 | Oracle k=5 | KL k=5 |
|---|---|---|---|---|
| 42 | 0.00% | 12.40% | 0.00% | 13.40% |
| 123 | 0.00% | 0.10% | 0.00% | 0.10% |
| 456 | 0.00% | 0.30% | 0.00% | 0.10% |

## Label controls (prompt length 5, k=10)

| Control | Oracle | KL | Missing (Oracle/KL) |
|---|---|---|---|
| shufflelabels | 0.00% +/- 0.00% | 3.90% +/- 6.41% | []/[] |
| randomlabels | 0.00% +/- 0.00% | 4.03% +/- 6.64% | []/[] |

### Label controls per-seed (prompt length 5, k=10)

| Seed | Oracle shuffle | KL shuffle | Oracle random | KL random |
|---|---|---|---|---|
| 42 | 0.00% | 11.30% | 0.00% | 11.70% |
| 123 | 0.00% | 0.40% | 0.00% | 0.30% |
| 456 | 0.00% | 0.00% | 0.00% | 0.10% |

## Stratified forget-set (prompt length 10, k=10)

| Bucket | KL | Missing (KL) |
|---|---|---|
| high_conf | 12.70% +/- 0.00% | [123, 456] |
| mid_conf | 14.00% +/- 0.00% | [123, 456] |
| low_conf | 13.60% +/- 0.00% | [123, 456] |

### Stratified forget-set per-seed (prompt length 10, k=10)

| Seed | KL high | KL mid | KL low |
|---|---|---|---|
| 42 | 12.70% | 14.00% | 13.60% |
| 123 | N/A | N/A | N/A |
| 456 | N/A | N/A | N/A |
