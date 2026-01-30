# K-shot Summary

Seeds: 42, 123, 456

## Default prompt length (10 tokens)

| K-shot | Oracle | KL | Missing (Oracle/KL) |
|---|---|---|---|
| 10 | 0.00% +/- 0.00% | 0.47% +/- 0.64% | []/[] |
| 25 | 0.00% +/- 0.00% | 0.43% +/- 0.40% | []/[] |
| 50 | 0.00% +/- 0.00% | 0.43% +/- 0.25% | []/[] |
| 100 | 0.00% +/- 0.00% | 1.90% +/- 0.75% | []/[] |

## Low-shot controls (prompt length 5)

| K-shot | Oracle | KL | Missing (Oracle/KL) |
|---|---|---|---|
| 1 | 0.00% +/- 0.00% | 34.17% +/- 47.64% | []/[] |
| 5 | 0.00% +/- 0.00% | 35.47% +/- 48.92% | []/[] |

### Low-shot controls (prompt length 5) per-seed

| Seed | Oracle k=1 | KL k=1 | Oracle k=5 | KL k=5 |
|---|---|---|---|---|
| 42 | 0.00% | 13.80% | 0.00% | 15.00% |
| 123 | 0.00% | 0.10% | 0.00% | 0.10% |
| 456 | 0.00% | 88.60% | 0.00% | 91.30% |

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
