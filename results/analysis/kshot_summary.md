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

## Label controls (prompt length 5, k=10)

| Control | Oracle | KL | Missing (Oracle/KL) |
|---|---|---|---|
| shufflelabels | 0.00% +/- 0.00% | 3.90% +/- 6.41% | []/[] |
| randomlabels | 0.00% +/- 0.00% | 4.03% +/- 6.64% | []/[] |
