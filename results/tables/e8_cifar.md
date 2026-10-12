### E8 Spatial inductive bias on CIFAR-10 (paper Table 7)

| Model | Data | Train loss ours | paper | Test acc ours | paper | #seeds |
|---|---|---|---|---|---|---|
| vmamba | regular | 0.458 ± 0.003 | 1.028 | 81.76 ± 0.27 | 69.33 | 3 |
| vit | regular | 0.727 ± 0.009 | 1.121 | 72.89 ± 0.42 | 67.50 | 3 |
| vmamba | shuffled | 1.149 ± 0.012 | 1.612 | 56.96 ± 0.43 | 46.44 | 3 |
| vit | shuffled | 0.903 ± 0.017 | 1.328 | 65.10 ± 0.49 | 59.72 | 3 |

Mean ± std over seeds. Shuffled = per-image permutation of the 8×8 grid of 4×4 patches.
