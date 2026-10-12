### E2 Hubness correction with CSLS (k = 10, transductive)

| Model | Task | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| climp-mamba1 | image retrieval | 38.2 → 42.7 | 70.5 → 74.5 | 81.8 → 84.7 |
| climp-mamba1 | text retrieval | 51.8 → 61.2 | 81.9 → 86.9 | 90.9 → 93.3 |
| openclip-vitb16 | image retrieval | 58.0 → 62.1 | 84.6 → 87.3 | 91.2 → 92.9 |
| openclip-vitb16 | text retrieval | 80.0 → 84.5 | 95.7 → 97.0 | 98.2 → 99.0 |

Values are before → after CSLS re-scoring.
