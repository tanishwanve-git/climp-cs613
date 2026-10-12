### E7 SugarCrepe accuracy (%)

| Model |  | add_att | add_obj | replace_att | replace_obj | replace_rel | swap_att | swap_obj | macro |
|---|---|---|---|---|---|---|---|---|---|
| climp-mamba1 |  | 75.4 | 83.2 | 75.6 | 89.4 | 67.2 | 55.3 | 55.5 | 71.7 |
| openclip-vitb16 |  | 81.8 | 89.2 | 85.0 | 96.1 | 71.1 | 67.7 | 64.9 | 79.4 |

Correct = sim(image, positive) > sim(image, negative). Chance = 50.
