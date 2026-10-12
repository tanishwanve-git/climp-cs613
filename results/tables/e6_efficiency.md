### E6 Vision-encoder memory and FLOPs vs resolution (paper Fig. 3)

| Model | Quantity | 224 | 384 | 512 | 768 | 896 |
|---|---|---|---|---|---|---|
| vmamba-b | activation_mb | 36.4 | 105.4 | 186.1 | 417.7 | 573.5 |
| vmamba-b | peak_mb | 383.7 | 454.1 | 535.8 | 772.1 | 930.2 |
| vmamba-b | gflops | 15.4 | 45.1 | 80.2 | 180.5 | 245.7 |
| vmamba-b | paper @896 (MB / GFLOPs) |  |  |  |  | 10.0 / 259.7 |
| vit-b16 | activation_mb | 6.3 | 15.9 | 24.1 | 55.0 | 73.7 |
| vit-b16 | peak_mb | 521.5 | 531.4 | 540.3 | 577.7 | 588.3 |
| vit-b16 | gflops | 17.6 | 55.5 | 107.1 | 295.3 | 450.0 |
| vit-b16 | paper @896 (MB / GFLOPs) |  |  |  |  | 50.4 / 457.9 |
| vit-b16-eager | activation_mb | 6.3 | 33.7 | 89.0 | 402.7 | 726.3 |
| vit-b16-eager | peak_mb | 521.2 | 540.4 | 602.5 | 922.1 | 1239.2 |
| vit-b16-eager | gflops | 17.6 | 55.5 | 107.1 | 295.3 | 450.0 |

Batch 1, fp16. activation = peak − (weights + input). FLOPs from fvcore (1 multiply-add = 1 FLOP). vit-b16 uses fused (memory-efficient) attention; vit-b16-eager materialises the attention matrix.
