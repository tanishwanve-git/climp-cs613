### Retrained CLIMP (ours, COCO 118k, frozen text tower) vs released CLIMP vs paper

| Metric | Ours retrained | Released (our eval) | Paper |
|---|---|---|---|
| NoCaps image R@5 @224 | 29.7 | 70.5 | 70.5 |
| NoCaps text R@5 @224 | 44.6 | 81.9 | 81.9 |
| NoCaps image R@5 @512 | 25.2 | 66.9 | 66.9 |
| NoCaps text R@5 @512 | 34.0 | 72.7 | 72.7 |
| NoCaps image R@5 @896 | 19.6 | 57.1 | 57.1 |
| NoCaps text R@5 @896 | 21.4 | 53.5 | 53.5 |
| NoCaps image R@5 change 224→896 (%) | -34 | -19 | -19 |
| NoCaps text R@5 change 224→896 (%) | -52 | -35 | -35 |
| Zero-shot imagenetv2 top-1 | 13.9 | 37.6 | 37.5 |
| Zero-shot imagenet-r top-1 | 13.2 | 46.6 | 46.2 |
| Zero-shot imagenet-a top-1 | 7.6 | 15.6 | 15.5 |
| Zero-shot imagenet-o top-1 | 22.5 | 49.8 | 48.1 |
| DOCCI full image R@1 @896 | 2.8 | 37.3 | 37.3 |
| DOCCI full text R@1 @896 | 2.5 | 23.8 | 23.8 |
| Alignment ↓ | 1.206 | 0.982 | 0.982 |
| Uniformity image ↓ | -3.437 | -3.184 | -3.180 |
| Uniformity text ↓ | -2.669 | -3.546 | -3.540 |
| Hubness image ↓ | 19.663 | 1.554 | 1.010 |
| Hubness text ↓ | 2.237 | 1.787 | 1.150 |

Ours retrained: VMamba-B from ImageNet-1K + frozen Mamba-1.4B, COCO train2017, 4 epochs, global batch 96 (paper: CC12M, 10 epochs, batch 2048, text tower trained).
