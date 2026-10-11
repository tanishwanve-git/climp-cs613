# Assumptions and deviations from the paper

One line each. Added as decisions are made.

- Hubness: k = 10, reported as the skewness of the k-occurrence distribution N_k (paper does not state k). If N_k is constant the skewness is defined as 0.
- CSLS hubness correction is transductive: the neighbourhood terms r_q / r_g are computed from the test queries and test gallery themselves.
- Geometry (E2) uses the first caption of each NoCaps image to form N matched image-text pairs; uniformity is computed on at most 5000 points subsampled with seed 0, t = 2.
- Retrieval naming follows the upstream code: "image retrieval" = text query -> images, "text retrieval" = image query -> texts. Paper Table 4 labels its columns I->T / T->I, but its values (70.5 / 81.9) equal the README's Image R@5 / Text R@5, so we compare our image retrieval to the column labelled I->T.
- Paper text (Sec. 4.5) says CLIMP text hubness is 1.13, but Table 8 gives 1.15 for Mamba-1 and 1.13 for Mamba-2; we compare against Table 8 (Mamba-1 = 1.15).
- Precision: CLIMP is evaluated under fp16 autocast. Upstream eval_nocaps.py uses torch.amp.autocast(device_type="cuda"), whose default dtype is fp16, so this matches the README setting.
- CLIMP text tokenisation uses padding="longest" instead of upstream's padding to the max caption length in characters; last-non-pad-token pooling makes the two equivalent (verified in the E1 smoke test).
- Baseline is OpenCLIP ViT-B/16 (laion2b_s34b_b88k), not the paper's RoPE-ViT/FlexViT/NaFlex + LLaMA baselines, which are not released. Its text tower always truncates at 77 tokens.
- E8 shuffled condition: a new random permutation of the 8x8 grid of 4x4 patches per training image (re-drawn every time the image is loaded); test images use a fixed per-image permutation seeded by (seed, image index). A single fixed permutation would be trivially undone by ViT's learned positional embeddings.
- E6 FLOPs come from fvcore, which counts one multiply-add as one FLOP. VMamba FLOPs use upstream VSSM.flops() (fvcore + selective_scan_flop_jit).
- Upstream CLIMP is pinned to commit 3ff1279bb87f905596e6187f0e4a14f2cd28e818 (main, cloned 2026-10-12).
