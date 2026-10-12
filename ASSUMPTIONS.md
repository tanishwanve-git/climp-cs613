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
- DOCCI: official files (docci_descriptions.jsonlines + docci_images.tar.gz, test split = 5000 images); google/docci on HF uses a loading script that current `datasets` cannot run.
- NoCaps is loaded from HF's auto-converted parquet of HuggingFaceM4/NoCaps (refs/convert/parquet): same rows/order as upstream's load_dataset call, whose loading script no longer runs in datasets>=3.
- E1 smoke runs at 224 and 896 (not only 224) so GPU-memory problems at 896 appear before full runs. Image batch size defaults to max(4, 32*(224/res)^2).
- E6 reports ViT-B/16 twice: fused SDPA attention (timm default) and eager attention (explicit LxL matrix); the paper does not say which its baselines used. ViT FLOPs are always counted on the eager graph because fvcore cannot see inside the fused kernel.
- Kaggle runs use torch 2.10 (CUDA 12.6) because it is the newest torch with official prebuilt mamba-ssm 2.3.2.post1 / causal-conv1d 1.6.2.post1 wheels; Kaggle's default torch would require a 40-60 min source build. Same kernels, no effect on results.
- CLIMP is loaded with the same steps as upstream load_climp (build MambaCLIP, strict load_state_dict of model.safetensors, eval) but the model is moved to the GPU before the checkpoint is read, to halve peak CPU RAM. One CLIMP model is reused for all input resolutions (VMamba's imgsize only affects positional embeddings, which CLIMP disables; E1 asserts identical parameter names/shapes at 224 and 896).
- E8 augmentation runs on the GPU batch-wise: 4-px black padding + random 32x32 crop, horizontal flip (p = 0.5), then normalisation, then (shuffled condition) a fresh per-image patch permutation. Same operations as the torchvision pipeline; only the random stream differs. 20 epochs (the paper does not state its epoch count, so we compare the pattern, not exact values).
- E3 full runs use ImageNet-V2/R/A/O (paper Table 2); CIFAR-10 is only a sanity check in the smoke run.
- E8 loads CIFAR-10 from the HuggingFace parquet mirror (uoft-cs/cifar10), verified byte-identical (pixels, labels, order) to torchvision's copy; the original Toronto server ran at ~200 B/s.
- CLIMP text batches are capped at ~4096 tokens (batch x longest caption), max 64 texts: upstream encode_text keeps all layers' hidden states (output_hidden_states=True), which ran out of T4 memory for long DOCCI captions at batch 64. Batching does not change the embeddings.
- The Mamba-2 tokenizer pads on the left (Mamba-1 on the right); the E1 smoke check confirms padding="longest" gives the same embeddings as upstream's padding for both (min cosine 1.00000).
