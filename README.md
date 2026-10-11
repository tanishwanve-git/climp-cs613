# CLIMP reproduction (CS613)

Reproduction and small extensions of **CLIMP: Contrastive Language-Image Mamba Pretraining**
([arXiv 2601.06891](https://arxiv.org/abs/2601.06891), [upstream code](https://github.com/NimrodShabtay/CLIMP)).
We evaluate the released checkpoints (no retraining at scale) against an OpenCLIP ViT-B/16 (LAION-2B) baseline.

| ID | Script | What | Paper |
|---|---|---|---|
| E1 | `src/exp_nocaps.py` | NoCaps retrieval at 224/512/896 (+ README correctness gate) | Table 4, README |
| E2 | `src/exp_geometry.py` | Alignment / uniformity / hubness + CSLS hubness correction (CPU) | Table 8 + new |
| E3 | `src/exp_zeroshot.py` | Zero-shot ImageNet-V2/R/A/O (+ CIFAR-10 sanity) | Table 2 |
| E4 | `src/exp_zeroshot.py --shuffle-grid G` | Patch-shuffle robustness on ImageNet-V2 | new |
| E5 | `src/exp_docci.py` | DOCCI long captions, 77 / 154 / full tokens | Table 6 + new |
| E6 | `src/exp_efficiency.py` | Vision-encoder memory + FLOPs vs resolution | Fig. 3 |
| E7 | `src/exp_sugarcrepe.py` | SugarCrepe compositionality | new |
| E8 | `src/exp_cifar_bias.py` | Tiny VMamba vs ViT on CIFAR-10, regular vs shuffled (training) | Table 7 |

Shared code: `src/common.py` (models, encoding, helpers), `src/metrics.py` (metrics). Tables/figures:
`src/make_results.py` reads only `results/*.json` and `paper_numbers.json`. Deviations: `ASSUMPTIONS.md`.

## Setup

```bash
git clone https://github.com/tanishwanve-git/climp-cs613.git && cd climp-cs613
git clone https://github.com/NimrodShabtay/CLIMP.git CLIMP && git -C CLIMP checkout 3ff1279   # never edited
pip install -r CLIMP/requirements.txt -r requirements.txt   # mamba-ssm / causal-conv1d need an NVIDIA GPU
```

Data and HF caches go to `$DATA_DIR` (default `./data`). Every script takes `--smoke` (tiny subset, writes to
`results_smoke/`) and `--out-dir`; each run writes one JSON with its config, versions, git hashes and runtime.

## Local checks (any machine, CPU)

```bash
python -m pytest tests -q
python src/exp_cifar_bias.py --smoke --models vit --n-train 512   # E8 ViT path on CPU
python src/make_results.py --smoke
```

The CLIMP paths (VMamba, Mamba) need CUDA, so their smoke tests run on Kaggle.

## Kaggle

Notebook: `kaggle/run_climp.ipynb`. Settings: **GPU T4 x2**, **Internet on**, secret **`HF_TOKEN`**.

1. `MODE = "smoke"` → Save & Run All (~30 min). Check every script passes, then pick `E8_EPOCHS` from the E8 timing.
2. `MODE = "full", STAGE = "A"` → E1 (stops if the README gate fails) → E2 → E6 → E3 → E4 (~3 h).
3. `MODE = "full", STAGE = "B"` → E5 → E7 → E8 (~2–2.5 h; E8 runs one model per GPU).
4. Download `results_A.zip`, `results_B.zip` from the Output tab, unzip both into this folder, then run
   `python src/make_results.py` to build `results/tables/` and `results/figures/`.
