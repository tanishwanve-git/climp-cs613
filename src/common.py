"""Shared helpers: model loading/encoding, grid shuffle, seeding, JSON output, run metadata.

Import this module before `datasets`/`transformers` so HF_HOME points into DATA_DIR.
"""
import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "CLIMP"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
sys.path.insert(0, str(UPSTREAM))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# preprocess: PIL image -> tensor. encode_images: [B,3,H,W] tensor -> [B,D].
# encode_texts: list[str] -> [N,D]. All outputs are L2-normalised float32 CPU tensors.
Model = namedtuple("Model", "name image_size preprocess encode_images encode_texts tokenizer module")
MODELS = ["climp-mamba1", "climp-mamba2", "openclip-vitb16"]


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true", help="tiny subset, writes to results_smoke/")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def out_dir(args):
    return Path(args.out_dir) if args.out_dir else ROOT / ("results_smoke" if args.smoke else "results")


def load_model(name, image_size=224, device="cuda"):
    if name.startswith("climp-"):
        return _load_climp(name.split("-")[1], image_size, device)
    if name == "openclip-vitb16":
        return _load_openclip(image_size, device)
    raise ValueError(f"unknown model {name}")


def autocast(device):
    # T4 has no bf16; upstream eval also runs under fp16 autocast.
    return torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda"))


def _load_climp(variant, image_size, device):
    from models import load_climp
    from data.utils import transform_image

    model = load_climp(variant, image_size=image_size, device=device)
    tok = model.text_tokenizer

    @torch.no_grad()
    def encode_images(pixels):
        with autocast(device):
            emb = model.encode_image(pixels.to(device))
        return F.normalize(emb.float(), dim=-1).cpu()

    @torch.no_grad()
    def encode_texts(texts, max_tokens=None, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding="longest", truncation=max_tokens is not None,
                      max_length=max_tokens, return_tensors="pt").to(device)
            with autocast(device):
                emb = model.encode_text(enc["input_ids"], enc["attention_mask"])
            out.append(F.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(out)

    return Model(f"climp-{variant}", image_size, transform_image(image_size), encode_images, encode_texts, tok, model)


def _load_openclip(image_size, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", force_image_size=image_size, device=device)
    model.eval()
    print(f"[openclip] image_size={image_size} positional_embedding={tuple(model.visual.positional_embedding.shape)}"
          f" (expect ({(image_size // 16) ** 2 + 1}, 768))")
    tok = open_clip.get_tokenizer("ViT-B-16")  # fixed 77-token context

    @torch.no_grad()
    def encode_images(pixels):
        with autocast(device):
            emb = model.encode_image(pixels.to(device))
        return F.normalize(emb.float(), dim=-1).cpu()

    @torch.no_grad()
    def encode_texts(texts, max_tokens=None, batch_size=256):
        out = []
        for i in range(0, len(texts), batch_size):
            with autocast(device):
                emb = model.encode_text(tok(texts[i:i + batch_size]).to(device))
            out.append(F.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(out)

    return Model("openclip-vitb16", image_size, preprocess, encode_images, encode_texts, tok, model)


class _Images(Dataset):
    def __init__(self, get_image, n, preprocess):
        self.get_image, self.n, self.preprocess = get_image, n, preprocess

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.preprocess(self.get_image(i).convert("RGB"))


def encode_image_set(model, get_image, n, batch_size=None, workers=4):
    """Encode images get_image(0..n-1) (PIL) with the model's own preprocessing.
    Default batch size shrinks with resolution (32 @224 -> 4 @896) so ViT attention fits in 16 GB."""
    batch_size = batch_size or max(4, round(32 * (224 / model.image_size) ** 2))
    dl = DataLoader(_Images(get_image, n, model.preprocess), batch_size=batch_size, num_workers=workers)
    return torch.cat([model.encode_images(x) for x in tqdm(dl, desc=f"{model.name}@{model.image_size}")])


def free_model():
    """Call after `del model` so the next (6-11 GB) model fits."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def grid_shuffle(x, grid, perm):
    """Split the last two dims of x ([..., C, H, W]) into a grid x grid array of equal blocks and
    reorder the blocks by `perm` (a permutation of range(grid*grid), row-major)."""
    *lead, c, h, w = x.shape
    bh, bw = h // grid, w // grid
    blocks = x.reshape(*lead, c, grid, bh, grid, bw).movedim(-3, -2)  # [..., C, g, g, bh, bw]
    blocks = blocks.reshape(*lead, c, grid * grid, bh, bw)[..., perm, :, :]
    return blocks.reshape(*lead, c, grid, grid, bh, bw).movedim(-2, -3).reshape(*lead, c, h, w)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def run_meta(args, t0, **extra):
    """Config block stored in every result JSON."""
    versions = {}
    for pkg in ["torch", "transformers", "open_clip", "timm", "mamba_ssm"]:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"args": vars(args), **extra, "versions": versions, "gpu": gpu,
            "git": _git_hash(ROOT), "upstream_git": _git_hash(UPSTREAM),
            "seconds": round(time.time() - t0, 1)}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"saved {path}")
