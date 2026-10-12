"""Retrain CLIMP without the released weights (scaled-down reproduction of the paper's training).

Starts from the paper's initialisations, VMamba-B ImageNet-1K and pretrained Mamba-1.4B, and trains contrastively on
COCO Captions (train2017) with the paper's loss / optimiser / schedule. The Mamba text tower is frozen, so its
last-token features are computed once and cached; training updates VMamba-B, both projections and the temperature.
Data caches and checkpoints are stored in a private HF repo so separate Kaggle runs can hand them over.

  python src/train_climp.py check                                       # weight checks + batch-size/speed probe
  python src/train_climp.py prep-images [--smoke]                       # COCO -> 256x256 JPEGs (CPU)
  python src/train_climp.py prep-text --part 0 --parts 2 [--smoke]      # frozen Mamba-1.4B caption features (1 GPU)
  python src/train_climp.py prep-upload [--smoke]
  torchrun --nproc_per_node 2 src/train_climp.py train --epochs 3 --batch-size 64 [--resume] [--smoke]
"""
import argparse
import datetime
import fcntl
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

from common import DATA_DIR, ROOT, autocast, run_meta, save_json, seed_all, token_batches

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from metrics import recall_at_k

TEXT_MODEL = "state-spaces/mamba-1.4b-hf"  # upstream MODEL_CONFIGS["mamba1"]
VMAMBA_IN1K = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth"
COCO = "http://images.cocodataset.org/{}"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # upstream transform_image
STORE_SIZE = 256        # training images are stored squashed to 256x256 (eval squashes to 224x224, upstream)
CAPS_PER_IMAGE = 2      # cached captions per training image (one is drawn per image per epoch)
N_VAL = 1000            # COCO val2017 images used to monitor training (5 captions each)
SMOKE_TRAIN = 2000      # smoke: train on val2017 images N_VAL..N_VAL+1999 (no 18 GB download)


def tag(args):
    return "smoke" if args.smoke else "full"


def work_dir(args):
    d = DATA_DIR / "climp_train" / tag(args)
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(args):
    return ROOT / ("results_trained_smoke" if args.smoke else "results_trained") / "train"


def download(url, path):
    """Download once; a file lock makes parallel prep jobs wait for each other instead of writing the same file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            print(f"downloading {url} ...", flush=True)
            part = str(path) + ".part"
            cmd = ["wget", "-q", "-c", "-O", part, url] if shutil.which("wget") else ["curl", "-sSL", "-C", "-", "-o", part, url]
            subprocess.run(cmd, check=True)
            os.replace(part, path)
    return path


# ---------------------------------------------------------------- HF artifact repo (hand-over between Kaggle runs)
def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN"))


def artifact_repo():
    api = hf_api()
    repo = os.environ.get("HF_TRAIN_REPO") or f"{api.whoami()['name']}/climp-cs613-train"
    api.create_repo(repo, private=True, exist_ok=True)
    return repo


def upload(local, remote):
    api = hf_api()
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=artifact_repo())
    print(f"uploaded {local} -> {remote}", flush=True)


def fetch(remote, local):
    """Download remote file from the artifact repo to `local` (skipped if present). Returns local path or None."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    local = Path(local)
    if local.exists():
        return local
    try:  # local_dir gives a real file (the HF cache would hold a symlink)
        path = hf_hub_download(artifact_repo(), remote, token=hf_api().token, local_dir=DATA_DIR / "hf_artifacts")
    except EntryNotFoundError:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, local)
    return local


# ---------------------------------------------------------------- model
def build_vision(grad_ckpt=False):
    """CLIMP's VMamba-B (upstream build_vmamba) initialised from the official ImageNet-1K checkpoint."""
    from models import build_vmamba
    model = build_vmamba(224)
    state = torch.load(download(VMAMBA_IN1K, DATA_DIR / "vmamba" / "vssm_base_0229_ckpt_epoch_237.pth"),
                       map_location="cpu", weights_only=False)["model"]
    r = model.load_state_dict(state, strict=False)
    assert not r.missing_keys and all(k.startswith("classifier.head.") for k in r.unexpected_keys), r
    for m in model.modules():  # VSS blocks read this flag in forward
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = grad_ckpt
    return model


class TrainableCLIMP(nn.Module):
    """The trainable part of upstream MambaCLIP, with identical parameter names, so the saved state_dict loads into
    MambaCLIP(model_type="mamba1") with only the (frozen, pretrained) text_encoder.* keys missing."""

    def __init__(self, vision, text_dim=2048, proj_dim=768, temperature=0.07):  # upstream MODEL_CONFIGS["mamba1"]
        super().__init__()
        self.vision_encoder = vision
        self.vision_projection = nn.Linear(vision.config.dims[-1], proj_dim)
        self.text_projection = nn.Linear(text_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, pixels, text_feats):
        img = self.vision_projection(self.vision_encoder(pixels))
        txt = self.text_projection(text_feats.to(self.text_projection.weight.dtype))
        return F.normalize(img, dim=-1), F.normalize(txt, dim=-1), self.logit_scale.exp()


def clip_loss(img, txt, scale):
    """Symmetric InfoNCE over the global batch. Each rank scores its own rows against all ranks' features
    (gathered with gradient); with equal per-rank batches the mean of the rank losses is the global loss and DDP's
    gradient averaging yields its exact gradient (open_clip's local_loss + gather_with_grad)."""
    labels = torch.arange(len(img), device=img.device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        import torch.distributed.nn as dist_nn
        all_img, all_txt = torch.cat(dist_nn.all_gather(img)), torch.cat(dist_nn.all_gather(txt))
        labels = labels + dist.get_rank() * len(img)
    else:
        all_img, all_txt = img, txt
    return (F.cross_entropy(scale * img @ all_txt.T, labels) + F.cross_entropy(scale * txt @ all_img.T, labels)) / 2


@torch.no_grad()
def text_features(backbone, tok, texts, device, max_batch=256):
    """Frozen Mamba last-token features, computed exactly as upstream MambaCLIP.encode_text does before its
    projection (last layer output at the last non-padding token, fp16 autocast)."""
    from models import find_last_one_and_extract_values
    lengths = [len(ids) for ids in tok(texts)["input_ids"]]
    out = []
    for i, j in token_batches(lengths, max_batch, token_budget=16384):
        enc = tok(texts[i:j], padding="longest", return_tensors="pt").to(device)
        with autocast(device):
            hidden = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                              output_hidden_states=False)[0]
        out.append(find_last_one_and_extract_values(enc["attention_mask"], hidden)[1].float().cpu())
    return torch.cat(out)


def load_text_backbone(device):
    from transformers import AutoTokenizer, MambaForCausalLM
    return MambaForCausalLM.from_pretrained(TEXT_MODEL).backbone.to(device).eval(), AutoTokenizer.from_pretrained(TEXT_MODEL)


# ---------------------------------------------------------------- step 0: checks
def cmd_check(args):
    """(1) ImageNet VMamba-B loads into CLIMP's vision tower; (2) did the authors change the text tower? (released
    CLIMP text_encoder vs pretrained Mamba-1.4B); (3) cached features + projection == upstream encode_text;
    (4) training batch-size / speed probe on one GPU."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from models import HF_REPOS, MambaCLIP
    t0, report = time.time(), {}

    vision = build_vision()
    report["vmamba_in1k_load"] = "ok: all CLIMP VMamba-B weights loaded; unused: classifier.head (ImageNet classes)"
    print("[check 1]", report["vmamba_in1k_load"], flush=True)

    climp = safe_open(hf_hub_download(HF_REPOS["mamba1"], "model.safetensors"), "pt")
    climp_keys = set(climp.keys())
    pre_dir = Path(snapshot_download(TEXT_MODEL, allow_patterns=["*.safetensors", "*.json"]))
    pre = {}
    for f in pre_dir.glob("*.safetensors"):
        h = safe_open(str(f), "pt")
        pre.update({k: (h, k) for k in h.keys()})
    rel, unmatched = [], []
    for k in sorted(k for k in climp_keys if k.startswith("text_encoder.")):
        src = "backbone." + k[len("text_encoder."):]
        if src not in pre:
            unmatched.append(k)
            continue
        a, b = climp.get_tensor(k).float(), pre[src][0].get_tensor(pre[src][1]).float()
        rel.append(((a - b).norm() / b.norm().clamp_min(1e-12)).item())
    in1k = vision.state_dict()
    vrel = [((climp.get_tensor("vision_encoder." + k).float() - v.float()).norm() / v.float().norm().clamp_min(1e-12)).item()
            for k, v in in1k.items() if v.is_floating_point() and "vision_encoder." + k in climp_keys]
    report["released_text_tower_vs_pretrained"] = {"n_tensors": len(rel), "n_unmatched": len(unmatched),
                                                   "n_identical": sum(r == 0 for r in rel),
                                                   "max_rel_diff": max(rel), "mean_rel_diff": float(np.mean(rel))}
    report["released_vision_tower_vs_in1k"] = {"n_tensors": len(vrel), "mean_rel_diff": float(np.mean(vrel))}
    print("[check 2] released CLIMP text tower vs pretrained Mamba-1.4B:", report["released_text_tower_vs_pretrained"])
    print("          released CLIMP vision tower vs ImageNet VMamba-B:", report["released_vision_tower_vs_in1k"], flush=True)

    backbone, tok = load_text_backbone(args.device)
    texts = ["A dog.", "Two people riding bicycles down a busy city street next to a red bus.",
             "a close up of a plate of food with broccoli", "A kitchen."]
    proj = nn.Linear(2048, 768).to(args.device)
    ours = F.normalize(proj(text_features(backbone, tok, texts, args.device).to(args.device)), dim=-1)
    enc = tok(texts, padding="longest", return_tensors="pt").to(args.device)
    shim = type("Shim", (), {"text_encoder": backbone, "text_projection": proj})()
    with torch.no_grad(), autocast(args.device):
        ref = F.normalize(MambaCLIP.encode_text(shim, enc["input_ids"], enc["attention_mask"]).float(), dim=-1)
    report["cached_text_path_min_cos"] = (ours * ref).sum(-1).min().item()
    print(f"[check 3] cached features + projection vs upstream encode_text: min cos {report['cached_text_path_min_cos']:.5f}")
    assert report["cached_text_path_min_cos"] > 0.999
    del backbone, climp
    torch.cuda.empty_cache()

    report["probe"] = []
    for grad_ckpt in [False, True]:
        model = TrainableCLIMP(build_vision(grad_ckpt)).to(args.device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
        scaler = torch.amp.GradScaler()
        for bs in [32, 48, 64, 96, 128]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                x, t = torch.randn(bs, 3, 224, 224, device=args.device), torch.randn(bs, 2048, device=args.device)
                times = []
                for _ in range(4):
                    torch.cuda.synchronize()
                    s = time.time()
                    with autocast(args.device):
                        loss = clip_loss(*model(x, t))
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    torch.cuda.synchronize()
                    times.append(time.time() - s)
                r = {"grad_ckpt": grad_ckpt, "batch": bs, "img_per_s": round(bs / np.mean(times[1:]), 1),
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
            except torch.OutOfMemoryError:
                r = {"grad_ckpt": grad_ckpt, "batch": bs, "oom": True}
                opt.zero_grad(set_to_none=True)
            report["probe"].append(r)
            print("[check 4] probe", r, flush=True)
            if r.get("oom"):
                break
        del model, opt
    save_json({"config": run_meta(args, t0, experiment="train_check"), "metrics": report},
              results_dir(args) / "train_check.json")


# ---------------------------------------------------------------- step 1: data
def coco_lists(args):
    """Training and monitoring image lists: [(zip_name, image_file, [captions])]."""
    ann = download(COCO.format("annotations/annotations_trainval2017.zip"), DATA_DIR / "coco" / "annotations_trainval2017.zip")

    def read(split):
        with zipfile.ZipFile(ann) as z:
            d = json.loads(z.read(f"annotations/captions_{split}2017.json"))
        caps = {}
        for a in sorted(d["annotations"], key=lambda a: a["id"]):
            caps.setdefault(a["image_id"], []).append(a["caption"].strip())
        files = {im["id"]: im["file_name"] for im in d["images"]}
        return [(split, files[i], caps[i]) for i in sorted(caps)]

    val = read("val")
    train = val[N_VAL:N_VAL + SMOKE_TRAIN] if args.smoke else read("train")
    return train, val[:N_VAL]


_zips = {}


def _load(item, size):
    split, name, _ = item
    if split not in _zips:
        _zips[split] = zipfile.ZipFile(DATA_DIR / "coco" / f"{split}2017.zip")
    img = Image.open(io.BytesIO(_zips[split].read(f"{split}2017/{name}"))).convert("RGB")
    return img.resize((size, size), Image.BILINEAR)  # torchvision Resize((s, s)) on PIL = PIL bilinear


def _jpeg(item):
    buf = io.BytesIO()
    _load(item, STORE_SIZE).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def cmd_prep_images(args):
    t0 = time.time()
    train, val = coco_lists(args)
    free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
    assert args.smoke or free_gb > 25, f"only {free_gb:.0f} GB free in {DATA_DIR}; need ~25 GB (19 GB zip + output)"
    for split in {s for s, _, _ in train + val}:
        download(COCO.format(f"zips/{split}2017.zip"), DATA_DIR / "coco" / f"{split}2017.zip")
    out = work_dir(args)
    with tarfile.open(out / "images.tar", "w") as tar, Pool(args.workers) as pool:
        for i, data in enumerate(pool.imap(_jpeg, train, chunksize=64)):
            info = tarfile.TarInfo(f"{i:06d}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            if i % 10000 == 0:
                print(f"  {i}/{len(train)} images ({time.time() - t0:.0f}s)", flush=True)
    with Pool(args.workers) as pool:  # monitoring set: exactly the eval preprocessing up to ToTensor (224x224)
        val_imgs = np.stack(pool.starmap(lambda_free_load, [(it, 224) for it in val]))
    np.save(out / "val_images.npy", val_imgs)
    meta = {"train": [{"file": f, "captions": c[:CAPS_PER_IMAGE]} for _, f, c in train],
            "val": [{"file": f, "captions": c[:5]} for _, f, c in val]}
    (out / "meta.json").write_text(json.dumps(meta))
    stats = {"n_train": len(train), "n_val": len(val), "images_tar_gb": round((out / "images.tar").stat().st_size / 1e9, 2),
             "captions_per_train_image_available": {"min": min(len(c) for _, _, c in train),
                                                    "max": max(len(c) for _, _, c in train)}}
    print("prep-images:", stats)
    save_json({"config": run_meta(args, t0, experiment="train_prep_images"), "metrics": stats},
              results_dir(args) / "prep_images.json")


def lambda_free_load(item, size):  # picklable for Pool.starmap
    return np.asarray(_load(item, size))


def cmd_prep_text(args):
    t0 = time.time()
    train, val = coco_lists(args)
    texts = [c for _, _, caps in train for c in caps[:CAPS_PER_IMAGE]] + [c for _, _, caps in val for c in caps[:5]]
    assert all(len(caps) >= CAPS_PER_IMAGE for _, _, caps in train) and all(len(caps) >= 5 for _, _, caps in val)
    lo, hi = len(texts) * args.part // args.parts, len(texts) * (args.part + 1) // args.parts
    backbone, tok = load_text_backbone(args.device)
    feats = text_features(backbone, tok, texts[lo:hi], args.device).half()
    torch.save({"lo": lo, "hi": hi, "features": feats}, work_dir(args) / f"text_part{args.part}.pt")
    print(f"prep-text part {args.part}: {hi - lo} captions in {time.time() - t0:.0f}s", flush=True)


def cmd_prep_upload(args):
    out = work_dir(args)
    meta = json.loads((out / "meta.json").read_text())
    parts = sorted(out.glob("text_part*.pt"), key=lambda p: torch.load(p)["lo"])
    feats = torch.cat([torch.load(p)["features"] for p in parts])
    n_train = len(meta["train"]) * CAPS_PER_IMAGE
    assert len(feats) == n_train + len(meta["val"]) * 5, "text parts do not cover all captions"
    torch.save({"train": feats[:n_train].view(-1, CAPS_PER_IMAGE, feats.shape[-1]),
                "val": feats[n_train:].view(-1, 5, feats.shape[-1])}, out / "text_features.pt")
    print(f"text features: train {tuple(feats[:n_train].shape)}, val {tuple(feats[n_train:].shape)}")
    for f in ["meta.json", "val_images.npy", "text_features.pt", "images.tar"]:
        if not args.no_upload:
            upload(out / f, f"{tag(args)}/data/{f}")


# ---------------------------------------------------------------- step 2: training
class TrainImages(Dataset):
    def __init__(self, root, n):
        import torchvision.transforms as T
        self.root, self.n = root, n
        # crop keeps >= 90% of the (squashed) image area, close to the eval view; no flip (captions mention left/right)
        self.tf = T.Compose([T.RandomResizedCrop(224, scale=(0.9, 1.0)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.tf(Image.open(self.root / f"{i:06d}.jpg").convert("RGB")), i


def rank_indices(n, epoch, seed, world, rank, batch):
    """This rank's sample indices for one epoch (same permutation on all ranks, drop_last)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
    steps = n // (batch * world)
    return perm[: steps * batch * world][rank::world].tolist(), steps


@torch.no_grad()
def monitor(model, val_images, val_text, device):
    """COCO val 1k-image retrieval (5 captions per image) with the current weights."""
    model.eval()
    x = torch.from_numpy(val_images).permute(0, 3, 1, 2).float().div(255)
    x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
    img = []
    for i in range(0, len(x), 64):
        with autocast(device):
            img.append(F.normalize(model.vision_projection(model.vision_encoder(x[i:i + 64].to(device))).float(), dim=-1))
    img = torch.cat(img).cpu()
    with autocast(device):
        txt = F.normalize(model.text_projection(val_text.flatten(0, 1).to(device).float()).float(), dim=-1).cpu()
    gt = torch.zeros(len(txt), len(img), dtype=torch.bool)
    gt[torch.arange(len(txt)), torch.arange(len(img)).repeat_interleave(val_text.shape[1])] = True
    sim = txt @ img.T
    model.train()
    return {"image_retrieval": {str(k): v for k, v in recall_at_k(sim, gt, (1, 5, 10)).items()},
            "text_retrieval": {str(k): v for k, v in recall_at_k(sim.T, gt.T, (1, 5, 10)).items()}}


def trainable_state(model):
    return {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}


def cmd_train(args):
    t0 = time.time()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=datetime.timedelta(minutes=60))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    barrier = (lambda: dist.barrier()) if world > 1 else (lambda: None)
    seed_all(args.seed + rank)
    out, res = work_dir(args), results_dir(args)
    t = tag(args)

    if rank == 0:  # data from the prep step (local or artifact repo)
        for f in ["meta.json", "val_images.npy", "text_features.pt", "images.tar"]:
            assert fetch(f"{t}/data/{f}", out / f), f"missing {t}/data/{f}: run prep first"
        if not (out / "images").exists():
            with tarfile.open(out / "images.tar") as tar:
                tar.extractall(out / "images.tmp")
            os.replace(out / "images.tmp", out / "images")
        if args.resume and not args.no_upload:
            fetch(f"{t}/train/ckpt_last.pt", out / "ckpt_last.pt")
    barrier()
    meta = json.loads((out / "meta.json").read_text())
    text = torch.load(out / "text_features.pt")
    val_images = np.load(out / "val_images.npy")
    n = len(meta["train"])

    model = TrainableCLIMP(build_vision(args.grad_ckpt), text_dim=text["train"].shape[-1]).to(device)
    decay = [p for name, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for name, p in model.named_parameters() if p.ndim < 2]  # norms, biases, logit_scale
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    _, steps_per_epoch = rank_indices(n, 0, args.seed, world, rank, args.batch_size)
    total = args.epochs * steps_per_epoch
    warmup = min(args.warmup, total // 10)
    lr_at = lambda s: args.lr * (s + 1) / warmup if s < warmup else \
        args.lr * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))

    step, log = 0, []
    ckpt_path = out / "ckpt_last.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, log = ck["step"], ck["log"]
        assert ck["total"] == total, f"resume needs the same epochs / batch size (ckpt total {ck['total']}, now {total})"
        if rank == 0:
            print(f"resumed at step {step}/{total}", flush=True)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local] if device.startswith("cuda") else None,
                                              static_graph=args.grad_ckpt) if world > 1 else model  # VMamba uses reentrant checkpointing

    def save(final=False):
        if rank != 0:
            return
        torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "total": total, "log": log}, ckpt_path)
        res.mkdir(parents=True, exist_ok=True)
        (res / "train_log.json").write_text(json.dumps(log, indent=1))
        if not args.no_upload:
            upload(ckpt_path, f"{t}/train/ckpt_last.pt")
            upload(res / "train_log.json", f"{t}/train/train_log.json")
        if final:
            from safetensors.torch import save_file
            save_file(trainable_state(model), out / "model.safetensors")
            if not args.no_upload:
                upload(out / "model.safetensors", f"{t}/final/model.safetensors")

    def check_and_log(kind):
        entry = {"kind": kind, "step": step, "epoch": round(step / steps_per_epoch, 3),
                 "hours": round((time.time() - t0) / 3600, 3), "metrics": monitor(model, val_images, text["val"], device)}
        log.append(entry)
        m = entry["metrics"]
        print(f"[monitor] step {step}: COCO-val 1k image R@1/5 {100 * m['image_retrieval']['1']:.1f}/"
              f"{100 * m['image_retrieval']['5']:.1f}, text R@1/5 {100 * m['text_retrieval']['1']:.1f}/"
              f"{100 * m['text_retrieval']['5']:.1f}", flush=True)

    if rank == 0 and step == 0:
        check_and_log("init")  # zero-step baseline: random projections
    barrier()
    last_check, window, win_t = time.time(), [], time.time()
    stop = False
    while step < total and not stop:
        epoch, in_epoch = divmod(step, steps_per_epoch)
        idx, _ = rank_indices(n, epoch, args.seed, world, rank, args.batch_size)
        idx = idx[in_epoch * args.batch_size:]
        caption = torch.randint(0, CAPS_PER_IMAGE, (n,), generator=torch.Generator().manual_seed(args.seed + 1000 + epoch))
        loader = DataLoader(TrainImages(out / "images", n), batch_size=args.batch_size, sampler=idx,
                            num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=False)
        model.train()
        for x, i in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            feats = text["train"][i, caption[i]]
            with autocast(device):
                loss = clip_loss(*ddp(x.to(device, non_blocking=True), feats.to(device, non_blocking=True)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))  # CLIP's temperature cap
            step += 1
            window.append(loss.item())
            if step % args.log_every == 0 or step == total:
                dt = time.time() - win_t
                entry = {"kind": "train", "step": step, "epoch": round(step / steps_per_epoch, 3),
                         "loss": float(np.mean(window)), "lr": lr_at(step - 1), "logit_scale": model.logit_scale.exp().item(),
                         "img_per_s": round(len(window) * args.batch_size * world / dt, 1),
                         "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.startswith("cuda") else 0,
                         "hours": round((time.time() - t0) / 3600, 3)}
                if rank == 0:
                    log.append(entry)
                    print(f"step {step}/{total} ep {entry['epoch']:.2f} loss {entry['loss']:.4f} lr {entry['lr']:.2e} "
                          f"scale {entry['logit_scale']:.1f} {entry['img_per_s']} img/s mem {entry['gpu_mem_gb']} GB "
                          f"{entry['hours']:.2f} h", flush=True)
                window, win_t = [], time.time()
            due = time.time() - last_check > args.check_every_min * 60
            over = (time.time() - t0) / 3600 > args.max_hours
            flags = torch.tensor([due, over], dtype=torch.float32, device=device)
            if world > 1:
                dist.broadcast(flags, 0)  # all ranks take the same decision
            if flags[0] or flags[1] or step == total:
                if rank == 0:
                    check_and_log("final" if step == total else "periodic")
                    save(final=step == total or bool(flags[1]))  # out of time: still export the latest weights
                barrier()
                last_check = time.time()
            if flags[1] and step < total:
                stop = True
                if rank == 0:
                    print(f"time budget of {args.max_hours} h reached at step {step}/{total}; checkpoint saved, "
                          f"rerun with --resume to continue", flush=True)
                break
        del loader
    if rank == 0:
        save_json({"config": run_meta(args, t0, experiment="train_climp", n_train=n, world=world,
                                      global_batch=args.batch_size * world, steps_per_epoch=steps_per_epoch,
                                      total_steps=total, finished=step == total),
                   "metrics": {"last": log[-1] if log else None}}, res / f"train_climp_step{step}.json")
    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["check", "prep-images", "prep-text", "prep-upload", "train"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64, help="per GPU")
    p.add_argument("--lr", type=float, default=5e-5, help="paper: peak 5e-5, cosine")
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=10.5, help="stop + checkpoint before Kaggle's 12 h limit")
    p.add_argument("--check-every-min", type=float, default=45)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-upload", action="store_true", help="keep artifacts local (dry runs)")
    args = p.parse_args()
    {"check": cmd_check, "prep-images": cmd_prep_images, "prep-text": cmd_prep_text, "prep-upload": cmd_prep_upload,
     "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
