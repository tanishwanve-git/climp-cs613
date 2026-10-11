"""E1 (paper Table 4 + README): NoCaps zero-shot retrieval at several input resolutions.
Image retrieval = text query -> images; text retrieval = image query -> captions (upstream definitions).
Saves embeddings to <out>/emb/ for E2. Smoke: 64 images at 224 and 896. Exits with code 1 if CLIMP @224 misses the README numbers by > 0.5.

  python src/exp_nocaps.py --models climp-mamba1 climp-mamba2 openclip-vitb16 --resolutions 224 512 896
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

from common import (MODELS, ROOT, autocast, base_parser, encode_image_set, free_model, load_model, out_dir,
                    run_meta, save_json, seed_all, with_resolution)
from datasets import load_dataset
from metrics import recall_at_k

KS = (1, 5, 10)
# HF's auto-converted parquet of HuggingFaceM4/NoCaps (the original loading script no longer runs in `datasets`>=3).
NOCAPS = "hf://datasets/HuggingFaceM4/NoCaps@refs%2Fconvert%2Fparquet/default/validation/{}.parquet"
GATE_TOL = 0.5  # percentage points


def padding_check(m, texts):
    """Min cosine between our padding="longest" text embeddings and upstream's padding to max length in characters."""
    enc = m.tokenizer(texts, max_length=max(len(t) for t in texts), padding="max_length", truncation=True,
                      return_tensors="pt").to(m.module.text_projection.weight.device)
    with torch.no_grad(), autocast(enc["input_ids"].device):
        upstream = F.normalize(m.module.encode_text(enc["input_ids"], enc["attention_mask"]).float(), dim=-1).cpu()
    return (m.encode_texts(texts) * upstream).sum(-1).min().item()


def vmamba_resolution_independent():
    """CLIMP reuses one set of weights at every resolution; check VMamba-B has identical parameter names/shapes at
    224 and 896 (i.e. nothing in the architecture depends on the input size)."""
    from models import build_vmamba
    a, b = build_vmamba(224).state_dict(), build_vmamba(896).state_dict()
    return a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)


def gate(name, metrics):
    readme = json.loads((ROOT / "paper_numbers.json").read_text())["nocaps_readme_224"][name]
    diffs = {f"{task}@{k}": round(100 * metrics[task][str(k)] - readme[task][str(k)], 2)
             for task in readme for k in KS}
    ok = all(abs(d) <= GATE_TOL for d in diffs.values())
    print(f"[GATE] {name} @224 vs README (ours - README, pp): {diffs} -> {'PASS' if ok else 'FAIL'}")
    return {"pass": ok, "diff_pp": diffs}


def main(args):
    seed_all(0)
    files = [NOCAPS.format("0000")] if args.smoke else [NOCAPS.format("*")]
    ds = load_dataset("parquet", data_files={"validation": files}, split="validation")
    assert list(ds["image_id"]) == list(range(len(ds))), "unexpected NoCaps row order"
    if args.smoke:
        ds = ds.select(range(64))
    captions = list(ds["annotations_captions"])
    texts = [c for caps in captions for c in caps]
    text_img = torch.tensor([i for i, caps in enumerate(captions) for _ in caps])
    gt = torch.zeros(len(texts), len(ds), dtype=torch.bool)
    gt[torch.arange(len(texts)), text_img] = True
    print(f"NoCaps validation: {len(ds)} images, {len(texts)} captions")
    images = ds.select_columns(["image"])
    gate_failed = False
    if any(n.startswith("climp") for n in args.models):
        assert vmamba_resolution_independent(), "VMamba weights depend on input resolution; cannot reuse the model"
        print("[check] VMamba-B parameters identical in name/shape at 224 and 896 -> one CLIMP model serves all resolutions")

    for name in args.models:
        text_emb, m = None, None
        for res in args.resolutions:
            t0 = time.time()
            m = load_model(name, res, args.device) if m is None else with_resolution(m, res, args.device)
            extra = {}
            if text_emb is None:  # text tower does not depend on image resolution: encode once per model
                text_emb = m.encode_texts(texts)
                if args.smoke and name.startswith("climp"):
                    extra["padding_check_min_cos"] = padding_check(m, texts[:16])
                    print(f"[padding check] {name}: tokenizer padding_side={m.tokenizer.padding_side}, "
                          f"min cos(longest, upstream) = {extra['padding_check_min_cos']:.5f} (need > 0.999)")
            img_emb = encode_image_set(m, lambda i: images[i]["image"], len(ds), args.batch_size, args.workers)
            free_model()

            sim = text_emb @ img_emb.T
            metrics = {"image_retrieval": {str(k): v for k, v in recall_at_k(sim, gt, KS).items()},
                       "text_retrieval": {str(k): v for k, v in recall_at_k(sim.T, gt.T, KS).items()}}
            print(f"{name} @{res}: " + ", ".join(f"{t} R@{k}={100 * v:.1f}" for t, d in metrics.items() for k, v in d.items()))
            if name.startswith("climp") and res == 224 and not args.smoke:
                extra["gate"] = gate(name, metrics)
                if not extra["gate"]["pass"]:
                    gate_failed = True
                    (out_dir(args) / "GATE_FAILED").parent.mkdir(parents=True, exist_ok=True)
                    (out_dir(args) / "GATE_FAILED").write_text(json.dumps(extra["gate"]))  # read by the other GPU queue
            out = out_dir(args)
            save_json({"config": run_meta(args, t0, experiment="nocaps", model=name, image_size=res,
                                          n_images=len(ds), n_texts=len(texts), **extra), "metrics": metrics},
                      out / f"nocaps_{name}_{res}.json")
            (out / "emb").mkdir(parents=True, exist_ok=True)
            torch.save({"image_emb": img_emb, "text_emb": text_emb, "text_to_image_index": text_img},
                       out / "emb" / f"nocaps_{name}_{res}.pt")
        del m
        free_model()
    if gate_failed:
        sys.exit("[GATE] FAILED: CLIMP does not reproduce the README numbers. Stop and debug before other runs.")


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--resolutions", nargs="+", type=int, default=[224, 512, 896])
    p.add_argument("--batch-size", type=int, default=None, help="default: 32 @224, scaled down with resolution")
    p.add_argument("--workers", type=int, default=4 if torch.cuda.is_available() else 0)
    args = p.parse_args()
    if args.smoke:
        args.resolutions = [224, 896]  # 896 included so out-of-memory problems show up in the smoke run
    main(args)
