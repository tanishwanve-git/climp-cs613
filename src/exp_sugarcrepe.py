"""E7 (new): SugarCrepe compositionality. An item is correct if sim(image, true caption) > sim(image, hard negative).

  python src/exp_sugarcrepe.py
"""
import json
import time
import urllib.request
import zipfile

import torch
from PIL import Image

from common import DATA_DIR, base_parser, encode_image_set, free_model, load_model, out_dir, run_meta, save_json, seed_all

SUBSETS = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]
JSON_URL = "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/{}.json"
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"


def fetch(n_per_subset=None):
    """Return {subset: [items]} and the local dir holding the COCO val2017 images they use."""
    root = DATA_DIR / "sugarcrepe"
    root.mkdir(parents=True, exist_ok=True)
    data = {}
    for s in SUBSETS:
        path = root / f"{s}.json"
        if not path.exists():
            urllib.request.urlretrieve(JSON_URL.format(s), path)
        items = list(json.loads(path.read_text()).values())
        data[s] = items[:n_per_subset] if n_per_subset else items
    zip_path, img_dir = DATA_DIR / "coco" / "val2017.zip", DATA_DIR / "coco" / "val2017"
    needed = {it["filename"] for items in data.values() for it in items}
    missing = [f for f in needed if not (img_dir / f).exists()]
    if missing:
        if not zip_path.exists():
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {COCO_URL} (~0.8 GB) ...")
            urllib.request.urlretrieve(COCO_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:  # extract only the images SugarCrepe uses
            for f in missing:
                z.extract(f"val2017/{f}", img_dir.parent)
    return data, img_dir


def main(args):
    seed_all(0)
    data, img_dir = fetch(32 if args.smoke else None)
    files = sorted({it["filename"] for items in data.values() for it in items})
    print("SugarCrepe items:", {s: len(v) for s, v in data.items()}, f"| {len(files)} unique COCO images")
    for name in args.models:
        t0 = time.time()
        m = load_model(name, 224, args.device)
        img_emb = dict(zip(files, encode_image_set(m, lambda i: Image.open(img_dir / files[i]), len(files),
                                                   workers=args.workers)))
        metrics = {}
        for s, items in data.items():
            pos = m.encode_texts([it["caption"] for it in items])
            neg = m.encode_texts([it["negative_caption"] for it in items])
            img = torch.stack([img_emb[it["filename"]] for it in items])
            metrics[s] = ((img * pos).sum(-1) > (img * neg).sum(-1)).float().mean().item()
        metrics["macro"] = sum(metrics[s] for s in SUBSETS) / len(SUBSETS)
        del m
        free_model()
        print(f"{name}: " + ", ".join(f"{k} {100 * v:.1f}" for k, v in metrics.items()))
        save_json({"config": run_meta(args, t0, experiment="sugarcrepe", model=name, image_size=224,
                                      n_items={s: len(v) for s, v in data.items()}), "metrics": metrics},
                  out_dir(args) / f"sugarcrepe_{name}.json")


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=["climp-mamba1", "openclip-vitb16"])
    p.add_argument("--workers", type=int, default=4 if torch.cuda.is_available() else 0)
    main(p.parse_args())
