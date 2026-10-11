"""E2 (paper Table 8 + new analysis): embedding geometry and hubness-corrected retrieval on NoCaps @224.
CPU only; reads the embeddings saved by exp_nocaps.py.

  python src/exp_geometry.py
"""
import time

import numpy as np
import torch

from common import base_parser, out_dir, run_meta, save_json
from metrics import alignment, csls, hubness, recall_at_k, uniformity

KS = (1, 5, 10)


def analyse(path):
    e = torch.load(path)
    img, txt, idx = e["image_emb"], e["text_emb"], e["text_to_image_index"]
    _, first = np.unique(idx.numpy(), return_index=True)  # first caption of each image -> N matched pairs
    pair_txt = txt[torch.from_numpy(first)]
    out = {"alignment": alignment(img, pair_txt),
           "uniformity_image": uniformity(img),
           "uniformity_text": uniformity(pair_txt),
           "hubness_image": hubness(pair_txt, img, k=10),   # queries = texts, gallery = images
           "hubness_text": hubness(img, pair_txt, k=10)}    # queries = images, gallery = texts

    sim = txt @ img.T  # full matrix: all captions vs all images
    gt = torch.zeros_like(sim, dtype=torch.bool)
    gt[torch.arange(len(txt)), idx] = True
    corrected = csls(sim, k=10)
    fmt = lambda r: {str(k): v for k, v in r.items()}
    out["csls"] = {
        "image_retrieval": {"before": fmt(recall_at_k(sim, gt, KS)), "after": fmt(recall_at_k(corrected, gt, KS))},
        "text_retrieval": {"before": fmt(recall_at_k(sim.T, gt.T, KS)), "after": fmt(recall_at_k(corrected.T, gt.T, KS))},
    }
    return out, {"n_images": len(img), "n_texts": len(txt)}


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=["climp-mamba1", "openclip-vitb16"])
    p.add_argument("--resolution", type=int, default=224)
    args = p.parse_args()
    t0 = time.time()
    metrics, sizes = {}, {}
    for name in args.models:
        metrics[name], sizes[name] = analyse(out_dir(args) / "emb" / f"nocaps_{name}_{args.resolution}.pt")
        g, c = metrics[name], metrics[name]["csls"]
        print(f"{name}: align {g['alignment']:.3f} | unif img {g['uniformity_image']:.2f} txt {g['uniformity_text']:.2f}"
              f" | hub img {g['hubness_image']:.2f} txt {g['hubness_text']:.2f}")
        for task in c:
            print(f"  CSLS {task}: " + ", ".join(f"R@{k} {100 * c[task]['before'][k]:.1f}->{100 * c[task]['after'][k]:.1f}"
                                                 for k in c[task]["before"]))
    save_json({"config": run_meta(args, t0, experiment="geometry", sizes=sizes), "metrics": metrics},
              out_dir(args) / "geometry.json")
