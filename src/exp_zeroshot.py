"""E3 (paper Table 2): zero-shot classification and robustness on CLIP-Benchmark webdatasets.
E4 (new): --shuffle-grid G splits each preprocessed image into a GxG grid and permutes the blocks
(permutation seeded by seed + image index, so both models see the same permutation).

  python src/exp_zeroshot.py --datasets cifar10 imagenetv2 imagenet-r imagenet-a imagenet-o
  python src/exp_zeroshot.py --datasets imagenetv2 --shuffle-grid 2 4 8
"""
import time

import torch
import torchvision
import webdataset as wds
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from common import DATA_DIR, base_parser, free_model, grid_shuffle, load_model, out_dir, run_meta, save_json, seed_all

REPOS = {"cifar10": "wds_vtab-cifar10", "imagenetv2": "wds_imagenetv2", "imagenet-r": "wds_imagenet-r",
         "imagenet-a": "wds_imagenet-a", "imagenet-o": "wds_imagenet-o"}


def fetch(name, smoke):
    """Download the test shards + classnames + templates of one clip-benchmark webdataset."""
    patterns = ["*.txt", "test/0.tar"] if smoke else ["*.txt", "test/*.tar"]
    path = DATA_DIR / "wds" / REPOS[name]
    snapshot_download(f"clip-benchmark/{REPOS[name]}", repo_type="dataset", allow_patterns=patterns, local_dir=path)
    classnames = (path / "classnames.txt").read_text().strip().split("\n")
    templates = (path / "zeroshot_classification_templates.txt").read_text().strip().split("\n")
    return sorted(str(p) for p in (path / "test").glob("*.tar")), classnames, templates


def classifier(m, classnames, templates):
    """Average the normalised embeddings of all templates per class, then re-normalise."""
    emb = m.encode_texts([t.replace("{c}", c) for c in classnames for t in templates])
    return torch.nn.functional.normalize(emb.view(len(classnames), len(templates), -1).mean(1), dim=-1)


def evaluate(m, shards, weights, grid, args, example_path=None):
    ds = wds.WebDataset(shards, shardshuffle=False).decode("pil").to_tuple("jpg;png;webp", "cls")
    ds = ds.map_tuple(lambda im: m.preprocess(im.convert("RGB")), int)
    dl = DataLoader(ds.batched(args.batch_size), batch_size=None, num_workers=min(args.workers, len(shards)))
    correct1 = correct5 = n = 0
    for x, y in dl:
        if args.smoke and n >= args.n_smoke:
            break
        if grid > 1:
            for j in range(len(x)):
                gen = torch.Generator().manual_seed(args.seed + n + j)
                x[j] = grid_shuffle(x[j], grid, torch.randperm(grid * grid, generator=gen))
            if example_path is not None and n == 0:
                orig = next(iter(DataLoader(ds.batched(1), batch_size=None)))[0]
                torchvision.utils.save_image(torch.cat([orig, x[:1]]), example_path, normalize=True, scale_each=True)
                print(f"saved {example_path}")
        top5 = (m.encode_images(x) @ weights.T).topk(5, dim=1).indices
        y = torch.as_tensor(y)
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(1).sum().item()
        n += len(y)
    return {"top1": correct1 / n, "top5": correct5 / n, "n": n}


def main(args):
    seed_all(args.seed)
    data = {d: fetch(d, args.smoke) for d in args.datasets}
    for name in args.models:
        t0 = time.time()
        m = load_model(name, 224, args.device)
        results = {g: {} for g in args.shuffle_grid}
        for d, (shards, classnames, templates) in data.items():
            weights = classifier(m, classnames, templates)
            for g in args.shuffle_grid:
                example = out_dir(args) / f"shuffle_example_{name}_g{g}.png" if args.smoke and g > 1 else None
                results[g][d] = evaluate(m, shards, weights, g, args, example)
                print(f"{name} G={g} {d}: top1 {100 * results[g][d]['top1']:.1f} top5 {100 * results[g][d]['top5']:.1f}"
                      f" (n={results[g][d]['n']}, {len(classnames)} classes, {len(templates)} templates)")
        del m
        free_model()
        for g, metrics in results.items():
            save_json({"config": run_meta(args, t0, experiment="zeroshot", model=name, image_size=224, shuffle_grid=g),
                       "metrics": metrics}, out_dir(args) / f"zeroshot_{name}_g{g}.json")


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=["climp-mamba1", "openclip-vitb16"])
    p.add_argument("--datasets", nargs="+", default=list(REPOS), choices=list(REPOS))
    p.add_argument("--shuffle-grid", nargs="+", type=int, default=[1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4 if torch.cuda.is_available() else 0)
    p.add_argument("--n-smoke", type=int, default=128)
    args = p.parse_args()
    if args.smoke and args.datasets == list(REPOS):
        args.datasets = ["cifar10", "imagenet-o"]
    main(args)
