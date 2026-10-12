"""E5 (paper Table 6 + new analysis): DOCCI long-caption retrieval. CLIMP encodes each description truncated to
77 / 154 tokens and at full length; the OpenCLIP baseline is limited to 77 tokens by its text tower.

  python src/exp_docci.py --resolution 896
"""
import json
import tarfile
import time
import urllib.request

import torch
from PIL import Image

from common import DATA_DIR, base_parser, encode_image_set, free_model, load_model, out_dir, run_meta, save_json, seed_all
from metrics import recall_at_k

URL = "https://storage.googleapis.com/docci/data/"
KS = (1, 5, 10)
TRUNCATIONS = {"77": 77, "154": 154, "full": None}


def fetch(limit=None):
    """Download descriptions and stream-extract test images (only images/test_*.jpg are written to disk).
    With `limit`, stop after that many test images (smoke test)."""
    root = DATA_DIR / "docci"
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    desc_path = root / "docci_descriptions.jsonlines"
    if not desc_path.exists():
        urllib.request.urlretrieve(URL + "docci_descriptions.jsonlines", desc_path)
    rows = [json.loads(line) for line in desc_path.read_text().splitlines()]
    test = {r["image_file"]: r["description"] for r in rows if r["split"] == "test"}
    want = limit or len(test)
    if len(list(img_dir.glob("test_*.jpg"))) < want:
        print(f"streaming {URL}docci_images.tar.gz (7.6 GB) and keeping test images ...")
        with urllib.request.urlopen(URL + "docci_images.tar.gz") as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
            got = 0
            for member in tar:
                name = member.name.split("/")[-1]
                if name in test:
                    with open(img_dir / name, "wb") as f:
                        f.write(tar.extractfile(member).read())
                    got += 1
                    if got >= want:
                        break
    files = sorted(p.name for p in img_dir.glob("test_*.jpg"))[:want]
    assert len(files) == want, f"expected {want} DOCCI test images, found {len(files)}"
    return [img_dir / f for f in files], [test[f] for f in files]


def retrieval(text_emb, img_emb):
    sim = text_emb @ img_emb.T
    gt = torch.eye(len(sim), dtype=torch.bool)  # one description per image
    return {"image_retrieval": {str(k): v for k, v in recall_at_k(sim, gt, KS).items()},
            "text_retrieval": {str(k): v for k, v in recall_at_k(sim.T, gt.T, KS).items()}}


def main(args):
    seed_all(0)
    paths, texts = fetch(64 if args.smoke else None)
    print(f"DOCCI test: {len(paths)} image-description pairs")
    for name in args.models:
        t0 = time.time()
        m = load_model(name, args.resolution, args.device)
        img_emb = encode_image_set(m, lambda i: Image.open(paths[i]), len(paths), workers=args.workers)
        extra, metrics = {}, {}
        if name.startswith("climp"):
            lengths = torch.tensor([len(ids) for ids in m.tokenizer(texts)["input_ids"]])
            extra["tokens"] = {"mean": lengths.float().mean().item(), "frac_over_77": (lengths > 77).float().mean().item(),
                               "frac_over_154": (lengths > 154).float().mean().item(), "max": lengths.max().item()}
            print(f"Mamba tokenizer lengths: {extra['tokens']} (paper: mean ~142, 94.4% over 77)")
            embs = {key: m.encode_texts(texts, max_tokens=TRUNCATIONS[key]) for key in args.tokens}
            if args.smoke and {"77", "full"} <= set(embs):  # truncation must actually change long captions
                long = lengths > 77
                cos = (embs["77"][long] * embs["full"][long]).sum(-1)
                extra["truncation_check_max_cos"] = cos.max().item()
                print(f"[truncation check] {long.sum().item()} captions > 77 tokens, max cos(77, full) = {cos.max():.5f}")
                assert cos.max() < 0.9999, "77-token and full-length embeddings are identical"
        else:
            embs = {"77": m.encode_texts(texts)}
        del m
        free_model()
        for key, text_emb in embs.items():
            metrics[key] = retrieval(text_emb, img_emb)
            print(f"{name} @{args.resolution} tokens={key}: " + ", ".join(
                f"{t} R@{k}={100 * v:.1f}" for t, d in metrics[key].items() for k, v in d.items()))
        save_json({"config": run_meta(args, t0, experiment="docci", model=name, image_size=args.resolution,
                                      n_pairs=len(paths), **extra), "metrics": metrics},
                  out_dir(args) / f"docci_{name}.json")


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=["climp-mamba1", "openclip-vitb16"])
    p.add_argument("--resolution", type=int, default=896)
    p.add_argument("--tokens", nargs="+", default=list(TRUNCATIONS), choices=list(TRUNCATIONS),
                   help="CLIMP text lengths (paper Table 6 uses full)")
    p.add_argument("--workers", type=int, default=4 if torch.cuda.is_available() else 0)
    args = p.parse_args()
    if args.smoke:
        args.resolution = 224
    main(args)
