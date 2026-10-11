"""E8 (paper Table 7): spatial inductive bias. Train a tiny 3-layer VMamba (~0.33M) and ViT (~0.35M) on
CIFAR-10 with regular vs patch-shuffled images; report final-epoch train loss and test accuracy.

  python src/exp_cifar_bias.py --models vmamba vit --conds regular shuffled --seeds 0 1 2 --epochs 25
"""
import time

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

from common import DATA_DIR, base_parser, grid_shuffle, out_dir, run_meta, save_json, seed_all

PAPER_PARAMS = {"vmamba": 0.33e6, "vit": 0.35e6}
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
GRID = 8  # 32px image / 4px patches = the models' own 8x8 patch grid


def build(name):
    if name == "vit":
        from timm.models.vision_transformer import VisionTransformer
        return VisionTransformer(img_size=32, patch_size=4, depth=3, num_classes=10, embed_dim=96, num_heads=3)
    from models import VMAMBA_BASE_CONFIG, VSSM  # upstream; needs CUDA (+ triton for the fast path)
    return VSSM(**dict(VMAMBA_BASE_CONFIG, depths=[3], dims=[84], patch_size=4, imgsize=32,
                       num_classes=10, drop_path_rate=0.0))


class Shuffled(Dataset):
    """Permute the 8x8 grid of 4x4 patches. Train: a fresh permutation every load.
    Test: a fixed permutation per image, seeded by (seed, index)."""

    def __init__(self, base, train, seed):
        self.base, self.train, self.seed = base, train, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        gen = None if self.train else torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        return grid_shuffle(x, GRID, torch.randperm(GRID * GRID, generator=gen)), y


def loaders(cond, seed, args):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    train = torchvision.datasets.CIFAR10(DATA_DIR / "cifar10", train=True, download=True,
                                         transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), *norm]))
    test = torchvision.datasets.CIFAR10(DATA_DIR / "cifar10", train=False, download=True, transform=T.Compose(norm))
    if args.n_train:
        train = Subset(train, range(args.n_train))
    if args.n_test:
        test = Subset(test, range(args.n_test))
    if cond == "shuffled":
        train, test = Shuffled(train, True, seed), Shuffled(test, False, seed)
    kw = dict(batch_size=128, num_workers=args.workers, pin_memory=True)
    return DataLoader(train, shuffle=True, drop_last=True, **kw), DataLoader(test, **kw)


def run(name, cond, seed, args):
    t0 = time.time()
    seed_all(seed)
    train_dl, test_dl = loaders(cond, seed, args)
    model = build(name).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params = {params:,} (paper {PAPER_PARAMS[name]:,.0f})")
    assert abs(params / PAPER_PARAMS[name] - 1) <= 0.10, "parameter count not within 10% of the paper"

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(args.device), y.to(args.device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        train_loss = total / n
        print(f"  {name}/{cond}/s{seed} epoch {epoch + 1}/{args.epochs} loss {train_loss:.3f} ({time.time() - t0:.0f}s)")

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x.to(args.device)).argmax(1).cpu() == y).sum().item()
    acc = correct / len(test_dl.dataset)
    print(f"  {name}/{cond}/s{seed}: final train loss {train_loss:.3f}, test acc {acc:.4f}")
    save_json({"config": run_meta(args, t0, experiment="cifar_bias", model=name, condition=cond, seed=seed,
                                  params=params, n_train=len(train_dl.dataset), n_test=len(test_dl.dataset)),
               "metrics": {"train_loss": train_loss, "test_acc": acc}},
              out_dir(args) / f"cifar_{name}_{cond}_s{seed}.json")


def save_example(args):
    """Write one CIFAR image next to its patch-shuffled version, for a visual check."""
    x, _ = torchvision.datasets.CIFAR10(DATA_DIR / "cifar10", train=False, download=True, transform=T.ToTensor())[0]
    y = grid_shuffle(x, GRID, torch.randperm(GRID * GRID, generator=torch.Generator().manual_seed(0)))
    path = out_dir(args) / "cifar_shuffle_example.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(F.interpolate(torch.stack([x, y]), scale_factor=8, mode="nearest"), path, padding=4)
    print(f"saved {path}")


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=["vmamba", "vit"], choices=["vmamba", "vit"])
    p.add_argument("--conds", nargs="+", default=["regular", "shuffled"], choices=["regular", "shuffled"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.seeds = 1, [0]
        args.n_train, args.n_test = args.n_train or 2000, args.n_test or 1000
        save_example(args)
    for name in args.models:
        for cond in args.conds:
            for seed in args.seeds:
                run(name, cond, seed, args)
