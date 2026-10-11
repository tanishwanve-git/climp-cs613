"""E8 (paper Table 7): spatial inductive bias. Train a tiny 3-layer VMamba (~0.33M) and ViT (~0.35M) on
CIFAR-10 with regular vs patch-shuffled images; report final-epoch train loss and test accuracy.
The whole dataset lives on the GPU and augmentation runs there batch-wise (Kaggle has only 4 CPU cores).

  python src/exp_cifar_bias.py --models vmamba vit --conds regular shuffled --seeds 0 1 2 --epochs 20
"""
import time

import torch
import torch.nn.functional as F
import torchvision

from common import DATA_DIR, base_parser, grid_shuffle, out_dir, run_meta, save_json, seed_all

PAPER_PARAMS = {"vmamba": 0.33e6, "vit": 0.35e6}
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
GRID = 8  # 32px image / 4px patches = the models' own 8x8 patch grid
BATCH = 128


def build(name):
    if name == "vit":
        from timm.models.vision_transformer import VisionTransformer
        return VisionTransformer(img_size=32, patch_size=4, depth=3, num_classes=10, embed_dim=96, num_heads=3)
    from models import VMAMBA_BASE_CONFIG, VSSM  # upstream; needs CUDA (+ triton for the fast path)
    return VSSM(**dict(VMAMBA_BASE_CONFIG, depths=[3], dims=[84], patch_size=4, imgsize=32,
                       num_classes=10, drop_path_rate=0.0))


def load_cifar(train, n, device):
    ds = torchvision.datasets.CIFAR10(DATA_DIR / "cifar10", train=train, download=True)
    x = torch.from_numpy(ds.data[:n]).permute(0, 3, 1, 2).contiguous().to(device)  # uint8 [N, 3, 32, 32]
    return x, torch.tensor(ds.targets[:n], device=device)


def normalize(x):
    mean = torch.tensor(MEAN, device=x.device)[:, None, None]
    std = torch.tensor(STD, device=x.device)[:, None, None]
    return (x - mean) / std


def augment(x):
    """uint8 batch -> RandomCrop(32, padding=4) with black padding + RandomHorizontalFlip, as floats in [0, 1]
    (the same ops as the torchvision train transform, applied before normalisation)."""
    b, dev = len(x), x.device
    x = F.pad(x.float() / 255, (4, 4, 4, 4))
    oy, ox = torch.randint(0, 9, (2, b, 1), device=dev)
    ar = torch.arange(32, device=dev)
    x = x[torch.arange(b, device=dev)[:, None, None], :, (oy + ar)[:, :, None], (ox + ar)[:, None, :]]  # [B,32,32,C]
    x = x.permute(0, 3, 1, 2)
    return torch.where(torch.rand(b, 1, 1, 1, device=dev) < 0.5, x.flip(-1), x)


def test_perms(n, seed):
    """Fixed per-image permutation for test images, seeded by (seed, index)."""
    return torch.stack([torch.randperm(GRID * GRID, generator=torch.Generator().manual_seed(seed * 1_000_003 + i))
                        for i in range(n)])


def run(name, cond, seed, args):
    t0 = time.time()
    seed_all(seed)
    train_x, train_y = load_cifar(True, args.n_train, args.device)
    test_x, test_y = load_cifar(False, args.n_test, args.device)
    test_x = normalize(test_x.float() / 255)
    if cond == "shuffled":
        test_x = grid_shuffle(test_x, GRID, test_perms(len(test_x), seed).to(args.device))
    model = build(name).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params = {params:,} (paper {PAPER_PARAMS[name]:,.0f})")
    assert abs(params / PAPER_PARAMS[name] - 1) <= 0.10, "parameter count not within 10% of the paper"

    steps = len(train_x) // BATCH  # drop_last
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_x), device=args.device)
        total = torch.zeros((), device=args.device)
        for i in range(steps):
            idx = order[i * BATCH:(i + 1) * BATCH]
            x = normalize(augment(train_x[idx]))
            if cond == "shuffled":  # a fresh random permutation per image, every time it is seen
                x = grid_shuffle(x, GRID, torch.rand(len(x), GRID * GRID, device=args.device).argsort(dim=1))
            loss = F.cross_entropy(model(x), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach()
        train_loss = (total / steps).item()
        print(f"  {name}/{cond}/s{seed} epoch {epoch + 1}/{args.epochs} loss {train_loss:.3f} ({time.time() - t0:.0f}s)")

    model.eval()
    with torch.no_grad():
        correct = sum((model(test_x[i:i + 500]).argmax(1) == test_y[i:i + 500]).sum().item()
                      for i in range(0, len(test_x), 500))
    acc = correct / len(test_x)
    print(f"  {name}/{cond}/s{seed}: final train loss {train_loss:.3f}, test acc {acc:.4f}")
    save_json({"config": run_meta(args, t0, experiment="cifar_bias", model=name, condition=cond, seed=seed,
                                  params=params, n_train=steps * BATCH, n_test=len(test_x)),
               "metrics": {"train_loss": train_loss, "test_acc": acc}},
              out_dir(args) / f"cifar_{name}_{cond}_s{seed}.json")


def save_example(args):
    """Row 1: 8 training images after crop/flip; row 2: the same images patch-shuffled (the training pipeline)."""
    torch.manual_seed(0)
    x = normalize(augment(load_cifar(True, 8, args.device)[0]))
    y = grid_shuffle(x, GRID, torch.rand(len(x), GRID * GRID, device=args.device).argsort(dim=1))
    std, mean = torch.tensor(STD, device=x.device)[:, None, None], torch.tensor(MEAN, device=x.device)[:, None, None]
    grid = torch.cat([x, y]) * std + mean
    path = out_dir(args) / "cifar_shuffle_example.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(F.interpolate(grid.cpu(), scale_factor=4, mode="nearest"), path, nrow=8, padding=4)
    print(f"saved {path}")


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="+", default=["vmamba", "vit"], choices=["vmamba", "vit"])
    p.add_argument("--conds", nargs="+", default=["regular", "shuffled"], choices=["regular", "shuffled"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.seeds = 1, [0]
        args.n_train, args.n_test = args.n_train or 2000, args.n_test or 1000
        save_example(args)
    for name in args.models:
        for cond in args.conds:
            for seed in args.seeds:
                run(name, cond, seed, args)
