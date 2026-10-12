import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_climp import TrainableCLIMP, clip_loss, rank_indices


def reference_loss(img, txt, scale):
    logits = scale * img @ txt.T
    labels = torch.arange(len(img))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def test_clip_loss_single_process():
    torch.manual_seed(0)
    img, txt = F.normalize(torch.randn(8, 16), dim=-1), F.normalize(torch.randn(8, 16), dim=-1)
    assert torch.allclose(clip_loss(img, txt, 10.0), reference_loss(img, txt, 10.0))


def _rank(rank, world, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    img, txt = torch.randn(8, 16), torch.randn(8, 16)  # same full batch on every rank
    b = len(img) // world
    i = img[rank * b:(rank + 1) * b].clone().requires_grad_()
    t = txt[rank * b:(rank + 1) * b].clone().requires_grad_()
    loss = clip_loss(F.normalize(i, dim=-1), F.normalize(t, dim=-1), 10.0)
    loss.backward()
    # DDP averages parameter gradients over ranks, i.e. divides these input gradients by world
    q.put((rank, loss.item(), (i.grad / world).tolist(), (t.grad / world).tolist()))  # plain lists: process exits
    dist.destroy_process_group()


def test_clip_loss_distributed_matches_full_batch():
    world, q = 2, mp.get_context("spawn").SimpleQueue()
    mp.spawn(_rank, args=(world, 29517, q), nprocs=world)
    out = sorted(q.get() for _ in range(world))
    torch.manual_seed(0)
    img, txt = torch.randn(8, 16).requires_grad_(), torch.randn(8, 16).requires_grad_()
    full = reference_loss(F.normalize(img, dim=-1), F.normalize(txt, dim=-1), 10.0)
    full.backward()
    assert abs(sum(o[1] for o in out) / world - full.item()) < 1e-5            # mean of rank losses = global loss
    assert torch.allclose(torch.tensor(sum((o[2] for o in out), [])), img.grad, atol=1e-6)  # exact global gradient
    assert torch.allclose(torch.tensor(sum((o[3] for o in out), [])), txt.grad, atol=1e-6)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 4, 4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 32))
        self.config = types.SimpleNamespace(dims=[32])

    def forward(self, x):
        return self.net(x)


def test_trainable_climp_learns_synthetic_pairs():
    torch.manual_seed(0)
    colour = torch.rand(256, 3)
    x = colour[:, :, None, None] + 0.05 * torch.randn(256, 3, 16, 16)  # one distinct colour per image
    feats = colour @ torch.randn(3, 48) + 0.01 * torch.randn(256, 48)  # caption feature = f(image colour)
    model = TrainableCLIMP(TinyVision(), text_dim=48, proj_dim=24)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(300):
        i = torch.randint(0, 256, (64,))
        loss = clip_loss(*model(x[i], feats[i]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) / 20 < 0.6 * sum(losses[:20]) / 20


def test_trainable_climp_names_match_upstream():
    names = {k.split(".")[0] for k in TrainableCLIMP(TinyVision()).state_dict()}
    assert names == {"vision_encoder", "vision_projection", "text_projection", "logit_scale"}  # MambaCLIP minus text


def test_rank_indices_partition_epoch():
    a, steps = rank_indices(100, epoch=3, seed=0, world=2, rank=0, batch=8)
    b, _ = rank_indices(100, epoch=3, seed=0, world=2, rank=1, batch=8)
    assert steps == 6 and len(a) == len(b) == 48 and not set(a) & set(b)
    assert rank_indices(100, 4, 0, 2, 0, 8)[0] != a  # new order each epoch
