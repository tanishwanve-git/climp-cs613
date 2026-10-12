import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import grid_shuffle, token_batches
from metrics import alignment, csls, hubness, recall_at_k, uniformity


def upstream_recall(scores, positive_pairs, k):
    # Copied from CLIMP/eval_nocaps.py: recall_at_k(...) > 0, then averaged over queries.
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    onehot = F.one_hot(topk_indices, num_classes=nb_images)
    nb_true_positive = (onehot * positive_pairs.view(nb_texts, 1, nb_images)).sum(dim=(1, 2))
    return ((nb_true_positive / nb_positive) > 0).float().mean().item()


def test_recall_identity():
    assert recall_at_k(torch.eye(20), torch.eye(20, dtype=bool), ks=(1,))[1] == 1.0


def test_recall_matches_upstream():
    torch.manual_seed(0)
    n_img, caps = 50, 5  # NoCaps-like: several captions per image
    sim = torch.randn(n_img * caps, n_img)
    gt = torch.zeros_like(sim, dtype=bool)
    gt[torch.arange(n_img * caps), torch.arange(n_img).repeat_interleave(caps)] = True
    ours_ir, ours_tr = recall_at_k(sim, gt), recall_at_k(sim.T, gt.T)
    for k in (1, 5, 10):
        assert ours_ir[k] == upstream_recall(sim, gt, k)
        assert ours_tr[k] == upstream_recall(sim.T, gt.T, k)


def test_alignment_zero():
    x = F.normalize(torch.randn(100, 16), dim=-1)
    assert alignment(x, x) == 0


def test_uniformity():
    assert abs(uniformity(F.normalize(torch.ones(50, 8), dim=-1))) < 1e-6  # collapsed = maximum (0)
    assert uniformity(F.normalize(torch.randn(500, 8), dim=-1)) < -1


def test_hubness_symmetric_and_flat():
    gallery = torch.eye(3)
    queries = gallery[[0, 1, 1, 2, 2, 2]]  # k-occurrence counts [1, 2, 3]: symmetric -> skew 0
    assert abs(hubness(queries, gallery, k=1)) < 1e-9
    assert hubness(torch.eye(5), torch.eye(5), k=1) == 0.0  # every item appears exactly once


def test_hubness_single_hub():
    gallery = torch.eye(10)
    queries = gallery[[0] * 30]  # item 0 is in every query's top-1
    assert hubness(queries, gallery, k=1) > 2


def test_csls_keeps_diagonal_ranking():
    sim = torch.eye(10) * 0.8 + 0.1
    gt = torch.eye(10, dtype=bool)
    assert recall_at_k(csls(sim, k=3), gt, ks=(1,))[1] == 1.0
    assert torch.equal(csls(sim, k=3).argmax(dim=1), sim.argmax(dim=1))


def test_grid_shuffle():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float).reshape(2, 3, 8, 8)
    assert torch.equal(grid_shuffle(x, 4, torch.arange(16)), x)  # identity permutation
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(0))
    y = grid_shuffle(x, 4, perm)
    assert torch.equal(grid_shuffle(y, 4, torch.argsort(perm)), x)  # inverse permutation restores
    assert torch.equal(y[..., 0:2, 0:2], x[..., (perm[0] // 4) * 2:(perm[0] // 4) * 2 + 2, (perm[0] % 4) * 2:(perm[0] % 4) * 2 + 2])


def test_grid_shuffle_batched_matches_per_image():
    torch.manual_seed(0)
    x = torch.randn(5, 3, 32, 32)
    perms = torch.rand(5, 64).argsort(dim=1)
    batched = grid_shuffle(x, 8, perms)
    assert torch.equal(batched, torch.stack([grid_shuffle(x[i], 8, perms[i]) for i in range(5)]))


def test_e8_augment_is_padded_crop_or_flip():
    from exp_cifar_bias import augment
    torch.manual_seed(0)
    x = torch.randint(0, 256, (16, 3, 32, 32), dtype=torch.uint8)
    y = augment(x)
    padded = F.pad(x.float() / 255, (4, 4, 4, 4))  # black padding, as torchvision RandomCrop(padding=4)
    for i in range(16):
        windows = [padded[i, :, oy:oy + 32, ox:ox + 32] for oy in range(9) for ox in range(9)]
        assert any(torch.equal(y[i], w) or torch.equal(y[i], w.flip(-1)) for w in windows)


def test_token_batches():
    assert token_batches([10] * 130, max_batch=64) == [(0, 64), (64, 128), (128, 130)]  # short texts: plain batches
    long = token_batches([300] * 40, max_batch=64, token_budget=4096)                    # 13 x 300 <= 4096
    assert long[0] == (0, 13) and all(j - i <= 13 for i, j in long) and long[-1][1] == 40
    mixed = [5, 5, 400, 5, 5]
    covered = [k for i, j in token_batches(mixed, 64, 1000) for k in range(i, j)]
    assert covered == list(range(5))                                                   # every text exactly once, in order
    assert all((j - i) * max(mixed[i:j]) <= 1000 for i, j in token_batches(mixed, 64, 1000))
    assert token_batches([5000], 64, 4096) == [(0, 1)]                                   # a single over-long text still runs
