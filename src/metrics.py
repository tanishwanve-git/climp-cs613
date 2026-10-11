"""Retrieval and embedding-geometry metrics. Pure torch (+ scipy for skewness); no model code.

All embedding inputs are assumed L2-normalised float tensors of shape [n, d].
"""
import torch
from scipy.stats import skew


def recall_at_k(sim, gt, ks=(1, 5, 10)):
    """sim: [n_queries, n_gallery] scores; gt: bool matrix of the same shape.
    Returns {k: fraction of queries with >= 1 positive among their top-k gallery items}.
    Same definition as upstream eval_nocaps.py (recall > 0, averaged over queries)."""
    top = sim.topk(max(ks), dim=1).indices
    hits = gt.gather(1, top)
    return {k: hits[:, :k].any(dim=1).float().mean().item() for k in ks}


def alignment(x, y):
    """Wang & Isola alignment (alpha = 2): mean squared distance between matched pairs x_i, y_i."""
    return (x - y).norm(dim=1).pow(2).mean().item()


def uniformity(x, t=2.0, max_n=5000):
    """Wang & Isola uniformity: log mean_{i != j} exp(-t ||x_i - x_j||^2). 0 = collapsed, more negative = more uniform."""
    if len(x) > max_n:
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:max_n]
        x = x[idx]
    return torch.pdist(x.float()).pow(2).mul(-t).exp().mean().log().item()


def hubness(queries, gallery, k=10):
    """Skewness of the k-occurrence distribution N_k(g) = #queries having gallery item g in their top-k."""
    top = (queries @ gallery.T).topk(k, dim=1).indices
    counts = torch.bincount(top.flatten(), minlength=len(gallery)).double()
    if counts.std() == 0:
        return 0.0
    return float(skew(counts.numpy()))


def csls(sim, k=10):
    """Cross-domain similarity local scaling: 2*sim - r_q - r_g, where r_q (r_g) is the mean of each
    row's (column's) top-k similarities. Transductive: uses the test queries/gallery as neighbourhoods."""
    r_q = sim.topk(min(k, sim.shape[1]), dim=1).values.mean(dim=1)
    r_g = sim.topk(min(k, sim.shape[0]), dim=0).values.mean(dim=0)
    return 2 * sim - r_q[:, None] - r_g[None, :]
