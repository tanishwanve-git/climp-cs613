"""E6 (paper Fig. 3): vision-encoder memory and FLOPs vs input resolution (batch 1, fp16 autocast, CUDA).
vmamba-b = CLIMP's vision tower (upstream build_vmamba, random weights). vit-b16 = timm ViT-B/16 with fused
(memory-efficient) attention; vit-b16-eager = same model with classic attention that materialises the L x L matrix.

  python src/exp_efficiency.py
"""
import time

import timm
import torch
from fvcore.nn import FlopCountAnalysis

from common import autocast, base_parser, free_model, out_dir, run_meta, save_json

MODELS = ["vmamba-b", "vit-b16", "vit-b16-eager"]


def build(name, res):
    if name == "vmamba-b":
        from models import build_vmamba  # upstream
        return build_vmamba(res)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, dynamic_img_size=True, num_classes=0)
    if name == "vit-b16-eager":
        for blk in model.blocks:
            blk.attn.fused_attn = False
    return model


def memory(model, res, device):
    """Peak allocated memory of one forward pass after warm-up; activation = peak - (weights + input)."""
    x = torch.randn(1, 3, res, res, device=device)
    with torch.no_grad(), autocast(device):
        model(x)  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        model(x)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / 2 ** 20, peak / 2 ** 20


def gflops(name, model, res, device):
    """fvcore counts one multiply-add as one FLOP."""
    if name == "vmamba-b":
        return model.flops(shape=(3, res, res), verbose=False) / 1e9  # upstream helper (fvcore + selective-scan handle)
    eager = build("vit-b16-eager", res).to(device).eval()  # fvcore cannot see inside the fused attention kernel
    with torch.no_grad():
        return FlopCountAnalysis(eager, torch.randn(1, 3, res, res, device=device)).unsupported_ops_warnings(False).total() / 1e9


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--resolutions", nargs="+", type=int, default=[224, 384, 512, 768, 896])
    args = p.parse_args()
    assert torch.cuda.is_available(), "E6 needs a CUDA GPU"
    if args.smoke:
        args.resolutions = [224, 512]
    t0 = time.time()
    metrics = {}
    for name in MODELS:
        metrics[name] = {}
        for res in args.resolutions:
            model = build(name, res).to(args.device).eval()
            act, peak = memory(model, res, args.device)
            metrics[name][str(res)] = {"activation_mb": act, "peak_mb": peak, "gflops": gflops(name, model, res, args.device),
                                       "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
            print(f"{name} @{res}: activation {act:.1f} MB, peak {peak:.1f} MB, {metrics[name][str(res)]['gflops']:.1f} GFLOPs")
            del model
            free_model()
    save_json({"config": run_meta(args, t0, experiment="efficiency"), "metrics": metrics}, out_dir(args) / "efficiency.json")
