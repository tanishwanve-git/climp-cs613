"""Build results/tables/*.md|.tex and results/figures/*.png from results/*.json and paper_numbers.json.
Missing results are written as TODO. Nothing here computes or estimates a number.

Expected files (all have {"config": ..., "metrics": ...}; recalls/accuracies are fractions in [0, 1]):
  nocaps_{model}_{res}.json       metrics = {"image_retrieval": {"1","5","10"}, "text_retrieval": {...}}
  geometry.json                   metrics = {model: {"alignment", "uniformity_image", "uniformity_text",
                                             "hubness_image", "hubness_text",
                                             "csls": {"image_retrieval"|"text_retrieval": {"before"|"after": {"1","5","10"}}}}}
  zeroshot_{model}_g{G}.json      metrics = {dataset: {"top1", "top5"}}
  docci_{model}.json              metrics = {"77"|"154"|"full": {"image_retrieval": {...}, "text_retrieval": {...}}}
  efficiency.json                 metrics = {model: {res: {"activation_mb", "peak_mb", "gflops"}}}
  sugarcrepe_{model}.json         metrics = {subset: acc, ..., "macro": acc}
  cifar_{model}_{cond}_s{seed}.json  metrics = {"train_loss", "test_acc"}
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CLIP_MODELS = ["climp-mamba1", "climp-mamba2", "openclip-vitb16"]
RES = ["224", "512", "896"]
KS = ["1", "5", "10"]
ZS_SETS = ["imagenetv2", "imagenet-r", "imagenet-a", "imagenet-o"]
SC_SUBSETS = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj", "macro"]
GRIDS = ["1", "2", "4", "8"]


def load(res_dir, name):
    path = res_dir / f"{name}.json"
    return json.loads(path.read_text())["metrics"] if path.exists() else None


def get(d, *keys):
    for k in keys:
        if d is None or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def fmt(v, scale=100, nd=1):
    return "TODO" if v is None else f"{v * scale:.{nd}f}"


def pfmt(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


def write_table(out, name, title, header, rows, note=""):
    md = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    if note:
        md += ["", note]
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%").replace("→", r"$\to$").replace("±", r"$\pm$")
    tex = [r"\begin{tabular}{" + "l" * 2 + "c" * (len(header) - 2) + "}", r"\toprule",
           " & ".join(map(esc, header)) + r" \\", r"\midrule"]
    tex += [" & ".join(map(esc, r)) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    (out / f"{name}.tex").write_text("\n".join(tex) + "\n")


def nocaps(res, paper, out):
    rows = []
    for m in CLIP_MODELS:
        for r in RES:
            d = load(res, f"nocaps_{m}_{r}")
            p = get(paper, "nocaps_readme_224", m) if r == "224" else None
            for task in ["image_retrieval", "text_retrieval"]:
                rows.append([m, r, task.replace("_", " ")] +
                            [f"{fmt(get(d, task, k))} ({pfmt(get(p, task, k))})" for k in KS])
    write_table(out, "e1_nocaps", "E1 NoCaps retrieval: ours (paper/README)", ["Model", "Res", "Task", "R@1", "R@5", "R@10"], rows,
                "Image retrieval = text query → images; text retrieval = image query → captions. Paper values: README @224.")
    rows = []
    for m in CLIP_MODELS:
        for r in RES:
            d, p = load(res, f"nocaps_{m}_{r}"), get(paper, "nocaps_table4_r5", m, r)
            rows.append([m, r, fmt(get(d, "image_retrieval", "5")), pfmt(p and p[0]),
                         fmt(get(d, "text_retrieval", "5")), pfmt(p and p[1])])
    write_table(out, "e1_nocaps_table4", "E1 NoCaps R@5 vs resolution (paper Table 4)",
                ["Model", "Res", "Image R@5 ours", "paper 'I→T'", "Text R@5 ours", "paper 'T→I'"], rows,
                "Paper Table 4 labels its columns I→T / T→I, but its values match the README's image / text retrieval R@5.")


def geometry(res, paper, out):
    d = load(res, "geometry")
    keys = ["alignment", "uniformity_image", "uniformity_text", "hubness_image", "hubness_text"]
    rows = [[m, "ours"] + [fmt(get(d, m, k), 1, 3) for k in keys] for m in ["climp-mamba1", "openclip-vitb16"]]
    rows.append(["climp-mamba1", "paper"] + [pfmt(get(paper, "geometry_table8", "climp-mamba1", k), 3) for k in keys])
    write_table(out, "e2_geometry", "E2 Embedding geometry on NoCaps @224 (paper Table 8)",
                ["Model", "Source", "Alignment", "Unif. img", "Unif. txt", "Hub. img", "Hub. txt"], rows,
                "Lower is better for all. Hubness = skewness of k-occurrence, k = 10.")
    rows = []
    for m in ["climp-mamba1", "openclip-vitb16"]:
        for task in ["image_retrieval", "text_retrieval"]:
            c = get(d, m, "csls", task)
            rows.append([m, task.replace("_", " ")] +
                        [f"{fmt(get(c, 'before', k))} → {fmt(get(c, 'after', k))}" for k in KS])
    write_table(out, "e2_csls", "E2 Hubness correction with CSLS (k = 10, transductive)",
                ["Model", "Task", "R@1", "R@5", "R@10"], rows, "Values are before → after CSLS re-scoring.")


def zeroshot(res, paper, out):
    rows = []
    for m in ["climp-mamba1", "openclip-vitb16"]:
        d, p = load(res, f"zeroshot_{m}_g1"), get(paper, "zeroshot_table2", m)
        for s in ZS_SETS:
            ps = get(p, s)
            rows.append([m, s, fmt(get(d, s, "top1")), pfmt(ps and ps[0]), fmt(get(d, s, "top5")), pfmt(ps and ps[1])])
    write_table(out, "e3_zeroshot", "E3 Zero-shot classification and robustness (paper Table 2)",
                ["Model", "Dataset", "Top-1 ours", "Top-1 paper", "Top-5 ours", "Top-5 paper"], rows)
    rows = []
    for m in ["climp-mamba1", "openclip-vitb16"]:
        accs = [get(load(res, f"zeroshot_{m}_g{g}"), "imagenetv2", "top1") for g in GRIDS]
        base = accs[0]
        rows.append([m, "imagenetv2"] + [fmt(a) + ("" if a is None or not base else f" ({100 * (a / base - 1):+.0f}%)")
                                         for a in accs])
    write_table(out, "e4_shuffle", "E4 ImageNet-V2 top-1 under G×G patch shuffle",
                ["Model", "Dataset"] + [f"G={g}" for g in GRIDS], rows, "Brackets: change relative to G = 1.")


def docci(res, paper, out):
    rows = []
    for m, variants in [("climp-mamba1", ["77", "154", "full"]), ("openclip-vitb16", ["77"])]:
        d = load(res, f"docci_{m}")
        for v in variants:
            for task in ["image_retrieval", "text_retrieval"]:
                rows.append([m, v, task.replace("_", " ")] + [fmt(get(d, v, task, k)) for k in KS])
    for task in ["image_retrieval", "text_retrieval"]:
        rows.append(["climp-mamba1 (paper)", "full", task.replace("_", " ")] +
                    [pfmt(get(paper, "docci_table6", "climp-mamba1", task, k)) for k in KS])
    write_table(out, "e5_docci", "E5 DOCCI long-caption retrieval @896 vs text truncation (paper Table 6)",
                ["Model", "Max tokens", "Task", "R@1", "R@5", "R@10"], rows)


def efficiency(res, paper, out):
    d = load(res, "efficiency")
    resolutions = sorted({int(r) for m in (d or {}).values() for r in m}) or [224, 384, 512, 768, 896]
    rows = []
    for m in ["vmamba-b", "vit-b16", "vit-b16-eager"]:
        for key in ["activation_mb", "peak_mb", "gflops"]:
            rows.append([m, key] + [fmt(get(d, m, str(r), key), 1, 1) for r in resolutions])
        p = get(paper, "efficiency_fig3_896", m)
        if p is None:
            continue
        rows.append([m, "paper @896 (MB / GFLOPs)"] + [""] * (len(resolutions) - 1) +
                    [f"{pfmt(get(p, 'memory_mb'))} / {pfmt(get(p, 'gflops'))}"])
    write_table(out, "e6_efficiency", "E6 Vision-encoder memory and FLOPs vs resolution (paper Fig. 3)",
                ["Model", "Quantity"] + [str(r) for r in resolutions], rows,
                "Batch 1, fp16. activation = peak − (weights + input). FLOPs from fvcore (1 multiply-add = 1 FLOP). "
                "vit-b16 uses fused (memory-efficient) attention; vit-b16-eager materialises the attention matrix.")


def sugarcrepe(res, paper, out):
    rows = [[m, ""] + [fmt(get(load(res, f"sugarcrepe_{m}"), s)) for s in SC_SUBSETS]
            for m in ["climp-mamba1", "openclip-vitb16"]]
    write_table(out, "e7_sugarcrepe", "E7 SugarCrepe accuracy (%)", ["Model", ""] + SC_SUBSETS, rows,
                "Correct = sim(image, positive) > sim(image, negative). Chance = 50.")


def cifar(res, paper, out):
    rows = []
    for cond in ["regular", "shuffled"]:
        for m in ["vmamba", "vit"]:
            runs = [load(res, f"cifar_{m}_{cond}_s{s}") for s in range(3)]
            runs = [r for r in runs if r is not None]
            ms = lambda key, sc, nd: "TODO" if not runs else (
                f"{statistics.mean(r[key] for r in runs) * sc:.{nd}f}" if len(runs) == 1 else
                f"{statistics.mean(r[key] for r in runs) * sc:.{nd}f} ± {statistics.stdev(r[key] for r in runs) * sc:.{nd}f}")
            p = get(paper, "cifar_table7", m, cond)
            rows.append([m, cond, ms("train_loss", 1, 3), pfmt(p and p[0], 3), ms("test_acc", 100, 2), pfmt(p and p[1], 2), str(len(runs))])
    write_table(out, "e8_cifar", "E8 Spatial inductive bias on CIFAR-10 (paper Table 7)",
                ["Model", "Data", "Train loss ours", "paper", "Test acc ours", "paper", "#seeds"], rows,
                "Mean ± std over seeds. Shuffled = per-image permutation of the 8×8 grid of 4×4 patches.")



# ---------------------------------------------------------------- figures (slide-ready; 16:9 friendly)
INK, INK2, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#8a8984"
COLOR = {"climp-mamba1": "#2a78d6", "climp-mamba2": "#1baf7a", "openclip-vitb16": "#eb6834",
         "vmamba-b": "#2a78d6", "vit-b16": "#eb6834", "vit-b16-eager": "#eb6834", "vmamba": "#2a78d6", "vit": "#eb6834"}
NAME = {"climp-mamba1": "CLIMP-Mamba1", "climp-mamba2": "CLIMP-Mamba2", "openclip-vitb16": "OpenCLIP ViT-B/16",
        "vmamba-b": "VMamba-B (CLIMP)", "vit-b16": "ViT-B/16, fused attention", "vit-b16-eager": "ViT-B/16, eager attention",
        "vmamba": "VMamba (0.33M)", "vit": "ViT (0.35M)"}
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def pct(v):
    return None if v is None else 100 * v


def end_label(ax, x, y, text, color, dy=0):
    """Direct label at the end of a line (identity never by colour alone)."""
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center", fontsize=11, color=INK2)


def bar_labels(ax, bars, fmt_="{:.1f}", inside=False):
    for b in bars:
        h = b.get_height()
        if h == h and h:  # skip NaN / 0
            ax.annotate(fmt_.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, color=INK2)


def fig_nocaps(res, paper):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    xs = [int(r) for r in RES]
    for ax, (task, title, pi) in zip(axes, [("image_retrieval", "Image retrieval (text → image) R@5", 0),
                                            ("text_retrieval", "Text retrieval (image → text) R@5", 1)]):
        rope = get(paper, "nocaps_table4_r5", "rope-vit (reference)")
        if rope:
            ys = [rope[r][pi] for r in RES]
            ax.plot(xs, ys, "--", color=PAPER, marker="o", mfc="white", label="RoPE-ViT + LLaMA (paper)")
            end_label(ax, xs[-1], ys[-1], "RoPE-ViT (paper)", PAPER)
        for m in CLIP_MODELS:
            ys = [pct(get(load(res, f"nocaps_{m}_{r}"), task, "5")) for r in RES]
            if any(y is not None for y in ys):
                ax.plot(xs, ys, color=COLOR[m], marker="o", label=NAME[m])
                end_label(ax, xs[-1], ys[-1], NAME[m], COLOR[m], dy={"climp-mamba1": 7, "climp-mamba2": -7}.get(m, 0))
        ax.set_xticks(xs, [f"{x}×{x}" for x in xs]), ax.set_xlim(180, 1150), ax.set_title(title)
        ax.set_xlabel("input resolution (trained at 224)")
    axes[0].set_ylabel("R@5 (%)")
    fig.tight_layout()
    return fig


def fig_efficiency(res, paper):
    d = load(res, "efficiency")
    if d is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, title in [(axes[0], "activation_mb", "Activation memory, batch 1 (MB)"), (axes[1], "gflops", "Compute (GFLOPs)")]:
        for m, ls in [("vmamba-b", "-"), ("vit-b16", "-"), ("vit-b16-eager", "--")]:
            rs = sorted(int(r) for r in d.get(m, {}))
            ys = [get(d, m, str(r), key) for r in rs]
            if key == "gflops" and m == "vit-b16-eager":
                continue  # same FLOPs as fused ViT
            ax.plot(rs, ys, ls, color=COLOR[m], marker="o", mfc="white" if ls == "--" else COLOR[m],
                    label=NAME[m] if key == "activation_mb" else NAME[m].split(",")[0])
            end_label(ax, rs[-1], ys[-1], f"{ys[-1]:.0f}", COLOR[m])
        notes = []
        for m in ["vmamba-b", "vit-b16"]:
            p = get(paper, "efficiency_fig3_896", m, "memory_mb" if key == "activation_mb" else "gflops")
            if p is not None:
                ax.plot([896], [p], "D", color=PAPER, mfc="white", ms=8, label="paper @896" if m == "vmamba-b" else None)
                notes.append(f"{'VMamba' if m == 'vmamba-b' else 'ViT'} {p:g}")
        if notes:
            ax.text(0.03, 0.56, "paper @896: " + ", ".join(notes), transform=ax.transAxes, fontsize=10.5, color=INK2)
        ax.set_title(title), ax.set_xlabel("input resolution"), ax.set_xlim(150, 1010)
        ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    return fig


def fig_shuffle(res, paper):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    gs = [int(g) for g in GRIDS]
    for m in ["climp-mamba1", "openclip-vitb16"]:
        accs = [get(load(res, f"zeroshot_{m}_g{g}"), "imagenetv2", "top1") for g in GRIDS]
        if accs[0]:
            rel = [None if a is None else 100 * a / accs[0] for a in accs]
            ax.plot(gs, rel, color=COLOR[m], marker="o", label=NAME[m])
            end_label(ax, gs[-1], rel[-1], f"{NAME[m]} ({100 * accs[0]:.1f}→{100 * accs[-1]:.1f})", COLOR[m],
                      dy=8 if m == "openclip-vitb16" else -8)
    ax.set_xscale("log", base=2), ax.set_xticks(gs, [f"{g}×{g}" for g in gs]), ax.set_xlim(0.8, 40)
    ax.set_ylim(0, 105), ax.set_ylabel("% of unshuffled accuracy"), ax.set_xlabel("shuffle grid (blocks permuted)")
    ax.set_title("Zero-shot ImageNet-V2 under patch shuffling")
    fig.tight_layout()
    return fig


def reproduction_pairs(res, paper):
    """(group, label, ours, paper) for every number recomputable from the released weights."""
    pairs = []
    for m in ["climp-mamba1", "climp-mamba2"]:
        for ri, r in enumerate(RES):
            p = get(paper, "nocaps_table4_r5", m, r)
            for ti, task in enumerate(["image_retrieval", "text_retrieval"]):
                pairs.append(("NoCaps (Table 4 + README)", f"{m} {r} {task} R@5", pct(get(load(res, f"nocaps_{m}_{r}"), task, "5")), p and p[ti]))
        for task in ["image_retrieval", "text_retrieval"]:
            for k in ["1", "10"]:
                pairs.append(("NoCaps (Table 4 + README)", f"{m} 224 {task} R@{k}",
                              pct(get(load(res, f"nocaps_{m}_224"), task, k)), get(paper, "nocaps_readme_224", m, task, k)))
    z = load(res, "zeroshot_climp-mamba1_g1")
    for s in ZS_SETS:
        p = get(paper, "zeroshot_table2", "climp-mamba1", s)
        for i, k in enumerate(["top1", "top5"]):
            pairs.append(("Zero-shot ImageNet (Table 2)", f"{s} {k}", pct(get(z, s, k)), p and p[i]))
    d = load(res, "docci_climp-mamba1")
    for task in ["image_retrieval", "text_retrieval"]:
        for k in KS:
            pairs.append(("DOCCI long captions (Table 6)", f"{task} R@{k}", pct(get(d, "full", task, k)),
                          get(paper, "docci_table6", "climp-mamba1", task, k)))
    return [q for q in pairs if q[2] is not None and q[3] is not None]


def fig_reproduction(res, paper):
    pairs = reproduction_pairs(res, paper)
    if not pairs:
        return None
    groups = list(dict.fromkeys(g for g, *_ in pairs))
    gcol = dict(zip(groups, ["#2a78d6", "#eb6834", "#1baf7a"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.plot([0, 100], [0, 100], color=GRID, lw=1.5, zorder=0)
    for g in groups:
        q = [(o, p) for gg, _, o, p in pairs if gg == g]
        a1.scatter([p for _, p in q], [o for o, _ in q], s=46, color=gcol[g], edgecolor="white", linewidth=1.2,
                   label=f"{g} ({len(q)})", zorder=3)
    a1.set_xlim(0, 100), a1.set_ylim(0, 100), a1.set_aspect("equal")
    a1.set_xlabel("paper / README (%)"), a1.set_ylabel("ours (%)"), a1.set_title("Every recomputable number")
    a1.legend(loc="upper left", fontsize=9.5)
    a2.axhspan(-0.5, 0.5, color="#f1f0ec", zorder=0)
    a2.axhline(0, color=MUTED, lw=1)
    for gi, g in enumerate(groups):
        q = [(lab, o - p) for gg, lab, o, p in pairs if gg == g]
        xs = [gi + (i - (len(q) - 1) / 2) * 0.6 / max(1, len(q) - 1) for i in range(len(q))]
        a2.scatter(xs, [dv for _, dv in q], s=46, color=gcol[g], edgecolor="white", linewidth=1.2, zorder=3)
        lab, dv = max(q, key=lambda t: abs(t[1]))
        if abs(dv) > 0.45:
            a2.annotate(lab.replace("climp-", "").replace("_retrieval", ""), (xs[[l for l, _ in q].index(lab)], dv),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK2)
    a2.set_xticks(range(len(groups)), [g.split(" (")[0] for g in groups], fontsize=10.5)
    a2.set_ylabel("ours − paper (pp)"), a2.set_title("Differences (shaded band = ±0.5 pp)")
    lim = max(2.0, max(abs(o - p) for *_, o, p in pairs) + 0.3)
    a2.set_ylim(-lim, lim), a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_cifar(res, paper):
    import numpy as np
    stats = {}
    for m in ["vmamba", "vit"]:
        for cond in ["regular", "shuffled"]:
            runs = [r for r in (load(res, f"cifar_{m}_{cond}_s{s}") for s in range(3)) if r]
            if runs:
                accs = [100 * r["test_acc"] for r in runs]
                stats[m, cond] = (statistics.mean(accs), statistics.stdev(accs) if len(accs) > 1 else 0)
    if not stats:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x, w = np.arange(2), 0.36
    for i, m in enumerate(["vmamba", "vit"]):
        mean = [stats.get((m, c), (np.nan, 0))[0] for c in ["regular", "shuffled"]]
        sd = [stats.get((m, c), (0, 0))[1] for c in ["regular", "shuffled"]]
        bars = ax.bar(x + (i - 0.5) * w, mean, w, yerr=sd, color=COLOR[m], edgecolor="white", linewidth=2,
                      error_kw={"ecolor": INK2, "capsize": 3, "lw": 1.2}, label=f"{NAME[m]}, ours (3 seeds)")
        bar_labels(ax, bars)
        pv = [get(paper, "cifar_table7", m, c) for c in ["regular", "shuffled"]]
        ax.plot(x + (i - 0.5) * w, [p and p[1] for p in pv], "D", color=PAPER, mfc="white", ms=8,
                label="paper Table 7" if i == 0 else None)
    ax.set_xticks(x, ["regular images", "patch-shuffled images"]), ax.set_ylim(0, 100)
    ax.set_ylabel("CIFAR-10 test accuracy (%)"), ax.set_title("Training from scratch: spatial inductive bias")
    ax.legend(loc="upper right", fontsize=9.5), ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_docci(res, paper):
    d, o = load(res, "docci_climp-mamba1"), load(res, "docci_openclip-vitb16")
    if d is None:
        return None
    keys = [k for k in ["77", "154", "full"] if k in d]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task, title in [(axes[0], "image_retrieval", "Image retrieval R@1"), (axes[1], "text_retrieval", "Text retrieval R@1")]:
        ys = [pct(get(d, k, task, "1")) for k in keys]
        ax.plot(range(len(keys)), ys, color=COLOR["climp-mamba1"], marker="o", label="CLIMP-Mamba1, caption cut to N tokens")
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.1f}", (i, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10.5, color=INK2)
        p = get(paper, "docci_table6", "climp-mamba1", task, "1")
        if p is not None:
            ax.plot([len(keys) - 1], [p], "D", color=PAPER, mfc="white", ms=9, label="paper (full captions)")
        oc = pct(get(o, "77", task, "1"))
        if oc is not None:
            ax.axhline(oc, color=COLOR["openclip-vitb16"], ls=":", lw=2, label="OpenCLIP (cannot read past 77)")
            ax.annotate(f"OpenCLIP {oc:.1f}", (0, oc), xytext=(0, 6), textcoords="offset points", fontsize=10.5, color=INK2)
        ax.set_xticks(range(len(keys)), [{"77": "77 (CLIP limit)", "154": "154", "full": "full (≤578)"}[k] for k in keys])
        ax.set_xlabel("caption length given to the text tower (tokens)"), ax.set_title(title), ax.set_xlim(-0.3, len(keys) - 0.6)
    axes[0].set_ylabel("R@1 (%)"), axes[0].set_ylim(0, 50)
    axes[1].legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    return fig


def fig_sugarcrepe(res, paper):
    import numpy as np
    data = {m: load(res, f"sugarcrepe_{m}") for m in ["climp-mamba1", "openclip-vitb16"]}
    if not any(data.values()):
        return None
    subsets = SC_SUBSETS
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x, w = np.arange(len(subsets)) + np.array([0] * (len(subsets) - 1) + [0.4]), 0.38
    for i, m in enumerate(data):
        vals = [pct(get(data[m], s)) or np.nan for s in subsets]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=COLOR[m], edgecolor="white", linewidth=2, label=NAME[m])
        bar_labels(ax, bars, "{:.0f}")
    ax.axhline(50, color=INK2, ls="--", lw=1.2)
    ax.annotate("chance\n(50%)", ((x[-2] + x[-1]) / 2, 50), xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=10, color=INK2)  # in the gap before MACRO
    ax.set_xticks(x, [s.replace("_", "\n").replace("att", "attribute").replace("obj", "object").replace("rel", "relation")
                      .replace("macro", "MACRO\n(average)") for s in subsets], fontsize=10.5)
    ax.set_ylim(40, 100), ax.set_ylabel("accuracy (%)"), ax.grid(axis="x", visible=False)
    ax.set_title("SugarCrepe: does the true caption beat a minimally edited one?")
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    fig.tight_layout()
    return fig


def fig_csls(res, paper):
    d = load(res, "geometry")
    if d is None:
        return None
    rows = [(m, t) for m in ["climp-mamba1", "openclip-vitb16"] for t in ["image_retrieval", "text_retrieval"]
            if get(d, m, "csls", t)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, (m, t) in enumerate(rows):
        b, a = pct(get(d, m, "csls", t, "before", "1")), pct(get(d, m, "csls", t, "after", "1"))
        y = len(rows) - 1 - i
        ax.plot([b, a], [y, y], color=COLOR[m], lw=2.5, zorder=2)
        ax.plot([b], [y], "o", color=COLOR[m], mfc="white", ms=9, zorder=3)
        ax.plot([a], [y], "o", color=COLOR[m], ms=9, zorder=3)
        ax.annotate(f"+{a - b:.1f}", (a, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=11, color=INK2)
    ax.set_yticks(range(len(rows)), [f"{NAME[m].split(' ')[0]} · {t.split('_')[0]} retrieval" for m, t in rows][::-1])
    ax.set_xlabel("NoCaps R@1 (%)   ○ cosine   ● CSLS"), ax.set_xlim(30, 95), ax.grid(axis="y", visible=False)
    ax.set_title("Hubness correction (CSLS, no training)")
    fig.tight_layout()
    return fig


def fig_zeroshot(res, paper):
    import numpy as np
    c, o = load(res, "zeroshot_climp-mamba1_g1"), load(res, "zeroshot_openclip-vitb16_g1")
    if c is None:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x, w = np.arange(len(ZS_SETS)), 0.27
    rope = [get(paper, "zeroshot_table2", "rope-vit (reference)", s) for s in ZS_SETS]
    series = [("RoPE-ViT + LLaMA (paper, CC12M)", [r and r[0] for r in rope], "#c9c8c2"),
              ("CLIMP-Mamba1 (ours)", [pct(get(c, s, "top1")) for s in ZS_SETS], COLOR["climp-mamba1"]),
              ("OpenCLIP ViT-B/16 (ours, LAION-2B)", [pct(get(o, s, "top1")) for s in ZS_SETS], COLOR["openclip-vitb16"])]
    for i, (lab, vals, col) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, [v if v is not None else np.nan for v in vals], w, color=col, edgecolor="white",
                      linewidth=2, label=lab)
        bar_labels(ax, bars)
    pc = [get(paper, "zeroshot_table2", "climp-mamba1", s) for s in ZS_SETS]
    ax.plot(x, [p and p[0] for p in pc], "D", color=INK2, mfc="white", ms=7, label="CLIMP-Mamba1 (paper)")
    ax.set_xticks(x, ["ImageNet-V2", "ImageNet-R", "ImageNet-A", "ImageNet-O"]), ax.set_ylim(0, 118), ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("top-1 accuracy (%)"), ax.grid(axis="x", visible=False), ax.legend(loc="upper left", ncol=2, fontsize=10)
    ax.set_title("Zero-shot robustness on ImageNet variants")
    fig.tight_layout()
    return fig


FIGURES = {"fig1_nocaps_resolution": fig_nocaps, "fig2_efficiency": fig_efficiency, "fig3_patch_shuffle": fig_shuffle,
           "fig4_reproduction": fig_reproduction, "fig5_cifar_bias": fig_cifar, "fig6_docci_tokens": fig_docci,
           "fig7_sugarcrepe": fig_sugarcrepe, "fig8_csls": fig_csls, "fig9_zeroshot": fig_zeroshot}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    res = ROOT / ("results_smoke" if args.smoke else "results")
    tables, figures = res / "tables", res / "figures"
    tables.mkdir(parents=True, exist_ok=True), figures.mkdir(parents=True, exist_ok=True)
    paper = json.loads((ROOT / "paper_numbers.json").read_text())
    nocaps(res, paper, tables)
    geometry(res, paper, tables)
    zeroshot(res, paper, tables)
    docci(res, paper, tables)
    efficiency(res, paper, tables)
    sugarcrepe(res, paper, tables)
    cifar(res, paper, tables)
    figs = {}
    for name, make in FIGURES.items():
        fig = make(res, paper)
        if fig is not None:
            fig.savefig(figures / f"{name}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            figs[name] = fig
    print(f"wrote {len(list(tables.glob('*.md')))} tables to {tables} and {len(figs)} figures to {figures}")
