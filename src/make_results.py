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
ZS_SETS = ["cifar10", "imagenetv2", "imagenet-r", "imagenet-a", "imagenet-o"]
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


def legend(ax):
    if ax.get_legend_handles_labels()[0]:  # nothing to label until results exist
        ax.legend()


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
    fig, ax = plt.subplots()
    for m in CLIP_MODELS:
        for task, ls in [("image_retrieval", "-"), ("text_retrieval", "--")]:
            ys = [get(load(res, f"nocaps_{m}_{r}"), task, "5") for r in RES]
            if any(y is not None for y in ys):
                ax.plot([int(r) for r in RES], [None if y is None else 100 * y for y in ys], ls, marker="o",
                        label=f"{m} {task.split('_')[0]}")
    ax.set_xlabel("input resolution"), ax.set_ylabel("R@5 (%)"), ax.set_title("NoCaps R@5 vs resolution")
    legend(ax)
    return fig


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
    rows, fig = [], plt.figure()
    for m in ["climp-mamba1", "openclip-vitb16"]:
        accs = [get(load(res, f"zeroshot_{m}_g{g}"), "imagenetv2", "top1") for g in GRIDS]
        base = accs[0]
        rows.append([m, "imagenetv2"] + [fmt(a) + ("" if a is None or not base else f" ({100 * (a / base - 1):+.0f}%)")
                                         for a in accs])
        if any(a is not None for a in accs):
            plt.plot([int(g) for g in GRIDS], [None if a is None else 100 * a for a in accs], marker="o", label=m)
    write_table(out, "e4_shuffle", "E4 ImageNet-V2 top-1 under G×G patch shuffle",
                ["Model", "Dataset"] + [f"G={g}" for g in GRIDS], rows, "Brackets: change relative to G = 1.")
    plt.xscale("log", base=2), plt.xlabel("shuffle grid G"), plt.ylabel("ImageNet-V2 top-1 (%)")
    plt.title("Patch-shuffle robustness")
    legend(plt.gca())
    return fig


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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in ["vmamba-b", "vit-b16", "vit-b16-eager"]:
        for ax, key in zip(axes, ["activation_mb", "gflops"]):
            ys = [get(d, m, str(r), key) for r in resolutions]
            if any(y is not None for y in ys):
                ax.plot(resolutions, ys, marker="o", label=m)
    for ax, lab in zip(axes, ["activation memory (MB)", "GFLOPs"]):
        ax.set_xlabel("input resolution"), ax.set_ylabel(lab)
        legend(ax)
    return fig


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    res = ROOT / ("results_smoke" if args.smoke else "results")
    tables, figures = res / "tables", res / "figures"
    tables.mkdir(parents=True, exist_ok=True), figures.mkdir(parents=True, exist_ok=True)
    paper = json.loads((ROOT / "paper_numbers.json").read_text())
    figs = {"fig1_nocaps_resolution": nocaps(res, paper, tables)}
    geometry(res, paper, tables)
    figs["fig3_patch_shuffle"] = zeroshot(res, paper, tables)
    docci(res, paper, tables)
    figs["fig2_efficiency"] = efficiency(res, paper, tables)
    sugarcrepe(res, paper, tables)
    cifar(res, paper, tables)
    for name, fig in figs.items():
        fig.savefig(figures / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {len(list(tables.glob('*.md')))} tables to {tables} and {len(figs)} figures to {figures}")
