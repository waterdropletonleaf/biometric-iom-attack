"""Exploratory data analysis of FVC2004 B for the report (assignment section II).

Produces vector PDFs sized for IEEE columns plus a LaTeX file with one
figure block per plot (title + one-sentence message in each caption, with the
numbers filled in from the data):

  results/eda/overleaf/          upload this: eda_figures.tex (\\begin{figure} blocks)
                                 + figures/*.pdf (\\columnwidth = 3.5in, \\textwidth = 7.16in)
  results/eda/figures/           the same PDFs plus PNG previews (for slides)
  results/eda/image_stats.csv    per-image statistics (table view of the figures)

Needs results/features/deepprint_flx.npz, results/features/nbis_xyt/ and
results/baseline/scores_flx.npz (run extract_deepprint.py and run_baseline.py).

Usage: python src/eda.py
"""
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from config import FEATURES_DIR, NBIS_BIN, RESULTS_DIR
from dataset import load_samples
from nbis_match import minutiae_count

OUT = RESULTS_DIR / "eda"
FIG = OUT / "figures"        # PDFs + PNG previews
UPLOAD = OUT / "overleaf"     # what to upload: eda_figures.tex + figures/*.pdf
COL_W, TEXT_W = 3.5, 6.5  # column width (two-column IEEE) / text width (one-column)
# Single-panel figures in a one-column document are shown at this fraction of
# \textwidth so they are not blown up to full page width.
ONECOL_SINGLE_WIDTH = "0.6\\textwidth"
# "!ht": here if it fits, else top of the next page; text keeps filling the gap.
# Use "H" (needs \usepackage{float}) to pin exactly, at the cost of blank gaps.
PLACEMENT = "!ht"

# Palette: reference categorical slots 1-4 for the DBs (validated on adjacent
# pairs; scatter plots add marker shapes because orange/yellow fail all-pairs).
DB_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DB_MARKERS = ["o", "s", "^", "D"]
DB_LABELS = ["DB1 (optical)", "DB2 (optical)", "DB3 (thermal)", "DB4 (synthetic)"]
GENUINE, IMPOSTOR = "#6250d6", "#9a9893"  # validated pair: dE 25.5, CVD dE 24.1
SINGLE = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FIGURES = []  # (filename, has_wide_variant, title, message) -> LaTeX blocks
HAS_WIDE = set()  # multi-panel figures that also have a side-by-side "_wide" layout


def save(fig, name, title, message, wide=False):
    """`wide=True` saves the side-by-side layout for one-column documents as
    <name>_wide.pdf; the caption is registered once, by the stacked layout."""
    FIG.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_wide" if wide else name
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=200)  # preview only
    plt.close(fig)
    if wide:
        HAS_WIDE.add(name)
    else:
        FIGURES.append((name, False, title, message))


# ---------------------------------------------------------------- statistics

def nfiq_score(img: np.ndarray) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "img.png"
        Image.fromarray(img).save(p, dpi=(500, 500))
        out = subprocess.run([str(NBIS_BIN / "nfiq"), str(p)], capture_output=True,
                             text=True, check=True)
    return int(out.stdout.split()[0])


def print_area(img: np.ndarray, block: int = 16, thresh: float = 12.0) -> float:
    """Share of the frame covered by fingerprint: 16x16 blocks whose intensity
    varies (ridges) rather than flat background (works for DB2's grey background)."""
    h, w = (s // block * block for s in img.shape)
    blocks = img[:h, :w].reshape(h // block, block, w // block, block).std(axis=(1, 3))
    return float((blocks > thresh).mean())


def image_stats(samples, names, emb, subj) -> dict:
    cache = OUT / "image_stats.csv"
    if cache.is_file():
        with cache.open() as f:
            rows = list(csv.DictReader(f))
        if [r["name"] for r in rows] == list(names):
            return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "name"}
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = e @ e.T
    own = np.array([sim[i][(subj == subj[i]) & (np.arange(len(subj)) != i)].mean()
                    for i in range(len(subj))])
    stats = {"nfiq": [], "area": [], "contrast": [], "minutiae": [], "own_sim": own}
    for s in samples:
        img = np.asarray(Image.open(s.path).convert("L"))
        stats["nfiq"].append(nfiq_score(img))
        stats["area"].append(100 * print_area(img))
        stats["contrast"].append(float(img.std()))
        stats["minutiae"].append(minutiae_count(FEATURES_DIR / "nbis_xyt" / f"{s.name}.xyt"))
    stats = {k: np.asarray(v, dtype=float) for k, v in stats.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", *stats])
        for i, n in enumerate(names):
            wr.writerow([n, *(round(float(stats[k][i]), 4) for k in stats)])
    return stats


# ---------------------------------------------------------------- figures

def fig_class_balance(subj, db, pairs, genuine, wide=False):
    i, j = pairs[:, 0], pairs[:, 1]
    n_gen = int(genuine.sum())
    n_within = int((~genuine & (db[i] == db[j])).sum())
    n_cross = int((~genuine & (db[i] != db[j])).sum())
    if wide:
        fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_W, 1.9), gridspec_kw={"width_ratios": [1.5, 1]})
    else:
        fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 2.45), gridspec_kw={"height_ratios": [0.85, 1]})
    counts = np.bincount(subj)
    a.bar(np.arange(40), counts, width=0.8, color=[DB_COLORS[s // 10] for s in range(40)])
    a.set(title="Images per finger", xlabel="Finger (subject id, grouped by DB)",
          ylabel="Images", ylim=(0, 10), xlim=(-1, 40))
    a.set_xticks([4.5, 14.5, 24.5, 34.5], ["DB1", "DB2", "DB3", "DB4"])
    a.grid(axis="x", visible=False)
    labels = ["Genuine", "Impostor, same DB", "Impostor, cross-DB"]
    vals = [n_gen, n_within, n_cross]
    b.barh(labels, vals, color=[GENUINE, IMPOSTOR, IMPOSTOR], height=0.6)
    b.set(title="Comparison pairs by class", xlabel="Pairs", xlim=(0, 47000))
    for y, v in enumerate(vals):
        b.text(v + 700, y, f"{v:,}", va="center", color=INK, fontsize=7)
    b.invert_yaxis()
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    ratio = (n_within + n_cross) / n_gen
    save(fig, "eda_class_balance",
         "Class balance",
         f"Every finger has exactly 8 images (80 per DB), so subjects are perfectly balanced, "
         f"but impostor pairs outnumber genuine pairs {ratio:.0f}:1 ({n_within + n_cross:,} vs "
         f"{n_gen:,}) and {100 * n_cross / (n_within + n_cross):.0f}\\% of impostors are cross-sensor "
         f"pairs, which are much easier to reject than same-sensor ones, so I report per-class error rates (FMR, FNMR, EER) and per-DB "
         f"results rather than accuracy, with no oversampling needed.",
         wide=wide)


def fig_nfiq(st, db):
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    x = np.arange(1, 6)
    w = 0.19
    for d in range(4):
        q = st["nfiq"][db == d]
        pct = [100 * (q == k).mean() for k in x]
        ax.bar(x + (d - 1.5) * w, pct, width=w * 0.88, color=DB_COLORS[d], label=DB_LABELS[d])
    ax.set(title="Fingerprint quality (NFIQ) by DB", xlabel="NFIQ score (1 = best, 5 = worst)",
           ylabel="Images in DB (%)", ylim=(0, 72))
    ax.set_xticks(x)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    good = {d: 100 * (st["nfiq"][db == d] <= 2).mean() for d in range(4)}
    worst = min(good, key=good.get)
    best = max(good, key=good.get)
    save(fig, "eda_nfiq",
         "Image quality by sensor",
         f"Quality differs by sensor: {good[best]:.0f}\\% of {DB_LABELS[best]} images score "
         f"NFIQ 1--2 against {good[worst]:.0f}\\% for {DB_LABELS[worst]}, so error rates should "
         f"be read per DB.")


NFIQ_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, 250-650


def fig_image_stats(st, db, wide=False):
    """Quality (NFIQ, stacked share per DB) plus box plots of image statistics."""
    if wide:
        fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 2.1))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.3))
    axes = axes.ravel()
    ax = axes[0]
    bottom = np.zeros(4)
    for k, c in zip(range(1, 6), NFIQ_RAMP):
        share = np.array([100 * (st["nfiq"][db == d] == k).mean() for d in range(4)])
        ax.bar(np.arange(1, 5), share, bottom=bottom, width=0.6, color=c,
               edgecolor="white", linewidth=0.6, label=str(k))
        bottom += share
    ax.set_xticks([1, 2, 3, 4], ["DB1", "DB2", "DB3", "DB4"])
    ax.set(ylabel="Images in DB (%)", ylim=(0, 100))
    ax.set_title("Quality (NFIQ, 1 = best)", pad=16)
    ax.legend(title=None, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.98),
              handlelength=0.9, handletextpad=0.3, columnspacing=0.6, fontsize=6.5,
              borderaxespad=0.1)
    ax.grid(axis="x", visible=False)
    specs = [("area", "Print area", "Frame covered (%)"),
             ("contrast", "Contrast", "Intensity SD"),
             ("minutiae", "Minutiae (mindtct)", "Minutiae")]
    for ax, (key, title, ylabel) in zip(axes[1:], specs):
        data = [st[key][db == d] for d in range(4)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops={"color": INK, "linewidth": 1},
                        whiskerprops={"color": INK2, "linewidth": 0.7},
                        capprops={"color": INK2, "linewidth": 0.7},
                        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK2,
                                    "markeredgecolor": "none"})
        for patch, c in zip(bp["boxes"], DB_COLORS):
            patch.set(facecolor=c, alpha=0.55, edgecolor=c, linewidth=0.8)
        ax.set_xticks([1, 2, 3, 4], ["DB1", "DB2", "DB3", "DB4"])
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    med = {k: [np.median(st[k][db == d]) for d in range(4)] for k in ("area", "minutiae")}
    good = [100 * (st["nfiq"][db == d] <= 2).mean() for d in range(4)]
    save(fig, "eda_image_stats",
         "Image quality and statistics by DB",
         f"The sensors produce visibly different images: DB2 has the fewest good-quality captures "
         f"(NFIQ 1--2: {good[1]:.0f}\\% vs {min(good[0], good[2], good[3]):.0f}--"
         f"{max(good[0], good[2], good[3]):.0f}\\% elsewhere), DB1 prints cover the least of the frame "
         f"(median {med['area'][0]:.0f}\\% vs {min(med['area'][1:]):.0f}--{max(med['area'][1:]):.0f}\\%), "
         f"and DB3's thermal sensor yields the most minutiae (median {med['minutiae'][2]:.0f} vs "
         f"{min(med['minutiae'][i] for i in (0, 1, 3)):.0f}--{max(med['minutiae'][i] for i in (0, 1, 3)):.0f}). Print area is estimated as the "
         f"share of 16$\\times$16-pixel blocks with ridge texture; contrast is the pixel "
         f"intensity SD.",
         wide=wide)


def fig_feature_values(tex, minu):
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True)
    bins = np.linspace(min(tex.min(), minu.min()), max(tex.max(), minu.max()), 70)
    for ax, x, name in [(axes[0], tex, "Texture half (256-d)"), (axes[1], minu, "Minutiae half (256-d)")]:
        ax.hist(x.ravel(), bins=bins, color=SINGLE, alpha=0.85)
        ax.set(title=f"DeepPrint feature values: {name}", ylabel="Count")
        ax.text(0.98, 0.9, f"mean {x.mean():.3f}, SD {x.std():.3f}", transform=ax.transAxes,
                ha="right", va="top", color=INK, fontsize=7)
    axes[1].set_xlabel("Feature value (each half L2-normalized)")
    fig.tight_layout()
    save(fig, "eda_feature_values",
         "DeepPrint feature distributions",
         f"Both 256-d halves of the DeepPrint vector are symmetric around zero, with "
         f"{100 * (np.concatenate([tex, minu]) < 0).mean():.0f}\\% of entries negative and no "
         f"extreme values; the shared SD of {tex.std():.3f} is fixed by each half being "
         f"normalized to unit length ($1/\\sqrt{{256}}$), not a property of the data.")


def fig_own_similarity(st, names):
    own = st["own_sim"]
    cut = own.mean() - 3 * own.std()
    n_out = int((own < cut).sum())
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    ax.hist(own, bins=40, color=SINGLE, alpha=0.85)
    ax.axvline(cut, color=INK, ls="--", lw=0.8)
    ax.text(cut - 0.01, ax.get_ylim()[1] * 0.9, f"mean $-$ 3 SD = {cut:.3f}\n{n_out} outliers",
            ha="right", va="top", fontsize=7, color=INK)
    ax.set(title="Similarity of each image to its own finger",
           xlabel="Mean DeepPrint cosine similarity to the other 7 impressions", ylabel="Images")
    fig.tight_layout()
    save(fig, "eda_own_similarity",
         "Per-image similarity to its own finger",
         f"Most impressions resemble the rest of their finger strongly (mean {own.mean():.3f}, "
         f"SD {own.std():.3f}); on visual inspection, the {n_out} below the 3-SD cutoff are "
         f"displaced, rotated, or over- or under-pressed captures rather than labelling errors, "
         f"so they were kept.")


VAR_NAMES = {"nfiq": "NFIQ (1 = best)", "area": "Print area", "contrast": "Contrast",
             "minutiae": "Minutiae", "own_sim": "Own-finger sim."}


def fig_correlation(st, db):
    """Spearman correlations computed within each DB and averaged. Pooling the four
    sensors mostly measures differences between sensors (Simpson's paradox): e.g.
    DB3 is both high-contrast and easy to match, which inflates the pooled
    contrast-similarity correlation."""
    keys = list(VAR_NAMES)
    within = np.mean([[[spearmanr(st[a][db == d], st[b][db == d])[0] for b in keys] for a in keys]
                      for d in range(4)], axis=0)
    pooled = np.array([[spearmanr(st[a], st[b])[0] for b in keys] for a in keys])
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    im = ax.imshow(within, cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [VAR_NAMES[k] for k in keys]
    ax.set_xticks(range(len(keys)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(keys)), labels)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{within[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(within[i, j]) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman $\\rho$ (mean within DB)")
    cb.outline.set_visible(False)
    ax.set_title("Correlation of per-image statistics, within DB")
    fig.tight_layout()
    o = keys.index("own_sim")
    own = {k: within[o, keys.index(k)] for k in keys if k != "own_sim"}
    strongest = max(own, key=lambda k: abs(own[k]))
    c, m = keys.index("contrast"), keys.index("minutiae")
    save(fig, "eda_correlation",
         "Correlation of image statistics",
         f"Within a sensor, no image statistic predicts well how closely an image matches its "
         f"own finger (all $|\\rho| \\le {max(abs(v) for v in own.values()):.2f}$; strongest: "
         f"{VAR_NAMES[strongest].split(' (')[0]}, $\\rho = {own[strongest]:.2f}$), while contrast "
         f"and minutiae count move together ($\\rho = {within[c, m]:.2f}$); correlations are "
         f"averaged within DBs because pooling the four sensors would suggest a misleading "
         f"contrast link ($\\rho = {pooled[o, c]:.2f}$) that only reflects DB3 being both "
         f"high-contrast and easy to match (Spearman, as NFIQ is ordinal).")


def fig_score_distributions(scores, genuine, wide=False):
    if wide:
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.0))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.35))
    specs = [("DeepPrint-512", "DeepPrint", "Cosine similarity", np.linspace(-0.3, 1, 70)),
             ("NBIS-Bozorth3", "NBIS Bozorth3", "Bozorth3 match score",
              np.arange(0, scores["NBIS-Bozorth3"].max() + 6, 5))]
    for ax, (key, title, xlabel, bins) in zip(axes, specs):
        s = scores[key]
        ax.hist(s[~genuine], bins=bins, density=True, color=IMPOSTOR, alpha=0.75, label="Impostor")
        ax.hist(s[genuine], bins=bins, density=True, color=GENUINE, alpha=0.75, label="Genuine")
        ax.set(title=f"Genuine vs impostor scores: {title}", xlabel=xlabel,
               ylabel="Density (per class)")
        ax.legend(loc="upper center" if key == "DeepPrint-512" else "upper right")
    fig.tight_layout()
    g, imp = scores["NBIS-Bozorth3"][genuine], scores["NBIS-Bozorth3"][~genuine]
    save(fig, "eda_score_distributions",
         "Score distributions by class",
         f"Both matchers separate the classes, but differently: DeepPrint's impostor scores form a "
         f"broad band that overlaps the genuine tail, while Bozorth3 impostors cluster near zero "
         f"(median {np.median(imp):.0f} vs {np.median(g):.0f} for genuine) with a long genuine tail; "
         f"each class is normalized separately, so the 45:1 impostor majority (Fig.~\\ref{{fig:eda-class-balance}}) is not "
         f"visible here.",
         wide=wide)


def fig_matcher_agreement(scores, genuine):
    dp, bz = scores["DeepPrint-512"], scores["NBIS-Bozorth3"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for mask, c, lab, s in [(~genuine, IMPOSTOR, "Impostor", 2), (genuine, GENUINE, "Genuine", 5)]:
        ax.scatter(dp[mask], bz[mask] + 1, s=s, color=c, alpha=0.5, linewidths=0,
                   label=lab, rasterized=True)
    ax.set_yscale("log")
    ax.set(title="DeepPrint vs NBIS score per pair", xlabel="DeepPrint cosine similarity",
           ylabel="Bozorth3 score + 1 (log)")
    ax.legend(loc="upper left", markerscale=3)
    fig.tight_layout()
    rho = spearmanr(dp[genuine], bz[genuine])[0]
    save(fig, "eda_matcher_agreement",
         "Agreement between matchers",
         f"The two matchers agree only loosely on genuine pairs (Spearman $\\rho = {rho:.2f}$), "
         f"showing that the learned and minutiae-based representations capture different "
         f"information, which is why both are used to judge reconstructions later.")


def tsne_2d(emb):
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return TSNE(n_components=2, perplexity=15, metric="cosine", init="pca",
                random_state=0).fit_transform(e)


def fig_tsne(emb, xy, subj, db, wide=False):
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if wide:
        fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 1.9), sharex=True, sharey=True,
                                 squeeze=False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.5), sharex=True, sharey=True)
    for d, ax in enumerate(axes.ravel()):
        for s in range(d * 10, d * 10 + 10):
            m = subj == s
            c = xy[m].mean(axis=0)
            for p in xy[m]:
                ax.plot([c[0], p[0]], [c[1], p[1]], color="#b5b4ae", lw=0.6, zorder=2)
        others = db != d
        ax.scatter(xy[others, 0], xy[others, 1], s=2, color="#d6d5d0", linewidths=0, zorder=1)
        ax.scatter(xy[db == d, 0], xy[db == d, 1], s=9, marker=DB_MARKERS[d], color=DB_COLORS[d],
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_title(DB_LABELS[d])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes[-1]:
        ax.set_xlabel("t-SNE 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("t-SNE 2")
    fig.suptitle("DeepPrint features (t-SNE), linked to finger centre",
                 fontsize=8.5, fontweight="bold", y=0.995)
    if wide:
        fig.subplots_adjust(left=0.03, right=0.995, bottom=0.1, top=0.78, wspace=0.06)
    else:
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.88, wspace=0.06, hspace=0.18)
    sim = e @ e.T
    np.fill_diagonal(sim, -9)
    nn = (subj[sim.argmax(1)] == subj).mean()
    save(fig, "eda_tsne",
         "Feature space by finger",
         f"Impressions of the same finger form distinct clusters (short links to each centre), "
         f"tightest in DB1 and looser in DB3--DB4, with a few long links from outlier "
         f"impressions; {100 * nn:.1f}\\% of images have a same-finger nearest neighbour, "
         f"confirming that the labels are consistent and the features are discriminative.",
         wide=wide)


# ---------------------------------------------------------------- LaTeX

# Figures included in the report (3-4 page limit). The others are still written
# to results/eda/figures/ as extras (eda_nfiq, eda_feature_values,
# eda_matcher_agreement, eda_tsne).
SECTIONS = [
    ("Univariate analysis", ["eda_image_stats", "eda_own_similarity"]),
    ("Bivariate analysis", ["eda_correlation", "eda_score_distributions"]),
    ("Class balance", ["eda_class_balance"]),
]
REPORT_FIGURES = [n for _, names in SECTIONS for n in names]


def write_latex():
    lines = ["% Generated by src/eda.py. Preamble needs \\usepackage{graphicx}.",
             "% Upload figures/*.pdf to your Overleaf project's figures/ folder.",
             "%",
             "% Works in one- and two-column documents: in two-column, every figure is one",
             "% column wide (multi-panel figures stacked); in one-column, multi-panel figures",
             f"% use side-by-side layouts at full width and single plots are {ONECOL_SINGLE_WIDTH}.",
             f"% Placement [{PLACEMENT}]: here if it fits, otherwise top of the next page. A float",
             "% barrier at the end keeps every figure before the next \\section; add",
             "% \\usepackage{placeins} to the preamble so it does not force a page break.",
             "\\makeatletter",
             "\\providecommand{\\edafig}[2]{%  #1 = file stem, #2 = has a side-by-side layout (y/n)",
             "  \\if@twocolumn\\includegraphics[width=\\columnwidth]{figures/#1.pdf}%",
             "  \\else\\if y#2\\includegraphics[width=\\textwidth]{figures/#1_wide.pdf}%",
             f"  \\else\\includegraphics[width={ONECOL_SINGLE_WIDTH}]{{figures/#1.pdf}}\\fi\\fi}}",
             "% Pack figures tightly: let text pages hold more figures, and stack any",
             "% figure-only page from the top instead of spreading figures down the page.",
             "\\renewcommand{\\topfraction}{0.9}\\renewcommand{\\textfraction}{0.1}",
             "\\renewcommand{\\floatpagefraction}{0.8}",
             "\\setcounter{topnumber}{3}\\setcounter{totalnumber}{4}",
             "\\setlength{\\@fptop}{0pt}\\setlength{\\@fpsep}{8pt plus 2pt}\\setlength{\\@fpbot}{0pt plus 1fil}",
             "\\makeatother", ""]
    by_name = {f[0]: f for f in FIGURES}
    ordered = []
    for heading, names in SECTIONS:
        ordered.append(heading)
        ordered += [by_name[n] for n in names]
    for item in ordered:
        if isinstance(item, str):
            lines += ["% " + "-" * 60, f"% {item}", "% " + "-" * 60, ""]
            continue
        name, wide, title, msg = item
        env = "figure"
        lines += [f"\\begin{{{env}}}[{PLACEMENT}]", "  \\centering",
                  f"  \\edafig{{{name}}}{{{'y' if name in HAS_WIDE else 'n'}}}",
                  f"  \\caption{{\\textbf{{{title}.}} {msg}}}",
                  f"  \\label{{fig:{name.replace('eda_', 'eda-').replace('_', '-')}}}",
                  f"\\end{{{env}}}", ""]
    # Float barrier: every figure above is output before the next section starts.
    # Uses \FloatBarrier (\usepackage{placeins}) if loaded, otherwise \clearpage.
    lines += ["% Keep all EDA figures inside this section: flush them before the next \\section.",
              "\\makeatletter",
              "\\@ifundefined{FloatBarrier}{\\clearpage}{\\FloatBarrier}",
              "\\makeatother", ""]
    UPLOAD.mkdir(parents=True, exist_ok=True)
    (UPLOAD / "eda_figures.tex").write_text("\n".join(lines))


def main():
    samples = load_samples()
    f = np.load(FEATURES_DIR / "deepprint_flx.npz")
    subj, db, names = f["subject"], f["db"], f["name"]
    assert list(names) == [s.name for s in samples]
    sc = np.load(RESULTS_DIR / "baseline" / "scores_flx.npz")
    scores = {k: sc[k] for k in ("DeepPrint-512", "NBIS-Bozorth3")}
    pairs, genuine = sc["pairs"], sc["genuine"]

    print("Computing per-image statistics (NFIQ, area, contrast, minutiae) ...", flush=True)
    st = image_stats(samples, names, f["embedding"], subj)

    fig_class_balance(subj, db, pairs, genuine)
    fig_class_balance(subj, db, pairs, genuine, wide=True)
    fig_nfiq(st, db)
    fig_image_stats(st, db)
    fig_image_stats(st, db, wide=True)
    fig_feature_values(f["texture"], f["minutia"])
    fig_own_similarity(st, names)
    fig_correlation(st, db)
    fig_score_distributions(scores, genuine)
    fig_score_distributions(scores, genuine, wide=True)
    fig_matcher_agreement(scores, genuine)
    print("Running t-SNE ...", flush=True)
    xy = tsne_2d(f["embedding"])
    fig_tsne(f["embedding"], xy, subj, db)
    fig_tsne(f["embedding"], xy, subj, db, wide=True)
    # Overleaf upload folder: the .tex plus the PDFs only (no PNG previews).
    shutil.rmtree(UPLOAD, ignore_errors=True)
    (UPLOAD / "figures").mkdir(parents=True)
    for name in REPORT_FIGURES:
        shutil.copy(FIG / f"{name}.pdf", UPLOAD / "figures")
        if name in HAS_WIDE:
            shutil.copy(FIG / f"{name}_wide.pdf", UPLOAD / "figures")
    write_latex()
    print(f"Wrote {len(FIGURES)} figures ({len(REPORT_FIGURES)} in the report); "
          f"Overleaf upload folder: {UPLOAD}")


if __name__ == "__main__":
    main()
