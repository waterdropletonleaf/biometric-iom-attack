"""Checkpoint 2, Section I: system + attack architecture diagram.

Draws the specific pipeline of this project (not a generic flowchart): the
enrolment/verification path, and the attack lane marking where the attack
enters and what the attacker controls. Vector PDF for LaTeX + a PNG preview.

Usage: python src/make_arch_diagram.py   ->  results/figures/architecture.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import RESULTS_DIR

OUT = RESULTS_DIR / "figures"

SYS = "#2a78d6"       # system pipeline
SYS_FILL = "#eaf1fb"
ATK = "#c0392b"       # attack lane
ATK_FILL = "#fbecea"
FUT = "#8a8a8a"       # future work (reconstruction)
FUT_FILL = "#f0f0f0"
INK = "#111111"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def box(ax, x, y, w, h, title, body, ec, fc, title_color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", weight="bold",
            fontsize=7.6, color=title_color or ec, zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=6.9,
                color=INK, zorder=3, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", dashed=False, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color, zorder=1,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(7.16, 4.3))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---- row labels ----
    ax.text(0.1, 8.55, "Enrolment / verification pipeline", fontsize=9, weight="bold", color=SYS)
    ax.text(0.1, 3.95, "Pre-image attack (this project)", fontsize=9, weight="bold", color=ATK)

    w, h, y = 2.86, 1.95, 6.3
    xs = [0.15, 3.16, 6.17, 9.18, 12.19]

    box(ax, xs[0], y, w, h, "Input",
        "FVC2004 Set B\nfingerprint (TIFF)\noptical / thermal /\nsynthetic sensor", SYS, SYS_FILL)
    box(ax, xs[1], y, w, h, "Preprocessing",
        "1. pad to square\n2. resize 299×299\n3. Gabor binarize\n(ridge width 1.8)", SYS, SYS_FILL)
    box(ax, xs[2], y, w, h, "Feature extraction",
        "DeepPrint CNN →\n512-d embedding\n(256 texture +\n256 minutiae)", SYS, SYS_FILL)
    box(ax, xs[3], y, w, h, "Template generation",
        "IoM hashing + key\nGRP m=300, q=16\nURP m=600,w=100,p=2\n→ argmax code", SYS, SYS_FILL)
    box(ax, xs[4], y, w, h, "Storage + matcher",
        "store IoM code + key;\nscore = # matching\nentries ≥ τ →\naccept / reject", SYS, SYS_FILL)

    for a, b in zip(xs[:4], xs[1:]):
        arrow(ax, a + w, y + h / 2, b, y + h / 2, SYS)

    # ---- attack entry: down from Storage into the solver (rightmost attack box) ----
    xentry = xs[4] + w / 2
    ax.text(xentry - 1.9, 4.95, "attacker steals\ntemplate + key\n(stolen-token model)",
            ha="right", va="center", fontsize=6.8, color=ATK, style="italic")
    arrow(ax, xentry, y, xentry, 3.05, ATK, lw=1.8)

    # ---- attack lane: flows right -> left (solver produces feature -> eval -> future) ----
    ya = 1.1
    box(ax, xs[4], ya, w, h, "Pre-image attack",
        "Newton solver on\nLSP constraints\nG(x)=LSP−|LSP|\n≤10 restarts", ATK, ATK_FILL)
    box(ax, 8.55, ya, 3.0, h, "Recovered feature",
        "512-d pre-image\npurity ≈ 0.835\n(cos to true feat.)", ATK, ATK_FILL)
    box(ax, 4.6, ya, 3.6, h, "Reconstruction",
        "feature inversion →\nimage; re-extract →\ncos 0.86 to true feat.\n(optimisation-based)", ATK, ATK_FILL)
    box(ax, 0.15, ya, 4.1, h, "Image-level attack success",
        "Type-1 (same sys) 100%\nType-2 (unprot.) 19–70%\nfuture: generative decoder\n(realistic img, transfer)", ATK, ATK_FILL)

    arrow(ax, xs[4], ya + h / 2, 11.55, ya + h / 2, ATK)        # solver -> recovered
    arrow(ax, 8.55, ya + h / 2, 8.20, ya + h / 2, ATK)          # recovered -> reconstruction
    arrow(ax, 4.6, ya + h / 2, 4.25, ya + h / 2, ATK)           # reconstruction -> eval

    # legend
    ax.text(0.15, 0.32, "Blue: system    Red: attack (implemented)",
            fontsize=7, color=INK)
    ax.text(15.05, 0.32, "τ = operating threshold (FMR ≤ 1%)", fontsize=6.8,
            color=INK, ha="right", style="italic")

    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "architecture.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "architecture.png", dpi=200)
    plt.close(fig)

    # Overleaf upload folder + .tex block, matching eda/section4 (full-width figure*)
    upload = OUT / "overleaf"
    (upload / "figures").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(OUT / "architecture.pdf", upload / "figures" / "architecture.pdf")
    (upload / "architecture.tex").write_text(
        "% Generated by src/make_arch_diagram.py. Needs \\usepackage{graphicx}.\n"
        "% Upload figures/architecture.pdf to your Overleaf project's figures/ folder.\n"
        "% Two-column template: figure* spans both columns. Single-column: figure + \\columnwidth.\n"
        "\\begin{figure*}[!t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{figures/architecture.pdf}\n"
        "  \\caption{\\textbf{System and attack architecture.} Top: the enrolment/verification "
        "pipeline (FVC2004 image $\\rightarrow$ preprocessing $\\rightarrow$ DeepPrint feature "
        "$\\rightarrow$ IoM template with key $\\rightarrow$ stored template and matcher). Bottom: "
        "the pre-image attack, entering at the stored template and key under the stolen-token "
        "model, running template\\,+\\,key $\\rightarrow$ Newton solver $\\rightarrow$ recovered "
        "feature $\\rightarrow$ reconstructed image $\\rightarrow$ image-level Type-1/Type-2.}\n"
        "  \\label{fig:architecture}\n"
        "\\end{figure*}\n")
    print(f"wrote {OUT/'architecture.pdf'}, .png, and {upload/'architecture.tex'}")


if __name__ == "__main__":
    main()
