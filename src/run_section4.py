"""Report Section IV: baseline performance with counts, threshold analysis,
error analysis, and the seeds behind every number.

Reads saved scores (no recomputation), so run the pipeline first:
  python src/extract_deepprint.py && python src/run_baseline.py
  python src/run_iom.py && python src/eda.py      # eda.py writes image_stats.csv
Then:
  python src/run_section4.py

Writes results/section4/:
  summary.json                  every rate with its counts, thresholds, error analysis
  overleaf/section4.tex         two tables + one figure, ready to \\input
  overleaf/figures/s4_threshold.pdf   FMR and FNMR against threshold
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR
from eda import COL_W, GRID, INK, INK2  # shared figure style (rcParams set on import)
from metrics import eer, fnmr_at_fmr, roc, save_json

OUT = RESULTS_DIR / "section4"
UPLOAD = OUT / "overleaf"
FMR_COLOR, FNMR_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1-2 (adjacent pair, validated)

# The chosen operating point (see choose_operating_point docstring).
OPERATING_FMR = 1e-2
SYSTEMS = [  # (label, file, array key)
    ("DeepPrint (unprotected)", "baseline/scores_flx.npz", "DeepPrint-512"),
    ("NBIS Bozorth3", "baseline/scores_flx.npz", "NBIS-Bozorth3"),
    ("GRP-IoM (key 0)", "iom/scores_grp.npz", "stolen_key0"),
    ("URP-IoM (key 0)", "iom/scores_urp.npz", "stolen_key0"),
]
# Every source of randomness in the pipeline, for the reproducibility statement.
SEEDS = {
    "IoM shared (stolen-token) keys": "torch.Generator seeds 0-4 (key 0 = attack target)",
    "IoM per-finger keys (genuine-token)": "10000 + subject id",
    "IoM reissued keys (revocability)": "20000 + subject id",
    "Unlinkability noise floor": "numpy default_rng(0), 20 repeats",
    "EDA t-SNE / scatter jitter": "random_state=0 / default_rng(0)",
    "DeepPrint inference": "deterministic (eval mode, no sampling); re-run is bit-identical",
    "NBIS mindtct / Bozorth3": "deterministic",
}


def counts_at(s, genuine, t):
    fm = int((s[~genuine] >= t).sum())
    fnm = int((s[genuine] < t).sum())
    n_imp, n_gen = int((~genuine).sum()), int(genuine.sum())
    return {"threshold": float(t), "fm": fm, "n_impostor": n_imp, "fmr": fm / n_imp,
            "fnm": fnm, "n_genuine": n_gen, "fnmr": fnm / n_gen}


def operating_points(s, genuine):
    _, t_eer = eer(s[genuine], s[~genuine])
    _, t1 = fnmr_at_fmr(s[genuine], s[~genuine], 1e-2)
    _, t01 = fnmr_at_fmr(s[genuine], s[~genuine], 1e-3)
    return {"EER": counts_at(s, genuine, t_eer),
            "FMR 1%": counts_at(s, genuine, t1),
            "FMR 0.1%": counts_at(s, genuine, t01)}


def error_analysis(scores, genuine, pairs):
    """Where the baseline fails at the operating point (DeepPrint and NBIS)."""
    f = np.load(RESULTS_DIR / "features" / "deepprint_flx.npz")
    db, subj, names = f["db"], f["subject"], f["name"]
    with (RESULTS_DIR / "eda" / "image_stats.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["name"] for r in rows] == list(names)
    nfiq = np.array([float(r["nfiq"]) for r in rows])
    own = np.array([float(r["own_sim"]) for r in rows])
    outlier = own < own.mean() - 3 * own.std()
    i, j = pairs[:, 0], pairs[:, 1]
    worse_q = np.maximum(nfiq[i], nfiq[j])

    res, fnm_sets = {}, {}
    for label, key in [("DeepPrint", "DeepPrint (unprotected)"), ("NBIS", "NBIS Bozorth3")]:
        s = scores[key]
        _, t = fnmr_at_fmr(s[genuine], s[~genuine], OPERATING_FMR)
        fnm, fm = genuine & (s < t), ~genuine & (s >= t)
        fnm_sets[label] = fnm
        same_db = db[i] == db[j]
        r = {"threshold": float(t),
             "fnm_by_db": {f"DB{d + 1}": [int((fnm & (db[i] == d)).sum()),
                                          int((genuine & (db[i] == d)).sum())] for d in range(4)},
             "fm_within_db": {f"DB{d + 1}": [int((fm & same_db & (db[i] == d)).sum()),
                                             int((~genuine & same_db & (db[i] == d)).sum())]
                              for d in range(4)},
             "fm_cross_db": [int((fm & ~same_db).sum()), int((~genuine & ~same_db).sum())],
             "fnm_by_worse_nfiq": {str(k): [int((fnm & (worse_q == k)).sum()),
                                            int((genuine & (worse_q == k)).sum())]
                                   for k in range(1, 6) if (genuine & (worse_q == k)).any()},
             "fnm_with_outlier_image": [int((fnm & (outlier[i] | outlier[j])).sum()),
                                        int((genuine & (outlier[i] | outlier[j])).sum())],
             "fnm_without_outlier": [int((fnm & ~(outlier[i] | outlier[j])).sum()),
                                     int((genuine & ~(outlier[i] | outlier[j])).sum())]}
        fm_fingers = np.bincount(np.concatenate([subj[i[fm]], subj[j[fm]]]), minlength=40)
        r["fingers_in_most_fm"] = [[f"db{s_ // 10 + 1}_{101 + s_ % 10}", int(fm_fingers[s_])]
                                   for s_ in np.argsort(-fm_fingers)[:3]]
        res[label] = r
    d, n = fnm_sets["DeepPrint"], fnm_sets["NBIS"]
    res["fnm_overlap"] = {"both": int((d & n).sum()), "DeepPrint_only": int((d & ~n).sum()),
                          "NBIS_only": int((n & ~d).sum())}
    # IoM hashes DeepPrint features: how many of its errors are DeepPrint's errors?
    s_dp = scores["DeepPrint (unprotected)"]
    _, t_dp = fnmr_at_fmr(s_dp[genuine], s_dp[~genuine], OPERATING_FMR)
    dp_fnm, dp_fm = genuine & (s_dp < t_dp), ~genuine & (s_dp >= t_dp)
    for label, key in [("GRP-IoM", "GRP-IoM (key 0)"), ("URP-IoM", "URP-IoM (key 0)")]:
        s = scores[key]
        _, t = fnmr_at_fmr(s[genuine], s[~genuine], OPERATING_FMR)
        fnm, fm = genuine & (s < t), ~genuine & (s >= t)
        res[label] = {"threshold": float(t),
                      "fnm_by_db": {f"DB{k + 1}": int((fnm & (db[i] == k)).sum()) for k in range(4)},
                      "fnm_shared_with_deepprint": [int((fnm & dp_fnm).sum()), int(fnm.sum())],
                      "fm_shared_with_deepprint": [int((fm & dp_fm).sum()), int(fm.sum())]}
    return res


def fig_threshold(scores, genuine, ops):
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.1))
    xlabels = ["Cosine similarity", "Bozorth3 score", "Matching entries (of 300)",
               "Matching entries (of 600)"]
    titles = ["DeepPrint", "NBIS Bozorth3", "GRP-IoM", "URP-IoM"]
    for ax, (label, *_), xl, title in zip(axes.ravel(), SYSTEMS, xlabels, titles):
        s = scores[label]
        t, fmr, fnmr = roc(s[genuine], s[~genuine])
        ax.plot(t, 100 * np.maximum(fmr, 1e-4), color=FMR_COLOR, lw=1.2, label="FMR")
        ax.plot(t, 100 * np.maximum(fnmr, 1e-4), color=FNMR_COLOR, lw=1.2, label="FNMR")
        e = ops[label]["EER"]
        ax.plot(e["threshold"], 100 * (e["fmr"] + e["fnmr"]) / 2, "o", color=INK, ms=3.5, zorder=4)
        ax.axvline(ops[label]["FMR 1%"]["threshold"], color=INK2, ls="--", lw=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 100)
        ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.minorticks_off()
        if label.startswith("NBIS"):
            ax.set_xlim(0, 60)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xl, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(color=GRID)
    for ax in axes[:, 0]:
        ax.set_ylabel("Error rate (%)", fontsize=7)
    handles = [plt.Line2D([], [], color=FMR_COLOR, lw=1.2, label="FMR (false match rate)"),
               plt.Line2D([], [], color=FNMR_COLOR, lw=1.2, label="FNMR (false non-match rate)"),
               plt.Line2D([], [], color=INK, marker="o", ls="none", ms=3.5, label="EER"),
               plt.Line2D([], [], color=INK2, ls="--", lw=0.7, label="Operating point (FMR ≤ 1%)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01),
               fontsize=6.5, columnspacing=1.0)
    fig.suptitle("FMR and FNMR against decision threshold", fontsize=8.5, fontweight="bold",
                 y=0.995)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.23, hspace=1.0, wspace=0.28)
    (UPLOAD / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(UPLOAD / "figures" / "s4_threshold.pdf")
    fig.savefig(OUT / "s4_threshold.png", dpi=200)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.2f}"


def write_latex(ops, err):
    L = ["% Generated by src/run_section4.py -- every number comes from results/ at this commit.",
         "% Needs \\usepackage{graphicx}. Upload figures/s4_threshold.pdf to figures/.", ""]
    # Table: rates with counts
    L += ["\\begin{table}[!htb]", "\\centering",
          "\\caption{Verification performance with counts (1,120 genuine and 49,920 impostor "
          "pairs; IoM uses shared key 0). Rates are FM/49,920 and FNM/1,120. Each EER row "
          "gives the EER and the two rates at the threshold where FMR and FNMR are closest; "
          "the EER is their mean. "
          "Bozorth3 scores are integers, so its FMR cannot hit 1\\% or 0.1\\% exactly.}",
          "\\label{tab:s4-counts}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
          "\\begin{tabular}{llrrr}", "\\hline",
          "System & Point & Thr. & FMR \\% (FM) & FNMR \\% (FNM) \\\\", "\\hline"]
    short = {"DeepPrint (unprotected)": "DeepPrint", "NBIS Bozorth3": "NBIS",
             "GRP-IoM (key 0)": "GRP-IoM", "URP-IoM (key 0)": "URP-IoM"}
    for label, *_ in SYSTEMS:
        for k, (pt, c) in enumerate(ops[label].items()):
            thr = f"{c['threshold']:.3f}" if c["threshold"] < 2 else f"{c['threshold']:.0f}"
            name = short[label] if k == 0 else ""
            pt_tex = pt.replace("%", "\\%")
            if pt == "EER":
                pt_tex = f"EER = {pct((c['fmr'] + c['fnmr']) / 2)}\\%"
            if pt == f"FMR {OPERATING_FMR * 100:g}%":
                pt_tex = f"\\textbf{{{pt_tex}}}"
            L.append(f"{name} & {pt_tex} & {thr} & {pct(c['fmr'])} ({c['fm']:,}) & "
                     f"{pct(c['fnmr'])} ({c['fnm']:,}) \\\\")
        L.append("\\hline")
    L += ["\\end{tabular}", "\\end{table}", ""]
    # Figure
    L += ["\\begin{figure}[!ht]", "  \\centering",
          "  \\includegraphics[width=\\columnwidth]{figures/s4_threshold.pdf}",
          "  \\caption{\\textbf{Threshold analysis.} FMR falls and FNMR rises as the threshold "
          "increases; the curves cross at the EER (dot). The dashed line marks the chosen "
          "operating point, FMR $\\le$ 1\\%, for every system (0.69\\% for Bozorth3, whose "
          "scores are integers). The Bozorth3 axis is cut at 60; its FNMR reaches 100\\% at "
          "the maximum score of 363.}",
          "  \\label{fig:s4-threshold}", "\\end{figure}", ""]
    # Table: error analysis
    dp, nb = err["DeepPrint"], err["NBIS"]

    def cell(v):
        return f"{v[0]} ({100 * v[0] / v[1]:.1f})"
    L += ["\\begin{table}[!htb]", "\\centering",
          "\\caption{Where the baselines fail at FMR $\\le$ 1\\%: false non-matches per DB (of 280 "
          "genuine pairs) and false matches between different fingers of the same DB (of "
          "2,880 pairs; cross-DB: of 38,400). Values: count (\\%).}",
          "\\label{tab:s4-errors}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
          "\\begin{tabular}{lcccc}", "\\hline",
          " & \\multicolumn{2}{c}{False non-matches} & \\multicolumn{2}{c}{False matches} \\\\",
          "DB & DeepPrint & NBIS & DeepPrint & NBIS \\\\", "\\hline"]
    for d in ("DB1", "DB2", "DB3", "DB4"):
        L.append(f"{d} & {cell(dp['fnm_by_db'][d])} & {cell(nb['fnm_by_db'][d])} & "
                 f"{cell(dp['fm_within_db'][d])} & {cell(nb['fm_within_db'][d])} \\\\")
    L.append(f"Cross-DB & -- & -- & {cell(dp['fm_cross_db'])} & {cell(nb['fm_cross_db'])} \\\\")
    L += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    UPLOAD.mkdir(parents=True, exist_ok=True)
    (UPLOAD / "section4.tex").write_text("\n".join(L))


def main():
    scores, genuine, pairs = {}, None, None
    for label, path, key in SYSTEMS:
        z = np.load(RESULTS_DIR / path)
        if pairs is None:
            pairs, genuine = z["pairs"], z["genuine"]
        assert (z["pairs"] == pairs).all() and (z["genuine"] == genuine).all()
        scores[label] = z[key].astype(float)
    ops = {label: operating_points(scores[label], genuine) for label, *_ in SYSTEMS}
    err = error_analysis(scores, genuine, pairs)
    fig_threshold(scores, genuine, ops)
    write_latex(ops, err)
    save_json({"operating_fmr": OPERATING_FMR, "operating_points": ops,
               "error_analysis": err, "seeds": SEEDS}, OUT / "summary.json")

    for label, *_ in SYSTEMS:
        print(label)
        for pt, c in ops[label].items():
            print(f"  {pt:<9} thr {c['threshold']:>8.3f}  FMR {pct(c['fmr'])}% ({c['fm']}/{c['n_impostor']})"
                  f"  FNMR {pct(c['fnmr'])}% ({c['fnm']}/{c['n_genuine']})")
    print("FNM overlap at operating point:", err["fnm_overlap"])
    for k in ("GRP-IoM", "URP-IoM"):
        e = err[k]
        print(f"{k}: FNM shared with DeepPrint {e['fnm_shared_with_deepprint']}, "
              f"FM shared {e['fm_shared_with_deepprint']}, FNM by DB {e['fnm_by_db']}")
    print(f"Wrote {OUT / 'summary.json'} and {UPLOAD}")


if __name__ == "__main__":
    main()
