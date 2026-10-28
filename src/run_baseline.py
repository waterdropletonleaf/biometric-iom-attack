"""Week 2: unprotected baseline on FVC2004 B (320 samples, 40 fingers).

Scores every pair with
  - DeepPrint: cosine similarity of the 512-d (texture || minutia) embedding,
    plus each 256-d branch on its own;
  - NBIS: Bozorth3 on mindtct minutiae.
Writes results/baseline/{summary.json, scores_*.npz, *.png}.

Usage: python src/run_baseline.py [--prep scaled|flx] [--skip-nbis]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from config import FEATURES_DIR, RESULTS_DIR, nbis_available
from dataset import load_samples
from metrics import evaluate, pair_indices, roc, save_json
from nbis_match import bozorth3_pairs, extract_all, minutiae_count

OUT = RESULTS_DIR / "baseline"


def cosine_scores(x: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return np.einsum("nd,nd->n", x[pairs[:, 0]], x[pairs[:, 1]])


def plot(results: dict, scores: dict, genuine_mask: np.ndarray, tag: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, s in scores.items():
        _, fmr, fnmr = roc(s[genuine_mask], s[~genuine_mask])
        ok = (fmr > 0) & (fnmr > 0) & (fmr < 1) & (fnmr < 1)
        ax.plot(norm.ppf(fmr[ok]), norm.ppf(fnmr[ok]),
                label=f"{name} (EER {100 * results[name]['all']['eer']:.2f}%)")
    ticks = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5])
    ax.set_xticks(norm.ppf(ticks), [f"{100 * t:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{100 * t:g}" for t in ticks])
    ax.set_xlim(norm.ppf(5e-5), norm.ppf(0.5))
    ax.set_ylim(norm.ppf(5e-4), norm.ppf(0.6))
    ax.set_xlabel("FMR (%)")
    ax.set_ylabel("FNMR (%)")
    ax.set_title("FVC2004 B, unprotected baseline (all pairs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"det_{tag}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(scores), figsize=(4.5 * len(scores), 3.5))
    for ax, (name, s) in zip(np.atleast_1d(axes), scores.items()):
        bins = np.linspace(s.min(), s.max(), 60)
        ax.hist(s[~genuine_mask], bins=bins, density=True, alpha=0.6, label="impostor")
        ax.hist(s[genuine_mask], bins=bins, density=True, alpha=0.6, label="genuine")
        ax.axvline(results[name]["all"]["eer_threshold"], color="k", ls="--", lw=1)
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"scores_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", choices=["scaled", "flx"], default="flx")
    ap.add_argument("--skip-nbis", action="store_true")
    args = ap.parse_args()

    feats = np.load(FEATURES_DIR / f"deepprint_{args.prep}.npz")
    samples = load_samples()
    subj, dbs, imp = feats["subject"], feats["db"], feats["impression"]
    assert (subj == [s.subject for s in samples]).all()

    gen, impo = pair_indices(subj, dbs, imp, "all")
    pairs = np.concatenate([gen, impo])
    genuine_mask = np.zeros(len(pairs), bool)
    genuine_mask[: len(gen)] = True

    scores = {
        "DeepPrint-512": cosine_scores(feats["embedding"], pairs),
        "DeepPrint-tex256": cosine_scores(feats["texture"], pairs),
        "DeepPrint-minu256": cosine_scores(feats["minutia"], pairs),
    }

    xyt_dir = RESULTS_DIR / "features" / "nbis_xyt"
    xyts = extract_all([s.path for s in samples], [s.name for s in samples], xyt_dir)
    counts = np.array([minutiae_count(x) for x in xyts])
    print(f"mindtct: minutiae per image mean {counts.mean():.1f}, "
          f"min {counts.min()}, max {counts.max()}")
    if not args.skip_nbis and nbis_available():
        scores["NBIS-Bozorth3"] = bozorth3_pairs(xyts, pairs).astype(float)
    else:
        # Without NBIS (or with --skip-nbis), keep the Bozorth3 scores already saved
        # for these exact pairs instead of dropping them from the results.
        saved = OUT / f"scores_{args.prep}.npz"
        old = np.load(saved) if saved.is_file() else None
        if old is not None and "NBIS-Bozorth3" in old and np.array_equal(old["pairs"], pairs):
            scores["NBIS-Bozorth3"] = old["NBIS-Bozorth3"]
            print(f"NBIS not run: reusing saved Bozorth3 scores from {saved}")
        else:
            print("NBIS not run and no saved Bozorth3 scores: NBIS rows omitted")

    OUT.mkdir(parents=True, exist_ok=True)
    results = {name: evaluate(s, genuine_mask, pairs, subj, dbs, imp)
               for name, s in scores.items()}
    np.savez(OUT / f"scores_{args.prep}.npz", pairs=pairs, genuine=genuine_mask, **scores)
    save_json({"prep": args.prep, "results": results}, OUT / f"summary_{args.prep}.json")
    plot(results, scores, genuine_mask, args.prep)

    print(f"\n{'system':<20}{'protocol':<8}{'EER%':>8}{'FNMR@1%':>10}"
          f"{'FNMR@0.1%':>11}{'d-prime':>9}")
    for name, r in results.items():
        for proto in ["all", "fvc", "db1", "db2", "db3", "db4"]:
            m = r[proto]
            print(f"{name:<20}{proto:<8}{100 * m['eer']:>8.2f}"
                  f"{100 * m['fnmr@fmr1%']:>10.2f}{100 * m['fnmr@fmr0.1%']:>11.2f}"
                  f"{m['d_prime']:>9.2f}")


if __name__ == "__main__":
    main()
