"""Feature-level attack success of the pre-image attack (Checkpoint 2, Sec. IV).

The pre-image attack recovers a feature vector; image reconstruction (Type-1/-2
on images) needs a trained decoder and is future work. At the feature level we
can already measure two attack success rates against the Checkpoint-1 thresholds:

  Type-1 (same protected system): hash the pre-image with the victim's key and
    compare to the stored IoM template. Accept if the matching score >= the IoM
    operating threshold. (The solver targets this template, so this is near 100%
    by construction; reported with counts for completeness.)

  Type-2 (unprotected system, impersonation): compare the recovered feature to
    the victim's OTHER impressions on the unprotected DeepPrint matcher. Accept
    if cosine >= the unprotected threshold. This tests whether the recovered
    feature impersonates the victim beyond the single template it came from, and
    is compared directly with the Checkpoint-1 genuine/impostor rates.

Baselines (from run_baseline/run_section4) on the same unprotected matcher:
  genuine TAR = 1 - FNMR, impostor accept = FMR, at the same threshold.

Thresholds come from results/section4/summary.json (FMR 1% and FMR 0.1%).
Outputs results/attack/eval.json and results/attack/det_attack.png.

Usage: python src/run_attack_eval.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import FEATURES_DIR, RESULTS_DIR
from iom import prepare

OUT = RESULTS_DIR / "attack"
VARIANTS = {"grp": "GRP-IoM", "urp": "URP-IoM (random init)", "urp_smart": "URP-IoM (smart init)"}
IOM_THR = {"grp": {"EER": 111, "FMR 1%": 155, "FMR 0.1%": 202},   # from section4 operating points
           "urp": {"EER": 116, "FMR 1%": 198, "FMR 0.1%": 311},
           "urp_smart": {"EER": 116, "FMR 1%": 198, "FMR 0.1%": 311}}
UNPROT_THR = {"EER": 0.7047266960144043, "FMR 1%": 0.8408539295196533, "FMR 0.1%": 0.9301478266716003}


def type2_scores(preimgs: torch.Tensor, feats: torch.Tensor, subj: np.ndarray):
    """Cosine of each pre-image to every OTHER impression of the same finger."""
    scores = []
    for i in range(len(preimgs)):
        others = np.where((subj == subj[i]) & (np.arange(len(subj)) != i))[0]
        cos = F.cosine_similarity(preimgs[i:i + 1], feats[others]).numpy()
        scores.append(cos)
    return np.concatenate(scores)


def main():
    feats_npz = np.load(FEATURES_DIR / "deepprint_flx.npz")
    feats = prepare(feats_npz["embedding"])            # L2-normalised true features
    subj_all = feats_npz["subject"]

    # Baseline unprotected genuine/impostor cosines (same matcher the attack targets)
    base = np.load(RESULTS_DIR / "baseline" / "scores_flx.npz")
    g = base["genuine"]
    dp = base["DeepPrint-512"]
    gen_scores, imp_scores = dp[g], dp[~g]

    results = {}
    attack_dists = {}
    for tag, label in VARIANTS.items():
        a = np.load(OUT / f"preimages_{tag}.npz")
        pre = torch.as_tensor(a["preimages"], dtype=torch.float32)
        subj = a["subject"]
        assert np.array_equal(subj, subj_all[:len(subj)]), "pre-image order must match features"
        scheme = "grp" if tag == "grp" else "urp"

        # Type-1: matching score vs own stored template (already in npz)
        t1 = {}
        for pt, thr in IOM_THR[tag].items():
            acc = int((a["scores"] >= thr).sum())
            t1[pt] = {"threshold": thr, "accepted": acc, "n": len(a["scores"]),
                      "asr": acc / len(a["scores"])}

        # Type-2: recovered feature vs victim's other impressions, unprotected matcher
        t2_scores = type2_scores(pre, feats, subj)
        t2 = {}
        for pt, thr in UNPROT_THR.items():
            acc = int((t2_scores >= thr).sum())
            gen_tar = float((gen_scores >= thr).mean())
            imp_acc = float((imp_scores >= thr).mean())
            t2[pt] = {"threshold": thr, "accepted": acc, "n": len(t2_scores),
                      "attack_asr": acc / len(t2_scores),
                      "baseline_genuine_tar": gen_tar, "baseline_impostor_accept": imp_acc}

        results[tag] = {"label": label, "type1_same_system": t1,
                        "type2_unprotected": t2,
                        "type2_score_mean": float(t2_scores.mean())}
        attack_dists[tag] = t2_scores

    # DET-style overlay: attack (Type-2) vs baseline genuine/impostor on unprotected cosine
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(imp_scores, bins=bins, density=True, alpha=0.5, color="#9a9893", label="Impostor (baseline)")
    ax.hist(gen_scores, bins=bins, density=True, alpha=0.5, color="#6250d6", label="Genuine (baseline)")
    ax.hist(attack_dists["grp"], bins=bins, density=True, alpha=0.6, histtype="step",
            linewidth=2, color="#2a78d6", label="Attack Type-2 GRP")
    ax.hist(attack_dists["urp"], bins=bins, density=True, alpha=0.6, histtype="step",
            linewidth=2, color="#eb6834", label="Attack Type-2 URP")
    ax.axvline(UNPROT_THR["FMR 1%"], color="k", ls="--", lw=1, label="Threshold (FMR 1%)")
    ax.set_xlabel("DeepPrint cosine similarity (unprotected matcher)")
    ax.set_ylabel("Density")
    ax.set_title("Pre-image attack vs baseline (Type-2, unprotected)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "det_attack.png", dpi=150, metadata={"CreationDate": None})
    plt.close(fig)

    (OUT / "eval.json").write_text(json.dumps(results, indent=2))

    print("=" * 72)
    for tag, r in results.items():
        print(f"{r['label']}  (recovered-feature purity to own impression shown in summary.json)")
        for pt in ("EER", "FMR 1%"):
            t1, t2 = r["type1_same_system"][pt], r["type2_unprotected"][pt]
            print(f"  [{pt:7}] Type-1 same-system ASR {100 * t1['asr']:5.1f}% "
                  f"({t1['accepted']}/{t1['n']})   "
                  f"Type-2 unprot ASR {100 * t2['attack_asr']:4.1f}% "
                  f"({t2['accepted']}/{t2['n']})  vs genuine "
                  f"{100 * t2['baseline_genuine_tar']:.0f}% / impostor "
                  f"{100 * t2['baseline_impostor_accept']:.0f}%")


if __name__ == "__main__":
    main()
