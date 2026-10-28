"""Week 3: implement and validate GRP-IoM and URP-IoM on DeepPrint embeddings.

For each scheme (Paik et al. parameters: GRP m=300 q=16; URP m=600 w=100 p=2):
  1. Stolen-token (one key shared by everyone, Paik's setting and the attack's
     threat model): verification over the same 'all' pairs as the baseline,
     repeated over several keys. Should stay close to the unprotected EER.
  2. Genuine-token (each finger has its own key; an impostor uses their own):
     the best-case setting usually reported for cancelable biometrics.
  3. Revocability / unlinkability: same finger hashed with two different keys
     should look like an impostor. Reports D<->sys and score means.
  4. Sanity check: different-key scores should sit at chance level
     (m/q for GRP, m/w for URP).
  5. Hashing time per template.
Saves the stolen-token templates of the first key as the week 4 attack target.

Usage: python src/run_iom.py [--keys 5]
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from config import FEATURES_DIR, RESULTS_DIR
from iom import hash_all, make_key, make_scheme, match_scores, prepare
from metrics import (dsys_noise_floor, evaluate, pair_indices, roc, save_json,
                     summarize, unlinkability_dsys)

OUT = RESULTS_DIR / "iom"
TEMPLATE_DIR = RESULTS_DIR / "templates"
SCHEMES = ["grp", "urp"]
# Token ranges kept disjoint so no two experiments reuse a key by accident.
STOLEN_TOKEN_BASE = 0          # shared keys: 0, 1, 2, ...
GENUINE_TOKEN_BASE = 10_000    # per-finger keys: 10_000 + subject
REVOKED_TOKEN_BASE = 20_000    # reissued per-finger keys: 20_000 + subject


def chance_level(scheme) -> float:
    return scheme.m / (scheme.q if hasattr(scheme, "q") else scheme.w)


def hash_per_subject(scheme, x, subj, token_base):
    """Hash each sample with its own finger's key (token_base + subject)."""
    codes = torch.empty((len(x), scheme.m), dtype=torch.long)
    for s in np.unique(subj):
        idx = torch.as_tensor(np.where(subj == s)[0])
        codes[idx] = hash_all(scheme, x[idx], make_key(scheme, token_base + int(s)))
    return codes


def run_scheme(name, x, subj, dbs, imp, pairs, genuine_mask, n_keys):
    scheme = make_scheme(name)
    res = {"params": {k: getattr(scheme, k) for k in ("m", "q", "w", "p")
                      if hasattr(scheme, k)},
           "chance_level": chance_level(scheme)}

    # 1. Stolen-token, repeated over n_keys shared keys.
    per_key, first_scores = [], None
    for k in range(n_keys):
        token = STOLEN_TOKEN_BASE + k
        key = make_key(scheme, token)
        t0 = time.perf_counter()
        codes = hash_all(scheme, x, key)
        hash_ms = 1000 * (time.perf_counter() - t0) / len(x)
        scores = match_scores(codes, codes, pairs).astype(float)
        per_key.append(evaluate(scores, genuine_mask, pairs, subj, dbs, imp))
        per_key[-1]["hash_ms_per_template"] = hash_ms
        if k == 0:
            first_scores = scores
            TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"scheme": name, "params": res["params"], "token": token,
                        "codes": codes, "subject": subj, "db": dbs, "impression": imp},
                       TEMPLATE_DIR / f"{name}_stolen_token{token}.pt")
    eers = np.array([r["all"]["eer"] for r in per_key])
    res["stolen_token"] = {
        "per_key": per_key,
        "eer_mean": float(eers.mean()),
        "eer_std": float(eers.std()),
        "fnmr@fmr1%_mean": float(np.mean([r["all"]["fnmr@fmr1%"] for r in per_key])),
        "fnmr@fmr0.1%_mean": float(np.mean([r["all"]["fnmr@fmr0.1%"] for r in per_key])),
        "hash_ms_per_template_mean": float(np.mean([r["hash_ms_per_template"] for r in per_key])),
    }

    # 2. Genuine-token: every finger has its own key.
    codes_own = hash_per_subject(scheme, x, subj, GENUINE_TOKEN_BASE)
    s_own = match_scores(codes_own, codes_own, pairs).astype(float)
    res["genuine_token"] = summarize(s_own[genuine_mask], s_own[~genuine_mask])

    # 3. Revocability / unlinkability: reissue every finger a new key.
    codes_new = hash_per_subject(scheme, x, subj, REVOKED_TOKEN_BASE)
    s_cross = match_scores(codes_own, codes_new, pairs).astype(float)
    mated, nonmated = s_cross[genuine_mask], s_cross[~genuine_mask]
    # Same sample, old key vs new key (the strongest linkage attempt).
    n = len(x)
    same_sample = match_scores(codes_own, codes_new, np.stack([np.arange(n)] * 2, 1))
    res["revocability"] = {
        "mated_cross_key_mean": float(mated.mean()),
        "nonmated_cross_key_mean": float(nonmated.mean()),
        "same_sample_cross_key_mean": float(same_sample.mean()),
        "same_key_genuine_mean": float(s_own[genuine_mask].mean()),
        "dsys_unlinkability": unlinkability_dsys(mated, nonmated),
        "dsys_noise_floor": dsys_noise_floor(nonmated, len(mated)),
        # How often a reissued template would be accepted by the old one at the
        # stolen-token EER threshold of the first key.
        "old_vs_new_accept_rate@eer_threshold": float(
            (same_sample >= per_key[0]["all"]["eer_threshold"]).mean()),
    }
    return res, first_scores, {"own": s_own, "cross_mated": mated,
                               "cross_nonmated": nonmated, "same_sample": same_sample}


def plot_det(curves: dict, genuine_mask):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, s in curves.items():
        _, fmr, fnmr = roc(s[genuine_mask], s[~genuine_mask])
        ok = (fmr > 0) & (fnmr > 0) & (fmr < 1) & (fnmr < 1)
        ax.plot(norm.ppf(fmr[ok]), norm.ppf(fnmr[ok]), label=label)
    ticks = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5])
    ax.set_xticks(norm.ppf(ticks), [f"{100 * t:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{100 * t:g}" for t in ticks])
    ax.set_xlim(norm.ppf(5e-5), norm.ppf(0.5))
    ax.set_ylim(norm.ppf(5e-4), norm.ppf(0.6))
    ax.set_xlabel("FMR (%)")
    ax.set_ylabel("FNMR (%)")
    ax.set_title("FVC2004 B: unprotected vs IoM (stolen token, key 0)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "det_iom.png", dpi=150)
    plt.close(fig)


def plot_distributions(name, stolen, extra, genuine_mask, chance):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    bins = np.arange(0, max(stolen.max(), extra["own"].max()) + 2) - 0.5
    ax = axes[0]
    ax.hist(stolen[~genuine_mask], bins=bins, density=True, alpha=0.6, label="impostor")
    ax.hist(stolen[genuine_mask], bins=bins, density=True, alpha=0.6, label="genuine")
    ax.set_title(f"{name.upper()}-IoM, stolen token")
    ax.set_xlabel("matching hash entries")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(extra["cross_nonmated"], bins=bins, density=True, alpha=0.6,
            label="different finger, different key")
    ax.hist(extra["cross_mated"], bins=bins, density=True, alpha=0.6,
            label="same finger, different key")
    ax.hist(extra["own"][genuine_mask], bins=bins, density=True, alpha=0.4,
            label="same finger, same key")
    ax.axvline(chance, color="k", ls="--", lw=1, label="chance (m/q or m/w)")
    ax.set_title(f"{name.upper()}-IoM, revocability")
    ax.set_xlabel("matching hash entries")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"scores_{name}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="deepprint_flx.npz")
    ap.add_argument("--keys", type=int, default=5, help="shared keys for stolen-token")
    args = ap.parse_args()

    feats = np.load(FEATURES_DIR / args.features)
    x = prepare(feats["embedding"])
    subj, dbs, imp = feats["subject"], feats["db"], feats["impression"]
    gen, impo = pair_indices(subj, dbs, imp, "all")
    pairs = np.concatenate([gen, impo])
    genuine_mask = np.zeros(len(pairs), bool)
    genuine_mask[: len(gen)] = True

    xn = x.numpy()
    unprotected = np.einsum("nd,nd->n", xn[pairs[:, 0]], xn[pairs[:, 1]])
    results = {"features": args.features, "n_keys": args.keys,
               "unprotected": summarize(unprotected[genuine_mask], unprotected[~genuine_mask])}
    curves = {f"Unprotected DeepPrint-512 (EER {100 * results['unprotected']['eer']:.2f}%)":
              unprotected}

    OUT.mkdir(parents=True, exist_ok=True)
    for name in SCHEMES:
        print(f"Running {name.upper()}-IoM ...", flush=True)
        res, stolen, extra = run_scheme(name, x, subj, dbs, imp, pairs, genuine_mask, args.keys)
        results[name] = res
        curves[f"{name.upper()}-IoM (EER {100 * res['stolen_token']['per_key'][0]['all']['eer']:.2f}%)"] = stolen
        plot_distributions(name, stolen, extra, genuine_mask, res["chance_level"])
        np.savez(OUT / f"scores_{name}.npz", pairs=pairs, genuine=genuine_mask,
                 stolen_key0=stolen, **extra)

    plot_det(curves, genuine_mask)
    save_json(results, OUT / "summary.json")

    u = results["unprotected"]
    print(f"\nUnprotected DeepPrint-512: EER {100 * u['eer']:.2f}%  "
          f"FNMR@1% {100 * u['fnmr@fmr1%']:.2f}%  FNMR@0.1% {100 * u['fnmr@fmr0.1%']:.2f}%")
    for name in SCHEMES:
        r = results[name]
        st, gt, rv = r["stolen_token"], r["genuine_token"], r["revocability"]
        print(f"\n{name.upper()}-IoM {r['params']}  (chance level {r['chance_level']:.2f})")
        print(f"  stolen token  EER {100 * st['eer_mean']:.2f} ± {100 * st['eer_std']:.2f}%  "
              f"FNMR@1% {100 * st['fnmr@fmr1%_mean']:.2f}%  "
              f"FNMR@0.1% {100 * st['fnmr@fmr0.1%_mean']:.2f}%  "
              f"({args.keys} keys, {st['hash_ms_per_template_mean']:.3f} ms/template)")
        for k, pk in enumerate(st["per_key"]):
            print(f"    key {k}: EER {100 * pk['all']['eer']:.2f}%  "
                  + "  ".join(f"{d} {100 * pk[d]['eer']:.2f}%" for d in ("db1", "db2", "db3", "db4")))
        print(f"  genuine token EER {100 * gt['eer']:.2f}%  "
              f"FNMR@0.1% {100 * gt['fnmr@fmr0.1%']:.2f}%  "
              f"(genuine mean {gt['genuine_mean']:.1f}, impostor mean {gt['impostor_mean']:.1f})")
        print(f"  revocability  same finger/new key mean {rv['mated_cross_key_mean']:.2f}, "
              f"different finger/new key mean {rv['nonmated_cross_key_mean']:.2f}, "
              f"same sample/new key mean {rv['same_sample_cross_key_mean']:.2f}")
        print(f"                D<->sys {rv['dsys_unlinkability']:.3f} "
              f"(estimator noise floor {rv['dsys_noise_floor']:.3f}), "
              f"old-vs-new accepted {100 * rv['old_vs_new_accept_rate@eer_threshold']:.2f}%")


if __name__ == "__main__":
    main()
