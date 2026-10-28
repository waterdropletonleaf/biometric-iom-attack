"""Verification protocol and metrics shared by all experiments.

Two pair protocols over the 320 samples:
  - "all":  every genuine pair (40 x C(8,2) = 1120) and every impostor pair
            (C(320,2) - 1120 = 49,920), including cross-DB impostors.
  - "fvc":  FVC2004 competition protocol applied per DB: all genuine pairs, and
            impostors only between the first impressions of different fingers
            of the same DB (4 x C(10,2) = 180).
"""
import json
from pathlib import Path

import numpy as np


def pair_indices(subjects: np.ndarray, dbs: np.ndarray, impressions: np.ndarray,
                 protocol: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """Return (genuine_pairs, impostor_pairs), each an (N, 2) array with i < j."""
    i, j = np.triu_indices(len(subjects), k=1)
    same = subjects[i] == subjects[j]
    genuine = np.stack([i[same], j[same]], 1)
    if protocol == "all":
        imp_mask = ~same
    elif protocol == "fvc":
        imp_mask = (~same & (dbs[i] == dbs[j])
                    & (impressions[i] == 0) & (impressions[j] == 0))
    else:
        raise ValueError(protocol)
    impostor = np.stack([i[imp_mask], j[imp_mask]], 1)
    return genuine, impostor


def roc(genuine: np.ndarray, impostor: np.ndarray):
    """FMR and FNMR at every threshold (accept if score >= t). Higher = more similar."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    g = np.sort(genuine)
    im = np.sort(impostor)
    fnmr = np.searchsorted(g, thresholds, side="left") / len(g)
    fmr = 1.0 - np.searchsorted(im, thresholds, side="left") / len(im)
    return thresholds, fmr, fnmr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold at which it occurs."""
    t, fmr, fnmr = roc(genuine, impostor)
    k = np.argmin(np.abs(fmr - fnmr))
    return float((fmr[k] + fnmr[k]) / 2), float(t[k])


def fnmr_at_fmr(genuine: np.ndarray, impostor: np.ndarray, target: float) -> tuple[float, float]:
    """Lowest FNMR with FMR <= target, and its threshold."""
    t, fmr, fnmr = roc(genuine, impostor)
    ok = fmr <= target
    if not ok.any():
        return 1.0, float("inf")
    k = np.argmin(np.where(ok, fnmr, np.inf))
    return float(fnmr[k]), float(t[k])


def decidability(genuine: np.ndarray, impostor: np.ndarray) -> float:
    return float(abs(genuine.mean() - impostor.mean())
                 / np.sqrt((genuine.var() + impostor.var()) / 2))


def summarize(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    e, e_t = eer(genuine, impostor)
    f100, t100 = fnmr_at_fmr(genuine, impostor, 1e-2)
    f1000, t1000 = fnmr_at_fmr(genuine, impostor, 1e-3)
    return {
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "eer": e,
        "eer_threshold": e_t,
        "fnmr@fmr1%": f100,
        "threshold@fmr1%": t100,
        "fnmr@fmr0.1%": f1000,
        "threshold@fmr0.1%": t1000,
        "d_prime": decidability(genuine, impostor),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def evaluate(scores: np.ndarray, genuine_mask: np.ndarray, pairs: np.ndarray,
             subj, dbs, imp) -> dict:
    """Metrics for protocol 'all', protocol 'fvc', and per-DB (within-DB impostors).
    `pairs`/`scores` must cover the 'all' protocol (genuine_mask marks genuine)."""
    res = {"all": summarize(scores[genuine_mask], scores[~genuine_mask])}
    i, j = pairs[:, 0], pairs[:, 1]
    fvc_imp = (~genuine_mask & (dbs[i] == dbs[j]) & (imp[i] == 0) & (imp[j] == 0))
    res["fvc"] = summarize(scores[genuine_mask], scores[fvc_imp])
    for d in range(4):
        in_db = (dbs[i] == d) & (dbs[j] == d)
        res[f"db{d + 1}"] = summarize(scores[genuine_mask & in_db],
                                      scores[~genuine_mask & in_db])
    return res


def unlinkability_dsys(mated: np.ndarray, nonmated: np.ndarray, bins: int = 30) -> float:
    """Global unlinkability measure D<->sys (Gomez-Barrero et al., IEEE TIFS 2018).

    `mated`: scores between templates of the same finger made with different
    keys; `nonmated`: different fingers, different keys. 0 = fully unlinkable,
    1 = fully linkable. Densities are estimated with a shared histogram
    (omega = 1, i.e. equal priors).
    """
    edges = np.histogram_bin_edges(np.concatenate([mated, nonmated]), bins=bins)
    p_m, _ = np.histogram(mated, edges)
    p_nm, _ = np.histogram(nonmated, edges)
    p_m = p_m / p_m.sum()
    p_nm = p_nm / p_nm.sum()
    d = np.zeros_like(p_m)
    only_mated = (p_nm == 0) & (p_m > 0)
    d[only_mated] = 1.0
    both = (p_nm > 0) & (p_m > p_nm)
    lr = p_m[both] / p_nm[both]
    d[both] = (lr - 1) / (lr + 1)
    return float((p_m * d).sum())


def dsys_noise_floor(nonmated: np.ndarray, n_mated: int, repeats: int = 20,
                     seed: int = 0) -> float:
    """D<->sys between a random subset of non-mated scores (same size as the
    mated set) and the rest: what the estimator reports with no linkage at all."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        perm = rng.permutation(len(nonmated))
        vals.append(unlinkability_dsys(nonmated[perm[:n_mated]], nonmated[perm[n_mated:]]))
    return float(np.mean(vals))
