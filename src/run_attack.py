"""Week 4-5: PMA-LSH pre-image attack on the saved IoM templates.

Reproduces Paik et al.'s attack (Analysis_LSH/eval_attack.py) against the
GRP-/URP-IoM templates saved by run_iom.py (stolen-token, shared key 0). For
each protected template it recovers a feature vector that reproduces the stored
hash, then measures:
  - matching score: hash(pre-image) vs the stored code (0..m); m means an exact
    pre-image, i.e. the recovered vector lands in the same argmax cells.
  - purity: cosine similarity between the pre-image and the TRUE DeepPrint
    feature (how close the recovery is to the real template; Paik's Table IV/IX).
  - time per template.

GRP uses Paik's initialisation trick (average loser->winner direction of the
key rows); URP is solved one template at a time from a random start, which is
why it needs restarts and runs slower (the week-2 goal re: URP convergence).

Usage: python src/run_attack.py [--scheme grp|urp|both] [--limit N]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import FEATURES_DIR, RESULTS_DIR, ROOT
from iom import make_key, make_scheme, prepare
from metrics import save_json

import sys
sys.path.insert(0, str(ROOT / "Analysis_LSH"))
from LSH import GRP, URP            # noqa: E402  (Paik et al.)

OUT = RESULTS_DIR / "attack"
TEMPLATE_DIR = RESULTS_DIR / "templates"


def Newton(LSPs, batch_size, device, init=None, l_type="s", mode="abs",
           tol=1e-3, n_iter=50, emb_size=512):
    """Paik et al.'s root-finding solver, copied verbatim from
    Analysis_LSH/eval_attack.py (the module there imports mxnet for the face
    pipeline, which we do not use). Minimises the total constraint violation
    G(x) = LSP(x) - |LSP(x)|; converges when every argmax constraint holds.
    Up to 10 random restarts; on URP it may need several (the slow-convergence
    behaviour this project investigates)."""
    find = False
    restart = 10
    while not find:
        if (l_type == "s") and (restart < 0):
            l_type = "m"
        if init is not None:
            x = init.expand(batch_size, emb_size)
        else:
            x = F.normalize(torch.randn(batch_size, emb_size))
        x = x.requires_grad_(True).to(device)
        for _ in range(n_iter):
            G_x = (LSPs(x) - LSPs(x).abs())
            G_x = G_x.pow(2) if mode == "pow" else G_x.abs()
            loss = G_x.mean(dim=0).sum() if l_type == "s" else G_x.mean(dim=0).mean()
            grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
            if grad.norm() == 0:
                find = True
                break
            with torch.no_grad():
                x = x - grad / (grad.norm() ** 2 + 1e-8) * loss
            x.requires_grad_(True).to(device)
        if grad.norm() < tol or init is not None:
            find = True
        else:
            restart -= 1
    return x.detach().cpu()


def grp_init(scheme: GRP, codes: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Paik's GRP init: for each block, the mean normalised direction from the
    non-selected (loser) key rows toward the selected (winner) row."""
    n = codes.shape[0]
    helper_r = key.reshape(scheme.m, scheme.q, scheme.emb_size)
    out = torch.empty((n, scheme.emb_size))
    for j in range(n):
        winners = torch.stack([helper_r[k, codes[j, k]] for k in range(scheme.m)])  # (m, 512)
        winners = winners.reshape(scheme.m, 1, scheme.emb_size)
        out[j] = F.normalize(winners - helper_r, dim=2).mean(dim=[0, 1])
    return out


@torch.no_grad()
def hash_one(scheme, x: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return scheme.hashing(x, key)[0]


def attack_scheme(name: str, true_feats: torch.Tensor, limit: int | None):
    data = torch.load(TEMPLATE_DIR / f"{name}_stolen_token{0}.pt", weights_only=False)
    scheme = make_scheme(name)
    key = make_key(scheme, data["token"])
    codes = data["codes"]
    n = len(codes) if limit is None else min(limit, len(codes))
    m = scheme.m

    preimgs = torch.empty((n, scheme.emb_size))
    scores = np.empty(n, dtype=int)
    purity = np.empty(n, dtype=float)
    times = np.empty(n, dtype=float)

    inits = grp_init(scheme, codes[:n], key) if isinstance(scheme, GRP) else None

    for i in range(n):
        t0 = time.perf_counter()
        template = (codes[i:i + 1], key)
        lsp = scheme.lsp(template)
        init = inits[i:i + 1] if inits is not None else None
        # URP's products make the search non-convex; squared loss ("pow") is what
        # Paik uses when a plain L1 loss stalls. GRP is linear -> "abs" is fine.
        sol = Newton(lsp, batch_size=1, device="cpu", init=init,
                     l_type="s", mode="abs", emb_size=scheme.emb_size)
        times[i] = time.perf_counter() - t0

        # URP (p=2): x and -x hash identically; align sign to the true feature so
        # purity reflects the recovered direction (as Paik's code does).
        cos = F.cosine_similarity(sol, true_feats[i:i + 1]).item()
        if isinstance(scheme, URP) and cos < 0:
            sol, cos = -sol, -cos
        preimgs[i] = sol
        scores[i] = int((hash_one(scheme, sol, key) == codes[i:i + 1]).sum())
        purity[i] = cos
        if (i + 1) % 40 == 0 or i + 1 == n:
            print(f"  {name} {i + 1}/{n}  score {scores[i]}/{m}  purity {cos:.3f}  "
                  f"{times[i]:.3f}s", flush=True)

    res = {
        "scheme": name, "params": data["params"], "n": n, "m": m,
        "matching_score_mean": float(scores.mean()),
        "exact_preimage_rate": float((scores == m).mean()),
        "purity_mean": float(purity.mean()), "purity_std": float(purity.std()),
        "time_mean_s": float(times.mean()), "time_std_s": float(times.std()),
        "time_total_s": float(times.sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"preimages_{name}.npz", preimages=preimgs.numpy(),
             scores=scores, purity=purity, times=times,
             subject=data["subject"][:n], db=data["db"][:n])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["grp", "urp", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="attack only the first N templates")
    args = ap.parse_args()

    feats = np.load(FEATURES_DIR / "deepprint_flx.npz")
    true_feats = prepare(feats["embedding"])

    names = ["grp", "urp"] if args.scheme == "both" else [args.scheme]
    results = {}
    for name in names:
        print(f"Attacking {name.upper()}-IoM ...", flush=True)
        results[name] = attack_scheme(name, true_feats, args.limit)

    save_json(results, OUT / "summary.json")
    print("\n" + "=" * 64)
    for name, r in results.items():
        print(f"{name.upper()}-IoM {r['params']}  (n={r['n']})")
        print(f"  matching score   {r['matching_score_mean']:.1f}/{r['m']}  "
              f"(exact pre-image {100 * r['exact_preimage_rate']:.1f}%)")
        print(f"  purity (cos)     {r['purity_mean']:.3f} ± {r['purity_std']:.3f}")
        print(f"  time / template  {r['time_mean_s']:.3f} ± {r['time_std_s']:.3f} s"
              f"   (total {r['time_total_s']:.1f} s)")


if __name__ == "__main__":
    main()
