"""Week 4-5: PMA-LSH pre-image attack on the saved IoM templates.

Reproduces Paik et al.'s attack (Analysis_LSH/eval_attack.py) against the
GRP-/URP-IoM templates saved by run_iom.py (stolen-token, shared key 0). For
each protected template it recovers a feature vector that reproduces the stored
hash, then measures:
  - matching score: hash(pre-image) vs the stored code (0..m); m means an exact
    pre-image, i.e. the recovered vector lands in the same argmax cells.
  - purity: cosine similarity between the pre-image and the TRUE DeepPrint
    feature (how close the recovery is to the real template; Paik's Table IV/IX).
  - time per template, and the number of solver restarts (goal 2: URP is slow
    because it restarts; this counts how often).

GRP uses Paik's initialisation trick (average loser->winner direction of the
key rows) and a single solve. URP is solved from a random start with up to 10
restarts (--urp-init none, Paik's setting), or from a smart init that seeds the
winning coordinate pairs and falls back to restarts if it stalls (--urp-init
smart). The smart init is this project's attempt to cut URP's restart cost.

Usage: python src/run_attack.py [--scheme grp|urp|both]
                                [--urp-init none|smart] [--limit N]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import FEATURES_DIR, RESULTS_DIR, ROOT
from iom import make_key, make_scheme, prepare

import sys
sys.path.insert(0, str(ROOT / "Analysis_LSH"))
from LSH import GRP, URP            # noqa: E402  (Paik et al.)

OUT = RESULTS_DIR / "attack"
TEMPLATE_DIR = RESULTS_DIR / "templates"


def Newton(LSPs, batch_size, device, init=None, l_type="s", mode="abs",
           tol=1e-3, n_iter=50, emb_size=512, init_fallback=False):
    """Paik et al.'s root-finding solver (Analysis_LSH/eval_attack.py), copied
    here because that module imports mxnet for the face pipeline. Minimises the
    total constraint violation G(x) = LSP(x) - |LSP(x)|; converges when every
    argmax constraint holds.

    Behaviour, unchanged from Paik except where noted:
      - init given, init_fallback=False (GRP): one solve from init, then stop.
      - init None (URP baseline): random start, retry until converged.
      - init given, init_fallback=True (URP smart init, new): try init first,
        then fall back to random restarts if it did not converge.
    Returns (x, stats) where stats = {attempts, restarts, converged}; restarts
    is attempts-1 (how many times it had to start over).
    """
    find = False
    restart = 10
    attempts = 0
    converged = False
    first = True
    grad = None
    while not find:
        attempts += 1
        if (l_type == "s") and (restart < 0):
            l_type = "m"
        if init is not None and first:
            x = init.expand(batch_size, emb_size)
        else:
            x = F.normalize(torch.randn(batch_size, emb_size))
        first = False
        x = x.requires_grad_(True).to(device)
        for _ in range(n_iter):
            G_x = (LSPs(x) - LSPs(x).abs())
            G_x = G_x.pow(2) if mode == "pow" else G_x.abs()
            loss = G_x.mean(dim=0).sum() if l_type == "s" else G_x.mean(dim=0).mean()
            grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
            if grad.norm() == 0:
                find = True
                converged = True
                break
            with torch.no_grad():
                x = x - grad / (grad.norm() ** 2 + 1e-8) * loss
            x.requires_grad_(True).to(device)
        if grad.norm() < tol:
            converged = True
            find = True
        elif (init is not None) and (not init_fallback):
            find = True          # Paik's GRP behaviour: single solve from init
        else:
            restart -= 1
    stats = {"attempts": attempts, "restarts": attempts - 1, "converged": bool(converged)}
    return x.detach().cpu(), stats


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


def urp_init(scheme: URP, codes: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Smart URP init (new, analogous to grp_init). URP's key is p permutations
    per block; block k's code c means the product x[key[k,0,c]]*...*x[key[k,p-1,c]]
    is the largest over the first w window positions. Seed those winning
    coordinates with +1 so the winning product starts largest; normalise. Sign
    is irrelevant (x and -x hash identically for p=2)."""
    n = codes.shape[0]
    R = key.reshape(scheme.m, scheme.p, scheme.emb_size)   # (m, p, emb)
    out = torch.zeros((n, scheme.emb_size))
    for j in range(n):
        # winning coordinate indices for every block: R[k, t, codes[j,k]] -> (m, p)
        winners = R.gather(2, codes[j].view(scheme.m, 1, 1)
                           .expand(scheme.m, scheme.p, 1)).squeeze(-1)
        out[j].scatter_add_(0, winners.reshape(-1), torch.ones(winners.numel()))
    return F.normalize(out, dim=1)


@torch.no_grad()
def hash_one(scheme, x: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return scheme.hashing(x, key)[0]


def attack_scheme(name: str, true_feats: torch.Tensor, limit, urp_init_mode: str):
    data = torch.load(TEMPLATE_DIR / f"{name}_stolen_token{0}.pt", weights_only=False)
    scheme = make_scheme(name)
    key = make_key(scheme, data["token"])
    codes = data["codes"]
    n = len(codes) if limit is None else min(limit, len(codes))
    m = scheme.m
    is_grp = isinstance(scheme, GRP)
    smart = (not is_grp) and urp_init_mode == "smart"
    tag = name if is_grp or urp_init_mode == "none" else f"{name}_smart"

    preimgs = torch.empty((n, scheme.emb_size))
    scores = np.empty(n, dtype=int)
    purity = np.empty(n, dtype=float)
    times = np.empty(n, dtype=float)
    restarts = np.empty(n, dtype=int)
    converged = np.empty(n, dtype=bool)

    # Seed the random restarts so the URP attack is reproducible run-to-run
    # (GRP is already deterministic: init given, no restarts). Without this,
    # URP purity/time drift between runs because F.normalize(torch.randn(...)).
    torch.manual_seed(0)

    if is_grp:
        inits = grp_init(scheme, codes[:n], key)
    elif smart:
        inits = urp_init(scheme, codes[:n], key)
    else:
        inits = None

    for i in range(n):
        t0 = time.perf_counter()
        lsp = scheme.lsp((codes[i:i + 1], key))
        init = inits[i:i + 1] if inits is not None else None
        # GRP: linear constraints, single solve from init (abs loss).
        # URP: non-convex p=2 products; random/ smart start + restarts (abs loss).
        sol, st = Newton(lsp, batch_size=1, device="cpu", init=init,
                         l_type="s", mode="abs", emb_size=scheme.emb_size,
                         init_fallback=smart)
        times[i] = time.perf_counter() - t0

        # URP (p=2): x and -x hash identically; align sign to the true feature so
        # purity reflects the recovered direction (as Paik's code does).
        cos = F.cosine_similarity(sol, true_feats[i:i + 1]).item()
        if isinstance(scheme, URP) and cos < 0:
            sol, cos = -sol, -cos
        preimgs[i] = sol
        scores[i] = int((hash_one(scheme, sol, key) == codes[i:i + 1]).sum())
        purity[i] = cos
        restarts[i] = st["restarts"]
        converged[i] = st["converged"]
        if (i + 1) % 40 == 0 or i + 1 == n:
            print(f"  {tag} {i + 1}/{n}  score {scores[i]}/{m}  purity {cos:.3f}  "
                  f"restarts {st['restarts']}  {times[i]:.3f}s", flush=True)

    res = {
        "scheme": name, "variant": tag, "params": data["params"], "n": n, "m": m,
        "urp_init": None if is_grp else urp_init_mode,
        "matching_score_mean": float(scores.mean()),
        "exact_preimage_rate": float((scores == m).mean()),
        "purity_mean": float(purity.mean()), "purity_std": float(purity.std()),
        "time_mean_s": float(times.mean()), "time_std_s": float(times.std()),
        "time_total_s": float(times.sum()),
        "restarts_mean": float(restarts.mean()), "restarts_max": int(restarts.max()),
        "solved_first_try_rate": float((restarts == 0).mean()),
        "converged_rate": float(converged.mean()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"preimages_{tag}.npz", preimages=preimgs.numpy(),
             scores=scores, purity=purity, times=times, restarts=restarts,
             converged=converged, subject=data["subject"][:n], db=data["db"][:n])
    return tag, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["grp", "urp", "both"], default="both")
    ap.add_argument("--urp-init", choices=["none", "smart"], default="none",
                    help="URP initialisation: Paik's random start, or the smart seed")
    ap.add_argument("--limit", type=int, default=None, help="attack only the first N templates")
    args = ap.parse_args()

    # Single-threaded so the URP attack is bit-reproducible: multi-threaded CPU
    # reductions give tiny FP differences that flip the grad-norm convergence
    # test and change restart counts run-to-run. With one thread + the seed in
    # attack_scheme, re-running reproduces the saved results exactly.
    torch.set_num_threads(1)

    feats = np.load(FEATURES_DIR / "deepprint_flx.npz")
    true_feats = prepare(feats["embedding"])

    names = ["grp", "urp"] if args.scheme == "both" else [args.scheme]
    ran = {}
    for name in names:
        label = name.upper() + ("" if name == "grp" else f" (init={args.urp_init})")
        print(f"Attacking {label}-IoM ...", flush=True)
        tag, res = attack_scheme(name, true_feats, args.limit, args.urp_init)
        ran[tag] = res

    # Merge into summary.json so grp / urp / urp_smart accumulate across runs.
    summary_path = OUT / "summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    summary.update(ran)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    for tag, r in ran.items():
        print(f"{tag.upper()}  {r['params']}  (n={r['n']})")
        print(f"  matching score   {r['matching_score_mean']:.1f}/{r['m']}  "
              f"(exact pre-image {100 * r['exact_preimage_rate']:.1f}%)")
        print(f"  purity (cos)     {r['purity_mean']:.3f} ± {r['purity_std']:.3f}")
        print(f"  restarts         mean {r['restarts_mean']:.2f}, max {r['restarts_max']}, "
              f"solved first try {100 * r['solved_first_try_rate']:.1f}%")
        print(f"  time / template  {r['time_mean_s']:.3f} ± {r['time_std_s']:.3f} s"
              f"   (total {r['time_total_s']:.1f} s)")


if __name__ == "__main__":
    main()
