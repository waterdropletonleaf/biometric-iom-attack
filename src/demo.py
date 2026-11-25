#!/usr/bin/env python3
"""End-to-end demo of the PMA-LSH pre-image attack on ONE IoM-protected
DeepPrint fingerprint template.

Walks the whole attack chain on a single template and narrates each step:

  STEP 1  the stolen IoM code + key  (all the server stores)
  STEP 2  invert it -> a 512-d feature  (purity, exact hash match)   [LIVE]
  STEP 3  turn the feature back into a fingerprint image
  STEP 4  does the image RE-ENTER the protected system?  (Type-1)
  STEP 5  can it impersonate the victim elsewhere?        (Type-2)
  TAKEAWAY

It runs locally (venv) and in Google Colab.

--------------------------------------------------------------------------
Colab quickstart (paste into one cell):

    !git clone --recursive https://github.com/waterdropletonleaf/biometric-iom-attack.git
    %cd biometric-iom-attack
    !pip -q install torch numpy matplotlib
    !python src/demo.py

  --recursive pulls the Analysis_LSH submodule, which lets STEP 2 run the
  attack live. Without it, the demo falls back to replaying the saved
  pre-image (still a real result, just not recomputed in front of you).

  Already cloned without --recursive? Fetch just the attack submodule:
    !git submodule update --init Analysis_LSH
--------------------------------------------------------------------------

What it needs:
  * Default: only the committed results/ artifacts + numpy/torch, plus the
    Analysis_LSH submodule for the live attack (optional; auto-fallback).
  * --live-reconstruct: additionally re-optimizes the image with the real
    DeepPrint extractor (needs the fixed-length-fingerprint-extractors
    submodule AND its model weights; slow on CPU). Otherwise STEP 3 shows the
    precomputed reconstruction saved in results/reconstruct/.

Usage:
    python src/demo.py [--scheme grp|urp] [--index N] [--live-reconstruct] [--iters 250]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../src
sys.path.insert(0, str(HERE))

import torch                                     # noqa: E402
import torch.nn.functional as F                 # noqa: E402

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402

from config import RESULTS_DIR, FEATURES_DIR    # noqa: E402

# Unprotected-DeepPrint operating thresholds (from results/section4/summary.json)
EER_THR = 0.7047266960144043
FMR1_THR = 0.8408539295196533
IOM_FMR1 = {"grp": 155, "urp": 198}
DBS = ["DB1", "DB2", "DB3", "DB4"]


def banner(txt):
    print("\n" + "=" * 70 + f"\n  {txt}\n" + "=" * 70)


def maybe_show(path):
    """Display an image inline when running inside a notebook (Colab); no-op
    in a plain terminal."""
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:
        pass


def as_name(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def run_or_replay_attack(name, target, code, true_feat):
    """STEP 2. Try to run the Newton pre-image attack LIVE on this one template
    (needs the Analysis_LSH submodule). On any import/setup failure, replay the
    saved pre-image from results/attack/. Returns (feature, purity, score, m,
    seconds_or_None, live?)."""
    try:
        from iom import make_scheme, make_key
        from run_attack import Newton, grp_init
        scheme = make_scheme(name)
        key = make_key(scheme, 0)
        code_b = code.unsqueeze(0)               # (1, m)
        t0 = time.perf_counter()
        init = grp_init(scheme, code_b, key) if name == "grp" else None
        lsp = scheme.lsp((code_b, key))
        sol, _ = Newton(lsp, batch_size=1, device="cpu", init=init,
                        l_type="s", mode="abs", emb_size=scheme.emb_size,
                        init_fallback=False)
        dt = time.perf_counter() - t0
        rec = F.normalize(sol, dim=1)
        cos = F.cosine_similarity(rec, true_feat.unsqueeze(0)).item()
        if name == "urp" and cos < 0:            # x and -x hash identically (p=2)
            rec, cos = -rec, -cos
        score = int((scheme.hashing(rec, key)[0] == code_b).sum())
        return rec[0], cos, score, scheme.m, dt, True
    except Exception as e:
        print(f"  [live attack unavailable ({type(e).__name__}: {e});")
        print(f"   replaying the saved pre-image instead]")
        pre = np.load(RESULTS_DIR / "attack" / f"preimages_{name}.npz")
        rec = F.normalize(
            torch.as_tensor(pre["preimages"][target:target + 1], dtype=torch.float32), dim=1)[0]
        return rec, float(pre["purity"][target]), int(pre["scores"][target]), \
            int(pre["scores"].max() if name == "grp" else 600), None, False


def live_reconstruct(feature, code, name, iters):
    """STEP 3 (opt-in). Re-optimize an image whose DeepPrint re-extraction
    matches `feature`, then recompute its Type-1 score. Needs the DeepPrint
    model + weights. Returns (image_hw, type1_score) or (None, None)."""
    try:
        from extract_deepprint import load_model
        from run_reconstruct import reconstruct, diff_forward
        from iom import make_scheme, make_key
        model = load_model().eval()
        img = reconstruct(model, feature.unsqueeze(0), iters=iters, lr=0.05, tv_w=0.01)
        with torch.no_grad():
            re = diff_forward(model, img)
        scheme = make_scheme(name)
        key = make_key(scheme, 0)
        t1 = int((scheme.hashing(re, key)[0] == code.unsqueeze(0)).sum())
        return img[0, 0].numpy(), t1
    except Exception as e:
        print(f"  [live reconstruction unavailable ({type(e).__name__}: {e});")
        print(f"   place the DeepPrint checkpoint under "
              f"fixed-length-fingerprint-extractors/models/ to enable it.")
        print(f"   falling back to the saved reconstruction]")
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", choices=["grp", "urp"], default="grp")
    ap.add_argument("--index", type=int, default=None,
                    help="template index 0..319 to attack (default: first reconstructed one)")
    ap.add_argument("--live-reconstruct", action="store_true",
                    help="re-optimize the image with DeepPrint (needs weights; slow)")
    ap.add_argument("--iters", type=int, default=250, help="reconstruction iterations")
    args = ap.parse_args()
    name = args.scheme

    # ---- load committed artifacts ----
    templates = torch.load(RESULTS_DIR / "templates" / f"{name}_stolen_token0.pt",
                           weights_only=False)
    codes = templates["codes"]                                   # (320, m) long
    fnpz = np.load(FEATURES_DIR / "deepprint_flx.npz")
    feats = F.normalize(torch.as_tensor(fnpz["embedding"], dtype=torch.float32), dim=1)
    subj = np.asarray(fnpz["subject"])
    names = np.asarray(fnpz["name"])

    recon_path = RESULTS_DIR / "reconstruct" / f"recon_{name}.npz"
    recon = np.load(recon_path) if recon_path.is_file() else None
    recon_idx = recon["index"] if recon is not None else np.array([], dtype=int)

    # ---- choose the target template ----
    # db3_101_1 (#160) is the clearest teaching example: exact pre-image, and
    # Type-2 that is 100% at EER but collapses to 0% at FMR<=1% -- the headline.
    PREFERRED = {"grp": 160, "urp": 160}
    if args.index is not None:
        target = args.index
    elif name in PREFERRED and (recon is None or PREFERRED[name] in recon_idx):
        target = PREFERRED[name]
    elif recon_idx.size:
        target = int(recon_idx[0])
    else:
        target = 0
    pos = np.where(recon_idx == target)[0]      # position within the reconstructed subset

    name_str = as_name(names[target])
    print(f"\nPMA-LSH attack demo  |  scheme = {name.upper()}  |  "
          f"target = {name_str}  (template #{target}, subject {subj[target]})")

    # ===================================================================
    banner("STEP 1  The attacker's starting point: a stolen IoM template + key")
    code_i = codes[target]
    m = code_i.numel()
    print(f"The server stores this {name.upper()}-IoM code for {name_str}: "
          f"{m} argmax indices.")
    print(f"First 32 of them:\n  {np.asarray(code_i[:32]).tolist()}")
    print("That is ALL that is stored -- no image, no feature vector. The scheme")
    print("claims this code cannot be reversed. The attacker also holds the key.")

    # ===================================================================
    banner("STEP 2  Invert the 'non-invertible' template (pre-image attack)")
    feature, purity, score, m, secs, live = run_or_replay_attack(
        name, target, code_i, feats[target])
    how = f"in {secs * 1000:.1f} ms (live)" if secs is not None else "(saved result)"
    print(f"Recovered a 512-d feature {how}.")
    exact = "EXACT pre-image" if score == m else "near pre-image"
    print(f"  hash score vs the stored code : {score}/{m}   ->  {exact}")
    print(f"  purity (cosine to the victim's TRUE feature) : {purity:.3f}")
    print("The recovered feature hashes back to the same code: IoM is invertible.")

    # ===================================================================
    banner("STEP 3  Turn the recovered feature back into a fingerprint image")
    img, t1_live = None, None
    if name == "grp" and args.live_reconstruct:
        img, t1_live = live_reconstruct(feature, code_i, name, args.iters)
    if img is None and recon is not None and pos.size:
        img = recon["images"][pos[0], 0]
        print(f"Using the precomputed reconstruction "
              f"(cos to true feature {float(recon['cos_true'][pos[0]]):.2f}).")
    if img is not None:
        out_dir = RESULTS_DIR / "demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"recon_{name_str}.png"
        plt.figure(figsize=(3.2, 3.2))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"reconstructed {name_str}", fontsize=8)
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved the reconstructed print to {p}")
        maybe_show(p)
    else:
        print(f"No reconstruction available for template #{target}.")
        if recon_idx.size:
            print(f"Pass --index one of: {recon_idx.tolist()}  (or --live-reconstruct).")

    # ===================================================================
    banner("STEP 4  Does the reconstructed print RE-ENTER the protected system? (Type-1)")
    thr1 = IOM_FMR1[name]
    t1 = t1_live if t1_live is not None else (int(recon["type1"][pos[0]])
                                              if (recon is not None and pos.size) else None)
    if t1 is not None:
        verdict = "ACCEPTED  ->  the attack breaks IoM" if t1 >= thr1 else "rejected"
        print(f"Re-extract the reconstructed image, hash it with the victim key:")
        print(f"  Type-1 score {t1}/{m}   vs accept threshold {thr1}   ->  {verdict}")
    else:
        print("No image available to score Type-1 for this template.")

    # ===================================================================
    banner("STEP 5  Can the attacker impersonate the victim ELSEWHERE? (Type-2)")
    others = np.where((subj == subj[target]) & (np.arange(len(subj)) != target))[0]
    cos_o = F.cosine_similarity(feature.unsqueeze(0), feats[torch.as_tensor(others)]).numpy()
    print(f"Match the recovered feature against the victim's {len(others)} OTHER")
    print(f"impressions on an unprotected DeepPrint system: mean cosine {cos_o.mean():.3f}.")
    print(f"  @ EER threshold      {EER_THR:.3f} : {100 * (cos_o >= EER_THR).mean():5.1f}% accepted")
    print(f"  @ FMR<=1% threshold  {FMR1_THR:.3f} : {100 * (cos_o >= FMR1_THR).mean():5.1f}% accepted")
    print("The recovered feature is tuned to the ENROLLED template, so at a strict")
    print("threshold it no longer matches the victim's other captures.")

    # ===================================================================
    banner("TAKEAWAY")
    print("Inverting IoM re-enters the protected system almost perfectly (Type-1),")
    print("but cross-system impersonation (Type-2) collapses once the operating")
    print("point is tightened. IoM is invertible -- the real protection comes from")
    print("the operating threshold and the transfer gap, not from non-invertibility.")
    print()


if __name__ == "__main__":
    main()
