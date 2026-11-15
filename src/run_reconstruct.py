"""Week 6-7: reconstruct fingerprint images from the recovered pre-images.

No public DeepPrint inverter exists and training a decoder needs a GPU + a
separate fingerprint corpus, so reconstruction here is **optimization-based
feature inversion**: for each recovered pre-image feature, optimise a 299x299
image so the frozen DeepPrint re-extracts that feature (no training data). The
network's eval forward is wrapped in torch.no_grad(), so we call its submodules
directly (eval mode kept, so BatchNorm/Dropout are unchanged) to get gradients
to the input.

Then, on the reconstructed images, we measure the image-level attack success:
  Type-1 (same protected system): re-extract -> hash with victim key -> matching
    score vs the stored IoM template >= IoM threshold.
  Type-2 (unprotected): re-extract -> cosine to the victim's OTHER impressions
    >= the unprotected threshold.
Both are compared with the Checkpoint-1 baseline rates.

CPU only, so ~1 min/image; use --limit for a subset. Seeded (reproducible).

Usage: python src/run_reconstruct.py [--scheme grp|urp] [--limit N] [--iters 250]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import FEATURES_DIR, RESULTS_DIR
from extract_deepprint import load_model
from iom import make_key, make_scheme, prepare

OUT = RESULTS_DIR / "reconstruct"
IOM_FMR1 = {"grp": 155, "urp": 198}
UNPROT_FMR1 = 0.8408539295196533
UNPROT_EER = 0.7047266960144043


def diff_forward(model, x):
    """Differentiable, eval-consistent DeepPrint forward (bypasses the no_grad
    wrapper in the eval forward; submodules stay in eval mode)."""
    s = model.stem(x)
    tex = model.texture_branch(s)
    minu = model.minutia_embedding(model.minutia_stem(s))
    return F.normalize(torch.cat([tex, minu], dim=1), dim=1)


def tv(x):
    return ((x[:, :, 1:] - x[:, :, :-1]).abs().mean()
            + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean())


def reconstruct(model, targets, iters, lr, tv_w, seed=0):
    """Optimise images whose re-extracted feature matches `targets` (N x 512)."""
    torch.manual_seed(seed)
    n = len(targets)
    z = torch.rand(n, 1, 299, 299, requires_grad=True)   # logits -> sigmoid = pixels
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        img = torch.sigmoid(z)
        e = diff_forward(model, img)
        loss = (1 - F.cosine_similarity(e, targets)).mean() + tv_w * tv(img)
        loss.backward()
        opt.step()
    return torch.sigmoid(z).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["grp", "urp"], default="grp")
    ap.add_argument("--limit", type=int, default=24, help="images to reconstruct (even; batched x2)")
    ap.add_argument("--balanced", action="store_true",
                    help="spread the selection evenly across the four DBs (else first N)")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tv", type=float, default=0.01)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    model = load_model().eval()
    pre = np.load(OUT.parent / "attack" / f"preimages_{args.scheme}.npz")
    targets_all = F.normalize(torch.as_tensor(pre["preimages"], dtype=torch.float32), dim=1)
    subj_all = pre["subject"]
    feats = prepare(np.load(FEATURES_DIR / "deepprint_flx.npz")["embedding"])
    scheme = make_scheme(args.scheme)
    key = make_key(scheme, 0)
    codes = torch.load(RESULTS_DIR / "templates" / f"{args.scheme}_stolen_token0.pt",
                       weights_only=False)["codes"]

    n = min(args.limit, len(targets_all))
    n -= n % 2
    if args.balanced:
        db = pre["db"]
        per = n // 4
        idx = np.concatenate([np.where(db == d)[0][:per] for d in range(4)])
    else:
        idx = np.arange(n)
    n = len(idx)
    targets = targets_all[idx]

    recon = []
    for b in range(0, n, 2):
        recon.append(reconstruct(model, targets[b:b + 2], args.iters, args.lr, args.tv))
        print(f"  reconstructed {b + 2}/{n}", flush=True)
    recon = torch.cat(recon)

    # Re-extract features from the reconstructions (true eval extractor)
    with torch.no_grad():
        re = torch.cat([diff_forward(model, recon[b:b + 2]) for b in range(0, n, 2)])

    cos_recovered = F.cosine_similarity(re, targets).numpy()
    cos_true = F.cosine_similarity(re, feats[idx]).numpy()

    # Type-1 (image-level): hash re-extracted feature, score vs stored template
    t1_scores = np.array([int((scheme.hashing(re[k:k + 1], key)[0] == codes[idx[k]:idx[k] + 1]).sum())
                          for k in range(n)])
    t1_thr = IOM_FMR1[args.scheme]

    # Type-2 (image-level, unprotected): re-extracted feature vs OTHER impressions
    t2 = []
    for k in range(n):
        i = idx[k]
        others = np.where((subj_all == subj_all[i]) & (np.arange(len(subj_all)) != i))[0]
        t2.append(F.cosine_similarity(re[k:k + 1], feats[others]).numpy())
    t2 = np.concatenate(t2)

    base = np.load(RESULTS_DIR / "baseline" / "scores_flx.npz")
    g = base["genuine"]; dp = base["DeepPrint-512"]

    res = {
        "scheme": args.scheme, "n": n, "iters": args.iters,
        "recon_cos_to_recovered_mean": float(cos_recovered.mean()),
        "recon_cos_to_true_mean": float(cos_true.mean()),
        "type1_image_asr_fmr1": {"threshold": t1_thr,
                                 "accepted": int((t1_scores >= t1_thr).sum()), "n": n,
                                 "asr": float((t1_scores >= t1_thr).mean())},
        "type2_image": {}, "baseline": {}}
    for pt, thr in [("EER", UNPROT_EER), ("FMR 1%", UNPROT_FMR1)]:
        res["type2_image"][pt] = {"threshold": thr, "accepted": int((t2 >= thr).sum()),
                                  "n": len(t2), "asr": float((t2 >= thr).mean())}
        res["baseline"][pt] = {"genuine_tar": float((dp[g] >= thr).mean()),
                               "impostor_accept": float((dp[~g] >= thr).mean())}

    res["n"] = n
    res["dbs_covered"] = sorted({int(pre["db"][i]) + 1 for i in idx})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"recon_{args.scheme}.npz", images=recon.numpy(),
             cos_recovered=cos_recovered, cos_true=cos_true, type1=t1_scores,
             subject=subj_all[idx], index=idx)
    (OUT / f"recon_{args.scheme}.json").write_text(json.dumps(res, indent=2))

    # Montage: a few reconstructions next to a real impression of the same finger
    k = min(6, n)
    fig, axes = plt.subplots(2, k, figsize=(1.5 * k, 3.2))
    import cv2
    from dataset import load_samples
    samples = load_samples()
    # space the montage across the selection so it shows different DBs
    show = np.linspace(0, n - 1, k).astype(int)
    for c, sidx in enumerate(show):
        axes[0, c].imshow(recon[sidx, 0].numpy(), cmap="gray"); axes[0, c].set_title(
            f"recon {samples[idx[sidx]].name}\ncos→true {cos_true[sidx]:.2f}", fontsize=6.5)
        img = cv2.imread(str(samples[idx[sidx]].path), cv2.IMREAD_GRAYSCALE)
        axes[1, c].imshow(img, cmap="gray"); axes[1, c].set_title("real impression", fontsize=7)
        for r in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle(f"{args.scheme.upper()} pre-image reconstructions ({args.iters} iters)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"montage_{args.scheme}.png", dpi=150, metadata={"CreationDate": None})
    plt.close(fig)

    t1, t2e, t2f = res["type1_image_asr_fmr1"], res["type2_image"]["EER"], res["type2_image"]["FMR 1%"]
    print("=" * 66)
    print(f"{args.scheme.upper()} reconstruction (n={n}, {args.iters} iters)")
    print(f"  recon re-extract cos: to recovered {res['recon_cos_to_recovered_mean']:.3f}, "
          f"to TRUE feature {res['recon_cos_to_true_mean']:.3f}")
    print(f"  Type-1 (image, FMR1% thr {t1['threshold']}): ASR {100*t1['asr']:.1f}% ({t1['accepted']}/{t1['n']})")
    print(f"  Type-2 (image) @EER:   ASR {100*t2e['asr']:.1f}%  vs genuine "
          f"{100*res['baseline']['EER']['genuine_tar']:.0f}% / impostor {100*res['baseline']['EER']['impostor_accept']:.0f}%")
    print(f"  Type-2 (image) @FMR1%: ASR {100*t2f['asr']:.1f}%  vs genuine "
          f"{100*res['baseline']['FMR 1%']['genuine_tar']:.0f}% / impostor {100*res['baseline']['FMR 1%']['impostor_accept']:.0f}%")


if __name__ == "__main__":
    main()
