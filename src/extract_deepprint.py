"""Extract DeepPrint (TexMinu, 256+256 = 512-d) embeddings for FVC2004 B.

Preprocessing follows flx's FVC2004 pipeline (pad to square, resize to 299,
Gabor binarization). flx tunes the binarizer's ridge width for DB1 only
(1.8 px at 640 px -> 299 px, i.e. ~3.8 px at native 500 dpi). The B subsets
have different image sizes, so `--prep scaled` rescales the ridge width per
image (3.8 * 299 / max(h, w)); `--prep flx` uses 1.8 for every DB as flx does.

Output: results/features/deepprint_<prep>.npz with
  texture (320, 256), minutia (320, 256), embedding (320, 512, both halves
  concatenated), subject, db, impression, name.

Usage: python src/extract_deepprint.py [--prep scaled|flx]
"""
import argparse
import time

import cv2
import numpy as np
import torch

from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.setup.config import INPUT_SIZE

from config import (DEEPPRINT_BRANCH_DIMS, DEEPPRINT_MODEL_DIR,
                    DEEPPRINT_NUM_TRAINING_SUBJECTS, FEATURES_DIR)
from dataset import load_samples

NATIVE_RIDGE_WIDTH = 3.8  # flx's value for 500 dpi optical (MCYT) at ~1:1 scale
FLX_FVC_RIDGE_WIDTH = 1.8


def load_model():
    extractor = get_DeepPrint_TexMinu(
        num_training_subjects=DEEPPRINT_NUM_TRAINING_SUBJECTS,
        num_dims=DEEPPRINT_BRANCH_DIMS,
    )
    extractor.load_best_model(str(DEEPPRINT_MODEL_DIR))
    return extractor.model.eval()


def preprocess(path, prep: str, binarizers: dict) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(path)
    if prep == "flx":
        width = FLX_FVC_RIDGE_WIDTH
    elif prep == "scaled":
        width = round(NATIVE_RIDGE_WIDTH * INPUT_SIZE / max(img.shape), 2)
    else:
        raise ValueError(prep)
    if width not in binarizers:
        binarizers[width] = LazilyAllocatedBinarizer(width)
    x = pad_and_resize_to_deepprint_input_size(img, fill=1.0)
    return binarizers[width](x)


@torch.no_grad()
def extract(prep: str, batch_size: int = 32) -> dict:
    samples = load_samples()
    model = load_model()
    binarizers = {}
    tex, minu = [], []
    t0 = time.time()
    for b in range(0, len(samples), batch_size):
        batch = torch.stack([preprocess(s.path, prep, binarizers)
                             for s in samples[b:b + batch_size]])
        out = model(batch)
        tex.append(out.texture_embeddings.numpy())
        minu.append(out.minutia_embeddings.numpy())
        print(f"  {min(b + batch_size, len(samples))}/{len(samples)}", flush=True)
    print(f"Extracted {len(samples)} samples in {time.time() - t0:.1f}s")
    tex, minu = np.concatenate(tex), np.concatenate(minu)
    return {
        "texture": tex,
        "minutia": minu,
        "embedding": np.concatenate([tex, minu], axis=1),
        "subject": np.array([s.subject for s in samples]),
        "db": np.array([s.db for s in samples]),
        "impression": np.array([s.impression for s in samples]),
        "name": np.array([s.name for s in samples]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", choices=["scaled", "flx"], default="flx")
    args = ap.parse_args()
    feats = extract(args.prep)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURES_DIR / f"deepprint_{args.prep}.npz"
    np.savez(out, **feats)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
