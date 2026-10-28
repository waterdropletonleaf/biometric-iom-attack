"""NBIS (mindtct + bozorth3) minutiae extraction and matching.

mindtct in this NBIS build cannot read FVC's TIFFs, so images are first
converted to 8-bit PNG tagged as 500 dpi.

Also used later to match reconstructed fingerprints against originals.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from config import BOZORTH3, MINDTCT


def _check_binaries(*binaries: Path):
    for b in binaries or (MINDTCT, BOZORTH3):
        if not b.is_file():
            raise FileNotFoundError(
                f"{b} not found; build NBIS with scripts/build_nbis.sh or set NBIS_BIN")


def extract_minutiae(image_path: Path, out_dir: Path, name: str) -> Path:
    """Run mindtct on one image; return the .xyt path (cached if present)."""
    xyt = out_dir / f"{name}.xyt"
    if xyt.is_file():
        return xyt
    _check_binaries(MINDTCT)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    Image.open(image_path).convert("L").save(png, dpi=(500, 500))
    subprocess.run([str(MINDTCT), str(png), str(out_dir / name)],
                   check=True, capture_output=True)
    # Keep .xyt/.min/.qm; drop the large intermediate maps and the PNG.
    for ext in (".brw", ".dm", ".hcm", ".lcm", ".lfm", ".png"):
        (out_dir / f"{name}{ext}").unlink(missing_ok=True)
    return xyt


def extract_all(image_paths: list[Path], names: list[str], out_dir: Path) -> list[Path]:
    return [extract_minutiae(p, out_dir, n) for p, n in zip(image_paths, names)]


def bozorth3_pairs(xyts: list[Path], pairs: np.ndarray) -> np.ndarray:
    """Bozorth3 score for each (probe, gallery) index pair, in one batch call."""
    _check_binaries(BOZORTH3)
    with tempfile.TemporaryDirectory() as tmp:
        mates = Path(tmp) / "mates.lis"
        with mates.open("w") as f:
            for i, j in pairs:
                f.write(f"{xyts[i]}\n{xyts[j]}\n")
        res = subprocess.run(
            [str(BOZORTH3), "-A", f"maxfiles={2 * len(pairs) + 10}", "-M", str(mates)],
            check=True, capture_output=True, text=True,
        )
    scores = np.array([int(line) for line in res.stdout.split()], dtype=np.int64)
    if len(scores) != len(pairs):
        raise RuntimeError(f"bozorth3 returned {len(scores)} scores for {len(pairs)} pairs")
    return scores


def minutiae_count(xyt: Path) -> int:
    with open(xyt) as f:
        return sum(1 for _ in f)
