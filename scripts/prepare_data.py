"""Place and verify the FVC2004 Set B images (the dataset is not in the repository).

FVC2004 is distributed by the Biometric System Laboratory, University of Bologna
(https://biolab.csr.unibo.it/ -> Databases -> FVC2004). Download the four Set B
archives, DB1_B.zip, DB2_B.zip, DB3_B.zip and DB4_B.zip (80 TIFFs each, 320 total),
into one folder, then run from the project root:

    python scripts/prepare_data.py ~/Downloads

This unpacks them into data/fvc2004/db<N>/DB<N>_B/<finger>_<impression>.tif and
checks every image against scripts/fvc2004_setB.sha256, the checksums of the exact
files used for the report. Already-extracted folders (DB1_B/ ... DB4_B/) in the
source folder also work. Run with --verify-only to check an existing data/fvc2004.
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "fvc2004"
MANIFEST = Path(__file__).resolve().parent / "fvc2004_setB.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def place(src: Path) -> None:
    for d in range(1, 5):
        name = f"DB{d}_B"
        out = DEST / f"db{d}" / name
        out.mkdir(parents=True, exist_ok=True)
        archive, folder = src / f"{name}.zip", src / name
        if archive.is_file():
            with zipfile.ZipFile(archive) as z:
                tifs = [m for m in z.namelist() if m.lower().endswith(".tif")]
                for m in tifs:
                    (out / Path(m).name).write_bytes(z.read(m))
            print(f"{name}: extracted {len(tifs)} images from {archive}")
        elif folder.is_dir():
            tifs = sorted(folder.glob("*.tif"))
            for t in tifs:
                shutil.copy2(t, out / t.name)
            print(f"{name}: copied {len(tifs)} images from {folder}")
        else:
            sys.exit(f"Missing {archive} (or folder {folder}). Download the FVC2004 "
                     f"Set B archives first; see the docstring of this script.")


def verify() -> bool:
    expected = dict(line.split()[::-1] for line in MANIFEST.read_text().splitlines())
    missing = [p for p in expected if not (DEST / p).is_file()]
    wrong = [p for p in expected if (DEST / p).is_file() and sha256(DEST / p) != expected[p]]
    if missing or wrong:
        print(f"FAILED: {len(missing)} missing, {len(wrong)} with different contents "
              f"(e.g. {(missing + wrong)[:3]})")
        return False
    print(f"OK: all {len(expected)} images present in {DEST} and identical to those "
          f"used for the report")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", nargs="?", type=Path,
                    help="folder containing DB1_B.zip ... DB4_B.zip (or DB1_B/ ... folders)")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        if args.source is None:
            ap.error("give the folder with the downloaded archives, or use --verify-only")
        place(args.source.expanduser())
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
