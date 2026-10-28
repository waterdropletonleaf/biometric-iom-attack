"""Shared paths and constants for the IoM attack project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fvc2004"
RESULTS_DIR = ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
LOGS_DIR = ROOT / "logs"

FLX_DIR = ROOT / "fixed-length-fingerprint-extractors"
DEEPPRINT_MODEL_DIR = FLX_DIR / "models"  # contains best_model.pyt

# The pre-trained Rohwedder checkpoint is DeepPrint_TexMinu with 256 texture
# + 256 minutia dims (512-d when concatenated), trained on 8000 subjects.
DEEPPRINT_NUM_TRAINING_SUBJECTS = 8000
DEEPPRINT_BRANCH_DIMS = 256

# NBIS binaries (mindtct, bozorth3, nfiq). Looked up in order: the NBIS_BIN env
# var, tools/nbis/bin (built by scripts/build_nbis.sh), then ~/nbis/bin.
# NBIS is optional: without it, scripts reuse the NBIS outputs saved in results/.
def _find_nbis_bin() -> Path:
    if "NBIS_BIN" in os.environ:
        return Path(os.environ["NBIS_BIN"])
    for cand in (ROOT / "tools" / "nbis" / "bin", Path.home() / "nbis" / "bin"):
        if (cand / "bozorth3").is_file():
            return cand
    return ROOT / "tools" / "nbis" / "bin"


NBIS_BIN = _find_nbis_bin()
MINDTCT = NBIS_BIN / "mindtct"
BOZORTH3 = NBIS_BIN / "bozorth3"


def nbis_available() -> bool:
    return MINDTCT.is_file() and BOZORTH3.is_file()

# FVC2004 "B" subsets: 10 fingers x 8 impressions each, 4 DBs -> 320 images.
DBS = ["db1", "db2", "db3", "db4"]
FINGERS_PER_DB = 10
IMPRESSIONS = 8
