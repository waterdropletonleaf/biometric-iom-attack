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

# NBIS binaries (mindtct, bozorth3). Override with the NBIS_BIN env var.
NBIS_BIN = Path(os.environ.get("NBIS_BIN", Path.home() / "nbis" / "bin"))
MINDTCT = NBIS_BIN / "mindtct"
BOZORTH3 = NBIS_BIN / "bozorth3"

# FVC2004 "B" subsets: 10 fingers x 8 impressions each, 4 DBs -> 320 images.
DBS = ["db1", "db2", "db3", "db4"]
FINGERS_PER_DB = 10
IMPRESSIONS = 8
