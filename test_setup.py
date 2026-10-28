"""Environment check. Run from the project root: python test_setup.py

Every line should start with OK. NBIS is optional ("INFO" if missing): without
it, the scripts reuse the NBIS outputs saved in results/.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def check(name, ok, extra=""):
    print(("OK   " if ok else "FAIL ") + name + (" " + extra if extra else ""))


for mod in ["numpy", "scipy", "sklearn", "matplotlib", "PIL", "cv2", "torch"]:
    try:
        importlib.import_module(mod)
        check(mod, True)
    except Exception as e:
        check(mod, False, str(e))

import torch  # noqa: E402

check("torch CPU run", torch.ones(2).sum().item() == 2.0,
      "cuda=" + str(torch.cuda.is_available()))

try:
    from flx.extractor.fixed_length_extractor import DeepPrintExtractor  # noqa: F401
    check("flx DeepPrintExtractor", True)
except Exception as e:
    check("flx DeepPrintExtractor", False,
          f"{e} (run: pip install -e fixed-length-fingerprint-extractors)")

from config import (DATA_DIR, DEEPPRINT_MODEL_DIR, NBIS_BIN,  # noqa: E402
                    nbis_available)
from dataset import load_samples  # noqa: E402

try:
    check("FVC2004 Set B images", len(load_samples()) == 320, "(320 TIFFs)")
except FileNotFoundError as e:
    check("FVC2004 Set B images", False,
          f"missing {e}; run: python scripts/prepare_data.py <folder with DB1_B.zip ... DB4_B.zip>")

weights = DEEPPRINT_MODEL_DIR / "best_model.pyt"
check("DeepPrint weights", weights.is_file(),
      "" if weights.is_file() else f"missing {weights} (see README.md)")

if nbis_available():
    print(f"OK   NBIS found in {NBIS_BIN}")
else:
    print(f"INFO NBIS not built ({NBIS_BIN}); optional, saved NBIS results will be reused. "
          "To rebuild: bash scripts/build_nbis.sh")
