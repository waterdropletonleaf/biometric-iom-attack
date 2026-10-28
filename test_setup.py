import importlib, os, shutil

def check(name, ok, extra=""):
    print(("OK   " if ok else "FAIL ") + name + (" " + extra if extra else ""))

for mod in ["numpy", "scipy", "sklearn", "matplotlib", "PIL", "torch"]:
    try:
        importlib.import_module(mod); check(mod, True)
    except Exception as e:
        check(mod, False, str(e))

import torch
check("torch CPU run", torch.ones(2).sum().item() == 2.0,
      "cuda=" + str(torch.cuda.is_available()))

try:
    from flx.extractor.fixed_length_extractor import DeepPrintExtractor
    check("flx DeepPrintExtractor", True)
except Exception as e:
    check("flx DeepPrintExtractor", False, str(e))

nbis_bin = os.environ.get("NBIS_BIN", os.path.expanduser("~/nbis/bin"))
for tool in ["mindtct", "bozorth3"]:
    found = shutil.which(tool) or (os.path.join(nbis_bin, tool) if os.path.isfile(os.path.join(nbis_bin, tool)) else None)
    check(tool, found is not None, found or "")
check("FVC2004 data folder", os.path.isdir("data/fvc2004"))
