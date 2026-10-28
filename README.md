# PMA-LSH pre-image attack on fingerprint IoM templates

Course project, progress report 1. This repository sets up the fingerprint recognition system
that Paik et al.'s pre-image attack will target: unprotected baselines (DeepPrint, NBIS), the
protected systems (GRP-IoM and URP-IoM), data validation, exploratory data analysis, and
threshold and error analysis on FVC2004 Set B. The attack itself is the next phase.

Every number in the report is produced by the scripts below; `results/` holds their outputs
at the commit cited in the report.

## 1. Set up the environment

Python 3.12 on macOS or Linux; everything runs on CPU.

```bash
git clone --recurse-submodules <this-repo-url>   # submodules: Paik et al.'s code, DeepPrint code
cd biometric-iom-attack
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt                   # pinned dependencies + the DeepPrint package
```

`requirements.txt` lists what the scripts import, pinned to the versions used for the report.
`requirements-lock.txt` pins the full environment, including sub-dependencies, if an exact
replica is needed (`pip install -r requirements-lock.txt` then `pip install -e
fixed-length-fingerprint-extractors`).

**DeepPrint weights (836 MB, not in the repository).** Download the pretrained 512-d model,
`best_model.pyt`, from the Google Drive link in
[`fixed-length-fingerprint-extractors/README.md`](fixed-length-fingerprint-extractors/README.md)
into `fixed-length-fingerprint-extractors/models/`. Its SHA-256 is
`2683a04427bacd54adc00cfdc97474625b1e11e5a9e6672c5129f033018f8a28`.

## 2. Place the dataset (not in the repository)

FVC2004 is distributed for research use by the Biometric System Laboratory, University of
Bologna (<https://biolab.csr.unibo.it/>, *Databases → FVC2004*). Download the four **Set B**
archives, `DB1_B.zip`, `DB2_B.zip`, `DB3_B.zip` and `DB4_B.zip` (80 TIFF images each), into one
folder, then:

```bash
python scripts/prepare_data.py ~/Downloads        # the folder holding the four zips
```

This unpacks them into `data/fvc2004/db<N>/DB<N>_B/` and checks all 320 images against
`scripts/fvc2004_setB.sha256`, the checksums of the exact files used for the report.
Already-extracted folders work too; `--verify-only` checks an existing `data/fvc2004/`.

Then check everything:

```bash
python test_setup.py      # every line OK; NBIS may say INFO (optional, see section 5)
```

## 3. Run the pipeline

Run from the project root, in this order (about 2 minutes in total):

| Step | Script | What it does | Report |
|---|---|---|---|
| Data loading | [`src/dataset.py`](src/dataset.py) | Indexes the 320 images (40 fingers × 8 impressions); used by every script below | Sec. II |
| Feature extraction | [`src/extract_deepprint.py`](src/extract_deepprint.py) | Preprocesses images and extracts 512-d DeepPrint features (pretrained model) | Sec. II |
| Baseline evaluation | [`src/run_baseline.py`](src/run_baseline.py) | Scores all 51,040 pairs with DeepPrint (cosine) and NBIS (Bozorth3); EER, FNMR at FMR | Sec. IV |
| Protected systems | [`src/run_iom.py`](src/run_iom.py) | GRP-/URP-IoM hashing (Paik et al.'s code); accuracy, revocability, unlinkability; saves attack targets | Sec. IV |
| EDA | [`src/eda.py`](src/eda.py) | Image quality and statistics, correlations, score distributions, class balance | Sec. III, Figs. 1–5 |
| Threshold and error analysis | [`src/run_section4.py`](src/run_section4.py) | Rates with counts, FMR/FNMR curves, operating point, failure breakdown | Sec. IV, Tables I–II, Fig. 6 |

```bash
python src/extract_deepprint.py
python src/run_baseline.py
python src/run_iom.py
python src/eda.py
python src/run_section4.py
```

**No training step.** The baseline is a *pretrained* CNN (DeepPrint, trained by Rohwedder et al.
on SFinGe and MCYT330, which do not overlap with FVC2004), and IoM hashing has no learned
parameters. All 320 images are therefore evaluation data.

Each script's docstring describes its inputs, outputs and options; `NOTES.md` has the
design decisions and a troubleshooting table. Shared code: `src/config.py` (paths),
`src/metrics.py` (pair protocol, EER, unlinkability), `src/iom.py` (IoM wrapper),
`src/nbis_match.py` (NBIS calls).

## 4. Reproducibility

- **Seeds.** The only randomness in reported numbers is the IoM key, generated from token 0
  by a seeded PyTorch generator (`src/iom.py`). DeepPrint inference and NBIS are deterministic:
  re-extracting all 320 features gives bit-identical results.
- **Checking a run.** `results/` is committed, so `git diff results/` after a run shows any
  change. Expect exactly one: the hashing timings (`hash_ms_per_template`) in
  `results/iom/summary.json`, which vary between runs and are not used in the report. On a
  fresh Linux clone everything else reproduced byte for byte. On a different CPU (e.g. Apple
  silicon) DeepPrint may differ in the last floating-point digits.
- **Cited numbers** are in `results/section4/summary.json`, `results/iom/summary.json`,
  `results/baseline/summary_flx.json` and `results/eda/image_stats.csv`.

## 5. NBIS (optional)

NIST's NBIS (MINDTCT, Bozorth3, NFIQ) produced the minutiae and quality scores, which are
saved in `results/`. **The scripts run without NBIS installed**: `run_baseline.py` reuses the
saved Bozorth3 scores and `eda.py` the saved per-image statistics. To regenerate them, build
NBIS from the included source (NIST, public domain):

```bash
# macOS: xcode-select --install && brew install cmake     Linux: build-essential cmake
bash scripts/build_nbis.sh            # ~1 min; installs to tools/nbis/bin, found automatically
rm -r results/features/nbis_xyt results/eda/image_stats.csv    # force re-extraction
```

The build script was tested on Linux (GCC 13), where it reproduces the saved minutiae exactly;
it includes the flags recent Apple compilers need but has not been tested on a Mac.

## Repository layout

```
src/                         pipeline scripts (above)
scripts/prepare_data.py      places and verifies the dataset
scripts/fvc2004_setB.sha256  checksums of the 320 images used
scripts/build_nbis.sh        optional NBIS build
results/                     outputs cited in the report
Analysis_LSH/                Paik et al.'s code (submodule)
fixed-length-fingerprint-extractors/   DeepPrint implementation (submodule; weights go in models/)
nbis_v5_0_0.zip              NBIS 5.0.0 source
requirements.txt             dependencies;  requirements-lock.txt: full pinned environment
test_setup.py                environment and data check
NOTES.md                     design decisions, results log, troubleshooting
```
