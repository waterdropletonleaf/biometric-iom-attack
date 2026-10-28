"""FVC2004 B-subset index (DB1_B..DB4_B concatenated, 320 samples).

Each finger in each DB is treated as a distinct identity, giving 40 subjects
with 8 impressions each. Subject ids are 0..39 (db-major), impressions 0..7.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DATA_DIR, DBS, FINGERS_PER_DB, IMPRESSIONS


@dataclass(frozen=True)
class Sample:
    index: int       # position in the 320-sample ordering
    db: int          # 0..3
    finger: int      # 0..9 within the DB (file prefix 101..110)
    impression: int  # 0..7
    path: Path

    @property
    def subject(self) -> int:
        return self.db * FINGERS_PER_DB + self.finger

    @property
    def name(self) -> str:
        return f"db{self.db + 1}_{101 + self.finger}_{self.impression + 1}"


def load_samples(data_dir: Path = DATA_DIR) -> list[Sample]:
    samples = []
    for d, db in enumerate(DBS):
        db_dir = data_dir / db / f"DB{d + 1}_B"
        for f in range(FINGERS_PER_DB):
            for i in range(IMPRESSIONS):
                path = db_dir / f"{101 + f}_{i + 1}.tif"
                if not path.is_file():
                    raise FileNotFoundError(path)
                samples.append(Sample(len(samples), d, f, i, path))
    return samples


def labels(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Return (subject_ids, db_ids) arrays aligned with `samples`."""
    return (
        np.array([s.subject for s in samples]),
        np.array([s.db for s in samples]),
    )
