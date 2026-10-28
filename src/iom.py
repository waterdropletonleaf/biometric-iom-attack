"""GRP-IoM and URP-IoM (Jin et al. 2017) for DeepPrint embeddings.

Hashing, scoring and the LSP functions used by the attack come straight from
Paik et al.'s implementation (Analysis_LSH/LSH.py) so results stay comparable.
The only addition is reproducible keys: a user token (int) seeds a PRNG that
generates the random matrix / permutations, as in the original IoM design.
Paik's setup() draws from torch's global RNG instead.

Parameters default to Paik et al.'s preset (Analysis_LSH/preset.py).
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from config import ROOT

sys.path.insert(0, str(ROOT / "Analysis_LSH"))
from LSH import GRP, URP  # noqa: E402  (Paik et al.)

EMB_SIZE = 512
GRP_PARAMS = {"m": 300, "q": 16}
URP_PARAMS = {"m": 600, "w": 100, "p": 2}


def make_scheme(name: str, **params):
    p = {"grp": GRP_PARAMS, "urp": URP_PARAMS}[name] | params
    if name == "grp":
        return GRP(p["m"], p["q"], EMB_SIZE, "cpu")
    return URP(p["m"], p["w"], p["p"], EMB_SIZE, "cpu")


def make_key(scheme, token: int) -> torch.Tensor:
    """Deterministic version of scheme.setup() driven by a user token."""
    g = torch.Generator().manual_seed(int(token))
    if isinstance(scheme, GRP):
        return torch.randn((scheme.m * scheme.q, scheme.emb_size), generator=g)
    if isinstance(scheme, URP):
        return torch.randn((scheme.m * scheme.p, scheme.emb_size), generator=g).argsort()
    raise TypeError(type(scheme))


def prepare(embeddings: np.ndarray) -> torch.Tensor:
    """L2-normalize features as Paik et al. do. (Both IoM variants are
    scale-invariant, so this only matters for comparing pre-images later.)"""
    return F.normalize(torch.as_tensor(embeddings, dtype=torch.float32), dim=1)


@torch.no_grad()
def hash_all(scheme, x: torch.Tensor, key: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """IoM codes (N, m). Batched because URP materializes N x m*p x 512."""
    return torch.cat([scheme.hashing(x[b:b + batch_size], key)[0]
                      for b in range(0, len(x), batch_size)])


def match_scores(codes_a: torch.Tensor, codes_b: torch.Tensor, pairs: np.ndarray) -> np.ndarray:
    """Number of equal hash entries for each pair (codes_a[i], codes_b[j]),
    i.e. Paik's LSH.score() for precomputed codes."""
    i, j = torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1])
    return (codes_a[i] == codes_b[j]).sum(dim=1).numpy()
