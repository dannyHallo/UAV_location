# ---------- seed_utils.py ----------------------------------------------------
"""
Small helper around numpy.random.SeedSequence that guarantees
collision-free child seeds in a deterministic way.
"""

from __future__ import annotations
from typing import List
import numpy as np

def spawn_child_seeds(parent_entropy: int, n_children: int) -> List[int]:
    """
    Expand one integer seed (`parent_entropy`) into `n_children`
    unique 32-bit child seeds using numpy.random.SeedSequence.

    Returns
    -------
    List[int]  -- length == n_children, each item ∈ [0, 2**32 – 1]
    """
    ss_parent = np.random.SeedSequence(parent_entropy)
    child_seqs = ss_parent.spawn(n_children)
    # compress the full 128-bit state into a plain 32-bit integer that the rest
    # of the code (PyTorch, random, etc.) can accept.
    child_seeds = [int(cs.generate_state(1, dtype="uint32")[0]) for cs in child_seqs]
    return child_seeds
# -----------------------------------------------------------------------------

