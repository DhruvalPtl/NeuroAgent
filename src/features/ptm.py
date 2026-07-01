"""
src/features/ptm.py
===================
Post-translational modification (PTM) encoding — dual-stream approach.

WHY dual-stream?
----------------
In the raw lab data, an acetylated lysine residue is encoded as 'X'
(a non-standard character) rather than 'K'.  Standard amino-acid encoders
(one-hot, physicochemical lookup tables, ESM-2 tokenisers) are built around
the 20 canonical amino acids.  If 'X' is passed through unchanged it will
either:
  (a) be treated as an unknown/padding token — losing the information that
      this IS a lysine (just modified), or
  (b) raise a KeyError / index-out-of-bounds error in lookup tables.

The dual-stream approach avoids both problems:
  Stream 1 — clean_sequence : 'X' → 'K', so every encoder sees only the
             20 standard amino acids and handles the residue correctly.
  Stream 2 — ptm_position_mask : binary array (1 where 'X' was, 0 elsewhere)
             that travels alongside the sequence through the pipeline, letting
             the feature encoder append PTM site information without
             corrupting the primary encoding.

The two streams are merged in src/features/encoder.py, which appends
ptm_position_mask as an extra per-residue feature column before pooling.

This is disease-agnostic: 'X'-as-acetylated-K is the only PTM currently
encoded at the residue level.  Other PTM types (e.g. phosphorylation) are
captured as a global scalar flag in the disease config and handled at the
row level by the is_acetylated / future is_phosphorylated columns — not here.
"""

from __future__ import annotations

import numpy as np


def encode_ptm_map(sequence: str) -> tuple[str, np.ndarray]:
    """Separate PTM markers from the amino-acid sequence (dual-stream).

    Parameters
    ----------
    sequence : str
        Raw amino-acid sequence, possibly containing 'X' characters
        marking K-acetylation sites (as used in the alpha-synuclein
        lab dataset).

    Returns
    -------
    clean_sequence : str
        The sequence with every 'X' replaced by 'K' (the underlying
        natural amino acid).  Safe to pass to any standard encoder.
    ptm_position_mask : np.ndarray, shape (len(sequence),), dtype float32
        Binary array: 1.0 at each position that was originally 'X',
        0.0 everywhere else.  Same length as the input sequence.
        dtype float32 matches the per-residue feature matrices produced
        by encoder.py, allowing direct column concatenation.

    Examples
    --------
    >>> clean, mask = encode_ptm_map("AAXAA")
    >>> clean
    'AAKAA'
    >>> mask.tolist()
    [0.0, 0.0, 1.0, 0.0, 0.0]

    >>> clean, mask = encode_ptm_map("ACDEF")   # no PTM
    >>> mask.sum()
    0.0

    >>> clean, mask = encode_ptm_map("XAKXM")   # two PTM sites
    >>> clean
    'KAKKM'
    >>> mask.tolist()
    [1.0, 0.0, 0.0, 1.0, 0.0]
    """
    if not isinstance(sequence, str):
        raise TypeError(
            f"sequence must be a str, got {type(sequence).__name__!r}"
        )
    if len(sequence) == 0:
        return "", np.zeros(0, dtype=np.float32)

    sequence_upper = sequence.upper()
    ptm_mask = np.array(
        [1.0 if ch == "X" else 0.0 for ch in sequence_upper],
        dtype=np.float32,
    )
    clean_sequence = sequence_upper.replace("X", "K")
    return clean_sequence, ptm_mask
