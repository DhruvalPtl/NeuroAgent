"""
src/features/esm2_encoder.py
============================
ESM-2 sequence embeddings + PTM modification vector for the ESM2+CORAL model.

Design decisions
----------------
1.  Model is loaded ONCE per model_name, cached at module level in
    ``_ESM2_CACHE``.  Re-loading a 8M-parameter transformer on every
    encode_features() call would add ~2s per call on CPU — unacceptable
    for a 200-epoch training loop.

2.  ESM-2 tokenizer treats 'X' as an unknown token (UNK).  We replace
    'X' → 'K' BEFORE tokenising (reusing ptm.py's clean step).  The
    modification information is carried separately in the 5-dim
    modification vector, so no information is lost.

3.  Mean-pool over non-special token positions:
    ESM-2 prepends a <cls> token (index 0) and appends a <eos> token.
    We slice [1:-1] to exclude them, then mean-pool over the sequence
    axis.  This gives a fixed 320-dim float32 vector regardless of
    sequence length — the correct pooling strategy documented in the
    ESM paper (Rives et al. 2021).

4.  torch.no_grad() + model.eval() + frozen weights:
    We are using ESM-2 as a frozen feature extractor, never updating
    its weights.  no_grad() ensures no gradient graph is built (memory
    and speed), eval() disables dropout/batchnorm train-mode side effects.

Modification vector (5-dim)
----------------------------
Captures acetylation (X) pattern information that is stripped from the
clean sequence before ESM-2 sees it:
  [0] has_mod          — 1.0 if any 'X' present, else 0.0
  [1] n_mods_norm      — count('X') / len(sequence)
  [2] mean_pos_norm    — mean(X positions) / len(sequence)
  [3] std_pos_norm     — std(X positions) / len(sequence)  (0.0 if 0 or 1 X)
  [4] max_pos_norm     — max(X positions) / len(sequence)  (0.0 if no X)

All values are in [0, 1].  NaN is never returned — the std of a single
or empty set of X positions is defined as 0.0.

Combined ESM2 feature (325-dim) = 320-dim ESM2 + 5-dim modification vector.
This is what ESM2CoralModel.encode_features() stacks into X.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.features.ptm import encode_ptm_map

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache: keyed by model_name string
# ---------------------------------------------------------------------------

_ESM2_CACHE: dict[str, tuple] = {}   # model_name → (model, tokenizer)

ESM2_EMBEDDING_DIM = 320             # facebook/esm2_t6_8M_UR50D hidden size
MOD_VECTOR_DIM     = 5               # modification vector dimension
COMBINED_DIM       = ESM2_EMBEDDING_DIM + MOD_VECTOR_DIM  # 325


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_esm2_embedding(
    sequence: str,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
) -> np.ndarray:
    """Return a 320-dim mean-pooled ESM-2 embedding for one sequence.

    Parameters
    ----------
    sequence : str
        Raw peptide sequence.  'X' characters (acetylated K) are replaced
        with 'K' before tokenisation so ESM-2 sees only canonical AAs.
    model_name : str
        HuggingFace model identifier.  Defaults to the 8M-parameter ESM-2
        model (fastest, still outperforms tabular baselines on this task).

    Returns
    -------
    np.ndarray, shape (320,), dtype float32
        Mean-pooled CLS-excluded token embeddings.
    """
    model, tokenizer = _load_esm2(model_name)

    # PTM: replace X → K so ESM-2 sees only canonical amino acids
    clean_seq, _ = encode_ptm_map(sequence)

    import torch  # lazy import — not available at module load for non-GPU machines

    inputs = tokenizer(
        clean_seq,
        return_tensors="pt",
        add_special_tokens=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (1, seq_len+2, hidden_dim)
    # slice [1:-1] to exclude <cls> and <eos> special tokens
    hidden = outputs.last_hidden_state[0, 1:-1, :]  # (seq_len, 320)

    if hidden.shape[0] == 0:
        # Degenerate case: empty sequence
        return np.zeros(ESM2_EMBEDDING_DIM, dtype=np.float32)

    embedding = hidden.mean(dim=0).cpu().numpy().astype(np.float32)  # (320,)
    return embedding


def get_modification_vector(sequence: str) -> np.ndarray:
    """Return a 5-dim vector capturing acetylation (X) pattern statistics.

    Parameters
    ----------
    sequence : str
        Raw peptide sequence, possibly containing 'X' (acetylated lysine).

    Returns
    -------
    np.ndarray, shape (5,), dtype float32
        [has_mod, n_mods_norm, mean_pos_norm, std_pos_norm, max_pos_norm].
        All values in [0, 1].  Returns all-zeros if no 'X' present.
        Never returns NaN.
    """
    seq = sequence.upper() if sequence else ""
    n = len(seq)

    x_positions = [i for i, ch in enumerate(seq) if ch == "X"]
    n_mods = len(x_positions)

    if n == 0 or n_mods == 0:
        return np.zeros(MOD_VECTOR_DIM, dtype=np.float32)

    has_mod     = 1.0
    n_mods_norm = n_mods / n
    positions   = np.array(x_positions, dtype=np.float32)
    mean_pos    = float(positions.mean()) / n
    max_pos     = float(positions.max()) / n
    # std is undefined/0 for a single point — guard explicitly
    std_pos     = (float(positions.std()) / n) if n_mods > 1 else 0.0

    return np.array(
        [has_mod, n_mods_norm, mean_pos, std_pos, max_pos],
        dtype=np.float32,
    )


def encode_esm2_features(sequence: str, model_name: str = "facebook/esm2_t6_8M_UR50D") -> np.ndarray:
    """Combine ESM-2 embedding + modification vector into a 325-dim vector.

    This is the per-row feature function called by ESM2CoralModel.encode_features().

    Parameters
    ----------
    sequence : str
        Raw peptide sequence (may contain 'X').

    Returns
    -------
    np.ndarray, shape (325,), dtype float32
    """
    emb = get_esm2_embedding(sequence, model_name=model_name)
    mod = get_modification_vector(sequence)
    return np.concatenate([emb, mod]).astype(np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_esm2(model_name: str) -> tuple:
    """Load ESM-2 model + tokenizer, caching in _ESM2_CACHE.

    Thread-safety note: in the current single-process pipeline this is safe.
    For multi-worker DataLoader usage, load inside each worker separately.
    """
    if model_name in _ESM2_CACHE:
        return _ESM2_CACHE[model_name]

    logger.info("Loading ESM-2 model '%s' (first call — will be cached)", model_name)

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for ESM-2 embeddings. "
            "Install with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()

    # Freeze all parameters — we never fine-tune ESM-2 in this project
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        "ESM-2 '%s' loaded: %d parameters (all frozen)",
        model_name,
        sum(p.numel() for p in model.parameters()),
    )

    _ESM2_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer
