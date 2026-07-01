"""
src/splitting/homology_split.py
================================
Homology-aware sequence splitting for protein aggregation datasets.

WHY this matters (the leakage problem)
---------------------------------------
The real lab data explores a protein region by systematic point-mutation
and acetylation variants.  For example, "AAGKTKEGVLYVGSK" may appear 6+
times with 1–2 residue changes.  If we naively split rows by random
shuffle, or even by pairwise similarity thresholding applied independently,
we get *homology leakage*:

  Naive pairwise: A ≁ C  → A and C land in different splits.  ✓ looks OK.
  But:            A ~ B  and  B ~ C  → if B is in train alongside A, the
                  model sees A-family patterns during training.  If C is
                  then placed in test, C's sequence (a mutation cousin of B,
                  which is a mutation cousin of A) leaks pattern signal.

The correct abstraction is TRANSITIVE CLOSURE via connected components:
  If A ~ B and B ~ C (above threshold), all three must land in the SAME
  split, regardless of whether A directly resembles C.  B is the bridge.

Implementation
--------------
1. Build a similarity graph: nodes = unique clean sequences, edges between
   pairs whose Levenshtein.ratio() ≥ threshold.
2. Compute connected components via Union-Find (no extra dependency).
3. Assign WHOLE clusters to train or test using a greedy row-count-weighted
   strategy: clusters are sorted smallest-first and added to test until the
   target test_size fraction (by ROW COUNT, not unique sequences) is reached.
   Remainder goes to train.

No external graph library required — Union-Find is O(n·α(n)) ≈ O(n).

Disease-agnostic: threshold comes from disease_config, never hardcoded.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

import pandas as pd
import Levenshtein

from src.features.ptm import encode_ptm_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) — fast connected-component detection
# ---------------------------------------------------------------------------

class _UnionFind:
    """Path-compressed, union-by-rank Disjoint Set Union structure.

    Provides near-O(1) amortised find/union.  Used here to compute
    transitive closure of the sequence similarity graph without building
    an explicit adjacency list (saves memory for large sequence sets).
    """

    def __init__(self, elements: list[Any]) -> None:
        self._parent: dict[Any, Any] = {e: e for e in elements}
        self._rank:   dict[Any, int] = {e: 0  for e in elements}

    def find(self, x: Any) -> Any:
        """Return the canonical representative of x's component."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])   # path compression
        return self._parent[x]

    def union(self, x: Any, y: Any) -> None:
        """Merge the components containing x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank — keeps tree shallow
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> dict[Any, int]:
        """Return {element: cluster_id} for all elements."""
        root_to_id: dict[Any, int] = {}
        result: dict[Any, int] = {}
        next_id = 0
        for element in self._parent:
            root = self.find(element)
            if root not in root_to_id:
                root_to_id[root] = next_id
                next_id += 1
            result[element] = root_to_id[root]
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_sequences(
    sequences: list[str],
    threshold: float,
) -> dict[str, int]:
    """Cluster sequences by transitive Levenshtein similarity.

    Two sequences are connected by an edge if their Levenshtein.ratio()
    (normalised edit similarity, range [0, 1]) is >= ``threshold``.
    Connected components — including transitively similar sequences that
    may not be DIRECTLY similar to each other — form one cluster each.

    Parameters
    ----------
    sequences : list[str]
        Amino-acid sequences (must already be PTM-cleaned, i.e. X → K,
        before calling this function).  Duplicates are deduplicated
        internally; the returned mapping covers every input sequence.
    threshold : float
        Similarity threshold in [0, 1].  Sequences with ratio >= threshold
        are placed in the same cluster.

    Returns
    -------
    dict[str, int]
        Mapping sequence → cluster_id (non-negative integer).
        Sequences in the same transitive-closure cluster share the same id.

    Raises
    ------
    ValueError
        If threshold is not in [0.0, 1.0].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be in [0.0, 1.0], got {threshold!r}"
        )

    unique_seqs: list[str] = list(dict.fromkeys(sequences))   # deduplicate, stable order
    n = len(unique_seqs)
    logger.debug("cluster_sequences: %d unique sequences, threshold=%.3f", n, threshold)

    uf = _UnionFind(unique_seqs)

    # O(n²) pairwise similarity — acceptable for typical wet-lab datasets
    # (hundreds to low thousands of unique sequences).  If n > 5000 consider
    # a k-mer LSH pre-filter before the exact Levenshtein pass.
    edges_added = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = Levenshtein.ratio(unique_seqs[i], unique_seqs[j])
            if sim >= threshold:
                uf.union(unique_seqs[i], unique_seqs[j])
                edges_added += 1

    cluster_map = uf.components()
    n_clusters = len(set(cluster_map.values()))
    logger.info(
        "cluster_sequences: %d sequences → %d clusters "
        "(%d similarity edges, threshold=%.3f)",
        n, n_clusters, edges_added, threshold,
    )
    return cluster_map


def split_train_test(
    df: pd.DataFrame,
    disease_config: dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long-format DataFrame into train and test sets at cluster level.

    GUARANTEE: No cluster is ever split across train and test.  All rows
    belonging to sequences in the same transitive-closure cluster land in
    the same split.  This eliminates homology leakage regardless of how
    similar the sequences are to one another.

    The test set fraction is approximate (clusters are indivisible), but
    the greedy assignment strategy keeps it within a few percent of the
    requested ``test_size``.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame as returned by load_dataset().  Must contain
        a ``peptide_sequence`` column.
    disease_config : dict
        Parsed disease YAML config.  Must contain
        ``homology_cluster_threshold``.
    test_size : float
        Approximate fraction of ROWS (not unique sequences) to assign to
        the test set.  Default: 0.2 (20 %).
    random_state : int
        Seed for reproducible cluster-to-split assignment when breaking
        ties in the greedy packing.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
        Both DataFrames retain ALL original columns.

    Raises
    ------
    ValueError
        If ``peptide_sequence`` column is absent or ``test_size`` is not
        in (0, 1).
    KeyError
        If ``homology_cluster_threshold`` is missing from disease_config.
    """
    if "peptide_sequence" not in df.columns:
        raise ValueError(
            "split_train_test() requires a 'peptide_sequence' column in df."
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(
            f"test_size must be strictly between 0 and 1, got {test_size!r}"
        )

    threshold: float = disease_config["homology_cluster_threshold"]

    # ------------------------------------------------------------------ #
    # 1. Cluster on CLEAN sequences (X → K) to cluster on biology,
    #    not on the raw representation.  This is critical: two sequences
    #    that are the same except one has acetylated K (X) and the other
    #    has unmodified K must land in the same cluster.
    # ------------------------------------------------------------------ #
    unique_raw_seqs: list[str] = df["peptide_sequence"].unique().tolist()
    raw_to_clean: dict[str, str] = {
        seq: encode_ptm_map(seq)[0] for seq in unique_raw_seqs
    }
    unique_clean_seqs: list[str] = list(dict.fromkeys(raw_to_clean.values()))

    clean_cluster_map: dict[str, int] = cluster_sequences(
        unique_clean_seqs, threshold
    )

    # Map every raw sequence to its cluster_id via its clean form
    raw_seq_cluster: dict[str, int] = {
        raw: clean_cluster_map[clean]
        for raw, clean in raw_to_clean.items()
    }

    # ------------------------------------------------------------------ #
    # 2. Count ROWS per cluster (weighted by concentration rows, not just
    #    unique sequences — important for greedy fraction calculation)
    # ------------------------------------------------------------------ #
    df = df.copy()
    df["_cluster_id"] = df["peptide_sequence"].map(raw_seq_cluster)

    cluster_row_counts: dict[int, int] = (
        df["_cluster_id"].value_counts().to_dict()
    )
    total_rows = len(df)
    target_test_rows = int(total_rows * test_size)

    # ------------------------------------------------------------------ #
    # 3. Greedy cluster assignment to test set
    #    Sort clusters smallest-first to pack closer to target fraction;
    #    shuffle within equal-size groups with seeded RNG for reproducibility.
    # ------------------------------------------------------------------ #
    rng = random.Random(random_state)
    all_cluster_ids: list[int] = list(cluster_row_counts.keys())
    all_cluster_ids.sort(key=lambda cid: (cluster_row_counts[cid], rng.random()))

    test_cluster_ids: set[int] = set()
    test_row_count = 0
    for cid in all_cluster_ids:
        if test_row_count >= target_test_rows:
            break
        test_cluster_ids.add(cid)
        test_row_count += cluster_row_counts[cid]

    # ------------------------------------------------------------------ #
    # 4. Build train / test DataFrames, drop internal helper column
    # ------------------------------------------------------------------ #
    is_test = df["_cluster_id"].isin(test_cluster_ids)
    test_df  = df[is_test].drop(columns=["_cluster_id"]).reset_index(drop=True)
    train_df = df[~is_test].drop(columns=["_cluster_id"]).reset_index(drop=True)

    actual_test_frac = len(test_df) / total_rows
    n_clusters = len(set(clean_cluster_map.values()))
    logger.info(
        "split_train_test: %d total rows → train=%d, test=%d "
        "(actual test frac=%.3f, requested=%.3f, %d clusters, threshold=%.3f)",
        total_rows, len(train_df), len(test_df),
        actual_test_frac, test_size, n_clusters, threshold,
    )
    return train_df, test_df
