"""
tests/test_splitting.py
========================
Tests for src/splitting/homology_split.py

Key correctness properties verified:

1. TRANSITIVE CLOSURE: If A~B and B~C (above threshold) but A is NOT
   directly similar enough to C, all three must land in the SAME cluster.
   This is the primary leakage-prevention mechanism.

2. ZERO LEAKAGE: No (train_seq, test_seq) pair may have Levenshtein.ratio()
   >= threshold.  Verified by brute-force O(n²) pairwise check that is
   INDEPENDENT of the clustering logic — it does not call cluster_sequences()
   to verify itself.

3. TEST FRACTION: Actual test-set row count is within a reasonable tolerance
   of the requested fraction (clusters are indivisible, so exact match is
   impossible; we accept ±15 percentage points).

4. CLUSTER INTEGRITY: Every row belonging to a clustered sequence lands
   in the same split (no cluster is ever split across train/test).

5. REAL DATA END-TO-END: Split on all 896 real lab rows with no crash,
   no leakage, sensible row counts.
"""

from __future__ import annotations

import pathlib
from itertools import product

import pandas as pd
import pytest
import Levenshtein

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_REAL_FILE = _REPO_ROOT / "data" / "raw" / "alpha_synuclein" / "real_lab_batch_001.xlsx"
_CONFIG_PATH = str(_REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml")

from src.splitting.homology_split import cluster_sequences, split_train_test, _UnionFind

import yaml


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alpha_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _minimal_config(threshold: float = 0.85) -> dict:
    return {
        "name": "test",
        "homology_cluster_threshold": threshold,
        "label_schema": [None, "Low", "Medium", "High"],
        "ptm_types": ["acetylation"],
        "raw_data_path": "data/raw/alpha_synuclein/",
        "sequence_column": "peptide_sequence",
        "label_column": "label_ordinal",
    }


def _make_df(seq_label_conc: list[tuple[str, int, float]]) -> pd.DataFrame:
    """Build a minimal long-format DataFrame for splitting tests."""
    rows = [
        {
            "sequence_id": str(i),
            "peptide_sequence": seq,
            "concentration": conc,
            "label_ordinal": label,
            "is_acetylated": "X" in seq,
            "source_file": "test.csv",
            "data_snapshot_hash": "deadbeef",
        }
        for i, (seq, label, conc) in enumerate(seq_label_conc)
    ]
    df = pd.DataFrame(rows)
    df["concentration"] = df["concentration"].astype(float)
    df["is_acetylated"] = df["is_acetylated"].astype(bool)
    df["label_ordinal"] = df["label_ordinal"].astype(int)
    return df


def _brute_force_leakage_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Return all (train_seq, test_seq, similarity) pairs that exceed threshold.

    This is a BRUTE-FORCE, INDEPENDENT verification of zero leakage.
    It does NOT use cluster_sequences() — it directly recomputes pairwise
    similarities between unique train sequences and unique test sequences,
    using Levenshtein.ratio() from the same library but via a completely
    separate code path.

    Returns an empty list if there is no leakage.
    """
    train_seqs = train_df["peptide_sequence"].unique().tolist()
    test_seqs  = test_df["peptide_sequence"].unique().tolist()

    violations: list[tuple[str, str, float]] = []
    for t_seq, v_seq in product(train_seqs, test_seqs):
        sim = Levenshtein.ratio(t_seq, v_seq)
        if sim >= threshold:
            violations.append((t_seq, v_seq, sim))
    return violations


# ===========================================================================
# 1. Union-Find internals
# ===========================================================================

class TestUnionFind:

    def test_initial_each_own_component(self):
        uf = _UnionFind(["A", "B", "C"])
        comps = uf.components()
        assert len(set(comps.values())) == 3

    def test_union_merges_components(self):
        uf = _UnionFind(["A", "B", "C"])
        uf.union("A", "B")
        comps = uf.components()
        assert comps["A"] == comps["B"]
        assert comps["C"] != comps["A"]

    def test_transitive_union(self):
        uf = _UnionFind(["A", "B", "C"])
        uf.union("A", "B")
        uf.union("B", "C")
        comps = uf.components()
        assert comps["A"] == comps["B"] == comps["C"]

    def test_find_is_idempotent(self):
        uf = _UnionFind(["X", "Y"])
        uf.union("X", "Y")
        assert uf.find("X") == uf.find("X")
        assert uf.find("X") == uf.find("Y")


# ===========================================================================
# 2. cluster_sequences — transitive closure correctness
# ===========================================================================

class TestClusterSequences:
    """
    Adversarial triple:
      A = "ACDEFGHIKLM"   (11 chars)
      B = "ACDEFGHIKEM"   (1 edit from A: L→E at pos 9)   ratio(A,B)=10/11=0.909
      C = "ACDEFGHIKEC"   (1 edit from B: M→C at pos 10)  ratio(B,C)=10/11=0.909
                                                            ratio(A,C)=9/11=0.818

    At threshold=0.85: A~B ✓, B~C ✓, A≁C ✓  → must all be in same cluster.
    """
    SEQ_A = "ACDEFGHIKLM"
    SEQ_B = "ACDEFGHIKEM"
    SEQ_C = "ACDEFGHIKEC"
    THRESHOLD = 0.85

    def test_adversarial_ratios_as_expected(self):
        """Confirm the adversarial triple has the intended similarity structure."""
        ab = Levenshtein.ratio(self.SEQ_A, self.SEQ_B)
        bc = Levenshtein.ratio(self.SEQ_B, self.SEQ_C)
        ac = Levenshtein.ratio(self.SEQ_A, self.SEQ_C)
        assert ab >= self.THRESHOLD, f"A~B ratio {ab:.4f} < threshold {self.THRESHOLD}"
        assert bc >= self.THRESHOLD, f"B~C ratio {bc:.4f} < threshold {self.THRESHOLD}"
        assert ac  < self.THRESHOLD, f"A~C ratio {ac:.4f} >= threshold {self.THRESHOLD} (not adversarial)"

    def test_transitive_closure_all_same_cluster(self):
        """A, B, C must all land in the SAME cluster (transitive closure)."""
        result = cluster_sequences([self.SEQ_A, self.SEQ_B, self.SEQ_C], self.THRESHOLD)
        assert result[self.SEQ_A] == result[self.SEQ_B] == result[self.SEQ_C], (
            f"Transitive closure failed: "
            f"cluster(A)={result[self.SEQ_A]}, "
            f"cluster(B)={result[self.SEQ_B]}, "
            f"cluster(C)={result[self.SEQ_C]}. "
            "All three must share one cluster (B bridges A and C)."
        )

    def test_unrelated_sequence_separate_cluster(self):
        """A sequence unrelated to A/B/C must get its own cluster."""
        unrelated = "WWWWWWWWWWW"   # all-Trp, highly distinct
        result = cluster_sequences(
            [self.SEQ_A, self.SEQ_B, self.SEQ_C, unrelated], self.THRESHOLD
        )
        abc_cluster = result[self.SEQ_A]
        assert result[unrelated] != abc_cluster, (
            "Unrelated sequence incorrectly merged into the A/B/C cluster."
        )

    def test_returns_dict_with_all_inputs(self):
        seqs = [self.SEQ_A, self.SEQ_B, self.SEQ_C]
        result = cluster_sequences(seqs, self.THRESHOLD)
        assert set(result.keys()) == set(seqs)

    def test_cluster_ids_are_integers(self):
        result = cluster_sequences([self.SEQ_A, self.SEQ_B], self.THRESHOLD)
        assert all(isinstance(v, int) for v in result.values())

    def test_threshold_1_all_singletons(self):
        """At threshold=1.0 only identical sequences cluster together."""
        seqs = ["ACDEF", "ACDEG", "GHIKL"]
        result = cluster_sequences(seqs, threshold=1.0)
        assert len(set(result.values())) == 3   # each in its own cluster

    def test_threshold_0_all_one_cluster(self):
        """At threshold=0.0 every pair is similar — one giant cluster."""
        seqs = ["ACDEF", "WWWWW", "RRRRR"]
        result = cluster_sequences(seqs, threshold=0.0)
        assert len(set(result.values())) == 1

    def test_duplicate_inputs_handled(self):
        """Duplicate sequences in input must not crash."""
        seqs = [self.SEQ_A, self.SEQ_A, self.SEQ_B]
        result = cluster_sequences(seqs, self.THRESHOLD)
        assert result[self.SEQ_A] == result[self.SEQ_B]

    def test_single_sequence(self):
        result = cluster_sequences(["ACDEF"], threshold=0.9)
        assert result == {"ACDEF": 0}

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            cluster_sequences(["ACDEF"], threshold=1.5)


# ===========================================================================
# 3. split_train_test — split properties
# ===========================================================================

class TestSplitTrainTest:

    # ------------------------------------------------------------------
    # 3a. Transitive-closure test on the adversarial triple
    # ------------------------------------------------------------------

    def test_adversarial_triple_in_same_split(self):
        """A, B, C (transitive-closure triple) must land in the same split."""
        SEQ_A = "ACDEFGHIKLM"
        SEQ_B = "ACDEFGHIKEM"
        SEQ_C = "ACDEFGHIKEC"
        cfg = _minimal_config(threshold=0.85)

        # Build a dataset with many concentrations so the cluster has weight
        concs = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
        rows = [(s, 0, c) for s in [SEQ_A, SEQ_B, SEQ_C] for c in concs]
        # Add filler sequences to give the splitter something to assign to test
        for i in range(10):
            filler = f"WWWWWWWWWW{i % 10}"
            for c in concs:
                rows.append((filler[:11], 1, c))

        df = _make_df(rows)
        train, test = split_train_test(df, cfg, test_size=0.3, random_state=0)

        abc_seqs = {SEQ_A, SEQ_B, SEQ_C}
        train_has_abc = train["peptide_sequence"].isin(abc_seqs).any()
        test_has_abc  = test["peptide_sequence"].isin(abc_seqs).any()
        assert not (train_has_abc and test_has_abc), (
            "Transitive-closure cluster {A, B, C} was split across train AND test.\n"
            f"A/B/C in train: {train_has_abc}, in test: {test_has_abc}"
        )

    # ------------------------------------------------------------------
    # 3b. Zero leakage — brute-force independent verification
    # ------------------------------------------------------------------

    def test_zero_leakage_brute_force(self):
        """No (train, test) sequence pair may exceed the similarity threshold.

        IMPORTANT: This check uses Levenshtein.ratio() directly on every
        cross-split pair — it does NOT call cluster_sequences() and is
        therefore a fully independent falsification of the splitting logic.
        """
        cfg = _minimal_config(threshold=0.85)
        # Sequences with varied similarity
        seqs = [
            "ACDEFGHIKLM",   # cluster with B, C
            "ACDEFGHIKEM",
            "ACDEFGHIKEC",
            "MNPQRSTVWYAC",  # unrelated
            "GHIKLMNPQRST",
            "WWWWWWWWWWWW",
        ]
        rows = [(s, 0, float(c)) for s in seqs for c in [0.5, 1.0, 2.0]]
        df = _make_df(rows)
        train, test = split_train_test(df, cfg, test_size=0.3, random_state=42)

        violations = _brute_force_leakage_check(
            train, test, threshold=cfg["homology_cluster_threshold"]
        )
        assert len(violations) == 0, (
            f"Leakage detected! {len(violations)} cross-split pair(s) "
            f"exceed threshold {cfg['homology_cluster_threshold']}:\n"
            + "\n".join(f"  train={t!r}, test={v!r}, sim={s:.4f}"
                        for t, v, s in violations[:5])
        )

    # ------------------------------------------------------------------
    # 3c. Test fraction within tolerance
    # ------------------------------------------------------------------

    def test_test_fraction_within_tolerance(self):
        """Actual test fraction must be within ±15 pp of the requested fraction."""
        cfg = _minimal_config(threshold=0.9)
        # Construct varied sequences so clusters are small and splitter
        # has flexibility to approach the target.
        base_seqs = [
            "ACDEFGHIKL",
            "MNPQRSTVWY",
            "GHIKLMNPQR",
            "WWWWWWWWWW",
            "AAAABBBBCC",
            "DDEEEFFFFF",
        ]
        rows = [(s, 0, float(c)) for s in base_seqs for c in [0.1, 0.5, 1.0, 2.0]]
        df = _make_df(rows)
        target = 0.2
        train, test = split_train_test(df, cfg, test_size=target, random_state=0)
        actual = len(test) / len(df)
        assert abs(actual - target) <= 0.15, (
            f"Test fraction {actual:.3f} is more than 15pp away from "
            f"target {target:.3f}."
        )

    # ------------------------------------------------------------------
    # 3d. Cluster integrity — no cluster split across train/test
    # ------------------------------------------------------------------

    def test_no_cluster_split_across_splits(self):
        """Every sequence in a cluster must be entirely in train OR entirely in test."""
        cfg = _minimal_config(threshold=0.9)
        seqs = [
            "ACDEFGHIKLM",
            "ACDEFGHIKEM",   # similar to above (same cluster at 0.9)
            "WWWWWWWWWWW",   # isolated cluster
        ]
        rows = [(s, 0, float(c)) for s in seqs for c in [0.5, 1.0, 2.0, 4.0]]
        df = _make_df(rows)

        # Build reference cluster map to check integrity
        from src.features.ptm import encode_ptm_map
        clean_seqs = [encode_ptm_map(s)[0] for s in seqs]
        cluster_map = cluster_sequences(clean_seqs, cfg["homology_cluster_threshold"])
        # Map raw → cluster via clean
        raw_to_cluster = {
            raw: cluster_map[encode_ptm_map(raw)[0]] for raw in seqs
        }

        train, test = split_train_test(df, cfg, test_size=0.4, random_state=7)

        for seq, cid in raw_to_cluster.items():
            # Find all sequences with the same cluster id
            same_cluster_seqs = {s for s, c in raw_to_cluster.items() if c == cid}
            in_train = set(train["peptide_sequence"].unique()) & same_cluster_seqs
            in_test  = set(test["peptide_sequence"].unique())  & same_cluster_seqs
            assert not (in_train and in_test), (
                f"Cluster {cid} (seqs={same_cluster_seqs}) was split: "
                f"train has {in_train}, test has {in_test}"
            )

    # ------------------------------------------------------------------
    # 3e. Row conservation
    # ------------------------------------------------------------------

    def test_total_rows_conserved(self):
        """train + test must equal total rows — no rows dropped or duplicated."""
        cfg = _minimal_config(threshold=0.9)
        seqs = ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"]
        rows = [(s, 0, float(c)) for s in seqs for c in [0.5, 1.0, 2.0]]
        df = _make_df(rows)
        train, test = split_train_test(df, cfg, test_size=0.3, random_state=1)
        assert len(train) + len(test) == len(df), (
            f"Row count mismatch: train={len(train)}, test={len(test)}, "
            f"total={len(df)}"
        )

    # ------------------------------------------------------------------
    # 3f. Column preservation
    # ------------------------------------------------------------------

    def test_all_columns_preserved(self):
        """All input columns must be present in both train and test."""
        cfg = _minimal_config(threshold=0.9)
        seqs = ["ACDEFGHIK", "LMNPQRSTV"]
        rows = [(s, 0, 1.0) for s in seqs]
        df = _make_df(rows)
        original_cols = set(df.columns)
        train, test = split_train_test(df, cfg, test_size=0.4, random_state=0)
        assert set(train.columns) == original_cols
        assert set(test.columns)  == original_cols

    # ------------------------------------------------------------------
    # 3g. PTM-cleaned clustering: X and K variants cluster together
    # ------------------------------------------------------------------

    def test_x_and_k_variants_cluster_together(self):
        """A sequence with X and its K counterpart must be in the same split."""
        cfg = _minimal_config(threshold=0.9)
        # These differ ONLY in X vs K — same sequence biologically
        seq_k = "ACDEFGHIKLM"
        seq_x = "ACDEFGHIXLM"   # K at pos 8 → X (acetylated)
        unrelated = "WWWWWWWWWWW"
        rows = [
            (seq_k, 0, 1.0), (seq_k, 0, 2.0),
            (seq_x, 0, 1.0), (seq_x, 0, 2.0),
        ] + [(unrelated, 0, float(c)) for c in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]]
        df = _make_df(rows)
        train, test = split_train_test(df, cfg, test_size=0.3, random_state=0)

        k_in_train = (train["peptide_sequence"] == seq_k).any()
        k_in_test  = (test["peptide_sequence"]  == seq_k).any()
        x_in_train = (train["peptide_sequence"] == seq_x).any()
        x_in_test  = (test["peptide_sequence"]  == seq_x).any()

        same_split = (k_in_train == x_in_train) and (k_in_test == x_in_test)
        assert same_split, (
            f"X/K variant pair split across train/test!\n"
            f"  {seq_k!r}: train={k_in_train}, test={k_in_test}\n"
            f"  {seq_x!r}: train={x_in_train}, test={x_in_test}"
        )

    # ------------------------------------------------------------------
    # 3h. Reproducibility
    # ------------------------------------------------------------------

    def test_deterministic_same_random_state(self):
        """Same inputs + same random_state must give identical splits."""
        cfg = _minimal_config(threshold=0.9)
        seqs = ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "IIIIIIIIII"]
        rows = [(s, 0, float(c)) for s in seqs for c in [0.5, 1.0, 2.0]]
        df = _make_df(rows)
        train1, test1 = split_train_test(df, cfg, test_size=0.3, random_state=99)
        train2, test2 = split_train_test(df, cfg, test_size=0.3, random_state=99)
        pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                      train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True),
                                      test2.reset_index(drop=True))

    # ------------------------------------------------------------------
    # 3i. Guard rails
    # ------------------------------------------------------------------

    def test_missing_column_raises(self):
        df = pd.DataFrame({"label_ordinal": [0, 1]})
        cfg = _minimal_config()
        with pytest.raises(ValueError, match="peptide_sequence"):
            split_train_test(df, cfg)

    def test_invalid_test_size_raises(self):
        seqs = ["ACDEF", "GHIKL"]
        rows = [(s, 0, 1.0) for s in seqs]
        df = _make_df(rows)
        cfg = _minimal_config()
        with pytest.raises(ValueError, match="test_size"):
            split_train_test(df, cfg, test_size=1.5)


# ===========================================================================
# 4. Real data end-to-end (requires real file)
# ===========================================================================

@pytest.mark.skipif(
    not _REAL_FILE.exists(),
    reason="Real lab file not present"
)
class TestRealDataSplit:

    @pytest.fixture(scope="class")
    @classmethod
    def split_result(cls, alpha_config):
        from src.ingest.loader import load_dataset
        df = load_dataset(alpha_config, sources=[str(_REAL_FILE)])
        train, test = split_train_test(df, alpha_config, test_size=0.2, random_state=42)
        return df, train, test, alpha_config

    def test_no_crash(self, split_result):
        _, train, test, _ = split_result
        assert train is not None and test is not None

    def test_row_conservation(self, split_result):
        df, train, test, _ = split_result
        assert len(train) + len(test) == len(df)

    def test_test_fraction_reasonable(self, split_result):
        df, train, test, _ = split_result
        actual = len(test) / len(df)
        assert 0.05 <= actual <= 0.50, (
            f"Test fraction {actual:.3f} is outside reasonable range [0.05, 0.50]. "
            "Check that clusters are not pathologically large."
        )

    def test_zero_leakage_real_data(self, split_result):
        """Brute-force leakage check on real data — independent of clustering."""
        df, train, test, cfg = split_result
        threshold = cfg["homology_cluster_threshold"]
        violations = _brute_force_leakage_check(train, test, threshold)
        assert len(violations) == 0, (
            f"Leakage detected in real-data split! "
            f"{len(violations)} cross-split pair(s) exceed threshold {threshold}.\n"
            "First violation: "
            + (f"train={violations[0][0]!r}, test={violations[0][1]!r}, "
               f"sim={violations[0][2]:.4f}" if violations else "N/A")
        )

    def test_report_split_stats(self, split_result, capsys):
        """Print split statistics for human review (always passes)."""
        df, train, test, cfg = split_result
        from src.features.ptm import encode_ptm_map
        from src.splitting.homology_split import cluster_sequences

        unique_seqs = df["peptide_sequence"].unique().tolist()
        clean_seqs = [encode_ptm_map(s)[0] for s in unique_seqs]
        cluster_map = cluster_sequences(
            list(dict.fromkeys(clean_seqs)),
            cfg["homology_cluster_threshold"],
        )
        n_clusters = len(set(cluster_map.values()))

        print(
            f"\n{'='*55}\n"
            f"Real-data split report (threshold={cfg['homology_cluster_threshold']})\n"
            f"{'='*55}\n"
            f"  Total rows       : {len(df)}\n"
            f"  Train rows       : {len(train)} "
            f"({len(train)/len(df)*100:.1f}%)\n"
            f"  Test  rows       : {len(test)}  "
            f"({len(test)/len(df)*100:.1f}%)\n"
            f"  Unique sequences : {len(unique_seqs)}\n"
            f"  Clusters found   : {n_clusters}\n"
            f"{'='*55}"
        )
        assert True   # always passes — informational only
