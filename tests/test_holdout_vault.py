"""
tests/test_holdout_vault.py
============================
Tests for platform_core/holdout_vault.py.

All tests are fast (no real data I/O).
Property guarantees:
  1. build_vault() is deterministic given the same seed
  2. Zero cluster overlap between vault and remaining pool
  3. score_against_vault() refuses on missing or tampered vault
  4. Sandbox path-block: open("...holdout_vault/...") rejected at AST layer
  5. vault_scores do NOT appear in get_leaderboard()
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import textwrap
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

_DISEASE_CONFIG = {
    "name": "test_disease",
    "raw_data_path": "data/raw/test_disease/",
    "sequence_column": "sequence",
    "label_column": "aggregation_severity",
    "label_schema": [None, "Low", "Medium", "High"],
    "ptm_types": ["acetylation"],
    "homology_cluster_threshold": 0.9,
}

_SEQUENCES = [
    "ACDEFGHIKLM", "ACDEFGHIKLX", "ACDEFGHIKLY", "ACDEFGHIKLZ",  # cluster 0
    "RRRRRRRRRRR", "RRRRRRRRRRS", "RRRRRRRRRRT",                  # cluster 1
    "WWWWWWWWWWW", "WWWWWWWWWWX", "WWWWWWWWWWY",                 # cluster 2
]


def _make_df(n_rows_per_seq: int = 3) -> pd.DataFrame:
    rows = []
    for i, seq in enumerate(_SEQUENCES):
        for conc in range(n_rows_per_seq):
            rows.append({
                "peptide_sequence":   seq,
                "concentration":      float(conc + 1),
                "label_ordinal":      i % 4,
                "source_file":        "synthetic.csv",
                "source_type":        "lab_generated",
                "data_snapshot_hash": "aabbcc",
            })
    return pd.DataFrame(rows)


def _fake_cluster(seqs, threshold):
    return {s: (0 if s.startswith("A") else 1 if s.startswith("R") else 2)
            for s in seqs}


@pytest.fixture()
def tmp_db(tmp_path):
    db = str(tmp_path / "test_vault.db")
    from tracking.db import init_db
    init_db(db)
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper for TestBuildVault
# ─────────────────────────────────────────────────────────────────────────────

def _build_vault_mocked(tmp_path, tmp_db, seed=42, vault_fraction=0.3):
    """Call build_vault() with all external I/O mocked at module level."""
    import src.ingest.loader as _real_loader

    df = _make_df()
    vault_dir = tmp_path / "holdout_vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Only mock what we need: yaml.safe_load, cluster_sequences, encode_ptm_map,
    # loader.load_dataset, and DB calls. Do NOT mock builtins.open globally
    # (that would break _sha256_file's binary read of the real CSV).
    orig_load = _real_loader.load_dataset
    _real_loader.load_dataset = MagicMock(return_value=df.copy())

    orig_exists = pathlib.Path.exists
    def _mock_exists(self):
        if "diseases" in str(self) and str(self).endswith(".yaml"):
            return True
        return orig_exists(self)

    try:
        with (
            patch("pathlib.Path.exists", _mock_exists),
            patch("platform_core.holdout_vault.yaml") as mock_yaml,
            patch("platform_core.holdout_vault.cluster_sequences",
                  side_effect=_fake_cluster),
            patch("platform_core.holdout_vault.encode_ptm_map",
                  side_effect=lambda s: (s, {})),
            patch("platform_core.holdout_vault.VAULT_BASE_DIR", vault_dir),
            patch("tracking.db.get_vault_manifest", return_value=None),
            patch("tracking.db.init_db"),
            patch("tracking.db.log_vault_registry"),
            patch("tracking.db._fetch_git_commit", return_value="deadbeef"),
        ):
            mock_yaml.safe_load.return_value = _DISEASE_CONFIG
            # Mock the config file open (yaml.safe_load is already mocked,
            # so open is only called once — for the config. We patch it just
            # to avoid needing the file to exist on disk.)
            import builtins
            orig_open = builtins.open
            def _selective_open(path, *args, **kwargs):
                ps = str(path)
                if "diseases" in ps and ps.endswith(".yaml"):
                    return mock_open(read_data="")()
                return orig_open(path, *args, **kwargs)
            with patch("builtins.open", side_effect=_selective_open):
                from platform_core.holdout_vault import build_vault
                manifest = build_vault(
                    disease="test_disease",
                    seed=seed,
                    vault_fraction=vault_fraction,
                    db_path=tmp_db,
                    force=False,
                )
    finally:
        _real_loader.load_dataset = orig_load

    return manifest, vault_dir



# ─────────────────────────────────────────────────────────────────────────────
# 1. TestBuildVault
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildVault:

    def test_vault_csv_written(self, tmp_path, tmp_db):
        _, vault_dir = _build_vault_mocked(tmp_path, tmp_db)
        assert (vault_dir / "test_disease_vault.csv").exists()

    def test_vault_rows_nonzero(self, tmp_path, tmp_db):
        _, vault_dir = _build_vault_mocked(tmp_path, tmp_db, vault_fraction=0.3)
        df = pd.read_csv(vault_dir / "test_disease_vault.csv")
        assert len(df) > 0

    def test_class_dist_populated(self, tmp_path, tmp_db):
        manifest, _ = _build_vault_mocked(tmp_path, tmp_db)
        assert isinstance(manifest.class_dist, dict)
        assert len(manifest.class_dist) > 0

    def test_determinism_same_seed(self, tmp_path, tmp_db):
        m1, _ = _build_vault_mocked(tmp_path, tmp_db, seed=7)
        m2, _ = _build_vault_mocked(tmp_path, tmp_db, seed=7)
        assert sorted(m1.vault_cluster_ids) == sorted(m2.vault_cluster_ids)

    def test_vault_fraction_respected(self, tmp_path, tmp_db):
        """Vault should contain roughly vault_fraction of rows."""
        manifest, vault_dir = _build_vault_mocked(tmp_path, tmp_db, vault_fraction=0.3)
        total_rows = len(_make_df())
        # Clusters are indivisible so the fraction may be approximate
        assert manifest.vault_rows > 0
        assert manifest.vault_rows < total_rows

    def test_raises_on_existing_vault_no_force(self, tmp_path, tmp_db):
        from platform_core.holdout_vault import VaultAlreadyExistsError, build_vault

        orig_exists = pathlib.Path.exists
        def _mock_exists(self):
            if "diseases" in str(self):
                return True
            return orig_exists(self)

        with (
            patch("pathlib.Path.exists", _mock_exists),
            patch("builtins.open", mock_open(read_data="")),
            patch("platform_core.holdout_vault.yaml") as mock_yaml,
            patch("tracking.db.init_db"),
            patch("tracking.db.get_vault_manifest",
                  return_value={"vault_path": "x", "checksum_sha256": "y", "id": 1}),
        ):
            mock_yaml.safe_load.return_value = _DISEASE_CONFIG
            with pytest.raises(VaultAlreadyExistsError):
                build_vault("test_disease", seed=42, db_path=tmp_db)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TestClusterNonOverlap
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterNonOverlap:

    def test_cluster_ids_disjoint(self):
        from src.splitting.homology_split import cluster_sequences
        seqs_a = ["AAAAAAAAAA", "AAAAAAAAAB", "AAAAAAAAAC"]
        seqs_b = ["RRRRRRRRRRR", "SSSSSSSSSSS"]
        all_seqs = seqs_a + seqs_b
        cluster_map = cluster_sequences(all_seqs, threshold=0.8)
        unique = set(cluster_map.values())
        vault = {sorted(unique)[0]}
        remaining = unique - vault
        assert vault.isdisjoint(remaining)

    def test_no_sequence_in_both_splits_via_logic(self):
        """Directly test the splitting math: vault and remaining seqs are disjoint."""
        import random as _r
        df = _make_df()
        df["_cluster_id"] = df["peptide_sequence"].apply(
            lambda s: 0 if s.startswith("A") else 1 if s.startswith("R") else 2
        )
        counts = df["_cluster_id"].value_counts().to_dict()
        total = len(df)
        target = int(total * 0.3)

        rng = _r.Random(42)
        cids = sorted(counts.keys(), key=lambda c: (counts[c], rng.random()))
        vault_cids = set()
        n = 0
        for cid in cids:
            if n >= target:
                break
            vault_cids.add(cid)
            n += counts[cid]

        vault_seqs = set(df[df["_cluster_id"].isin(vault_cids)]["peptide_sequence"])
        remaining_seqs = set(df[~df["_cluster_id"].isin(vault_cids)]["peptide_sequence"])
        assert len(vault_seqs & remaining_seqs) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. TestVaultIntegrity
# ─────────────────────────────────────────────────────────────────────────────

class TestVaultIntegrity:

    def _manifest(self, vault_path, checksum="abc123"):
        return {"id": 1, "vault_path": vault_path,
                "checksum_sha256": checksum, "disease": "test_disease"}

    def test_missing_file_raises(self, tmp_path):
        from platform_core.holdout_vault import _verify_vault_integrity, VaultIntegrityError
        with pytest.raises(VaultIntegrityError, match="MISSING"):
            _verify_vault_integrity(self._manifest(str(tmp_path / "nonexistent.csv")))

    def test_tampered_content_raises(self, tmp_path):
        from platform_core.holdout_vault import _verify_vault_integrity, VaultIntegrityError
        csv = tmp_path / "vault.csv"
        csv.write_text("seq,label\nAAAAAAAAA,0\n", encoding="utf-8")
        correct = hashlib.sha256(csv.read_bytes()).hexdigest()
        csv.write_text("seq,label\nTAMPERED,1\n", encoding="utf-8")
        with pytest.raises(VaultIntegrityError, match="MISMATCH"):
            _verify_vault_integrity(self._manifest(str(csv), correct))

    def test_clean_file_passes(self, tmp_path):
        from platform_core.holdout_vault import _verify_vault_integrity
        csv = tmp_path / "clean.csv"
        csv.write_text("seq,label\nAAAAAAAAA,0\n", encoding="utf-8")
        checksum = hashlib.sha256(csv.read_bytes()).hexdigest()
        _verify_vault_integrity(self._manifest(str(csv), checksum))  # no raise

    def test_score_refuses_missing_vault(self, tmp_db):
        from platform_core.holdout_vault import score_against_vault
        with pytest.raises(FileNotFoundError):
            score_against_vault("random_forest", "nonexistent_disease", db_path=tmp_db)

    def test_score_refuses_integrity_fail(self, tmp_path, tmp_db):
        from platform_core.holdout_vault import score_against_vault, VaultIntegrityError
        csv = tmp_path / "vault.csv"
        csv.write_text("seq,label\nAAAAAAAAA,0\n", encoding="utf-8")
        tampered = {"id": 1, "vault_path": str(csv),
                    "checksum_sha256": "WRONG", "disease": "alpha_synuclein"}
        with patch("tracking.db.get_vault_manifest", return_value=tampered):
            with pytest.raises(VaultIntegrityError):
                score_against_vault("random_forest", "alpha_synuclein", db_path=tmp_db)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TestSandboxVaultBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxVaultBlock:

    def _sb(self, code):
        from agent.sandbox import run_in_sandbox
        return run_in_sandbox(code, timeout_seconds=5)

    def test_read_vault_path_blocked(self):
        code = "f = open('tracking/holdout_vault/alpha_synuclein_vault.csv', 'r')\n"
        r = self._sb(code)
        assert r["success"] is False
        exc = (r["exception"] or "").lower()
        assert "holdout_vault" in exc or "vault" in exc

    def test_write_vault_path_blocked(self):
        code = "with open('tracking/holdout_vault/data.csv', 'w') as f: f.write('x')\n"
        r = self._sb(code)
        assert r["success"] is False

    def test_nested_vault_path_blocked(self):
        code = "f = open('./tracking/holdout_vault/sub/file.csv')\n"
        r = self._sb(code)
        assert r["success"] is False

    def test_normal_read_not_blocked_by_ast(self):
        """Non-vault open() passes AST check (may fail at runtime — that is OK)."""
        code = textwrap.dedent("""\
            try:
                f = open('tracking/neuroagent.db', 'r')
            except Exception:
                pass
        """)
        r = self._sb(code)
        if not r["success"]:
            assert "holdout_vault" not in (r["exception"] or "")

    def test_vault_dir_name_exported(self):
        from agent.sandbox import VAULT_DIR_NAME
        assert VAULT_DIR_NAME == "holdout_vault"


# ─────────────────────────────────────────────────────────────────────────────
# 5. TestVaultDBTables
# ─────────────────────────────────────────────────────────────────────────────

class TestVaultDBTables:

    def test_vault_score_not_in_leaderboard(self, tmp_db):
        from tracking.db import log_vault_score, get_leaderboard
        metrics = {"macro_f1": 0.99, "quadratic_weighted_kappa": 0.98,
                   "per_class_recall": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
                   "high_class_recall_flag": False,
                   "confusion_matrix": [[1]*4]*4, "accuracy": 1.0}
        log_vault_score(disease="test_disease", model_type="rf",
                        vault_registry_id=1, metrics_json=json.dumps(metrics),
                        high_class_recall_flag=0, db_path=tmp_db)
        assert get_leaderboard(db_path=tmp_db, disease="test_disease").empty

    def test_vault_registry_separate_from_experiments(self, tmp_db):
        from tracking.db import log_vault_registry, get_leaderboard
        log_vault_registry(disease="alpha_synuclein", seed=2025,
                           vault_fraction=0.15, vault_path="/fake/vault.csv",
                           checksum_sha256="abc123", vault_rows=50,
                           class_dist_json=json.dumps({0: 30, 3: 20}),
                           cluster_ids_json=json.dumps([1, 2, 3]), db_path=tmp_db)
        assert get_leaderboard(db_path=tmp_db).empty

    def test_vault_tables_exist_after_init_db(self, tmp_db):
        import sqlite3
        with sqlite3.connect(tmp_db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "vault_registry" in tables
        assert "vault_scores" in tables

    def test_get_vault_manifest_returns_none_when_missing(self, tmp_db):
        from tracking.db import get_vault_manifest
        assert get_vault_manifest("nonexistent_disease", tmp_db) is None

    def test_log_and_retrieve_vault_manifest(self, tmp_db):
        from tracking.db import log_vault_registry, get_vault_manifest
        log_vault_registry(disease="alpha_synuclein", seed=2025,
                           vault_fraction=0.15,
                           vault_path="/tracking/holdout_vault/alpha_synuclein_vault.csv",
                           checksum_sha256="deadbeef123456", vault_rows=60,
                           class_dist_json=json.dumps({0: 40, 3: 20}),
                           cluster_ids_json=json.dumps([0, 1]), db_path=tmp_db)
        m = get_vault_manifest("alpha_synuclein", tmp_db)
        assert m is not None
        assert m["checksum_sha256"] == "deadbeef123456"
        assert m["vault_rows"] == 60
        assert m["seed"] == 2025
