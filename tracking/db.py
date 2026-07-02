"""
tracking/db.py
==============
SQLite-backed experiment tracking for NeuroAgent.

Design rationale
----------------
SQLite is chosen deliberately over a remote database service:
  - Zero infrastructure: runs anywhere the venv runs, including Colab/Jupyter.
  - The DB file is a single flat file that can be shared, backed up, or
    versioned with DVC without any server setup.
  - For the scale of this project (hundreds of experiments per milestone,
    not millions of rows), SQLite's single-writer model is not a bottleneck.

Schema philosophy
-----------------
metrics_json stores the FULL metrics dict (macro_f1, QWK, confusion_matrix,
per_class_recall, …) as serialised JSON.  high_class_recall_flag is also
denormalised into its own INTEGER column so the dashboard / SQL queries can
filter `WHERE high_class_recall_flag = 1` without parsing JSON on every row.
This is intentional redundancy — the source of truth is always metrics_json;
high_class_recall_flag is a materialised shortcut.

git_commit is auto-fetched so every experiment row is permanently tied to
the exact code that produced it.  If git is unavailable (detached HEAD,
no git binary) the column stores "unknown" rather than crashing.

hypothesis_debate_json and code_diff_summary are NULL for all Step 8
experiments but the columns exist now so the schema never needs a
migration when Step 10 wires in the agent debate loop.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition — single source of truth for column names / types
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    disease                 TEXT    NOT NULL,
    model_type              TEXT    NOT NULL,
    hyperparams_json        TEXT    NOT NULL,
    data_snapshot_hash      TEXT    NOT NULL,
    train_rows              INTEGER NOT NULL,
    test_rows               INTEGER NOT NULL,
    metrics_json            TEXT    NOT NULL,
    high_class_recall_flag  INTEGER NOT NULL DEFAULT 0,
    git_commit              TEXT    NOT NULL DEFAULT 'unknown',
    status                  TEXT    NOT NULL DEFAULT 'completed',
    hypothesis_debate_json  TEXT,
    code_diff_summary       TEXT,
    error_message           TEXT
);
"""

# Additive migrations — each is a guarded ALTER TABLE that is a no-op if the
# column already exists.  SQLite has no "ADD COLUMN IF NOT EXISTS" syntax, so
# we catch the OperationalError that signals the column already exists.
_DDL_MIGRATIONS: list[str] = [
    "ALTER TABLE experiments ADD COLUMN error_message TEXT;",
    "ALTER TABLE experiments ADD COLUMN target_type TEXT NOT NULL DEFAULT 'per_concentration';",
]

_REQUIRED_FIELDS = frozenset({
    "disease",
    "model_type",
    "hyperparams_json",
    "data_snapshot_hash",
    "train_rows",
    "test_rows",
    "metrics_json",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(db_path: str = "tracking/neuroagent.db") -> None:
    """Create the SQLite database and schema if they do not already exist.

    Idempotent — safe to call at the start of every pipeline run.  Uses
    ``CREATE TABLE IF NOT EXISTS`` so existing data is never touched.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file.  Parent directories must exist; the file
        is created automatically if absent.
    """
    with _connect(db_path) as conn:
        conn.executescript(_DDL)
        conn.commit()
        _run_migrations(conn)
    logger.info("init_db: schema initialised at %r", db_path)


def log_experiment(
    db_path: str = "tracking/neuroagent.db",
    **fields: Any,
) -> int:
    """Insert one experiment record and return its auto-assigned id.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file (must already be initialised via init_db).
    **fields : Any
        Experiment fields.  Required keys:
            disease             str
            model_type          str
            hyperparams_json    str | dict   (dict is JSON-serialised automatically)
            data_snapshot_hash  str
            train_rows          int
            test_rows           int
            metrics_json        str | dict   (dict is JSON-serialised automatically)
        Optional keys (have safe defaults):
            high_class_recall_flag  int  (0/1; auto-extracted from metrics_json
                                         if not explicitly provided)
            git_commit              str  (auto-fetched from git if not provided)
            status                  str  (default "completed")
            timestamp               str  (auto-set to UTC now if not provided)
            hypothesis_debate_json  str | None
            code_diff_summary       str | None

    Returns
    -------
    int
        The ``id`` (ROWID) of the newly inserted row.

    Raises
    ------
    ValueError
        If any required field is missing.
    """
    _validate_required_fields(fields)

    # Normalise dict → JSON for jsonb-ish columns
    fields = dict(fields)
    for col in ("hyperparams_json", "metrics_json"):
        if isinstance(fields[col], dict):
            fields[col] = json.dumps(fields[col])

    # Auto-fill optional fields
    if "timestamp" not in fields:
        fields["timestamp"] = datetime.now(timezone.utc).isoformat()

    if "git_commit" not in fields:
        fields["git_commit"] = _fetch_git_commit()

    if "high_class_recall_flag" not in fields:
        try:
            metrics = json.loads(fields["metrics_json"])
            fields["high_class_recall_flag"] = int(
                bool(metrics.get("high_class_recall_flag", False))
            )
        except (json.JSONDecodeError, TypeError):
            fields["high_class_recall_flag"] = 0

    fields.setdefault("status", "completed")
    fields.setdefault("hypothesis_debate_json", None)
    fields.setdefault("code_diff_summary", None)

    columns = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    sql = f"INSERT INTO experiments ({columns}) VALUES ({placeholders})"

    with _connect(db_path) as conn:
        cur = conn.execute(sql, list(fields.values()))
        conn.commit()
        row_id: int = cur.lastrowid

    logger.info(
        "log_experiment: inserted row id=%d (model=%s, disease=%s, status=%s)",
        row_id, fields.get("model_type"), fields.get("disease"),
        fields.get("status"),
    )
    return row_id


def get_leaderboard(
    db_path: str = "tracking/neuroagent.db",
    disease: str | None = None,
    sort_by: str = "macro_f1",
) -> pd.DataFrame:
    """Retrieve all experiments as a DataFrame, sorted by the requested metric.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file.
    disease : str | None
        If provided, filter to experiments whose ``disease`` field matches
        exactly.  None returns all experiments.
    sort_by : str
        Name of a key inside ``metrics_json`` to sort by (descending).
        Common values: ``"macro_f1"``, ``"quadratic_weighted_kappa"``,
        ``"accuracy"`` (discouraged — see Step 7 docstring).
        Falls back to ``macro_f1`` if the key is absent from a row.

    Returns
    -------
    pd.DataFrame
        One row per experiment, with all DB columns plus a ``sort_value``
        column containing the extracted sort metric.  Sorted descending.
        Returns an empty DataFrame (not an error) if no rows match.
    """
    with _connect(db_path) as conn:
        if disease is not None:
            sql = "SELECT * FROM experiments WHERE disease = ?"
            rows = conn.execute(sql, [disease]).fetchall()
            cols = [d[0] for d in conn.execute(sql, [disease]).description] \
                if rows else _column_names(conn)
        else:
            rows = conn.execute("SELECT * FROM experiments").fetchall()
            cols = [d[0] for d in conn.execute(
                "SELECT * FROM experiments LIMIT 0"
            ).description]

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=cols)

    # Extract sort metric from metrics_json for ordering
    def _extract(metrics_str: str) -> float:
        try:
            return float(json.loads(metrics_str).get(sort_by, 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    df["sort_value"] = df["metrics_json"].apply(_extract)
    df = df.sort_values("sort_value", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    """Return an sqlite3 connection with WAL journal for robustness."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _column_names(conn: sqlite3.Connection) -> list[str]:
    """Return column names from the experiments table."""
    cur = conn.execute("PRAGMA table_info(experiments)")
    return [row[1] for row in cur.fetchall()]


def _validate_required_fields(fields: dict[str, Any]) -> None:
    missing = _REQUIRED_FIELDS - set(fields.keys())
    if missing:
        raise ValueError(
            f"log_experiment: missing required field(s): {sorted(missing)}.\n"
            f"Required fields: {sorted(_REQUIRED_FIELDS)}"
        )


def _fetch_git_commit() -> str:
    """Return the current HEAD commit SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    logger.warning("_fetch_git_commit: could not determine HEAD commit.")
    return "unknown"


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive schema migrations idempotently.

    Each statement in _DDL_MIGRATIONS is executed once.  If the column
    already exists, SQLite raises OperationalError("duplicate column name:
    …") which we catch and ignore — this makes the function safe to call on
    both fresh and existing databases.
    """
    for sql in _DDL_MIGRATIONS:
        try:
            conn.execute(sql)
            conn.commit()
            logger.debug("Migration applied: %s", sql.strip())
        except sqlite3.OperationalError as exc:
            # "duplicate column name" → column already exists, skip
            if "duplicate column" in str(exc).lower():
                logger.debug("Migration already applied (skipping): %s", sql.strip())
            else:
                raise

