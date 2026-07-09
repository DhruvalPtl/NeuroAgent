"""
tracking/dashboard.py
=====================
Streamlit leaderboard dashboard for NeuroAgent experiment tracking.

Run:
    streamlit run tracking/dashboard.py

Features
--------
- Leaderboard table with disease filter + sort-by selector
- Red warning badge on any row where high_class_recall_flag = 1
- Row expander: full hyperparams, confusion matrix heatmap,
  per-class recall bar chart, git commit reference
- Auto-refreshes when the DB is updated (5-second poll interval)

Designed for offline lab use — no auth, no external services.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Resolve project root regardless of where streamlit is invoked from
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tracking.db import get_leaderboard, init_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_DB = os.path.join(_DIR, "neuroagent.db")
_SORT_OPTIONS = {
    "Macro F1 (primary)":          "macro_f1",
    "Quadratic Weighted Kappa":    "quadratic_weighted_kappa",
    "Accuracy (⚠ ref only)":       "accuracy",
}
_CLASS_LABELS = {0: "No", 1: "Low", 2: "Medium", 3: "High"}
_HIGH_RECALL_WARN = "⚠ LOW HIGH-RECALL"
_WARN_COLOR = "#FF4B4B"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroAgent — Experiment Leaderboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium, dark-adjacent styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .warn-badge {
        display: inline-block;
        background: #FF4B4B22;
        color: #FF4B4B;
        border: 1px solid #FF4B4B66;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }
    .ok-badge {
        display: inline-block;
        background: #00C85322;
        color: #00C853;
        border: 1px solid #00C85366;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .metric-pill {
        display: inline-block;
        background: #1E1E2E;
        border-radius: 8px;
        padding: 6px 16px;
        margin: 4px;
        font-size: 0.85rem;
        color: #CDD6F4;
    }
    .metric-pill strong { color: #89DCEB; }
    .stExpander { border: 1px solid #313244 !important; border-radius: 10px !important; }
    div[data-testid="metric-container"] { background: #181825; border-radius: 10px; padding: 12px; }
    .git-ref {
        font-family: 'Courier New', monospace;
        font-size: 0.78rem;
        color: #A6ADC8;
        background: #181825;
        border-radius: 4px;
        padding: 2px 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/"
    "DNA_simple.svg/200px-DNA_simple.svg.png",
    width=60,
)
st.sidebar.title("NeuroAgent 🧬")
st.sidebar.caption("Protein aggregation research platform")
st.sidebar.markdown("---")

db_path = st.sidebar.text_input("Database path", value=_DEFAULT_DB)
sort_label = st.sidebar.selectbox("Sort by", list(_SORT_OPTIONS.keys()), index=0)
sort_by = _SORT_OPTIONS[sort_label]
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠ **Accuracy** is shown for reference only.  "
    "A model predicting all class-0 achieves ≈75 % accuracy.  "
    "Use **Macro F1** for model comparison."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=5)
def _load(db_path: str, disease: str | None, sort_by: str) -> pd.DataFrame:
    try:
        init_db(db_path)
        return get_leaderboard(db_path, disease=disease or None, sort_by=sort_by)
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🧬 NeuroAgent — Experiment Leaderboard")
st.caption(
    "Tracks every training run with metrics, hyperparameters, "
    "data snapshot hash, and git commit reference."
)

# Disease filter (populated from DB)
raw_df = _load(db_path, None, sort_by)
diseases = ["All"] + sorted(raw_df["disease"].unique().tolist()) \
    if not raw_df.empty else ["All"]
disease_sel = st.sidebar.selectbox("Disease filter", diseases, index=0)
filter_disease = None if disease_sel == "All" else disease_sel

# Target type filter — CRITICAL: never compare per_concentration vs max_label
target_type_options = ["All", "per_concentration", "max_label"]
target_type_sel = st.sidebar.selectbox(
    "Target type",
    target_type_options,
    index=0,
    help=(
        "⚠ per_concentration and max_label metrics are NOT comparable. "
        "Always filter to one type before comparing models."
    ),
)
st.sidebar.caption(
    "⚠ **per_concentration** and **max_label** use different training sets. "
    "Never compare their metrics on the same table."
)

df = _load(db_path, filter_disease, sort_by)
if target_type_sel != "All" and not df.empty and "target_type" in df.columns:
    df = df[df["target_type"] == target_type_sel].reset_index(drop=True)

# Source-type filter — distinguishes runs trained on lab-only vs lab+external data
_SOURCE_TYPE_OPTIONS = ["All", "lab_generated", "external_public", "mixed"]
source_type_sel = st.sidebar.selectbox(
    "Training data source",
    _SOURCE_TYPE_OPTIONS,
    index=0,
    help=(
        "Filter by the source_type of the training data used:\n"
        "• lab_generated — only real lab measurements (default, always safe)\n"
        "• external_public — trained with opt-in public database data\n"
        "• mixed — trained with both lab + external data\n"
        "Rows without a source_type annotation are shown under 'All'."
    ),
)
st.sidebar.caption(
    "⚠ **lab_generated** and **external_public** metrics may not be directly "
    "comparable — external data uses crude binary labels ({0, 3} only)."
)

if source_type_sel != "All" and not df.empty and "training_source_type" in df.columns:
    df = df[df["training_source_type"] == source_type_sel].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Summary metrics row
# ---------------------------------------------------------------------------
if not df.empty:
    best = df.iloc[0]
    best_metrics = json.loads(best["metrics_json"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total experiments", len(df))
    c2.metric(
        "Best Macro F1",
        f"{best_metrics.get('macro_f1', 0):.4f}",
        help="Highest macro F1 in the current filter",
    )
    c3.metric(
        "Best QWK",
        f"{best_metrics.get('quadratic_weighted_kappa', 0):.4f}",
    )
    warn_count = int(df["high_class_recall_flag"].sum())
    c4.metric(
        "⚠ Low-recall warnings",
        warn_count,
        delta=f"{warn_count} models miss High aggregators"
        if warn_count else None,
        delta_color="inverse",
    )
    st.markdown("---")

# ---------------------------------------------------------------------------
# Leaderboard table + expanders
# ---------------------------------------------------------------------------
if df.empty:
    st.info(
        "No experiments recorded yet.  "
        "Run the pipeline to generate your first experiment row.\n\n"
        "Quick start:\n"
        "```python\n"
        "from platform_core.launcher import start_agent\n"
        "start_agent(budget_experiments_per_day=5)\n"
        "```"
    )
else:
    st.subheader(f"Experiments — sorted by {sort_label}")

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        metrics   = json.loads(row["metrics_json"])
        hyperparams = json.loads(row["hyperparams_json"])
        is_flagged = bool(row["high_class_recall_flag"])
        mf1  = metrics.get("macro_f1", 0.0)
        qwk  = metrics.get("quadratic_weighted_kappa", 0.0)
        acc  = metrics.get("accuracy", 0.0)
        pcr  = metrics.get("per_class_recall", {})

        # Build expander header
        flag_html = (
            f'<span class="warn-badge">{_HIGH_RECALL_WARN}</span>'
            if is_flagged
            else '<span class="ok-badge">\u2713 OK</span>'
        )
        tt = row.get("target_type", "per_concentration") or "per_concentration"
        tt_color = "#89DCEB" if tt == "per_concentration" else "#F9E2AF"
        tt_badge = (
            f'<span style="display:inline-block;background:{tt_color}22;'
            f'color:{tt_color};border:1px solid {tt_color}66;border-radius:5px;'
            f'padding:1px 8px;font-size:0.75rem;font-weight:600">{tt}</span>'
        )
        header = (
            f"#{rank}  **{row['model_type']}**  \u00b7  `{row['disease']}`  "
            f"\u00b7  F1 = **{mf1:.4f}**  \u00b7  QWK = **{qwk:.4f}**  "
            f"\u00b7  {row['timestamp'][:10]}"
        )

        with st.expander(header, expanded=(rank == 1)):
            st.markdown(
                flag_html + "&nbsp;&nbsp;" + tt_badge,
                unsafe_allow_html=True,
            )

            # Metric pills row
            pill_html = "".join([
                f'<span class="metric-pill">Macro F1 <strong>{mf1:.4f}</strong></span>',
                f'<span class="metric-pill">QWK <strong>{qwk:.4f}</strong></span>',
                f'<span class="metric-pill">Accuracy* <strong>{acc:.4f}</strong></span>',
                f'<span class="metric-pill">Train rows <strong>{row["train_rows"]}</strong></span>',
                f'<span class="metric-pill">Test rows <strong>{row["test_rows"]}</strong></span>',
            ])
            st.markdown(pill_html, unsafe_allow_html=True)
            st.markdown("")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                # Per-class recall bar chart
                st.markdown("**Per-class Recall**")
                if pcr:
                    recall_df = pd.DataFrame({
                        "Class": [f"{_CLASS_LABELS.get(int(k), k)} ({k})" for k in sorted(pcr)],
                        "Recall": [pcr[k] for k in sorted(pcr)],
                    })
                    st.bar_chart(recall_df.set_index("Class"), height=180)

                # Hyperparameters
                st.markdown("**Hyperparameters**")
                st.json(hyperparams)

            with col_right:
                # Confusion matrix heatmap
                cm_data = metrics.get("confusion_matrix")
                if cm_data:
                    st.markdown("**Confusion Matrix**")
                    try:
                        import matplotlib.pyplot as plt
                        import matplotlib.colors as mcolors

                        cm_arr = np.array(cm_data)
                        fig, ax = plt.subplots(figsize=(4, 3.2))
                        fig.patch.set_facecolor("#1E1E2E")
                        ax.set_facecolor("#1E1E2E")

                        cmap = plt.cm.get_cmap("YlOrRd")
                        im = ax.imshow(cm_arr, cmap=cmap, aspect="auto")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                        labels = [_CLASS_LABELS.get(i, str(i)) for i in range(4)]
                        ax.set_xticks(range(4))
                        ax.set_yticks(range(4))
                        ax.set_xticklabels(labels, color="#CDD6F4", fontsize=9)
                        ax.set_yticklabels(labels, color="#CDD6F4", fontsize=9)
                        ax.set_xlabel("Predicted", color="#CDD6F4", fontsize=9)
                        ax.set_ylabel("True", color="#CDD6F4", fontsize=9)
                        ax.tick_params(colors="#CDD6F4")

                        for i in range(4):
                            for j in range(4):
                                val = int(cm_arr[i, j])
                                text_color = "white" if cm_arr[i, j] < cm_arr.max() * 0.6 else "#1E1E2E"
                                ax.text(j, i, str(val), ha="center", va="center",
                                        color=text_color, fontsize=10, fontweight="bold")

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except ImportError:
                        # matplotlib not available: fall back to plain table
                        cm_df = pd.DataFrame(
                            cm_data,
                            index=[f"True {_CLASS_LABELS.get(i,i)}" for i in range(4)],
                            columns=[f"Pred {_CLASS_LABELS.get(i,i)}" for i in range(4)],
                        )
                        st.dataframe(cm_df)

                # Metadata
                st.markdown("**Run metadata**")
                st.markdown(
                    f"**Status:** `{row['status']}`  \n"
                    f"**Data hash:** `{row['data_snapshot_hash'][:16]}…`  \n"
                    f"**Git commit:** <span class='git-ref'>{row['git_commit'][:12]}</span>  \n"
                    f"**Timestamp:** `{row['timestamp']}`",
                    unsafe_allow_html=True,
                )

            if row.get("code_diff_summary"):
                st.markdown("**Code diff summary**")
                st.code(row["code_diff_summary"], language="diff")

        st.markdown("")   # breathing room between expanders

# ---------------------------------------------------------------------------
# Auto-refresh loop
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
