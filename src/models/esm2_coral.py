"""
src/models/esm2_coral.py
=========================
ESM-2 + CORAL ordinal MLP classifier — NeuroAgent's deep learning baseline.

Architecture (matching the side-project spec)
----------------------------------------------
Input: 325-dim vector (320 ESM-2 mean-pool + 5 modification stats)

  LayerNorm(325)
  → Linear(325, 256) → GELU → Dropout(dropout_1)
  → Linear(256, 128) → GELU → Dropout(dropout_2)
  → Linear(128, 3)   — CORAL head (3 ordinal bias thresholds for 4 classes)

CORAL loss (implemented inline — no coral-pytorch dependency)
--------------------------------------------------------------
CORAL (COnsistent RAnk Logits, Cao et al. 2020) frames K-class ordinal
regression as K-1 binary tasks sharing the same linear weights but having
independent bias terms (thresholds).

For K=4 classes, we have 3 thresholds θ₁ < θ₂ < θ₃.
  logit_k(x) = f(x) - θ_k          (f(x) is the shared 128→1 linear output)
  P(Y > k | x) = σ(logit_k(x))

Loss = − Σ_k [ y_binary_k * log σ(logit_k) + (1-y_binary_k) * log(1-σ(logit_k)) ]
where y_binary_k = 1 if true_label > k, else 0.

Class prediction: ŷ = Σ_k 1[σ(logit_k) > 0.5]   (cumulative decode)
Probabilities:    P(Y=k) = P(Y>k-1) - P(Y>k)  (clamped to [0,1])

Why CORAL?
  Standard cross-entropy ignores ordinal structure: predicting class 0
  for a true class 3 sample is penalised equally to predicting class 2.
  CORAL encodes the constraint that P(Y>0) ≥ P(Y>1) ≥ P(Y>2), making
  the model aware that classes are ordered (No < Low < Medium < High).

Design note on target_type
---------------------------
CORAL was designed and validated in the side project on single-max-label
sequences.  The model ignores the ``include_concentration`` flag passed by
the pipeline and always produces a 325-dim embedding.  A UserWarning is
emitted if the caller uses target_type='per_concentration', because CORAL's
ordinal-threshold design does not inherently model dose — treating multiple
concentration rows as independent training samples would break the ordinal
assumption and silently inflate training set size.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.registry import register_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORAL utilities (inline implementation — no coral-pytorch dependency)
# ---------------------------------------------------------------------------

def _labels_to_coral_targets(y: "torch.Tensor", num_classes: int = 4) -> "torch.Tensor":
    """Convert ordinal class labels to CORAL binary target matrix.

    For K classes → K-1 binary columns.
    coral_targets[i, k] = 1 if y[i] > k else 0.

    Parameters
    ----------
    y : torch.Tensor, shape (N,)
        Integer class labels in {0, 1, ..., K-1}.
    num_classes : int
        Total number of ordinal classes.

    Returns
    -------
    torch.Tensor, shape (N, K-1), dtype float32
    """
    import torch
    n_thresholds = num_classes - 1
    targets = torch.zeros(len(y), n_thresholds, dtype=torch.float32)
    for k in range(n_thresholds):
        targets[:, k] = (y > k).float()
    return targets


def _coral_loss(
    logits: "torch.Tensor",
    coral_targets: "torch.Tensor",
    sample_weights: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """CORAL loss: binary cross-entropy summed over K-1 thresholds.

    Parameters
    ----------
    logits : torch.Tensor, shape (N, K-1)
        Raw logits from the CORAL head.
    coral_targets : torch.Tensor, shape (N, K-1)
        Binary targets from ``_labels_to_coral_targets``.
    sample_weights : torch.Tensor, shape (N,) or None
        Per-sample weights for class-imbalance correction.

    Returns
    -------
    torch.Tensor, scalar
        Weighted mean CORAL loss.
    """
    import torch
    import torch.nn.functional as F

    # Per-sample, per-threshold BCE losses: shape (N, K-1)
    losses = F.binary_cross_entropy_with_logits(
        logits, coral_targets, reduction="none"
    )
    # Sum across thresholds: shape (N,)
    per_sample = losses.sum(dim=1)

    if sample_weights is not None:
        return (per_sample * sample_weights).mean()
    return per_sample.mean()


def _coral_decode(proba_thresholds: np.ndarray) -> np.ndarray:
    """Convert CORAL sigmoid threshold probabilities to class predictions.

    predicted_class = sum of (P(Y > k) > 0.5) for k in {0, ..., K-2}

    Parameters
    ----------
    proba_thresholds : np.ndarray, shape (N, K-1)
        Sigmoid-activated threshold probabilities.

    Returns
    -------
    np.ndarray, shape (N,), dtype int
        Predicted ordinal class labels.
    """
    return (proba_thresholds > 0.5).sum(axis=1).astype(int)


def _coral_to_class_proba(proba_thresholds: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Convert CORAL threshold probabilities to per-class probabilities.

    P(Y=0) = 1 - P(Y>0)
    P(Y=k) = P(Y>k-1) - P(Y>k)   for 0 < k < K-1
    P(Y=K-1) = P(Y>K-2)

    Parameters
    ----------
    proba_thresholds : np.ndarray, shape (N, K-1)

    Returns
    -------
    np.ndarray, shape (N, K), dtype float32
        Each row sums to approximately 1.0.
    """
    n = proba_thresholds.shape[0]
    K = num_classes
    proba = np.zeros((n, K), dtype=np.float32)
    # P(Y=0) = 1 - P(Y>0)
    proba[:, 0] = 1.0 - proba_thresholds[:, 0]
    # P(Y=k) = P(Y>k-1) - P(Y>k)  for 1 <= k <= K-2
    for k in range(1, K - 1):
        proba[:, k] = proba_thresholds[:, k - 1] - proba_thresholds[:, k]
    # P(Y=K-1) = P(Y>K-2)
    proba[:, K - 1] = proba_thresholds[:, K - 2]
    # Clamp to [0,1] and renormalise (numerical safety)
    proba = np.clip(proba, 0.0, 1.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return proba / row_sums


# ---------------------------------------------------------------------------
# MLP backbone (pure PyTorch nn.Module)
# ---------------------------------------------------------------------------

def _build_mlp(
    input_dim: int = 325,
    dropout_1: float = 0.3,
    dropout_2: float = 0.2,
    num_classes: int = 4,
) -> "torch.nn.Module":
    """Construct the LayerNorm → Linear → GELU → Dropout MLP with CORAL head."""
    import torch.nn as nn

    n_thresholds = num_classes - 1

    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 256),
        nn.GELU(),
        nn.Dropout(dropout_1),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(dropout_2),
        nn.Linear(128, n_thresholds),   # CORAL head: 3 outputs for 4 classes
    )


# ---------------------------------------------------------------------------
# Registered model
# ---------------------------------------------------------------------------

@register_model("esm2_coral")
class ESM2CoralModel(BaseModel):
    """ESM-2 (frozen) + CORAL ordinal MLP classifier.

    Parameters
    ----------
    learning_rate : float
        AdamW learning rate.  Default 3e-4 matches the side-project spec.
    weight_decay : float
        AdamW L2 regularisation.  Default 1e-4.
    batch_size : int
        Mini-batch size for the training loop.  Default 16.
    max_epochs : int
        Maximum training epochs before forced stop.  Default 200.
    patience : int
        Early-stopping patience (epochs without validation macro-F1
        improvement).  Default 30.
    dropout_1 : float
        Dropout rate after the first hidden layer (256 units).  Default 0.3.
    dropout_2 : float
        Dropout rate after the second hidden layer (128 units).  Default 0.2.
    val_fraction : float
        Fraction of the training data held out for early stopping.
        Default 0.15 (stratified split).
    esm2_model_name : str
        HuggingFace ESM-2 checkpoint.  Default is the 8M-parameter model.
    random_state : int
        Seed for the internal train/val split and weight initialisation.
    """

    name: str = "esm2_coral"

    _PARAM_NAMES = frozenset({
        "learning_rate", "weight_decay", "batch_size", "max_epochs",
        "patience", "dropout_1", "dropout_2", "val_fraction",
        "esm2_model_name", "random_state",
    })

    def __init__(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        max_epochs: int = 200,
        patience: int = 30,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
        val_fraction: float = 0.15,
        esm2_model_name: str = "facebook/esm2_t6_8M_UR50D",
        random_state: int = 42,
    ) -> None:
        self.learning_rate    = learning_rate
        self.weight_decay     = weight_decay
        self.batch_size       = batch_size
        self.max_epochs       = max_epochs
        self.patience         = patience
        self.dropout_1        = dropout_1
        self.dropout_2        = dropout_2
        self.val_fraction     = val_fraction
        self.esm2_model_name  = esm2_model_name
        self.random_state     = random_state
        self._mlp             = None   # set by fit()
        self._num_classes     = 4      # fixed for this dataset

    # ------------------------------------------------------------------
    # encode_features override
    # ------------------------------------------------------------------

    def encode_features(
        self,
        df: "pd.DataFrame",
        disease_config: dict,
        include_concentration: bool = True,
    ) -> np.ndarray:
        """Encode peptide rows to 325-dim ESM-2+modification vectors.

        This model ALWAYS produces 325-dim output regardless of
        ``include_concentration``.  CORAL's ordinal-threshold design
        was validated on max-label sequences; feeding concentration rows
        would break the ordinal assumption by treating multiple
        concentration rows from the same peptide as independent samples.

        A UserWarning is emitted if ``include_concentration=True`` is
        passed to alert the caller that the concentration dimension is
        being silently ignored.
        """
        if include_concentration:
            warnings.warn(
                "ESM2CoralModel.encode_features: 'include_concentration=True' "
                "was requested, but this model is designed for max_label "
                "(target_type='max_label').  Concentration is ignored — the "
                "CORAL ordinal head does not model dose-response.  "
                "Consider running with target_type='max_label'.",
                UserWarning,
                stacklevel=2,
            )

        from src.features.esm2_encoder import encode_esm2_features

        rows = [
            encode_esm2_features(str(row["peptide_sequence"]),
                                  model_name=self.esm2_model_name)
            for _, row in df.iterrows()
        ]
        return np.vstack(rows).astype(np.float32)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the CORAL MLP on (X, y) with internal train/val split.

        Parameters
        ----------
        X : np.ndarray, shape (N, 325)
        y : np.ndarray, shape (N,), integer labels in {0, 1, 2, 3}
        """
        import torch
        import torch.nn as nn
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)

        # ----------------------------------------------------------------
        # 1. Internal stratified train / validation split
        # ----------------------------------------------------------------
        n = len(y)
        # Fall back to no validation split if dataset too small
        min_val = max(1, int(n * self.val_fraction))
        if n - min_val < self._num_classes or min_val < self._num_classes:
            # Dataset too small for stratified split — use all for training,
            # disable early stopping
            X_tr, y_tr = X, y
            X_val, y_val = X[:min_val], y[:min_val]
            do_early_stop = False
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.val_fraction,
                random_state=self.random_state,
            )
            tr_idx, val_idx = next(sss.split(X, y))
            X_tr, y_tr   = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            do_early_stop = True

        # ----------------------------------------------------------------
        # 2. Sample weights (inverse class frequency — same as XGBoostModel)
        # ----------------------------------------------------------------
        from src.models.xgboost_model import _compute_sample_weights
        sw_tr = _compute_sample_weights(y_tr, n_classes=self._num_classes)

        # ----------------------------------------------------------------
        # 3. Build MLP + optimiser + scheduler
        # ----------------------------------------------------------------
        input_dim = X_tr.shape[1]
        self._mlp = _build_mlp(
            input_dim=input_dim,
            dropout_1=self.dropout_1,
            dropout_2=self.dropout_2,
            num_classes=self._num_classes,
        )

        optimiser = torch.optim.AdamW(
            self._mlp.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.max_epochs
        )

        # ----------------------------------------------------------------
        # 4. Tensors
        # ----------------------------------------------------------------
        X_tr_t   = torch.from_numpy(X_tr.astype(np.float32))
        y_tr_t   = torch.from_numpy(y_tr.astype(np.int64))
        sw_tr_t  = torch.from_numpy(sw_tr.astype(np.float32))
        X_val_t  = torch.from_numpy(X_val.astype(np.float32))
        y_val_t  = torch.from_numpy(y_val.astype(np.int64))

        best_val_f1 = -1.0
        best_state  = None
        patience_ct = 0

        # ----------------------------------------------------------------
        # 5. Training loop
        # ----------------------------------------------------------------
        for epoch in range(self.max_epochs):
            self._mlp.train()

            # Shuffle mini-batches
            perm = torch.randperm(len(X_tr_t))
            X_tr_t   = X_tr_t[perm]
            y_tr_t   = y_tr_t[perm]
            sw_tr_t  = sw_tr_t[perm]

            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, len(X_tr_t), self.batch_size):
                end = start + self.batch_size
                xb  = X_tr_t[start:end]
                yb  = y_tr_t[start:end]
                wb  = sw_tr_t[start:end]

                logits  = self._mlp(xb)
                targets = _labels_to_coral_targets(yb, self._num_classes)
                loss    = _coral_loss(logits, targets, sample_weights=wb)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()

            # Early stopping on validation macro-F1
            if do_early_stop:
                self._mlp.eval()
                with torch.no_grad():
                    val_logits = self._mlp(X_val_t)
                    val_proba  = torch.sigmoid(val_logits).cpu().numpy()
                val_preds  = _coral_decode(val_proba)
                val_f1     = f1_score(y_val, val_preds, average="macro",
                                       zero_division=0)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state  = {k: v.clone() for k, v in self._mlp.state_dict().items()}
                    patience_ct = 0
                else:
                    patience_ct += 1

                if epoch % 20 == 0:
                    logger.debug(
                        "Epoch %d/%d  loss=%.4f  val_macro_f1=%.4f  patience=%d/%d",
                        epoch + 1, self.max_epochs,
                        epoch_loss / max(n_batches, 1), val_f1,
                        patience_ct, self.patience,
                    )

                if patience_ct >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (best val_macro_f1=%.4f)",
                        epoch + 1, best_val_f1,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            self._mlp.load_state_dict(best_state)

        self._mlp.eval()
        logger.info(
            "ESM2CoralModel.fit: %d samples, input_dim=%d, "
            "best_val_macro_f1=%.4f",
            n, input_dim, best_val_f1,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        import torch
        self._mlp.eval()
        with torch.no_grad():
            logits = self._mlp(torch.from_numpy(X.astype(np.float32)))
            proba  = torch.sigmoid(logits).cpu().numpy()
        return _coral_decode(proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        import torch
        self._mlp.eval()
        with torch.no_grad():
            logits = self._mlp(torch.from_numpy(X.astype(np.float32)))
            proba  = torch.sigmoid(logits).cpu().numpy()
        return _coral_to_class_proba(proba, num_classes=self._num_classes)

    def get_params(self) -> dict[str, Any]:
        return {
            "learning_rate":   self.learning_rate,
            "weight_decay":    self.weight_decay,
            "batch_size":      self.batch_size,
            "max_epochs":      self.max_epochs,
            "patience":        self.patience,
            "dropout_1":       self.dropout_1,
            "dropout_2":       self.dropout_2,
            "val_fraction":    self.val_fraction,
            "esm2_model_name": self.esm2_model_name,
            "random_state":    self.random_state,
        }

    def set_params(self, **params: Any) -> None:
        unknown = set(params) - self._PARAM_NAMES
        if unknown:
            raise ValueError(
                f"ESM2CoralModel.set_params: unknown parameter(s) "
                f"{sorted(unknown)}. Valid: {sorted(self._PARAM_NAMES)}"
            )
        for k, v in params.items():
            setattr(self, k, v)
        self._mlp = None   # invalidate fitted state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if self._mlp is None:
            raise RuntimeError(
                "ESM2CoralModel.predict called before fit(). "
                "Call fit(X, y) first."
            )
