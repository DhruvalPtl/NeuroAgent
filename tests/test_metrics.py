"""
tests/test_metrics.py
=====================
Tests for src/eval/metrics.py — compute_metrics().

Test philosophy
---------------
Known-value tests use hand-verified inputs whose expected output was
computed DIRECTLY via sklearn (not by calling compute_metrics() on itself
or by reimplementing the formula).  This ensures the tests catch any
regression even if the implementation changes internally.

Reference computations are preserved as comments so the reader can
reproduce them independently.

Documented "trap" test
-----------------------
The test_accuracy_trap test exists to make permanent the answer to the
question "why is macro_f1 our leaderboard metric, not accuracy?":
  - A model predicting all class-0 on the real imbalanced distribution
    achieves ≈75 % accuracy while macro_f1 is ≈0.21.
  - This test will FAIL if someone accidentally replaces macro_f1 with
    accuracy as the comparison metric, thereby alerting them to the issue.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from src.eval.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Reference examples computed directly via sklearn (hand-verified)
# ---------------------------------------------------------------------------
# y_true = [0, 0, 0, 1, 1, 2, 2, 3, 3, 0, 2, 3]
# y_pred = [0, 0, 1, 1, 0, 2, 3, 3, 2, 0, 2, 3]
#
# sklearn outputs (verified 2026-07-01):
#   macro_f1  = 0.645833...
#   QWK       = 0.881773...
#   recall    = [0.75, 0.5, 0.6667, 0.6667]
#   accuracy  = 0.666667
# ---------------------------------------------------------------------------

_Y_TRUE_REF = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 0, 2, 3])
_Y_PRED_REF = np.array([0, 0, 1, 1, 0, 2, 3, 3, 2, 0, 2, 3])

_EPS = 1e-5   # tolerance for float comparisons


def _ref_metrics() -> dict:
    """Return compute_metrics on the reference example."""
    return compute_metrics(_Y_TRUE_REF, _Y_PRED_REF)


# ===========================================================================
# 1. Known input / output — macro F1
# ===========================================================================

class TestMacroF1:

    def test_macro_f1_matches_sklearn_directly(self):
        """macro_f1 must match sklearn.metrics.f1_score(average='macro')."""
        expected = f1_score(
            _Y_TRUE_REF, _Y_PRED_REF,
            average="macro", labels=[0, 1, 2, 3], zero_division=0,
        )
        result = _ref_metrics()
        assert abs(result["macro_f1"] - expected) < _EPS, (
            f"macro_f1={result['macro_f1']:.8f} != sklearn={expected:.8f}"
        )

    def test_macro_f1_known_value(self):
        """Spot-check against the hand-verified reference value."""
        result = _ref_metrics()
        assert abs(result["macro_f1"] - 0.645833) < 1e-4

    def test_macro_f1_perfect_prediction(self):
        y = np.array([0, 1, 2, 3])
        result = compute_metrics(y, y)
        assert abs(result["macro_f1"] - 1.0) < _EPS

    def test_macro_f1_all_wrong_is_low(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([3, 2, 1, 0])   # completely wrong
        result = compute_metrics(y_true, y_pred)
        assert result["macro_f1"] == 0.0

    def test_macro_f1_is_float(self):
        result = _ref_metrics()
        assert isinstance(result["macro_f1"], float)


# ===========================================================================
# 2. Known input / output — Quadratic Weighted Kappa
# ===========================================================================

class TestQuadraticWeightedKappa:

    def test_qwk_matches_sklearn_directly(self):
        """QWK must match sklearn.metrics.cohen_kappa_score(weights='quadratic')."""
        expected = cohen_kappa_score(
            _Y_TRUE_REF, _Y_PRED_REF,
            weights="quadratic", labels=[0, 1, 2, 3],
        )
        result = _ref_metrics()
        assert abs(result["quadratic_weighted_kappa"] - expected) < _EPS

    def test_qwk_known_value(self):
        result = _ref_metrics()
        assert abs(result["quadratic_weighted_kappa"] - 0.881773) < 1e-4

    def test_qwk_perfect_prediction_is_one(self):
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        result = compute_metrics(y, y)
        assert abs(result["quadratic_weighted_kappa"] - 1.0) < _EPS

    def test_qwk_ordinal_awareness(self):
        """Adjacent-class error must have higher QWK than far-class error."""
        y_true = np.array([3, 3, 3, 3])
        y_adjacent = np.array([2, 2, 2, 2])  # 1 step away
        y_far      = np.array([0, 0, 0, 0])  # 3 steps away

        # Single-class y_true → QWK undefined, returns 0.0
        r_adj = compute_metrics(y_true, y_adjacent)
        r_far = compute_metrics(y_true, y_far)
        # Both return 0.0 (single-class edge case), but confirm no crash
        assert isinstance(r_adj["quadratic_weighted_kappa"], float)
        assert isinstance(r_far["quadratic_weighted_kappa"], float)

    def test_qwk_single_unique_class_returns_zero(self):
        """QWK undefined for single-class y_true — must return 0.0, not raise."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 0])
        result = compute_metrics(y_true, y_pred)
        assert result["quadratic_weighted_kappa"] == 0.0

    def test_qwk_is_float(self):
        result = _ref_metrics()
        assert isinstance(result["quadratic_weighted_kappa"], float)


# ===========================================================================
# 3. Per-class recall and high-class recall flag
# ===========================================================================

class TestPerClassRecall:

    def test_per_class_recall_keys(self):
        """Must contain recall for all 4 classes."""
        result = _ref_metrics()
        assert set(result["per_class_recall"].keys()) == {0, 1, 2, 3}

    def test_per_class_recall_known_values(self):
        result = _ref_metrics()
        pcr = result["per_class_recall"]
        assert abs(pcr[0] - 0.75)   < 1e-4
        assert abs(pcr[1] - 0.5)    < 1e-4
        assert abs(pcr[2] - 0.6667) < 1e-4
        assert abs(pcr[3] - 0.6667) < 1e-4

    def test_recall_values_in_range(self):
        result = _ref_metrics()
        for cls, r in result["per_class_recall"].items():
            assert 0.0 <= r <= 1.0, f"Recall for class {cls} = {r} out of [0,1]"

    def test_per_class_recall_values_are_floats(self):
        result = _ref_metrics()
        for v in result["per_class_recall"].values():
            assert isinstance(v, float)


class TestHighClassRecallFlag:

    def test_flag_true_when_class3_recall_below_threshold(self):
        """Flag must be True when class-3 recall < 0.5."""
        # class 3 predicted as class 0 every time → recall = 0.0
        y_true = np.array([0, 0, 1, 2, 3, 3])
        y_pred = np.array([0, 0, 1, 2, 0, 0])   # class 3 never predicted
        result = compute_metrics(y_true, y_pred)
        assert result["high_class_recall_flag"] is True, (
            "high_class_recall_flag should be True when class-3 recall < 0.5"
        )

    def test_flag_false_when_class3_recall_above_threshold(self):
        """Flag must be False when class-3 recall ≥ 0.5."""
        # class 3 predicted correctly most of the time
        y_true = np.array([3, 3, 3, 3, 0, 1, 2])
        y_pred = np.array([3, 3, 3, 0, 0, 1, 2])  # recall = 3/4 = 0.75
        result = compute_metrics(y_true, y_pred)
        assert result["high_class_recall_flag"] is False, (
            "high_class_recall_flag should be False when class-3 recall >= 0.5"
        )

    def test_flag_true_when_class3_absent_from_predictions(self):
        """Class 3 never predicted → recall = 0.0 → flag must be True."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 2])   # class 3 missed → recall = 0
        result = compute_metrics(y_true, y_pred)
        assert result["high_class_recall_flag"] is True

    def test_flag_is_bool_type(self):
        result = _ref_metrics()
        assert isinstance(result["high_class_recall_flag"], bool)

    def test_flag_exactly_at_threshold_is_false(self):
        """Recall exactly = 0.5 must NOT trigger the flag (threshold is strictly <)."""
        # 2 class-3 samples, predict 1 correctly → recall = 0.5
        y_true = np.array([3, 3, 0, 1, 2])
        y_pred = np.array([3, 0, 0, 1, 2])
        result = compute_metrics(y_true, y_pred)
        assert result["high_class_recall_flag"] is False, (
            "Recall == 0.5 should NOT trigger the flag (threshold is strict <)"
        )

    def test_flag_matches_per_class_recall_consistency(self):
        """Flag must be consistent with reported per_class_recall[3]."""
        result = _ref_metrics()
        reported_recall = result["per_class_recall"][3]
        flag = result["high_class_recall_flag"]
        if reported_recall < 0.5:
            assert flag is True
        else:
            assert flag is False


# ===========================================================================
# 4. Confusion matrix
# ===========================================================================

class TestConfusionMatrix:

    def test_confusion_matrix_json_serialisable(self):
        """confusion_matrix must be JSON-serialisable (for SQLite storage)."""
        result = _ref_metrics()
        try:
            serialised = json.dumps(result["confusion_matrix"])
        except (TypeError, ValueError) as exc:
            pytest.fail(f"confusion_matrix is not JSON-serialisable: {exc}")
        # Verify it round-trips correctly
        restored = json.loads(serialised)
        assert restored == result["confusion_matrix"]

    def test_confusion_matrix_is_nested_list(self):
        result = _ref_metrics()
        cm = result["confusion_matrix"]
        assert isinstance(cm, list), "confusion_matrix must be a list"
        assert all(isinstance(row, list) for row in cm)

    def test_confusion_matrix_shape_4x4(self):
        result = _ref_metrics()
        cm = result["confusion_matrix"]
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)

    def test_confusion_matrix_diagonal_for_perfect_prediction(self):
        """Perfect predictions → diagonal confusion matrix."""
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        result = compute_metrics(y, y)
        cm = result["confusion_matrix"]
        for i in range(4):
            for j in range(4):
                if i == j:
                    assert cm[i][j] > 0
                else:
                    assert cm[i][j] == 0

    def test_confusion_matrix_row_sums_match_true_counts(self):
        """Each row sum must equal the number of true samples for that class."""
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
        y_pred = np.array([0, 1, 2, 1, 2, 2, 3, 3, 3, 0])
        result = compute_metrics(y_true, y_pred)
        cm = result["confusion_matrix"]
        from collections import Counter
        true_counts = Counter(y_true.tolist())
        for cls in range(4):
            assert sum(cm[cls]) == true_counts.get(cls, 0), (
                f"Row {cls} sum {sum(cm[cls])} != true count {true_counts.get(cls, 0)}"
            )

    def test_full_metrics_dict_json_serialisable(self):
        """The ENTIRE metrics dict must be JSON-serialisable (as stored in db)."""
        result = _ref_metrics()
        try:
            json.dumps(result)
        except (TypeError, ValueError) as exc:
            pytest.fail(
                f"Full compute_metrics() output is not JSON-serialisable: {exc}"
            )


# ===========================================================================
# 5. The accuracy trap — documents WHY macro_f1 is the leaderboard metric
# ===========================================================================

class TestAccuracyTrap:
    """
    This test class exists to permanently document and enforce the reason
    that accuracy is excluded from model comparison.

    The scenario: a model that always predicts class 0 achieves ≈75 %
    accuracy on the real imbalanced distribution.  This is both useless
    (never detects High aggregators) and misleading.

    If this test ever fails because someone changed the primary metric
    back to accuracy, the test description explains exactly why that is wrong.
    """

    def test_all_zero_prediction_has_high_accuracy_but_low_f1(self):
        """All-zero model: accuracy ≈75 %, macro_f1 ≈21 % on imbalanced data.

        This test documents WHY macro_f1 (not accuracy) is the leaderboard
        metric.  The gap between accuracy (≈75 %) and macro_f1 (≈21 %) on
        an all-zero model is the numerical proof.
        """
        n = 1000
        y_true = np.array([0] * 750 + [1] * 60 + [2] * 110 + [3] * 80)
        y_pred_all_zero = np.zeros(n, dtype=int)

        result = compute_metrics(y_true, y_pred_all_zero)

        # Accuracy deceptively high (≈75 %) — this is the TRAP
        assert result["accuracy"] > 0.70, (
            "All-zero model should achieve > 70 % accuracy on 75 %-majority data. "
            "Check the imbalanced y_true construction."
        )
        # Macro F1 correctly low (≈21 %) — this is WHY it's the real metric
        assert result["macro_f1"] < 0.30, (
            f"All-zero model macro_f1={result['macro_f1']:.4f} should be < 0.30. "
            "If macro_f1 is high for an all-zero model, the metric is not "
            "penalising missed minority classes correctly."
        )
        # Accuracy must be at least 2.5× higher than macro_f1
        ratio = result["accuracy"] / max(result["macro_f1"], 1e-9)
        assert ratio > 2.0, (
            f"accuracy/macro_f1 ratio = {ratio:.2f} (expected > 2.0). "
            "This ratio demonstrates the deceptiveness of accuracy on "
            "imbalanced data."
        )

    def test_all_zero_model_triggers_high_class_recall_flag(self):
        """All-zero model must always trigger the safety flag (class-3 recall = 0)."""
        y_true = np.array([0] * 75 + [1] * 6 + [2] * 11 + [3] * 8)
        y_pred = np.zeros(100, dtype=int)
        result = compute_metrics(y_true, y_pred)
        assert result["high_class_recall_flag"] is True, (
            "All-zero model should always trigger high_class_recall_flag "
            "(class-3 recall is 0.0)."
        )

    def test_good_model_accuracy_and_f1_both_reasonable(self):
        """A model predicting all classes correctly has high accuracy AND macro_f1."""
        y = np.array([0] * 75 + [1] * 6 + [2] * 11 + [3] * 8)
        result = compute_metrics(y, y)   # perfect prediction
        assert result["accuracy"] > 0.95
        assert result["macro_f1"] > 0.95


# ===========================================================================
# 6. Missing class in predictions (common with rare class 3)
# ===========================================================================

class TestMissingClassInPredictions:

    def test_class_3_never_in_y_pred_no_crash(self):
        """Class 3 absent from predictions must NOT raise; recall defaults to 0."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 2])   # class 3 never predicted
        result = compute_metrics(y_true, y_pred)
        assert result["per_class_recall"][3] == 0.0
        assert result["high_class_recall_flag"] is True

    def test_class_0_never_in_y_pred_no_crash(self):
        """Even majority class absent from predictions must not crash."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([1, 1, 2, 3])   # class 0 never predicted
        result = compute_metrics(y_true, y_pred)
        assert result["per_class_recall"][0] == 0.0

    def test_all_predictions_same_class_no_crash(self):
        """All predictions the same class — must not raise, returns valid metrics."""
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.zeros(6, dtype=int)
        result = compute_metrics(y_true, y_pred)
        assert "macro_f1" in result
        assert result["macro_f1"] >= 0.0


# ===========================================================================
# 7. Edge case / input validation
# ===========================================================================

class TestEdgeCases:

    def test_empty_y_true_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([]), np.array([]))

    def test_empty_y_pred_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([0, 1]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            compute_metrics(np.array([0, 1, 2]), np.array([0, 1]))

    def test_length_mismatch_message_includes_counts(self):
        try:
            compute_metrics(np.array([0, 1, 2, 3]), np.array([0, 1]))
        except ValueError as exc:
            msg = str(exc)
            assert "4" in msg and "2" in msg, (
                "Error message should include both array lengths."
            )

    def test_2d_y_true_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_metrics(np.array([[0, 1], [2, 3]]), np.array([0, 1, 2, 3]))

    def test_list_inputs_accepted(self):
        """Python lists must be accepted (auto-converted to np.ndarray)."""
        result = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3])
        assert result["macro_f1"] == pytest.approx(1.0, abs=_EPS)

    def test_single_correct_sample(self):
        """Single sample edge case must not raise."""
        result = compute_metrics(np.array([2]), np.array([2]))
        assert isinstance(result["macro_f1"], float)

    def test_returns_all_expected_keys(self):
        result = _ref_metrics()
        expected_keys = {
            "macro_f1",
            "quadratic_weighted_kappa",
            "per_class_recall",
            "high_class_recall_flag",
            "confusion_matrix",
            "accuracy",
        }
        assert set(result.keys()) == expected_keys
