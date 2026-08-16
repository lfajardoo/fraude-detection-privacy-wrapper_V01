"""
Batch evaluator — orchestrates all metrics over a labeled transaction set.

Responsibility: wire model_client + dp_layer + governor to the metrics package
and produce a structured evaluation report. Keeps metrics logic in metrics/,
keeps HTTP in app.py; this module only orchestrates.

To extend: add new metric calls inside evaluate() and include results in the
returned dict. The metrics/ package is the only place to add new formulas.
"""

import logging
from typing import Any, Dict, List, Optional

from client import model_client
from governor import GovernanceDecision, governor
from privacy_governor_simple import gobernar_politica
from privacy_layer import dp_layer
from metrics import (
    classification_report,
    noise_summary,
    privacy_utility_summary,
)

logger = logging.getLogger(__name__)

_LABEL_KEY = "label"


class BatchEvaluator:
    """
    Evaluate the full privacy pipeline over a labeled transaction batch.

    Each item in labeled_transactions must be a flat dict that contains:
      - all model feature fields (Time, V1…V14, Amount)
      - a "label" key with the ground-truth class (0 = legitimate, 1 = fraud)

    The label is stripped before sending to the model.
    """

    def evaluate(
        self,
        labeled_transactions: List[Dict[str, Any]],
        mode: str = "governed",
        threshold: float = 0.5,
        mechanism: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the pipeline over the batch and return a full metrics report.

        Args:
            labeled_transactions: transactions with ground-truth "label" field.
            mode: "raw" | "legacy" | "governed" — which DP policy to apply.
            threshold: fraud_probability cutoff to binarise predictions.
            mechanism: override DP mechanism ("laplace" / "gaussian" / None).

        Returns:
            Structured report with noise, classification, and privacy metrics.
        """
        if not labeled_transactions:
            raise ValueError("labeled_transactions must not be empty")

        y_true: List[int] = []
        y_pred_original: List[int] = []
        y_pred_dp: List[int] = []
        noise_pairs_prob: List[tuple] = []   # (original_prob, noisy_prob)
        risk_noise_pairs: List[tuple] = []   # (original_prob, |noise|)

        errors: List[str] = []

        for idx, item in enumerate(labeled_transactions):
            if _LABEL_KEY not in item:
                errors.append(f"Item {idx}: missing '{_LABEL_KEY}' field")
                continue

            true_label = int(item[_LABEL_KEY])
            features = {k: v for k, v in item.items() if k != _LABEL_KEY}

            try:
                original = model_client.predict(features)
            except Exception as exc:
                errors.append(f"Item {idx}: model error — {exc}")
                continue

            orig_prob = float(original["fraud_probability"])

            # Apply DP using the requested mode
            dp_result = self._apply_dp(original.copy(), mode, mechanism)
            noisy_prob = float(dp_result["fraud_probability"])

            y_true.append(true_label)
            y_pred_original.append(1 if orig_prob >= threshold else 0)
            y_pred_dp.append(1 if noisy_prob >= threshold else 0)
            noise_pairs_prob.append((orig_prob, noisy_prob))
            risk_noise_pairs.append((orig_prob, abs(orig_prob - noisy_prob)))

        if not y_true:
            raise RuntimeError(
                f"No transactions processed successfully. Errors: {errors}"
            )

        orig_report = classification_report(y_true, y_pred_original)
        dp_report = classification_report(y_true, y_pred_dp)

        # Utility Retention keyed by each classification metric
        utility_key = "f1"  # primary utility proxy (imbalance-robust)
        pu_summary = privacy_utility_summary(
            utility_original=orig_report[utility_key],
            utility_with_dp=dp_report[utility_key],
            risk_noise_pairs=risk_noise_pairs,
        )

        return {
            "n_transactions": len(y_true),
            "n_errors": len(errors),
            "errors": errors,
            "mode": mode,
            "threshold": threshold,
            "noise_metrics": {
                "fraud_probability": noise_summary(noise_pairs_prob),
            },
            "classification_original": orig_report,
            "classification_with_dp": dp_report,
            "utility_retention": {
                metric: (
                    dp_report[metric] / orig_report[metric]
                    if orig_report.get(metric) and orig_report[metric] != 0
                    else None
                )
                for metric in ("accuracy", "f1", "informedness", "mcc")
            },
            "privacy_utility": pu_summary,
        }

    # ------------------------------------------------------------------
    def _apply_dp(
        self,
        prediction: Dict[str, Any],
        mode: str,
        mechanism: Optional[str],
    ) -> Dict[str, Any]:
        """Apply the DP policy matching `mode` to a single prediction dict."""
        if mode == "raw":
            return prediction

        if mode == "legacy":
            decision = governor.decide(mechanism=mechanism)
            return dp_layer.apply_dp_to_response(prediction, decision=decision)

        # default: governed
        policy = gobernar_politica(
            fraud_probability=float(prediction["fraud_probability"]),
            confidence_score=float(prediction["confidence_score"]),
            iteracion=0,
        )
        decision = GovernanceDecision(
            mechanism=mechanism or governor._MECHANISM,
            epsilon=policy["epsilon_in"],
            delta=governor._DELTA,
            budget_allowed=policy["politica"] != "P4",
            epsilon_in=policy["epsilon_in"],
            epsilon_out=policy["epsilon_out"],
        )
        return dp_layer.apply_dp_to_response(prediction, decision=decision)


# Module-level singleton — same pattern as model_client / dp_layer
batch_evaluator = BatchEvaluator()
