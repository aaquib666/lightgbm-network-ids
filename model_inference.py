"""
model_inference.py — LightGBM Prediction Engine

Dissertation Reference:
    Chapter 5, Section 5.4.4 — Model Training Module (LGBMClassifier)

Loads the pre-trained LightGBM .pkl model at startup and exposes a
predict() method that returns (label, confidence) with inference latency
measurement (~6 µs/flow benchmark per dissertation).

Startup verification (Issue 5):
    - Prints model.feature_name_ for alignment inspection.
    - Compares model feature names against IDS FEATURE_COLUMNS.
    - If names differ only by whitespace→underscore mapping, an automatic
      column-rename mapping is applied at prediction time.
    - If an unmappable mismatch exists, startup is aborted.
"""

import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd

from feature_extractor import FEATURE_COLUMNS
from logger import IDSLogger


def _build_column_mapping(
    ids_columns: List[str],
    model_columns: List[str],
) -> Optional[Dict[str, str]]:
    """Try to build a mapping from IDS column names to model column names.

    Returns a dict {ids_name -> model_name} if every IDS column can be
    matched to exactly one model column (via normalisation: lowercase +
    spaces/underscores stripped).  Returns None on failure.
    """
    def _norm(name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "")

    model_lookup = {_norm(m): m for m in model_columns}
    mapping: Dict[str, str] = {}

    for ids_col in ids_columns:
        key = _norm(ids_col)
        if key in model_lookup:
            mapping[ids_col] = model_lookup[key]
        else:
            return None  # unmappable column

    return mapping


class ModelInference:
    """Wraps the pre-trained LightGBM binary classifier for real-time inference."""

    def __init__(self, model_path: str, threshold: float, logger: IDSLogger):
        self._threshold = threshold
        self._logger = logger
        self._column_mapping: Optional[Dict[str, str]] = None

        logger.log("MODEL", f"Loading LightGBM model from {model_path}")
        self._model = joblib.load(model_path)

        # ── Issue 5: feature alignment verification ──────────────────────
        try:
            model_features: List[str] = list(self._model.feature_name_)
            logger.log("MODEL", f"Model features:  {model_features}")
            logger.log("MODEL", f"IDS features:    {list(FEATURE_COLUMNS)}")

            if model_features == list(FEATURE_COLUMNS):
                logger.log("MODEL", "Feature alignment: EXACT MATCH ✓")
            else:
                # Attempt automatic mapping
                mapping = _build_column_mapping(FEATURE_COLUMNS, model_features)
                if mapping is not None:
                    logger.log("MODEL",
                               "Feature alignment: auto-mapped "
                               "(whitespace/underscore normalisation) ✓")
                    for ids_col, model_col in mapping.items():
                        if ids_col != model_col:
                            logger.log("MODEL",
                                       f"  {ids_col}  →  {model_col}")
                    self._column_mapping = mapping
                else:
                    logger.log("MODEL",
                               "ERROR: Feature alignment mismatch detected!")
                    logger.log("MODEL",
                               "The IDS feature columns do not match the "
                               "model's training features and cannot be "
                               "auto-mapped.")
                    logger.log("MODEL",
                               "Update FEATURE_COLUMNS in feature_extractor.py "
                               "to match the model, then restart.")
                    sys.exit(1)

        except AttributeError:
            logger.log("MODEL", "Model does not expose feature_name_ — "
                        "ensure manual feature alignment.")

        logger.log("MODEL", f"LightGBM model loaded. Features: 9 | "
                    f"Threshold: {threshold:.2f}")

    def predict(self, feature_row: pd.DataFrame, *,
                debug: bool = False) -> Tuple[float, float]:
        """
        Run inference on a single flow feature vector.

        Args:
            feature_row: DataFrame with shape (1, 9) matching training columns.
            debug:       If True, print prediction confidence before returning.

        Returns:
            (confidence, latency_us)
            - confidence:  probability for the attack class (0.0–1.0)
            - latency_us:  inference wall-clock time in microseconds

        Note:
            The NORMAL / SUSPICIOUS / ATTACK classification decision is
            made by the hybrid detection policy in ids_main.py, not here.
        """
        # Apply column renaming if the model expects different names
        if self._column_mapping is not None:
            feature_row = feature_row.rename(columns=self._column_mapping)

        t0 = time.perf_counter()

        proba = self._model.predict_proba(feature_row)
        attack_prob = float(proba[0][1])

        latency_us = (time.perf_counter() - t0) * 1e6

        if debug:
            print(f"  [DEBUG] Raw confidence: {attack_prob:.4f} | "
                  f"Latency: {latency_us:.1f} µs",
                  flush=True)

        return attack_prob, latency_us

