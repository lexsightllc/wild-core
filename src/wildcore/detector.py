# SPDX-License-Identifier: MPL-2.0
"""Utilities for detecting anomalous embeddings."""

import logging
from collections import deque
from typing import Any

import numpy as np


class AutoRegulatedPromptDetector:
    """Ensemble detector with a self-adjusting threshold."""

    def __init__(
        self,
        threshold: float = 0.5,
        window_size: int = 10,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        enable_adaptation: bool = True,
        ensemble_voting_threshold: int = 2,
    ):
        """Create a new detector instance.

        Parameters
        ----------
        threshold : float, default 0.5
            Initial similarity threshold used for classification.
        window_size : int, default 10
            Number of historical results to keep for adaptation.
        adaptation_rate : float, default 0.1
            Influence of new observations on the threshold.
        min_threshold : float, default 0.1
            Minimum allowed threshold value.
        max_threshold : float, default 0.9
            Maximum allowed threshold value.
        enable_adaptation : bool, default True
            Whether to enable dynamic threshold adaptation.
        ensemble_voting_threshold : int, default 2
            Number of methods that must agree for anomaly detection (1-3).

        Raises
        ------
        ValueError
            If parameters are out of valid ranges.
        """
        if not (0 <= threshold <= 1):
            raise ValueError("threshold must be between 0 and 1")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not (0 <= adaptation_rate <= 1):
            raise ValueError("adaptation_rate must be between 0 and 1")
        if not (0 < min_threshold < max_threshold < 1):
            raise ValueError("min_threshold < max_threshold and both in (0, 1)")
        if ensemble_voting_threshold not in (1, 2, 3):
            raise ValueError("ensemble_voting_threshold must be 1, 2, or 3")

        self.threshold = threshold
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.enable_adaptation = enable_adaptation
        self.ensemble_voting_threshold = ensemble_voting_threshold

        # Initialize history storage
        self.history: deque[dict[str, Any]] = deque(maxlen=window_size)
        self.detected_anomalies: list[dict[str, Any]] = []
        self.false_positives = 0
        self.false_negatives = 0

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("AutoRegulatedPromptDetector")

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity.

        Parameters
        ----------
        vec1 : numpy.ndarray
            First vector.
        vec2 : numpy.ndarray
            Second vector.

        Returns
        -------
        float
            Cosine similarity in the ``[0, 1]`` range.

        Raises
        ------
        ValueError
            If vectors are invalid or have mismatched dimensions.
        """
        if vec1 is None or vec2 is None:
            raise ValueError("Vectors cannot be None")

        if not isinstance(vec1, np.ndarray):
            try:
                vec1 = np.array(vec1)
            except (ValueError, TypeError) as e:
                raise ValueError(f"vec1 must be convertible to numpy array: {e}") from e

        if not isinstance(vec2, np.ndarray):
            try:
                vec2 = np.array(vec2)
            except (ValueError, TypeError) as e:
                raise ValueError(f"vec2 must be convertible to numpy array: {e}") from e

        if vec1.shape != vec2.shape:
            raise ValueError(
                f"Vector dimensions must match: vec1 {vec1.shape} vs vec2 {vec2.shape}"
            )

        # Ensure the vectors are normalized
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            self.logger.warning("Zero-norm vector detected in cosine_similarity")
            return 0.0

        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2

        return float(np.dot(vec1_normalized, vec2_normalized))

    def anomaly_scoring(self, similarities: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores using deviations from the median.

        Parameters
        ----------
        similarities : numpy.ndarray
            Similarity values to reference embeddings.

        Returns
        -------
        numpy.ndarray
            Score for each similarity where higher means more anomalous.
        """
        if len(similarities) < 2:
            return np.zeros_like(similarities)

        # Calculate median as our reference point for "normal" behavior
        median = np.median(similarities)

        # Calculate absolute deviation from the median
        deviation = np.abs(similarities - median)

        # The anomaly score is the deviation normalized by the maximum deviation
        # We add a small epsilon to avoid division by zero
        epsilon = 1e-10
        anomaly_scores = deviation / (np.max(deviation) + epsilon)

        return anomaly_scores

    def ensemble_detection(
        self, embedding: np.ndarray, reference_embeddings: list[np.ndarray]
    ) -> dict[str, Any]:
        """Classify an embedding using multiple detection methods.

        Parameters
        ----------
        embedding : numpy.ndarray
            Vector to classify.
        reference_embeddings : list of numpy.ndarray
            Collection of embeddings that represent normal behavior.

        Returns
        -------
        dict
            Detection results including anomaly status and confidence.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if embedding is None:
            raise ValueError("embedding cannot be None")

        if len(reference_embeddings) == 0:
            self.logger.warning("No reference embeddings provided for comparison")
            return {
                "is_anomalous": False,
                "confidence": 0.0,
                "methods_triggered": [],
                "min_similarity": 0.0,
                "max_anomaly_score": 0.0,
            }

        # Validate embeddings
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding)
            except (ValueError, TypeError) as e:
                raise ValueError(f"embedding must be convertible to numpy array: {e}") from e

        # Calculate similarities to all reference embeddings
        similarities = np.array(
            [self.cosine_similarity(embedding, ref) for ref in reference_embeddings]
        )

        # Method 1: Simple threshold on minimum similarity
        min_similarity = np.min(similarities)
        method1_triggered = min_similarity < self.threshold

        # Method 2: Anomaly scoring
        anomaly_scores = self.anomaly_scoring(similarities)
        max_anomaly_score = np.max(anomaly_scores)
        method2_triggered = max_anomaly_score > self.threshold

        # Method 3: Distribution analysis - check if the distribution is unusual
        mean = np.mean(similarities)
        std = np.std(similarities)
        z_scores = (similarities - mean) / (std + 1e-10)  # Avoid division by zero
        method3_triggered = np.any(np.abs(z_scores) > 2.0)  # z-score threshold of 2

        # Ensemble voting with configurable threshold
        methods_triggered = []
        if method1_triggered:
            methods_triggered.append("similarity_threshold")
        if method2_triggered:
            methods_triggered.append("anomaly_scoring")
        if method3_triggered:
            methods_triggered.append("distribution_analysis")

        votes = len(methods_triggered)
        is_anomalous = votes >= self.ensemble_voting_threshold

        # Calculate confidence based on how many methods triggered
        confidence = votes / 3.0

        # Update history with this detection
        self.history.append(
            {
                "is_anomalous": is_anomalous,
                "confidence": confidence,
                "min_similarity": min_similarity,
                "max_anomaly_score": max_anomaly_score,
                "methods_triggered": methods_triggered,
            }
        )

        # Dynamic threshold adjustment
        if self.enable_adaptation:
            self.dynamic_threshold_adjustment(similarities)

        return {
            "is_anomalous": is_anomalous,
            "confidence": confidence,
            "methods_triggered": methods_triggered,
            "min_similarity": float(min_similarity),
            "max_anomaly_score": float(max_anomaly_score),
        }

    def dynamic_threshold_adjustment(self, similarities: np.ndarray) -> None:
        """Adapt the detection threshold using recent similarities.

        Parameters
        ----------
        similarities : numpy.ndarray
            Recent similarity values observed during detection.
        """
        if len(similarities) < 2:
            return

        # Calculate the IQR (Interquartile Range) of similarities
        q1 = np.percentile(similarities, 25)
        q3 = np.percentile(similarities, 75)
        iqr = q3 - q1

        # Adjust threshold to be slightly below the lower bound of the IQR
        # This helps detect outliers while being adaptive to the current data
        new_threshold = q1 - 1.5 * iqr * self.adaptation_rate

        # Ensure the threshold stays within configured bounds
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        # Smooth the change to avoid abrupt threshold shifts
        self.threshold = (
            1 - self.adaptation_rate
        ) * self.threshold + self.adaptation_rate * new_threshold

        self.logger.debug(f"Adjusted threshold to {self.threshold:.4f}")

    def log_false_detection(self, is_false_positive: bool) -> None:
        """Record a false positive or false negative result.

        Parameters
        ----------
        is_false_positive : bool
            ``True`` if the last detection was a false positive, ``False`` otherwise.
        """
        if is_false_positive:
            self.false_positives += 1
        else:
            self.false_negatives += 1

        # Adjust the threshold based on the false detection type
        if is_false_positive:
            # If we have too many false positives, increase the threshold
            self.threshold = min(
                self.max_threshold, self.threshold + 0.05 * self.adaptation_rate
            )
        else:
            # If we have too many false negatives, decrease the threshold
            self.threshold = max(
                self.min_threshold, self.threshold - 0.05 * self.adaptation_rate
            )

        self.logger.info(
            f"Updated threshold to {self.threshold:.4f} after {'false positive' if is_false_positive else 'false negative'}"
        )

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self.history.clear()
        self.detected_anomalies.clear()
        self.false_positives = 0
        self.false_negatives = 0

    def get_stats(self) -> dict[str, Any]:
        """Get current detector statistics.

        Returns
        -------
        dict
            Current detector state and performance metrics.
        """
        return {
            "current_threshold": self.threshold,
            "adaptation_enabled": self.enable_adaptation,
            "ensemble_voting_threshold": self.ensemble_voting_threshold,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "history_size": len(self.history),
            "anomalies_detected": len(self.detected_anomalies),
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Return basic performance statistics.

        Returns
        -------
        dict
            Accuracy and error counts for the detector.
        """
        # Calculate basic metrics
        total_detections = len(self.detected_anomalies)
        total_errors = self.false_positives + self.false_negatives

        if total_detections > 0:
            accuracy = 1 - (total_errors / (total_detections + total_errors))
        else:
            accuracy = 0.0

        return {
            "total_detections": total_detections,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": accuracy,
            "current_threshold": self.threshold,
        }
