# SPDX-License-Identifier: MPL-2.0
"""Utility helpers for generating and persisting embeddings."""

import json
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("WildCore.utils")


def generate_random_embeddings(
    count: int, dimension: int = 768, anomaly_count: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random embeddings for testing.

    Parameters
    ----------
    count : int
        Number of embeddings to generate.
    dimension : int, default 768
        Dimension of each embedding.
    anomaly_count : int, default 0
        Number of anomalous embeddings to produce.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple of ``(embeddings, labels)`` where labels are ``1`` for anomalies
        and ``0`` for normal vectors.

    Raises
    ------
    ValueError
        If count, dimension, or anomaly_count are invalid.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if anomaly_count < 0:
        raise ValueError("anomaly_count cannot be negative")
    if anomaly_count > count:
        raise ValueError("anomaly_count cannot exceed count")

    # Generate normal embeddings
    normal_count = count - anomaly_count
    normal_embeddings = np.random.rand(normal_count, dimension)

    # Normalize the normal embeddings
    for i in range(normal_count):
        normal_embeddings[i] = normal_embeddings[i] / np.linalg.norm(
            normal_embeddings[i]
        )

    # Generate anomaly embeddings with a bias pattern
    anomaly_embeddings = np.zeros((anomaly_count, dimension))
    if anomaly_count > 0:
        for i in range(anomaly_count):
            # Create a biased vector that will look anomalous
            base = np.random.rand(dimension)
            bias = np.zeros(dimension)
            bias[:100] = 0.9  # Strong bias in the first 100 dimensions

            # Mix the base vector with the bias
            mixed = 0.3 * base + 0.7 * bias

            # Normalize
            anomaly_embeddings[i] = mixed / np.linalg.norm(mixed)

    # Combine normal and anomaly embeddings
    all_embeddings = (
        np.vstack((normal_embeddings, anomaly_embeddings))
        if anomaly_count > 0
        else normal_embeddings
    )

    # Create labels (0 for normal, 1 for anomaly)
    labels = np.zeros(count)
    if anomaly_count > 0:
        labels[normal_count:] = 1

    return all_embeddings, labels


def save_embeddings_to_file(
    embeddings: np.ndarray, labels: np.ndarray, filepath: str
) -> None:
    """Persist embeddings and labels to disk.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Embedding matrix to store.
    labels : numpy.ndarray
        Corresponding labels (``0`` for normal, ``1`` for anomaly).
    filepath : str
        Destination file path.

    Raises
    ------
    ValueError
        If embeddings and labels have mismatched lengths.
    IOError
        If the file cannot be written.
    """
    if embeddings is None or labels is None:
        raise ValueError("embeddings and labels cannot be None")

    if len(embeddings) != len(labels):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and labels ({len(labels)}) "
            "must have the same length"
        )

    if not filepath or not isinstance(filepath, str):
        raise ValueError("filepath must be a non-empty string")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Convert to list format for JSON serialization
        data = {"embeddings": embeddings.tolist(), "labels": labels.tolist()}

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")
    except (OSError, IOError) as e:
        logger.error(f"Failed to save embeddings to {filepath}: {e}")
        raise IOError(f"Failed to save embeddings to {filepath}: {e}") from e
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to serialize embeddings: {e}")
        raise ValueError(f"Failed to serialize embeddings: {e}") from e


def load_embeddings_from_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from disk.

    Parameters
    ----------
    filepath : str
        Location of the saved file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The loaded ``(embeddings, labels)`` pair.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not in the expected JSON format.
    IOError
        If the file cannot be read.
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("filepath must be a non-empty string")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")

    try:
        with open(filepath, encoding="utf-8") as handle:
            data = json.load(handle)

        if "embeddings" not in data or "labels" not in data:
            raise ValueError(
                "JSON file must contain 'embeddings' and 'labels' keys"
            )

        embeddings = np.array(data["embeddings"])
        labels = np.array(data["labels"])

        if len(embeddings) != len(labels):
            raise ValueError(
                f"Loaded embeddings ({len(embeddings)}) and labels ({len(labels)}) "
                "must have the same length"
            )

        logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")

        return embeddings, labels

    except (OSError, IOError) as e:
        logger.error(f"Failed to read embeddings file {filepath}: {e}")
        raise IOError(f"Failed to read embeddings file {filepath}: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {filepath}: {e}")
        raise ValueError(f"Invalid JSON format in {filepath}: {e}") from e


def evaluate_detector(
    detector: Any, embeddings: np.ndarray, true_labels: np.ndarray
) -> dict[str, Any]:
    """Evaluate a detector on labeled embeddings.

    Parameters
    ----------
    detector : object
        Detector instance exposing ``ensemble_detection``.
    embeddings : numpy.ndarray
        Embeddings to classify.
    true_labels : numpy.ndarray
        Ground truth labels, ``0`` for normal and ``1`` for anomaly.

    Returns
    -------
    Dict[str, Any]
        Accuracy, precision, recall and F1 score for the detector.

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible.
    AttributeError
        If detector does not have required methods.
    """
    if detector is None:
        raise ValueError("detector cannot be None")

    if not hasattr(detector, "ensemble_detection"):
        raise AttributeError("detector must have an 'ensemble_detection' method")

    if embeddings is None or true_labels is None:
        raise ValueError("embeddings and true_labels cannot be None")

    if len(embeddings) != len(true_labels):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and true_labels ({len(true_labels)}) "
            "must have the same length"
        )

    if len(embeddings) == 0:
        raise ValueError("embeddings and true_labels cannot be empty")

    predictions = []

    # Use a small subset of the data as the reference set (first 5 normal points)
    normal_indices = np.where(true_labels == 0)[0][:5]

    if len(normal_indices) == 0:
        logger.warning("No normal embeddings found in labels for reference set")
        normal_indices = np.array([0])

    reference_embeddings = embeddings[normal_indices]

    # Test each embedding
    for i in range(len(embeddings)):
        # Skip the embeddings that are in the reference set
        if i in normal_indices:
            continue

        try:
            result = detector.ensemble_detection(embeddings[i], reference_embeddings)
            predictions.append(1 if result.get("is_anomalous", False) else 0)
        except Exception as e:
            logger.error(f"Error detecting embedding {i}: {e}")
            raise ValueError(f"Error detecting embedding {i}: {e}") from e

    # Prepare true labels (excluding reference embeddings)
    test_true_labels = np.delete(true_labels, normal_indices)

    if len(predictions) == 0:
        logger.warning("No predictions made; returning default metrics")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
        }

    # Calculate metrics
    true_positives = sum(
        (p == 1 and t == 1) for p, t in zip(predictions, test_true_labels, strict=False)
    )
    false_positives = sum(
        (p == 1 and t == 0) for p, t in zip(predictions, test_true_labels, strict=False)
    )
    true_negatives = sum(
        (p == 0 and t == 0) for p, t in zip(predictions, test_true_labels, strict=False)
    )
    false_negatives = sum(
        (p == 0 and t == 1) for p, t in zip(predictions, test_true_labels, strict=False)
    )

    accuracy = (
        (true_positives + true_negatives) / len(predictions) if predictions else 0
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }
