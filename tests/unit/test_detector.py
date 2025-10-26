# SPDX-License-Identifier: MPL-2.0
"""Unit tests for the :mod:`wildcore.detector` module."""

from __future__ import annotations

import numpy as np
import pytest

from wildcore.detector import AutoRegulatedPromptDetector


@pytest.fixture()
def detector() -> AutoRegulatedPromptDetector:
    """Return a detector instance with default settings."""
    return AutoRegulatedPromptDetector()


@pytest.fixture()
def rng() -> np.random.Generator:
    """Return a seeded random number generator for deterministic tests."""
    return np.random.default_rng(seed=2024)


def test_anomaly_scoring_calculates_correctly(detector: AutoRegulatedPromptDetector) -> None:
    """An obvious outlier should produce the highest anomaly score."""
    similarities = np.array([0.2, 0.9, 0.3])
    anomaly_scores = detector.anomaly_scoring(similarities)

    assert anomaly_scores.shape == (3,)
    assert int(np.argmax(anomaly_scores)) == 1


def test_dynamic_threshold_adjustment(detector: AutoRegulatedPromptDetector) -> None:
    """The detector should raise its threshold when high similarities are observed."""
    initial_threshold = detector.threshold
    detector.dynamic_threshold_adjustment(np.array([0.8, 0.85, 0.9, 0.95]))

    assert detector.threshold > initial_threshold


def test_ensemble_detection_handles_normal_and_anomalous_embeddings(
    detector: AutoRegulatedPromptDetector,
    rng: np.random.Generator,
) -> None:
    """Normal embeddings should pass while clearly anomalous ones are flagged."""
    normal_dim = 10
    reference_vector = np.ones(normal_dim) / np.sqrt(normal_dim)
    reference_embeddings = []
    for _ in range(5):
        perturbed = reference_vector + 0.01 * rng.normal(size=normal_dim)
        reference_embeddings.append(perturbed / np.linalg.norm(perturbed))

    normal_embedding = reference_embeddings[0].copy()

    anomalous_embedding = reference_vector.copy()
    anomalous_embedding[:3] = -reference_vector[:3]
    anomalous_embedding = anomalous_embedding / np.linalg.norm(anomalous_embedding)

    normal_result = detector.ensemble_detection(normal_embedding, reference_embeddings)
    assert normal_result["is_anomalous"] is False

    anomalous_result = detector.ensemble_detection(
        anomalous_embedding, reference_embeddings
    )
    assert anomalous_result["is_anomalous"] is True
    assert len(anomalous_result["methods_triggered"]) > 0


def test_cosine_similarity_computes_expected_values(
    detector: AutoRegulatedPromptDetector,
) -> None:
    """Cosine similarity should align with well known vector relationships."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert np.isclose(detector.cosine_similarity(v1, v2), 0.0)

    v3 = np.array([0.5, 0.5, 0.5])
    v4 = np.array([1.0, 1.0, 1.0])
    assert np.isclose(detector.cosine_similarity(v3, v4), 1.0)

    v5 = np.array([1.0, 0.0])
    v6 = np.array([1.0, 1.0])
    assert np.isclose(detector.cosine_similarity(v5, v6), 1 / np.sqrt(2), atol=1e-6)
