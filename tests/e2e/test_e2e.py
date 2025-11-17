# SPDX-License-Identifier: MPL-2.0
"""End-to-end tests for WildCore complete workflows."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from wildcore.agent import SecuritySimulationAgent
from wildcore.detector import AutoRegulatedPromptDetector
from wildcore.utils import (
    evaluate_detector,
    generate_random_embeddings,
    load_embeddings_from_file,
    save_embeddings_to_file,
)


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    def test_full_simulation_and_detection_pipeline(self) -> None:
        """Test complete pipeline: generate data -> detect anomalies -> evaluate."""
        # Step 1: Generate embeddings with known anomalies
        embeddings, labels = generate_random_embeddings(
            count=100, dimension=768, anomaly_count=25
        )

        # Step 2: Create detector with custom configuration
        detector = AutoRegulatedPromptDetector(
            threshold=0.5,
            adaptation_rate=0.1,
            ensemble_voting_threshold=2,
            enable_adaptation=True,
        )

        # Step 3: Evaluate detector
        metrics = evaluate_detector(detector, embeddings, labels)

        # Verify all metrics are present and valid
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, float))

    def test_agent_based_data_generation_and_detection(self) -> None:
        """Test using agent to generate data for detection."""
        agent = SecuritySimulationAgent()
        detector = AutoRegulatedPromptDetector()

        # Generate reference data (normal behavior)
        normal_embeddings = []
        for _ in range(20):
            agent.reset()
            normal_embeddings.append(agent.generate_embedding())

        # Generate test data (mixed normal and anomalous)
        test_data = []
        test_labels = []

        for _ in range(15):
            agent.reset()
            test_data.append(agent.generate_embedding())
            test_labels.append(0)  # Normal

        for _ in range(10):
            agent.reset()
            agent.take_role("malicious")
            test_data.append(agent.generate_embedding())
            test_labels.append(1)  # Anomalous

        # Convert to numpy arrays
        test_embeddings = np.array(test_data)
        test_labels_array = np.array(test_labels)

        # Evaluate detector
        metrics = evaluate_detector(detector, test_embeddings, test_labels_array)

        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0

    def test_persistence_and_recovery_workflow(self) -> None:
        """Test saving, loading, and reusing detector state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train detector
            detector1 = AutoRegulatedPromptDetector(threshold=0.5)

            # Generate some embeddings and detections
            embeddings, labels = generate_random_embeddings(
                count=30, dimension=20, anomaly_count=10
            )

            # Save embeddings
            embeddings_path = os.path.join(tmpdir, "test_embeddings.json")
            save_embeddings_to_file(embeddings, labels, embeddings_path)

            # Load and verify
            loaded_embeddings, loaded_labels = load_embeddings_from_file(
                embeddings_path
            )

            # Evaluate on loaded data
            metrics = evaluate_detector(detector1, loaded_embeddings, loaded_labels)

            assert metrics["accuracy"] >= 0
            assert len(loaded_embeddings) == len(embeddings)
            assert len(loaded_labels) == len(labels)

    def test_multi_agent_security_simulation(self) -> None:
        """Test simulating multiple agents with different roles."""
        agents = [
            SecuritySimulationAgent(),
            SecuritySimulationAgent(),
            SecuritySimulationAgent(),
        ]

        # Assign different roles
        agents[0].take_role("analyst")
        agents[1].take_role("malicious")
        agents[2].take_role("unauthorized")

        # Generate embeddings from each
        embeddings_by_role = {}
        for i, agent in enumerate(agents):
            embeddings = [agent.generate_embedding() for _ in range(10)]
            role = agent.roles[0] if agent.roles else "unknown"
            embeddings_by_role[role] = np.array(embeddings)

        # Verify we have distinct role-based data
        assert len(embeddings_by_role) == 3
        assert all(arr.shape == (10, 768) for arr in embeddings_by_role.values())

    def test_adaptive_detection_with_feedback_loop(self) -> None:
        """Test detector adaptation based on false detections."""
        detector = AutoRegulatedPromptDetector(
            threshold=0.5, adaptation_rate=0.2, enable_adaptation=True
        )

        # Generate test embeddings
        embeddings, labels = generate_random_embeddings(
            count=50, dimension=30, anomaly_count=15
        )

        initial_threshold = detector.threshold

        # Simulate detection feedback loop
        for i in range(10):
            idx = np.random.randint(0, len(embeddings))
            embedding = embeddings[idx]
            true_label = labels[idx]

            # Get reference (first 5 normal)
            normal_indices = np.where(labels == 0)[0][:5]
            references = embeddings[normal_indices]

            # Detect
            result = detector.ensemble_detection(embedding, references)

            # Provide feedback (simulate some false detections)
            if i % 3 == 0:
                detector.log_false_detection(is_false_positive=True)
            else:
                detector.log_false_detection(is_false_positive=False)

        # Verify threshold has adapted
        assert detector.threshold != initial_threshold
        assert detector.false_positives > 0 or detector.false_negatives > 0

    def test_detector_reset_and_reuse(self) -> None:
        """Test resetting detector for clean re-evaluation."""
        detector = AutoRegulatedPromptDetector()

        # First evaluation
        embeddings1, labels1 = generate_random_embeddings(count=20, anomaly_count=5)
        metrics1 = evaluate_detector(detector, embeddings1, labels1)

        # Reset detector
        detector.reset()

        # Verify reset
        assert detector.false_positives == 0
        assert detector.false_negatives == 0
        assert len(detector.history) == 0

        # Second evaluation with different data
        embeddings2, labels2 = generate_random_embeddings(count=30, anomaly_count=10)
        metrics2 = evaluate_detector(detector, embeddings2, labels2)

        # Both evaluations should be valid (metrics don't carry over)
        assert "accuracy" in metrics1
        assert "accuracy" in metrics2


class TestRobustness:
    """Test robustness against edge cases and unusual inputs."""

    def test_detector_with_identical_embeddings(self) -> None:
        """Detector should handle identical embeddings gracefully."""
        detector = AutoRegulatedPromptDetector()

        # All identical embeddings
        identical = np.ones((10, 20))
        identical = identical / np.linalg.norm(identical[0])

        # Should not crash
        result = detector.ensemble_detection(identical[0], list(identical[1:]))
        assert "is_anomalous" in result
        assert "confidence" in result

    def test_detector_with_extreme_values(self) -> None:
        """Detector should handle extreme embedding values."""
        detector = AutoRegulatedPromptDetector()

        # Very large values
        large = np.ones(100) * 1e10
        large = large / np.linalg.norm(large)

        # Very small values
        small = np.ones(100) * 1e-10
        small = small / np.linalg.norm(small)

        references = [np.random.randn(100) / np.sqrt(100) for _ in range(5)]

        # Should not crash on extreme values
        result_large = detector.ensemble_detection(large, references)
        result_small = detector.ensemble_detection(small, references)

        assert "is_anomalous" in result_large
        assert "is_anomalous" in result_small

    def test_single_embedding_evaluation(self) -> None:
        """Detector should work with single embedding."""
        detector = AutoRegulatedPromptDetector()

        embeddings, labels = generate_random_embeddings(count=1, anomaly_count=0)

        # Should handle single embedding gracefully
        metrics = evaluate_detector(detector, embeddings, labels)
        assert "accuracy" in metrics

    def test_large_batch_processing(self) -> None:
        """Detector should handle large batches efficiently."""
        detector = AutoRegulatedPromptDetector()

        # Generate large batch
        embeddings, labels = generate_random_embeddings(
            count=500, dimension=100, anomaly_count=100
        )

        # Should process without issues
        metrics = evaluate_detector(detector, embeddings, labels)

        assert "accuracy" in metrics
        assert metrics["true_positives"] + metrics["true_negatives"] <= 500
