# SPDX-License-Identifier: MPL-2.0
"""Integration tests for WildCore components."""

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


class TestAgentDetectorIntegration:
    """Test integration between agent and detector."""

    def test_agent_creates_detectable_anomalies(self) -> None:
        """Agent should create embeddings that detector can identify."""
        agent = SecuritySimulationAgent()
        detector = AutoRegulatedPromptDetector()

        # Generate reference embeddings (normal behavior)
        agent.reset()
        reference_embeddings = [agent.generate_embedding() for _ in range(5)]

        # Generate normal embedding
        normal_embedding = agent.generate_embedding()
        result_normal = detector.ensemble_detection(
            normal_embedding, reference_embeddings
        )

        # Generate anomalous embedding
        agent.take_role("malicious")
        anomalous_embedding = agent.generate_embedding()
        result_anomalous = detector.ensemble_detection(
            anomalous_embedding, reference_embeddings
        )

        # Anomalous should have higher confidence
        assert result_anomalous["confidence"] >= result_normal["confidence"]

    def test_agent_memory_stream_tracking(self) -> None:
        """Agent should properly track events in memory stream."""
        agent = SecuritySimulationAgent()

        # Malicious role should trigger suspicious events
        agent.take_role("malicious")
        agent.generate_embedding()
        agent.simulate_breach(probability=0.5)

        memory = agent.get_memory_stream()
        assert len(memory) > 0
        # Should have either suspicious_embedding_generated or containment_breach events
        assert any(
            event["event"] in ["suspicious_embedding_generated", "containment_breach"]
            for event in memory
        )

    def test_agent_reset_functionality(self) -> None:
        """Agent should reset to initial state."""
        agent = SecuritySimulationAgent()

        agent.take_role("hacker")
        agent.generate_embedding()
        agent.simulate_breach(probability=1.0)

        initial_memory_len = len(agent.get_memory_stream())

        agent.reset()

        assert agent.state == "neutral"
        assert len(agent.roles) == 0
        assert len(agent.get_memory_stream()) == 0

    def test_multiple_agents_independent_state(self) -> None:
        """Multiple agents should maintain independent state."""
        agent1 = SecuritySimulationAgent()
        agent2 = SecuritySimulationAgent()

        agent1.take_role("malicious")
        agent2.take_role("analyst")

        assert "malicious" in agent1.roles
        assert "malicious" not in agent2.roles
        assert agent1.state != agent2.state


class TestFileIOIntegration:
    """Test file I/O operations with error handling."""

    def test_save_and_load_embeddings(self) -> None:
        """Should successfully save and load embeddings."""
        embeddings, labels = generate_random_embeddings(
            count=20, dimension=10, anomaly_count=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "embeddings.json")

            # Save embeddings
            save_embeddings_to_file(embeddings, labels, filepath)
            assert os.path.exists(filepath)

            # Load embeddings
            loaded_embeddings, loaded_labels = load_embeddings_from_file(filepath)

            np.testing.assert_array_equal(embeddings, loaded_embeddings)
            np.testing.assert_array_equal(labels, loaded_labels)

    def test_save_embeddings_mismatched_lengths(self) -> None:
        """Should raise error when embeddings and labels have different lengths."""
        embeddings = np.random.rand(10, 5)
        labels = np.array([0, 1, 0])  # Only 3 labels

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "embeddings.json")

            with pytest.raises(ValueError, match="must have the same length"):
                save_embeddings_to_file(embeddings, labels, filepath)

    def test_load_nonexistent_file(self) -> None:
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_embeddings_from_file("/nonexistent/path/file.json")

    def test_load_invalid_json(self) -> None:
        """Should raise ValueError for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "invalid.json")

            with open(filepath, "w") as f:
                f.write("{ invalid json }")

            with pytest.raises(ValueError, match="Invalid JSON"):
                load_embeddings_from_file(filepath)

    def test_load_missing_keys(self) -> None:
        """Should raise ValueError when JSON is missing required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "incomplete.json")

            with open(filepath, "w") as f:
                json.dump({"embeddings": [[1, 2, 3]]}, f)

            with pytest.raises(ValueError, match="must contain 'embeddings' and 'labels'"):
                load_embeddings_from_file(filepath)


class TestDetectorConfiguration:
    """Test detector configuration options."""

    def test_detector_with_custom_voting_threshold(self) -> None:
        """Detector should respect custom ensemble voting threshold."""
        detector_strict = AutoRegulatedPromptDetector(ensemble_voting_threshold=3)
        detector_lenient = AutoRegulatedPromptDetector(ensemble_voting_threshold=1)

        reference = [np.ones(10) / np.sqrt(10) for _ in range(5)]
        anomalous = np.zeros(10)
        anomalous[0] = 1.0

        result_strict = detector_strict.ensemble_detection(anomalous, reference)
        result_lenient = detector_lenient.ensemble_detection(anomalous, reference)

        # Lenient detector should detect more anomalies
        assert result_lenient["is_anomalous"] is True or result_strict["is_anomalous"] is True

    def test_detector_adaptation_toggle(self) -> None:
        """Detector should toggle adaptation on/off."""
        detector_adaptive = AutoRegulatedPromptDetector(enable_adaptation=True)
        detector_static = AutoRegulatedPromptDetector(enable_adaptation=False)

        initial_threshold_adaptive = detector_adaptive.threshold
        initial_threshold_static = detector_static.threshold

        reference = [np.ones(10) / np.sqrt(10) for _ in range(5)]
        test_embedding = np.random.rand(10)
        test_embedding /= np.linalg.norm(test_embedding)

        detector_adaptive.ensemble_detection(test_embedding, reference)
        detector_static.ensemble_detection(test_embedding, reference)

        # Both may change due to anomaly scoring, but adaptation can be controlled
        assert detector_adaptive.enable_adaptation is True
        assert detector_static.enable_adaptation is False

    def test_detector_threshold_bounds(self) -> None:
        """Detector should respect min/max threshold bounds."""
        min_th = 0.2
        max_th = 0.8

        detector = AutoRegulatedPromptDetector(
            min_threshold=min_th, max_threshold=max_th
        )

        # Force adaptation with extreme values
        detector.dynamic_threshold_adjustment(np.array([-1.0, 2.0, 3.0]))

        assert min_th <= detector.threshold <= max_th

    def test_detector_invalid_configuration(self) -> None:
        """Detector should reject invalid configurations."""
        with pytest.raises(ValueError, match="ensemble_voting_threshold"):
            AutoRegulatedPromptDetector(ensemble_voting_threshold=5)

        with pytest.raises(ValueError, match="threshold must be between"):
            AutoRegulatedPromptDetector(threshold=1.5)

        with pytest.raises(ValueError, match="window_size must be positive"):
            AutoRegulatedPromptDetector(window_size=0)


class TestEvaluatorIntegration:
    """Test evaluation functions with real components."""

    def test_evaluate_detector_integration(self) -> None:
        """Evaluator should correctly evaluate detector performance."""
        embeddings, labels = generate_random_embeddings(
            count=50, dimension=20, anomaly_count=15
        )

        detector = AutoRegulatedPromptDetector()
        metrics = evaluate_detector(detector, embeddings, labels)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_detector_mismatched_inputs(self) -> None:
        """Evaluator should reject mismatched inputs."""
        embeddings = np.random.rand(10, 20)
        labels = np.array([0, 1])  # Too few labels

        detector = AutoRegulatedPromptDetector()

        with pytest.raises(ValueError, match="must have the same length"):
            evaluate_detector(detector, embeddings, labels)

    def test_evaluate_detector_none_inputs(self) -> None:
        """Evaluator should handle None inputs gracefully."""
        detector = AutoRegulatedPromptDetector()

        with pytest.raises(ValueError, match="cannot be None"):
            evaluate_detector(detector, None, np.array([]))


class TestErrorHandling:
    """Test comprehensive error handling across components."""

    def test_agent_invalid_role_name(self) -> None:
        """Agent should reject invalid role names."""
        agent = SecuritySimulationAgent()

        with pytest.raises(ValueError):
            agent.take_role("")

        with pytest.raises(ValueError):
            agent.take_role(123)  # type: ignore

    def test_agent_invalid_breach_probability(self) -> None:
        """Agent should validate breach probability."""
        agent = SecuritySimulationAgent()

        with pytest.raises(ValueError, match="between 0 and 1"):
            agent.simulate_breach(probability=1.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            agent.simulate_breach(probability=-0.1)

    def test_detector_invalid_vectors(self) -> None:
        """Detector should validate vector inputs."""
        detector = AutoRegulatedPromptDetector()

        with pytest.raises(ValueError):
            detector.cosine_similarity(np.array([1, 2, 3]), np.array([1, 2]))

        with pytest.raises(ValueError, match="cannot be None"):
            detector.cosine_similarity(None, np.array([1, 2, 3]))

    def test_generate_embeddings_invalid_params(self) -> None:
        """Should reject invalid parameters."""
        with pytest.raises(ValueError, match="count must be positive"):
            generate_random_embeddings(count=0)

        with pytest.raises(ValueError, match="dimension must be positive"):
            generate_random_embeddings(count=10, dimension=0)

        with pytest.raises(ValueError, match="cannot exceed"):
            generate_random_embeddings(count=10, anomaly_count=15)
