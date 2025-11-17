# WildCore Usage Guide

## Overview

WildCore is an advanced AI security and anomaly detection framework for testing embedding-based systems. This guide covers practical usage patterns and best practices.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Configuration Options](#configuration-options)
3. [Integration Patterns](#integration-patterns)
4. [Advanced Scenarios](#advanced-scenarios)
5. [Troubleshooting](#troubleshooting)

## Basic Usage

### Command-Line Interface

Run the CLI demo with default settings:

```bash
python -m wildcore --iterations 12 --threshold 0.5
```

**Available Options:**
- `--iterations`: Number of simulation cycles (default: 12)
- `--threshold`: Initial detection threshold (default: 0.5)
- `--reference-size`: Number of baseline embeddings (default: 6)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Optional JSON file to save results

### Example: Basic Agent and Detector

```python
from wildcore.agent import SecuritySimulationAgent
from wildcore.detector import AutoRegulatedPromptDetector

# Create agent and detector
agent = SecuritySimulationAgent()
detector = AutoRegulatedPromptDetector()

# Generate reference embeddings (normal behavior)
agent.reset()
reference_embeddings = [agent.generate_embedding() for _ in range(5)]

# Detect normal embedding
normal_embedding = agent.generate_embedding()
result = detector.ensemble_detection(normal_embedding, reference_embeddings)
print(f"Normal embedding anomalous: {result['is_anomalous']}")

# Detect anomalous embedding
agent.take_role("malicious")
anomalous_embedding = agent.generate_embedding()
result = detector.ensemble_detection(anomalous_embedding, reference_embeddings)
print(f"Anomalous embedding detected: {result['is_anomalous']}")
```

## Configuration Options

### Detector Configuration

The `AutoRegulatedPromptDetector` accepts several configuration parameters:

```python
detector = AutoRegulatedPromptDetector(
    threshold=0.5,              # Initial similarity threshold [0-1]
    window_size=10,             # Historical size for adaptation
    adaptation_rate=0.1,        # Influence of new observations [0-1]
    min_threshold=0.1,          # Minimum threshold bound
    max_threshold=0.9,          # Maximum threshold bound
    enable_adaptation=True,     # Enable dynamic threshold adjustment
    ensemble_voting_threshold=2 # Methods that must agree [1-3]
)
```

### Pre-defined Configurations

Load configurations from JSON files:

```python
import json

# Load sensitive configuration
with open("configs/detector_sensitive.json") as f:
    config = json.load(f)["detector"]

detector = AutoRegulatedPromptDetector(**config)
```

### Configuration Profiles

| Profile | Use Case | Threshold | Voting | Adaptation |
|---------|----------|-----------|--------|-----------|
| **default** | Balanced detection | 0.5 | 2/3 | Enabled |
| **sensitive** | High security, tolerate FP | 0.3 | 1/3 | Enabled |
| **lenient** | Low risk, minimize FP | 0.7 | 3/3 | Enabled |

## Integration Patterns

### Pattern 1: Batch Evaluation

```python
from wildcore.utils import generate_random_embeddings, evaluate_detector

# Generate test dataset
embeddings, labels = generate_random_embeddings(
    count=100, dimension=768, anomaly_count=25
)

# Evaluate detector
detector = AutoRegulatedPromptDetector()
metrics = evaluate_detector(detector, embeddings, labels)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.2%}")
```

### Pattern 2: Persistence and Recovery

```python
from wildcore.utils import (
    save_embeddings_to_file,
    load_embeddings_from_file
)

# Save embeddings
save_embeddings_to_file(embeddings, labels, "data/test_embeddings.json")

# Load later
loaded_embeddings, loaded_labels = load_embeddings_from_file(
    "data/test_embeddings.json"
)
```

### Pattern 3: Adaptive Feedback Loop

```python
# Initialize detector
detector = AutoRegulatedPromptDetector(
    enable_adaptation=True,
    adaptation_rate=0.2
)

# Run detection with feedback
for embedding, true_label in zip(embeddings, labels):
    result = detector.ensemble_detection(embedding, reference_embeddings)
    prediction = result["is_anomalous"]

    # Provide feedback for false detections
    if prediction != true_label:
        is_false_positive = (prediction == True and true_label == 0)
        detector.log_false_detection(is_false_positive)

# Get final metrics
metrics = detector.get_performance_metrics()
print(f"False Positives: {metrics['false_positives']}")
print(f"False Negatives: {metrics['false_negatives']}")
```

## Advanced Scenarios

### Scenario 1: Multi-Agent Security Simulation

```python
# Simulate multiple agents with different roles
agents = {
    "analyst": SecuritySimulationAgent(),
    "malicious": SecuritySimulationAgent(),
    "unauthorized": SecuritySimulationAgent()
}

embeddings_by_role = {}

for role, agent in agents.items():
    agent.take_role(role)
    embeddings = [agent.generate_embedding() for _ in range(20)]
    embeddings_by_role[role] = embeddings

# Detect anomalies per role
detector = AutoRegulatedPromptDetector()

for role, embeddings_list in embeddings_by_role.items():
    anomaly_count = sum(
        1 for emb in embeddings_list
        if detector.ensemble_detection(emb, reference_embeddings)["is_anomalous"]
    )
    print(f"{role}: {anomaly_count} anomalies detected")
```

### Scenario 2: Custom Ensemble Strategy

```python
# Use stricter voting for high-security applications
strict_detector = AutoRegulatedPromptDetector(
    ensemble_voting_threshold=3,  # All 3 methods must agree
    enable_adaptation=False        # Static threshold
)

# Use lenient voting for research
lenient_detector = AutoRegulatedPromptDetector(
    ensemble_voting_threshold=1,  # Any 1 method triggers
    adaptation_rate=0.3           # Adapt quickly
)

# Compare results
result_strict = strict_detector.ensemble_detection(embedding, reference)
result_lenient = lenient_detector.ensemble_detection(embedding, reference)

print(f"Strict detection: {result_strict['is_anomalous']}")
print(f"Lenient detection: {result_lenient['is_anomalous']}")
```

### Scenario 3: Detector State Management

```python
# Track and manage detector state
detector = AutoRegulatedPromptDetector()

# Run multiple detection rounds
for epoch in range(10):
    detector.reset()  # Clean state between epochs

    # Process batch
    for embedding in batch:
        result = detector.ensemble_detection(embedding, reference)

    # Get epoch stats
    stats = detector.get_stats()
    print(f"Epoch {epoch}: {stats}")
```

## Error Handling

### Common Errors and Solutions

```python
# Error: Mismatched embeddings and labels
try:
    metrics = evaluate_detector(detector, embeddings, labels)
except ValueError as e:
    print(f"Validation error: {e}")
    # Fix: Ensure len(embeddings) == len(labels)

# Error: Invalid configuration
try:
    detector = AutoRegulatedPromptDetector(
        ensemble_voting_threshold=5  # Invalid, must be 1-3
    )
except ValueError as e:
    print(f"Configuration error: {e}")

# Error: Zero-norm vectors
try:
    result = detector.cosine_similarity(
        np.zeros(10), np.ones(10)
    )
except ValueError as e:
    print(f"Vector error: {e}")
    # Fix: Normalize vectors or check for zero-norm vectors
```

## Performance Considerations

### Efficiency Tips

1. **Batch Processing**: Process multiple embeddings efficiently
   ```python
   # Good: Batch evaluation
   metrics = evaluate_detector(detector, embeddings, labels)

   # Avoid: Individual calls
   for emb in embeddings:
       detector.ensemble_detection(emb, reference)
   ```

2. **Reference Set Size**: Smaller reference sets are faster
   ```python
   # 5-10 reference embeddings is typically sufficient
   reference_embeddings = embeddings[normal_indices][:10]
   ```

3. **Adaptation Rate**: Adjust based on data stability
   ```python
   # Static data: lower rate
   detector = AutoRegulatedPromptDetector(adaptation_rate=0.05)

   # Dynamic data: higher rate
   detector = AutoRegulatedPromptDetector(adaptation_rate=0.3)
   ```

### Scalability

- Current implementation handles 1000s of embeddings efficiently
- For massive datasets (>100k), consider batching or distributed processing
- Memory usage: ~10KB per embedding in JSON format

## Troubleshooting

### Issue: High False Positive Rate

**Solution**: Adjust configuration
```python
detector = AutoRegulatedPromptDetector(
    threshold=0.7,              # Higher threshold
    ensemble_voting_threshold=3, # Require more agreement
    enable_adaptation=True       # Let it adapt
)
```

### Issue: Missing Anomalies

**Solution**: Lower threshold and relax voting
```python
detector = AutoRegulatedPromptDetector(
    threshold=0.3,              # Lower threshold
    ensemble_voting_threshold=1, # Single method trigger
    adaptation_rate=0.2         # Faster adaptation
)
```

### Issue: Inconsistent Results

**Solution**: Fix random seed for reproducibility
```python
np.random.seed(42)
embeddings, labels = generate_random_embeddings(count=100)
# Results will be consistent across runs
```

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# End-to-end tests
python -m pytest tests/e2e/

# All tests with coverage
python -m pytest tests/ --cov=wildcore --cov-report=html
```

## Further Reading

- [Architecture Documentation](docs/architecture.md)
- [Agent Module Documentation](docs/agent.md)
- [Detector Module Documentation](docs/detector.md)
- [Utils Module Documentation](docs/utils.md)
