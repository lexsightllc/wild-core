"""Command line entry point for the WildCore demo pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict

import numpy as np

from .agent import SecuritySimulationAgent
from .detector import AutoRegulatedPromptDetector
from .utils import generate_random_embeddings


@dataclass
class SimulationSummary:
    """Aggregate results from a simulation run."""

    iterations: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float


def simulate(iterations: int, threshold: float, reference_size: int, seed: int) -> SimulationSummary:
    """Run a deterministic simulation using the detector and agent."""
    np.random.seed(seed)
    agent = SecuritySimulationAgent()
    detector = AutoRegulatedPromptDetector(threshold=threshold)

    reference_embeddings_array, _ = generate_random_embeddings(reference_size, agent._dimension)
    reference_embeddings = [emb for emb in reference_embeddings_array]

    stats = {
        "iterations": iterations,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
    }

    for index in range(iterations):
        is_attack_round = (index + 1) % 3 == 0
        role = "malicious" if is_attack_round else "analyst"
        agent.take_role(role)
        embedding = agent.generate_embedding(role=role)

        detection = detector.ensemble_detection(embedding, reference_embeddings)

        if is_attack_round and detection["is_anomalous"]:
            stats["true_positives"] += 1
        elif is_attack_round and not detection["is_anomalous"]:
            stats["false_negatives"] += 1
        elif not is_attack_round and detection["is_anomalous"]:
            stats["false_positives"] += 1
        else:
            stats["true_negatives"] += 1

    total = sum(
        stats[key]
        for key in ("true_positives", "false_positives", "true_negatives", "false_negatives")
    )
    accuracy = (
        (stats["true_positives"] + stats["true_negatives"]) / total if total else 0.0
    )

    return SimulationSummary(accuracy=accuracy, **stats)


def main() -> None:
    """Parse CLI arguments and run a simulation."""
    parser = argparse.ArgumentParser(description="WildCore security simulation")
    parser.add_argument("--iterations", type=int, default=12, help="Number of simulation steps to run.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Initial similarity threshold.")
    parser.add_argument("--reference-size", type=int, default=6, help="Number of baseline embeddings to create.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic randomness.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write the simulation summary as JSON.",
    )

    args = parser.parse_args()

    log_level = os.getenv("WILDCORE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    summary = simulate(
        iterations=args.iterations,
        threshold=args.threshold,
        reference_size=args.reference_size,
        seed=args.seed,
    )

    logging.getLogger("wildcore").info("Simulation complete: %s", summary)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(asdict(summary), handle, indent=2)
    else:
        print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
