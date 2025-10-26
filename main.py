# SPDX-License-Identifier: MPL-2.0
"""Convenience entry point for running the WildCore simulation locally."""

from __future__ import annotations

import json
from pathlib import Path

from wildcore.__main__ import simulate


def main() -> None:
    """Execute a small demonstration simulation and write results to disk."""
    summary = simulate(iterations=9, threshold=0.55, reference_size=6, seed=2024)
    output_path = Path("artifacts")
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "simulation-summary.json"
    target.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
    print(f"Simulation summary written to {target}")


if __name__ == "__main__":
    main()
