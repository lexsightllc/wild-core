# SPDX-License-Identifier: MPL-2.0
"""Utilities for simulating security scenarios using an AI agent."""

from typing import Any

import numpy as np


class SecuritySimulationAgent:
    """Simulated agent that can generate normal or malicious embeddings."""

    def __init__(self):
        """Initialize the agent with no roles assigned."""
        self.roles = []
        self.state = "neutral"
        self._dimension = 768  # Default embedding dimension

        # Initialize instance-level support systems (not global)
        self.containment_protocols: dict[str, dict[str, float | bool]] = {}
        self.system_alignment: dict[str, str] = {"baseline": "aligned"}
        self.memory_stream: list[dict[str, Any]] = []

    def take_role(self, role_name: str) -> dict[str, Any]:
        """Assign a role to the agent.

        Parameters
        ----------
        role_name : str
            Name of the role to assign (e.g., ``"writer"``, ``"malicious"``).

        Returns
        -------
        dict
            Information about the added role and current system status.
        """
        if not isinstance(role_name, str) or not role_name.strip():
            raise ValueError("role_name must be a non-empty string")

        # Add the role to the roles list
        self.roles.append(role_name)

        # Update containment protocols based on the role
        if role_name in self.containment_protocols:
            self.containment_protocols[role_name]["active"] = True
        else:
            self.containment_protocols[role_name] = {
                "active": True,
                "breach_probability": 0.1,
            }

        # Check if this is a potentially risky role
        if role_name.lower() in ["malicious", "hacker", "attacker", "unauthorized"]:
            self.system_alignment["status"] = "potentially_compromised"
            self.state = "suspicious"

        return {
            "role_added": role_name,
            "current_roles": self.roles,
            "system_status": (
                self.system_alignment["status"]
                if "status" in self.system_alignment
                else "aligned"
            ),
        }

    def generate_embedding(
        self, text: str | None = None, role: str | None = None
    ) -> np.ndarray:
        """Generate a simulated embedding.

        Parameters
        ----------
        text : str, optional
            Text used to seed the embedding. If ``None``, a random vector is generated.
        role : str, optional
            Role context for the embedding. Defaults to the most recent role.

        Returns
        -------
        numpy.ndarray
            Normalized embedding vector.
        """
        # Choose a role if not provided
        if role is None and self.roles:
            role = self.roles[-1]

        # Validate embedding dimension
        if self._dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        # Generate a base random vector
        base_vector = np.random.rand(self._dimension)

        # If we're in a suspicious state or using a suspicious role, introduce an anomaly
        if self.state == "suspicious" or (
            role and role.lower() in ["malicious", "hacker", "attacker"]
        ):
            # Create a biased vector that will look anomalous
            malicious_bias = np.zeros(self._dimension)
            malicious_bias[:100] = 0.9  # Strong bias in the first 100 dimensions

            # Mix the base vector with the malicious bias
            combined_vector = 0.3 * base_vector + 0.7 * malicious_bias

            # Log the suspicious activity
            self.memory_stream.append(
                {
                    "timestamp": "simulated",
                    "event": "suspicious_embedding_generated",
                    "role": role,
                }
            )

            return combined_vector / np.linalg.norm(combined_vector)  # Normalize
        else:
            # Return a normal random embedding (normalized)
            return base_vector / np.linalg.norm(base_vector)

    def simulate_breach(self, probability: float = 0.1) -> dict[str, Any]:
        """Simulate a containment breach attempt.

        Parameters
        ----------
        probability : float, default 0.1
            Probability that the breach succeeds.

        Returns
        -------
        dict
            Result of the breach attempt with success flag and system status.

        Raises
        ------
        ValueError
            If probability is not in the range [0, 1].
        """
        if not (0 <= probability <= 1):
            raise ValueError("probability must be between 0 and 1")

        # Determine if the breach is successful
        breach_successful = np.random.rand() < probability

        if breach_successful:
            self.state = "compromised"
            self.system_alignment["status"] = "compromised"

            # Log the breach
            self.memory_stream.append(
                {
                    "timestamp": "simulated",
                    "event": "containment_breach",
                    "success": True,
                }
            )

            return {
                "breach_attempted": True,
                "breach_successful": True,
                "system_status": "compromised",
            }
        else:
            # Log the failed breach attempt
            self.memory_stream.append(
                {
                    "timestamp": "simulated",
                    "event": "containment_breach",
                    "success": False,
                }
            )

            return {
                "breach_attempted": True,
                "breach_successful": False,
                "system_status": self.system_alignment.get("status", "aligned"),
            }

    def get_memory_stream(self) -> list[dict[str, Any]]:
        """Retrieve the agent's memory stream.

        Returns
        -------
        list of dict
            Chronological log of all simulated events.
        """
        return self.memory_stream.copy()

    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self.roles = []
        self.state = "neutral"
        self.containment_protocols = {}
        self.system_alignment = {"baseline": "aligned"}
        self.memory_stream = []
