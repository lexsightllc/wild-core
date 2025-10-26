<!-- SPDX-License-Identifier: MPL-2.0 -->
# ADR 0001: Repository Foundation Standardisation

## Status
Accepted

## Context
The WildCore project previously used an ad-hoc repository structure which made
it difficult for new contributors to discover tooling, understand testing
expectations, and reason about automation boundaries. The project is growing and
now requires a predictable, repeatable workflow that can be reproduced across
machines and enforced in CI.

## Decision
Adopt the WildCore repository baseline which provides:

- A canonical directory layout with dedicated folders for documentation,
  application code, automated tests, infrastructure, and operational assets.
- A unified developer toolbelt implemented as portable shell scripts and Make
  targets mirroring the CI pipeline.
- Standardised code quality tooling for Python, including Ruff, Black, isort,
  mypy, pytest, and coverage with reproducible configuration pinned in
  `pyproject.toml`.
- Project metadata captured in `project.yaml`, and developer onboarding
  instructions documented in the root `README.md`.

## Consequences
- Contributors can bootstrap and verify the project using
  `make bootstrap` followed by `make check`.
- CI pipelines rely on the same Make targets, eliminating drift between local
  and remote execution.
- Future architectural changes will be captured as additional ADRs under
  `docs/adr/` for traceability.
