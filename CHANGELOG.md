# Changelog

All notable changes to this project will be documented in this file. This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Canonical repository layout with infrastructure, configuration, assets, and documentation scaffolding.
- Unified developer toolbelt under `scripts/` with matching Make targets and CI integration.
- Project metadata (`project.yaml`), MkDocs configuration, and initial architectural decision record.
- Docker build assets and example `docker-compose.yml` for local orchestration.
- Pre-commit configuration enforcing Ruff, Black, isort, and mypy.

### Changed
- Updated GitHub Actions workflow to run `make check` across Python 3.10 and 3.11.
- Tightened dependency management with pinned development toolchain versions and reproducible lock files.
- Refined README onboarding flow and documented developer tasks, tooling, and structure expectations.
- Migrated unit tests into a deterministic `tests/unit` package and seeded randomness for reproducibility.

### Removed
- Legacy flake8 configuration in favour of Ruff-based linting.

## [0.1.0] - 2025-07-03
### Added
- Core package `wildcore` with simulation agent (`SecuritySimulationAgent`) and self-regulated anomaly detector (`AutoRegulatedPromptDetector`).
- Comprehensive `README.md` with installation guide, project structure and usage example.
- Automated CI workflow (`.github/workflows/ci.yml`) covering multi-version testing and linting.
- Development tooling configuration via `pyproject.toml` (Black, Isort, Ruff, MyPy, Flake8).
- Documentation in `docs/` covering agent, detector, utilities and project index.
- Unit tests for detector in `tests/`.

### Changed
- Adopted modern `src/` layout for Python packaging.
- Added optional dependency group `full` for heavy ML packages (PyTorch, Transformers, etc.) to keep base install lightweight.

### Fixed
- Removed emojis and informal language for a professional presentation.

[0.1.0]: https://github.com/ochoaughini/WildCore/releases/tag/v0.1.0
