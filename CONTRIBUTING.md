<!-- SPDX-License-Identifier: MPL-2.0 -->
# Contributing to WildCore

We are excited about your interest in contributing! Here are some guidelines to help you get started.

## Reporting Bugs

* Use the GitHub **Issues** section to report bugs.
* Be as detailed as possible. Include your Python version, dependencies, and the steps to reproduce the bug.

## Suggesting Enhancements

* Use the **Issues** section to suggest new features. Explain the use case and why the enhancement would be useful.

## Pull Request Process

1. **Fork** the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Run `make bootstrap` once to set up the development toolchain and git hooks.
3. Implement your changes with accompanying tests, documentation, and changelog updates.
4. Verify locally with `make check` (or the individual Make targets if you need to iterate).
5. Commit using [Conventional Commits](https://www.conventionalcommits.org/) and push your branch.
6. Open a **Pull Request** with a clear summary of the change.

### Licensing of Contributions

By submitting code, documentation, or other material to WildCore you agree
that your contribution is provided under the Mozilla Public License 2.0
(MPL-2.0). Inbound=outbound terms apply: contributions are accepted under the
same license that governs the project, ensuring continued availability under
MPL-2.0 while permitting broader aggregation under compatible terms. Please
include SPDX headers (`SPDX-License-Identifier: MPL-2.0`) in new files; the
automated tooling and pre-commit hooks will help enforce this requirement.

## Code Style & Quality

* Follow PEP 8 and the automated tooling enforced by `make lint` and `make fmt`.
* Include docstrings for public functions, classes, and modules.
* Write deterministic unit tests under `tests/unit/` and mirror source structure.
* Avoid introducing new TODO/FIXME comments—capture outstanding work in GitHub issues instead.

## Development Setup

```bash
git clone https://github.com/ochoaughini/WildCore.git
cd WildCore
make bootstrap
source .venv/bin/activate
```

Run checks individually or via the aggregated command:

```bash
make check
```

## Code Review

All submissions require review. Reviewers focus on correctness, security, maintainability, and documentation. Please be responsive to feedback and keep pull requests scoped for easier review.

## Community

* Be respectful and inclusive.
* Follow our [Code of Conduct](CODE_OF_CONDUCT.md).
* Help others who have questions.

Thank you for contributing to WildCore!
