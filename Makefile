SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

.PHONY: help bootstrap dev lint fmt typecheck test e2e coverage build package release update-deps security-scan sbom gen-docs migrate clean check

help:
	@echo "WildCore developer targets"
	@echo "  make bootstrap     # set up the development environment"
	@echo "  make dev           # run the demo CLI"
	@echo "  make lint          # lint Python code"
	@echo "  make fmt           # format code"
	@echo "  make typecheck     # run mypy"
	@echo "  make test          # execute unit/integration tests"
	@echo "  make e2e           # execute end-to-end tests"
	@echo "  make coverage      # generate coverage report"
	@echo "  make build         # build distribution artefacts"
	@echo "  make package       # alias for build"
	@echo "  make release       # run release pipeline"
	@echo "  make update-deps   # refresh dependency locks"
	@echo "  make security-scan # run security scanning"
	@echo "  make sbom          # generate SBOM"
	@echo "  make gen-docs      # build documentation"
	@echo "  make migrate       # run database migrations"
	@echo "  make clean         # remove build artefacts"
	@echo "  make check         # run the full verification suite"

bootstrap:
	scripts/bootstrap

dev:
	scripts/dev

lint:
	scripts/lint

fmt:
	scripts/fmt

typecheck:
	scripts/typecheck

test:
	scripts/test

e2e:
	scripts/e2e

coverage:
	scripts/coverage

build:
	scripts/build

package:
	scripts/package

release:
	scripts/release

update-deps:
	scripts/update-deps

security-scan:
	scripts/security-scan

sbom:
	scripts/sbom

gen-docs:
	scripts/gen-docs

migrate:
	scripts/migrate

clean:
	scripts/clean

check:
	scripts/check
