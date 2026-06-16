# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: a Kedro 1.4 project (`transfer_risk`) with eight modular
  pipelines — data, models, similarity, attacks, transfer, risk, reporting, plus
  a `smoke` wiring check — over a typed Data Catalog with kedro-viz layer
  metadata.
- Pure-core package `transfer_risk.lib` (linear/minibatch CKA, Diagonal Box
  Similarity, deterministic seeding, threshold calibration) as signature +
  docstring stubs.
- House-style tooling: uv (`uv_build`), ruff, mypy strict, pytest + coverage,
  two-stage pre-commit with nbstripout, and a justfile.
- Experiment tracking via kedro-mlflow on a local SQLite backend
  (`sqlite:///mlflow.db`; MLflow 3.x deprecates the `./mlruns` file store).
- A Quarto documentation site (the blog series), published to GitHub Pages by CI.
- GitHub Actions: lint/type/test on Ubuntu (with a gated macOS-14 job) and a docs
  publish workflow.

### Notes

- Scope is structure + tooling only: pipeline nodes and the `lib` core raise
  `NotImplementedError`. The coverage gate is 0 and ramps to 90% on
  `transfer_risk.lib` as that pure core is implemented.
- The heavy ML stack (torch, transformers, datasets, TextAttack, a CKA library)
  is deferred and added per pipeline as the next phases land. TextAttack's
  compatibility with Python 3.13 is verified when the `attacks` pipeline is built.
