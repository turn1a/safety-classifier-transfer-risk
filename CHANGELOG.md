# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: a Kedro 1.4 project (`transfer_risk`) with eight modular pipelines — data, models, similarity, attacks, transfer, risk, reporting, plus a `smoke` wiring check — over a typed Data Catalog with kedro-viz layer metadata.
- Pure-core package `transfer_risk.lib`: linear/minibatch CKA, Diagonal Box Similarity, deterministic seeding, threshold calibration, plus the transfer-rate and ablation statistics — implemented and unit-tested against real invariants (90% coverage gate).
- The seven domain pipelines (data, models, similarity, attacks, transfer, risk, reporting) implemented end-to-end: HuggingFace data harmonisation, surrogate fine-tuning + the from-scratch BiLSTM, CKA/DBS similarity, the TextAttack sweep, transfer evaluation, and the risk regression + ablation.
- Raw training sources pulled through `HFDataset` catalog entries; trained surrogates persisted through a custom `SurrogateModelDataset` (the `surrogate.{name}` factory).
- MLflow tracking of flattened params, run metrics (transfer rate, similarity-vs-transfer Spearman, ablation effect/p-value, calibrated thresholds), and artifacts (figures, tables); rich console + rotating-file logging (`conf/logging.yml`); `.env`-based runtime config; an `interrogate` docstring gate (public + private); and `just setup-data` for the offline NLP assets.
- House-style tooling: uv (`uv_build`), ruff, mypy strict, pytest + coverage, two-stage pre-commit with nbstripout, and a justfile.
- Experiment tracking via kedro-mlflow on a local SQLite backend (`sqlite:///mlflow.db`; MLflow 3.x deprecates the `./mlruns` file store).
- A Quarto documentation site (the blog series), published to GitHub Pages by CI.
- GitHub Actions: lint/type/test on Ubuntu (with a gated macOS-14 job) and a docs publish workflow.

### Changed

- The `attacks` pipeline now runs TextAttack in-process in the main environment (transformers 5, Python 3.13) instead of shelling out to a pinned Python 3.11 subenv. It uses a minimal fork, [turn1a/TextAttack](https://github.com/turn1a/TextAttack) (branch `transformers-5-compat`), that makes the three top-level `flair` imports lazy and adds a torch-native sentence-transformers encoder in place of the TensorFlow Universal Sentence Encoder (`semantic_encoder: use` restores the original, behind the `textattack[tensorflow]` extra). All five recipes run — DeepWordBug, PWWS, TextFooler, BAE, BERT-Attack — none dropped for a missing dependency.
- The attack sweep parallelises across CPU cores: each `(surrogate, recipe)` pair is an independent worker process (`num_workers` param, single-threaded each), cutting the full sweep from ~6h to ~2h. The fork also gained MPS device detection (a separate upstream improvement), though MPS gives no gain for these small per-example attacks (memory-bandwidth-bound), so workers run on CPU.
- `HF_HUB_DOWNLOAD_TIMEOUT` is set in `.env` so a stalled HuggingFace model download errors and retries instead of hanging the run indefinitely.

### Notes

- All seven domain pipelines and the `transfer_risk.lib` core are implemented; the coverage gate is 90% on `lib`. The project still **measures and compares** transferability risk; it never **certifies** robustness (Vassilev 2025).
- The heavy ML stack (torch, transformers 5, datasets, scikit-learn, TextAttack via the turn1a fork, sentence-transformers) runs in one main environment on Python 3.13 with MPS — no per-stage subenvs.
