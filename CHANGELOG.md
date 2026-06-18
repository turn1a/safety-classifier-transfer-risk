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
- Results blog post (second in the series) reporting the headline measurement against `protectai/deberta-v3-base-prompt-injection-v2`: CKA similarity predicts attack transfer (Spearman ρ = 0.72 per attack, 0.79 per surrogate), the same-backbone anchor tops both CKA and transfer, and the high-similarity surrogate half transfers ~3× the low-similarity half. Run scope (38 of 50 cells, all four perturbation families) and the inconclusive packaged ablation are reported honestly.

### Changed

- The `attacks` pipeline now runs TextAttack in-process in the main environment (transformers 5, Python 3.13) instead of shelling out to a pinned Python 3.11 subenv. It uses a minimal fork, [turn1a/TextAttack](https://github.com/turn1a/TextAttack) (branch `transformers-5-compat`), that makes the three top-level `flair` imports lazy and adds a torch-native sentence-transformers encoder in place of the TensorFlow Universal Sentence Encoder (`semantic_encoder: use` restores the original, behind the `textattack[tensorflow]` extra). All five recipes run — DeepWordBug, PWWS, TextFooler, BAE, BERT-Attack — none dropped for a missing dependency.
- The attack sweep parallelises each `(surrogate, recipe)` pair across CPU worker processes (`num_workers`, single-threaded each). The per-example greedy search is memory-bandwidth-bound on Apple Silicon: CPU gives ~2-3× over single-process and saturates near the perf-core count, and MPS was measured ~30% slower (the search issues many small forwards, where GPU kernel-launch overhead beats the batch gain) — so the sweep runs on CPU. Model loading is forced offline (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`) to drop the per-load HuggingFace round-trips, and the goal-function query batch is enlarged to amortise the model-weight stream.
- Attack tractability knobs: prompts are truncated to `max_prompt_chars` before the search (its cost scales with word count and the prompts are long-tailed), and `eval_set_size` / `query_budget` are tuned so the sweep completes in hours rather than days. These bound the search cost without changing any per-example prediction; raising them recovers full precision at higher cost. The runner cleans up worker processes on SIGINT/SIGTERM and writes results incrementally, so a sweep is resumable.
- `HF_HUB_DOWNLOAD_TIMEOUT` is set in `.env` so a stalled HuggingFace model download errors and retries instead of hanging the run indefinitely.

### Fixed

- Surrogate fine-tuning now loads backbones in fp32 explicitly (`dtype=torch.float32`). `microsoft/deberta-v3-base` ships a float16 checkpoint, and transformers>=5 loads in the checkpoint dtype by default, so training it in raw fp16 (no loss scaling) diverged to NaN within an epoch and silently produced a degenerate `deberta-base-ft-seed` surrogate (validation accuracy at the class baseline, all weights NaN). Other backbones ship fp32 and were unaffected, which is why only the same-backbone-as-target anchor was hit. The three inference load sites (CKA representations, the attack/transfer wrapper, and the persisted-checkpoint dataset) also force fp32, both for numerical consistency across the pool and because fp16 ops are slow or unimplemented on the CPU the attack sweep runs on.
- Fine-tuning catches a non-finite loss at the step it appears and retries that model once on CPU (a safety net for any future divergence), recording the device used in the manifest. The risk regression drops any surrogate whose similarity features are non-finite before fitting, since scikit-learn rejects NaN inputs.
- Node modules import their annotation types (`pandas`, `matplotlib.figure.Figure`, `typing.Any`) at module scope instead of under `if TYPE_CHECKING:`. Kedro 1.4's parameter-type validation calls `typing.get_type_hints` on every node function at startup, which evaluates the annotations against module globals; with the imports deferred the names did not resolve and Kedro logged a `kedro.validation.type_extractor` warning per node. Ruff's `TC001/2/3` are ignored for `pipelines/**/nodes.py` to keep these imports at runtime.

### Notes

- All seven domain pipelines and the `transfer_risk.lib` core are implemented; the coverage gate is 90% on `lib`. The project still **measures and compares** transferability risk; it never **certifies** robustness (Vassilev 2025).
- The heavy ML stack (torch, transformers 5, datasets, scikit-learn, TextAttack via the turn1a fork, sentence-transformers) runs in one main environment on Python 3.13 with MPS — no per-stage subenvs.
