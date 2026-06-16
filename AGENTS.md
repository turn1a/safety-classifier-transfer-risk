# AGENTS.md — safety-classifier-transfer-risk

Contributor and agent guide. `CLAUDE.md` is a symlink to this file. The full
method and the phase-by-phase build plan live in [SPEC.md](SPEC.md); this file
covers conventions. Rules are non-negotiable unless marked otherwise.

## 1. Purpose

Measure and compare adversarial transferability risk for text safety classifiers
(v1: prompt-injection detection). The pipeline computes CKA similarity between a
target classifier and surrogates, attacks the surrogates with TextAttack, measures
transfer to the frozen target, and fits a risk regression. It **measures and
compares** leakiness; it never **certifies** robustness (Vassilev 2025). Keep that
boundary in every doc and result.

## 2. Scope — v1 non-goals (deferred, leave extension points only)

- GCG / nanoGCG suffix attacks (v2, the LLM-judge tier).
- Jailbreak, CBRNE, toxicity target categories (v2/v3). Keep the design
  category-agnostic so adding one is config, not a rewrite.
- Multi-turn / Crescendo attacks (out of scope; document as a limitation).
- Any agentic wrapper. Build the deterministic core.

Excluded entirely (do not build, even later): CSAM, election/political-content,
bias/fairness, and profanity classifiers; any image-modality work.

## 3. Project layout

```
src/transfer_risk/
├── lib/                  # PURE algorithms (no I/O, no Kedro): cka, dbs, seeds, thresholds
├── pipelines/<stage>/    # one Kedro pipeline per stage: nodes.py + pipeline.py
│   ├── data models similarity attacks transfer risk reporting
│   └── smoke             # the one implemented (wiring-check) pipeline
├── pipeline_registry.py  # find_pipelines(); __default__ = data … reporting (smoke excluded)
└── settings.py
conf/base/                # catalog.yml, parameters_<stage>.yml, mlflow.yml
data/01_raw … 08_reporting/   # Kedro data layers (gitignored except .gitkeep)
tests/{lib,pipelines}/     # unit tests mirror lib/; registry test builds every pipeline
refs/                      # the three reference papers (git-lfs)
```

## 4. Architecture — pure lib vs Kedro nodes

- **Pure modules** (`transfer_risk.lib.*`) hold the security-relevant maths: CKA,
  DBS, deterministic seeding, threshold calibration. No I/O, no network, no
  imports of `kedro`, `mlflow`, `transformers`, or `textattack`. Unit-tested in
  isolation.
- **Nodes** (`pipelines/<stage>/nodes.py`) are thin: they read inputs from the
  catalog, call `lib` functions, and return artifacts. Heavy I/O (model loading,
  attack running) lives here, not in `lib`.
- **Pipelines** (`pipeline.py`) wire nodes to catalog datasets and `params:`. No
  business logic in pipeline assembly.

This mirrors the pure/glue split: the deterministic core is small, tested, and
import-clean; the orchestration is everything else.

## 5. Development environment

- **Python 3.13** (newest stable minus one), managed by **`uv`** exclusively.
  Never `pip`, never `poetry`.
- Bootstrap: `just install` (= `uv sync` + `uv run pre-commit install --install-hooks`).
- Recipes (all in `justfile`): `just fmt | lint | type | test | check | hooks | run | viz | viz-build | docs | mlflow-ui`. `just` with no argument runs
  `check` (lint + type + test).
- Telemetry is opted out (`.telemetry` + `DO_NOT_TRACK=1` in the justfile).

> Environment note: the venv runs a native **arm64** Python 3.13, so the MPS /
> PyTorch path works (verified: `torch.backends.mps.is_available()` is `True`). If
> you recreate the venv, force the native build —
> `uv venv --python cpython-3.13-macos-aarch64-none && uv sync` — otherwise uv may
> pick up a stray x86_64 interpreter and lose MPS.

## 6. uv-first workflow (mandated)

> Always use `uv` to create, add, modify, or run things. Do not hand-edit the
> `dependencies` / `[dependency-groups]` arrays in `pyproject.toml`; let `uv`
> populate them.

- Add a runtime dep: `uv add <pkg>`. Dev dep: `uv add --dev <pkg>`. Remove:
  `uv remove [--dev] <pkg>`. Upgrade: `uv lock --upgrade-package <pkg>` then
  `uv sync`.
- Run anything Python via `uv run …`. Never invoke `python`/`pytest`/`mypy`/
  `ruff`/`kedro` directly.
- `uv.lock` is committed and authoritative. `[tool.*]` config edits are fine.
- The heavy ML stack (torch, transformers, datasets, scikit-learn already present
  transitively, textattack, a CKA library) is added per pipeline as it lands.
  Pin TextAttack only after confirming it installs on 3.13 — fall back to a
  maintained fork if not.

## 7. Code quality (non-negotiable)

- **Google-style docstrings** on every module, class, and public function.
- `just lint` (ruff) must pass with zero diagnostics; `just type` (mypy strict on
  `src/`) must pass. Placeholder nodes carry a scoped `ARG` ignore (see
  `pyproject.toml`); remove it as each node is implemented.
- No unseeded randomness anywhere. Derive all seeds from the one root seed via
  `transfer_risk.lib.seeds`.

## 8. Kedro conventions

- Adding a target or surrogate is a single entry in `conf/base/parameters_models.yml`
  plus a catalog factory match — never a pipeline code change (SPEC.md §5).
- All tunable knobs live in `conf/base/parameters_<stage>.yml`, read as
  `params:<stage>`. No magic numbers in nodes.
- Every artifact is a catalog entry with a `kedro-viz` layer. Intermediate data
  stays under `data/NN_<layer>/` and is gitignored.
- `pipeline_registry.register_pipelines` keeps `__default__` = the full chain and
  excludes `smoke`.

## 9. Tests and coverage

- `tests/lib/` mirrors `transfer_risk.lib` (the invariants in SPEC.md §12: CKA
  self-similarity, rotation/scaling invariance; DBS box edge cases; thresholds
  `0 < r2 < r1 < 1`). `tests/pipelines/test_pipeline_registry.py` asserts every
  pipeline builds.
- Coverage gate (`pyproject.toml`) is **0** in the scaffold and ramps to **90%**
  on `transfer_risk.lib` once the pure core is implemented. The heavy torch /
  TextAttack I/O nodes stay outside the gate — gate the deterministic core, not
  the glue.
- Tests run on the pre-push hook, not pre-commit, to keep commits fast.

## 10. Reproducibility and tracking

- One root seed → `SeedSequence` → per-component seeds. Commit `uv.lock`.
- Runs are tracked by kedro-mlflow in `sqlite:///mlflow.db` (MLflow 3.x deprecates
  the `./mlruns` file store). Log params, metrics, artifacts, and the git SHA.

## 11. Git

- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`).
- Keep `CHANGELOG.md` (Keep a Changelog format) current.
- Notebooks are exploratory only; outputs are stripped by the nbstripout hook. The
  source of truth is `src/` + `conf/`, never a notebook.
