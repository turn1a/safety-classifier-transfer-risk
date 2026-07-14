# AGENTS.md: safety-classifier-transfer-risk

Contributor and agent guide. `CLAUDE.md` is a symlink to this file. [SPEC.md](SPEC.md) preserves the original implementation protocol and its audited updates. These rules apply unless a user explicitly narrows the work.

## Purpose

Measure transfer risk for text safety classifiers, initially a prompt-injection detector. The project computes CKA between a frozen target and surrogate pool, attacks surrogates with TextAttack, and audits the target on originals and perturbations.

The primary audited outcome is a verified target injection-to-benign outcome: the source attack succeeds, the target predicts injection on the original, and the target predicts benign after perturbation. The historical conditional target-benign rate is P[target benign on perturbed prompt | source success]. It is a separate estimand and must never be described as a verified target prediction change.

This project measures bounded transfer behavior. It does not certify robustness.

## Scope

v1 covers a public archived prompt-injection target, ten surrogates, CKA and corrected DBS, five TextAttack recipes, target auditing, a source-excluded sensitivity, and a training-only CKA sensitivity.

Do not add:

- GCG or nanoGCG suffix attacks.
- LLM-judge targets.
- Multi-turn attacks.
- Agentic wrappers.
- New task categories without explicit user direction.

Do not build CSAM, election content, bias or fairness classifiers, profanity classifiers, or image modalities.

## Project layout

```text
src/transfer_risk/
├── lib/                         # pure CKA, DBS, seeds, audit and statistics helpers
├── pipelines/
│   ├── data models similarity attacks transfer risk reporting
│   ├── audit                    # target inference plus audit finalization
│   ├── audit_reporting          # target-free public audit report
│   ├── similarity_audit_prepare # saved training split boundary
│   └── similarity_audit         # serial training-only CKA sensitivity
├── pipeline_registry.py
└── settings.py
conf/base/                       # catalog, parameters, MLflow
docs/artifacts/                  # catalog-generated aggregate report artifacts
tests/                           # pure and pipeline tests
infra/                           # Terraform cloud sweep documentation
```

`__default__` contains the original measurement chain. `audit`, `audit_finalization`, `audit_reporting`, `similarity_audit_prepare`, and `similarity_audit` are explicit stages outside it.

## Pure core and Kedro nodes

- Pure modules in `transfer_risk.lib` contain deterministic algorithms. They do not perform I/O, network access, model loading, or Kedro orchestration.
- Nodes read catalog inputs, invoke pure helpers where appropriate, run heavy inference or attacks, and return catalog outputs.
- Pipeline assembly wires datasets and parameters. Do not place business logic in `pipeline.py`.
- Corrected DBS uses the Bresenham-centered diagonal-box implementation and is recomputed from saved CKA matrices before report export.

## Audit and public artifact policy

- `audit` performs frozen-target inference on deduplicated successful-source originals and perturbations. It emits raw aggregate cells, source aggregates, and non-text context.
- `audit_finalization` recomputes corrected DBS and final summaries from saved aggregates without target inference.
- `audit_reporting` must remain target-free. It publishes aggregate-only cells, source rows, summaries, and figures.
- `similarity_audit_prepare` materializes the saved training split. `similarity_audit` runs serially on MPS, recomputes CKA over a training-only probe, and creates no attacks or training.
- Every public artifact must be catalog-owned and wrapped in `MlflowArtifactDataset`. Do not publish raw prompts, perturbations, model checkpoints, or private CKA matrices.
- `qualitative_examples.json` is a redacted audit artifact. It may contain only surrogate, recipe, edit count, categorical label, change summary, and audit note. Do not quote, paraphrase, or recreate underlying prompt text in public documentation.
- Public docs must label original conditional target-benign artifacts as historical or secondary when true target flip artifacts are present.

## Statistical writing

- Treat the ten surrogate summaries as designed-pool units of analysis. Do not present them as random samples.
- State that enumerated p-values condition on an exchangeability null for the designed pool. Do not present them as population or general-predictor validation.
- The M1 versus M2 decision rule is a project rule: at least five percentage points and strict p < 0.05.
- A p value of 0.05 in a three-versus-three enumeration equals the 1-in-20 floor and does not satisfy the strict decision rule.
- Tree regressors remain exploratory machinery. Do not report feature importance as research evidence.
- Preserve source and contamination limits. `jackhhao/jailbreak-classification` is both a canonical source and a target-card training source. Record-level decontamination is unavailable.

## Development environment

- Use Python 3.13 and `uv` exclusively. Do not use `pip` or Poetry.
- Run Python tooling through `uv run`.
- Use `just install`, `just check`, `just docs`, and the documented cloud or audit recipes.
- `.env` owns environment variables. Never place credentials in source or command history.
- Derive all randomness from the root seed through `transfer_risk.lib.seeds`.

## Kedro and catalog conventions

- Adding a surrogate is configuration plus matching catalog entries, never a pipeline-code edit.
- Tunable values belong in `conf/base/parameters_<stage>.yml`.
- Every data, model, S3 boundary, report, and public artifact crosses through the catalog.
- Cloud paths resolve through catalog configuration. Do not add `aws s3 sync`, `from_pretrained`, `load_dataset`, or `save_pretrained` calls inside nodes.
- Dynamic models, similarity, and attacks fan out by configured surrogate and attack shard. Resume through `--only-missing-outputs`.

## Quality and tests

- Write Google-style docstrings on public and private functions, classes, and modules.
- `just lint` and `just type` must pass for source changes.
- Tests assert behavioral invariants. CKA tests cover self-similarity, rotation, and scaling. DBS tests cover diagonal and full-box behavior. Audit tests cover aggregate-only output and target-free reporting.
- The coverage gate applies to `transfer_risk.lib`; heavy model and attack nodes use focused pipeline tests.

## Reproducibility and provenance

- `uv.lock` is authoritative.
- Kedro-MLflow records parameters, metrics, artifacts, and source provenance.
- The manifest's execution ref identifies cloud-run provenance only. Current source links can describe audited post-run reporting code that postdates the cloud run.
- The completed production sweep used an ARM64 `r8g.48xlarge` Graviton4 spot instance with 192 vCPUs. Final reductions, target auditing, and reporting ran locally.

## Documentation

- Keep `README.md`, `CHANGELOG.md`, `SPEC.md`, this file, and relevant docs pages current with behavior changes.
- Use one paragraph per line in Markdown.
- Keep technical claims concrete. State the outcome, denominator, artifact, and limitation before interpretation.
- Use straight quotes where practical. Do not use rhetorical em or en dashes.
- Do not claim target prediction changes from the historical conditional target-benign metric. Use verified target injection-to-benign terminology only when an audit baseline is present.
- Keep measurement and certification language to one sentence per research document.
