# safety-classifier-transfer-risk

`safety-classifier-transfer-risk` is a reproducible Kedro pipeline for measuring how surrogate-crafted text attacks relate to a frozen prompt-injection detector. It combines layer-wise CKA, five TextAttack recipes, a post-hoc target outcome audit, catalog-managed artifacts, and a public Quarto report.

## Result teaser

The target audit found 1,054 verified injection-to-benign outcomes among 4,009 successful-source originals that the target initially predicted as injection, a rate of 0.2629. Across ten surrogate-level summaries, macro true target flip rate versus mean CKA has rho = 0.8303 and enumerated two-sided p = 0.00471 under the exchangeability null for the designed pool. M1 minus M2 is +33.0511 percentage points with p = 0.05 from the 1-in-20 group-label enumeration, which does not satisfy this report's strict p < 0.05 decision rule.

## Read the work

- [Overview](https://turn1a.github.io/safety-classifier-transfer-risk/)
- [Full paper](https://turn1a.github.io/safety-classifier-transfer-risk/paper.html)
- [Implementation case study](https://turn1a.github.io/safety-classifier-transfer-risk/pipeline.html)
- [Public run metrics](docs/artifacts/run_metrics.json)
- [Public master results](docs/artifacts/master_results_table.csv)
- [Target audit summary](docs/artifacts/target_audit_summary.json)
- [Target audit cells](docs/artifacts/target_audit_cells.csv)
- [Redacted qualitative audit](docs/artifacts/qualitative_examples.json)
- [Training-only CKA sensitivity](docs/artifacts/training_probe_similarity_sensitivity.json)
- [Results manifest](docs/artifacts/results_manifest.json)

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), [just](https://just.systems), and the Quarto CLI for documentation rendering.

```bash
just install
just check
just docs
```

These commands verify the local code and render the site from generated artifacts without running models or attacks.

The following optional commands operate on saved project artifacts. They are not a full fresh reproduction:

```bash
just audit-target
just audit-report
just prepare-training-cka
just cka-train-sensitivity
```

`just audit-target` runs frozen-target inference over deduplicated saved attack texts. `just audit-report` renders safe aggregates without loading the target. `just prepare-training-cka` materializes the saved training split, and `just cka-train-sensitivity` recomputes training-only CKA serially on MPS without attacks or training. Full experimental reproduction is expensive and requires the configured data and model access, including gated Hugging Face models where applicable. See the [original implementation protocol](SPEC.md) for the staged workflow.

## Architecture and reproducibility

The [pure core](src/transfer_risk/lib/) holds CKA, DBS, seeds, association, ablation, and target-audit aggregation logic. Kedro nodes handle model execution and artifact lineage, while the [catalog](conf/base/catalog.yml) owns datasets, model sources, checkpoints, ONNX graphs, partitions, audit outputs, reports, and cloud locations.

The attack graph expands by surrogate, recipe, and example shard. `ParallelRunner` executes disjoint shards and `kedro run --only-missing-outputs` resumes persisted outputs. One root seed derives component seeds, `uv.lock` records the environment, and MLflow records run metadata. The [site artifacts](docs/artifacts/) retain historical conditional cells, audited true-flip aggregates, corrected DBS values, training-only CKA sensitivity outputs, and run metadata.

This repository measures bounded transfer behavior for a named target and attack suite. It does not certify robustness.

## Cloud sweep

See [infra/README.md](infra/README.md) for the catalog-backed cloud workflow, completed-run context, and execution controls.

## License

MIT. See [LICENSE](LICENSE).
