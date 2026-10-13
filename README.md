# safety-classifier-transfer-risk

`safety-classifier-transfer-risk` measures how text attacks generated against surrogate classifiers transfer to a public prompt-injection detector. It provides a Kedro pipeline, CKA and DBS calculations, five adapted TextAttack recipes, a target-baseline audit, saved-aggregate reporting, and a Quarto paper.

## Audited results

The audit found 1,054 verified injection-to-benign prediction changes among 4,009 successful source-attack records with target injection baselines, a pooled conditional rate of 26.3%. Across ten surrogate macro summaries, mean CKA has rho = 0.830 and exact two-sided p = 0.0047 under an exchangeability null for the designed pool. M1 minus M2 is 33.1 percentage points, with p = 0.05. The three-versus-three design cannot meet the project's strict p < 0.05 rule.

The historical BiLSTM has batch-dependent padding behavior. A saved-outcome sensitivity excluding it gives rho = 0.800 and p = 0.014 among nine transformers. The paper also documents semantic-constraint changes, mixed task labels, shared training sources, success-conditioned cohorts, fresh-head seeding, and model provenance limits. These results are exploratory classifier prediction measurements.

## Paper and results

- [Overview](https://turn1a.github.io/safety-classifier-transfer-risk/)
- [Paper](https://turn1a.github.io/safety-classifier-transfer-risk/paper.html)
- [Implementation](https://turn1a.github.io/safety-classifier-transfer-risk/pipeline.html)
- [Target audit summary](docs/artifacts/target_audit_summary.json)
- [Target audit cells](docs/artifacts/target_audit_cells.csv)
- [Source aggregates](docs/artifacts/target_audit_sources.csv)
- [BiLSTM-excluded sensitivity](docs/artifacts/surrogate_excluded_sensitivity.json)
- [Training-only CKA sensitivity](docs/artifacts/training_probe_similarity_sensitivity.json)
- [Historical conditional target-benign results](docs/artifacts/master_results_table.csv)
- [Public run metrics](docs/artifacts/run_metrics.json)
- [Redacted qualitative notes](docs/artifacts/qualitative_examples.json)
- [Results manifest](docs/artifacts/results_manifest.json)

## Local checks and rendering

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), [just](https://just.systems), and the Quarto CLI.

```bash
just install
just check
just docs
```

These commands install the locked dependencies and commit hooks, test the code, and render saved paper artifacts. They do not execute the experiment. To regenerate manuscript tables, figures, the synthetic DBS explanation, and the pool sensitivity from public aggregates and configuration:

```bash
just paper-report
just docs
```

Every commit runs both register passes on changed Markdown, Quarto source, text files, and SVG labels. A commit-message hook checks every commit message. The language checker and rules require a local `~/dev/vl-branding` checkout. SVG labels are checked as text, with independent visual styling. The independent read-through uses the Codex CLI with the model pinned in `.register-model`, currently `gpt-6.1-sol`. It runs in a temporary directory with a read-only sandbox and requires a complete structured verdict. A failed or incomplete read blocks approval; the script has no alternate-model retry. Approvals are cached against the subject bytes, model, rules, and reader implementation.

```bash
just register --all
```

Public artifacts retain full numerical precision. Paper tables round display values without rewriting the source JSON or CSV. See [SPEC.md](SPEC.md) for the historical protocol and audited deviations.

## Optional stages using saved private artifacts

The review recomputed the audit, corrected DBS, and training-probe reductions from saved inputs using the following commands. They perform no model inference or attacks:

```bash
just refresh-target-audit
just audit-report
just refresh-training-summary
just paper-report
```

The following separate stages require model inference or additional private inputs:

```bash
just audit-target
just audit-report
just prepare-training-cka
just cka-train-sensitivity
```

`audit-target` runs frozen-target inference over saved attack texts. `audit-report` publishes existing aggregates. `prepare-training-cka` materializes the saved training split; `cka-train-sensitivity` recomputes training-only CKA serially on MPS. These stages require their saved inputs. Full experimental reproduction is expensive and requires model access, including gated Hugging Face models.

## Implementation and provenance

The [pure core](src/transfer_risk/lib/) contains deterministic algorithms. [Catalog configuration](conf/base/catalog.yml) owns data, model sources, checkpoints, attack partitions, and report outputs. The dynamic attack graph expands by surrogate, recipe, and example shard, with resumption through `--only-missing-outputs`.

The saved run uses root seed `20260616` and the environment in `uv.lock`. Its implementation has two known defects: BiLSTM predictions and representations depend on batch padding, and fresh transformer-head initialization preceded the training seed. Main retains that historical model and attack code, with corrected saved-data reporting. The paper's measurements have not been regenerated with fixed training code.

The separate `codex/methodology-recompute` branch corrects padding for newly trained BiLSTMs, seeds fresh transformer heads, and makes torch attack token limits explicit. A rerun could change CKA ordering, group membership, source successes, target flip counts, and correlations; the direction is unknown. The token-limit change has no demonstrated effect on this run because all saved limits were already 512. [Recomputation notes](docs/recomputation.md) distinguish the fixes from unresolved task, semantic-validity, and provenance limits. Use an isolated worktree and separate output paths for any future rerun; switching branches does not isolate saved model and attack files.

The study measures bounded classifier transfer behavior and does not certify robustness. [infra/README.md](infra/README.md) documents the completed cloud sweep and its execution controls.

## License

MIT. See [LICENSE](LICENSE).
