# safety-classifier-transfer-risk

`safety-classifier-transfer-risk` measures how text attacks generated against surrogate classifiers transfer to a public prompt-injection detector. It provides a Kedro pipeline, CKA and DBS calculations, five adapted TextAttack recipes, a target-baseline audit, saved-aggregate reporting, and a Quarto paper.

## Audited results

The audit found 1,054 verified injection-to-benign prediction changes among 4,009 successful source-attack records with target injection baselines, a pooled conditional rate of 26.3%. Across ten surrogate macro summaries, mean CKA has rho = 0.830 and exact two-sided p = 0.0047 under an exchangeability null for the designed pool. M1 minus M2 is 33.1 percentage points, with p = 0.05. The three-versus-three design cannot meet the project's strict p < 0.05 rule.

The historical BiLSTM has batch-dependent padding behavior. Excluding it gives rho = 0.800 (p = 0.014) for conditional transfer among nine transformers, but rho = 0.400 (p = 0.29) for source-success-gated flips per attempted trial. Local inference over all returned transformer candidates adds 405 changes from failed source searches, for 1,403 target changes in 8,595 trials (16.3%) and rho = 0.733 (p = 0.031) under that separate policy. The outcome definition materially affects the association; no held-out surrogate selector is validated.

Clean-input checks expose different operating points: deepset flags 60 of 104 canonically benign test inputs, while ProtectAI and Prompt Guard 22M flag none. The broad 278-pair semantic/intent audit remains available. A separate staged review covers all 1,318 exact target-changing pairs from the nine-transformer analysis. Its first stage samples 300 changed trial occurrences uniformly without replacement from 1,403, then deduplicates them to 291 display pairs; 1,027 pairs remain for the full review. Actual human judgments are pending. Local embedding and bidirectional entailment screens are diagnostics only; they establish neither validity nor downstream effectiveness.

## Paper and results

- [Overview](https://turn1a.github.io/safety-classifier-transfer-risk/)
- [Paper](https://turn1a.github.io/safety-classifier-transfer-risk/paper.html)
- [Paper PDF](https://turn1a.github.io/safety-classifier-transfer-risk/paper.pdf)
- [Study guide: models, datasets, and metrics](https://turn1a.github.io/safety-classifier-transfer-risk/guide.html)
- [Validity checks and private review instructions](https://turn1a.github.io/safety-classifier-transfer-risk/validation.html)
- [Implementation](https://turn1a.github.io/safety-classifier-transfer-risk/pipeline.html)
- [Target audit summary](docs/artifacts/target_audit_summary.json)
- [Target audit cells](docs/artifacts/target_audit_cells.csv)
- [Source aggregates](docs/artifacts/target_audit_sources.csv)
- [BiLSTM-excluded sensitivity](docs/artifacts/surrogate_excluded_sensitivity.json)
- [Outcome and equal-trial selection sensitivity](docs/artifacts/pool_objective_sensitivity.json)
- [Distinct-original policy coverage](docs/artifacts/original_policy_coverage.json)
- [Common-source checkpoint comparison](docs/artifacts/checkpoint_comparison.json)
- [Clean-input confusion counts](docs/artifacts/detector_baseline_summary.json)
- [Staged flip-review preparation](docs/artifacts/semantic_flip_review_preparation.json)
- [Computational pair screens](docs/artifacts/semantic_screen_summary.json)
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

Every commit runs both register passes on changed Markdown, Quarto source, text files, and SVG labels. A commit-message hook checks every commit message. The language checker and rules require a local `~/dev/vl-branding` checkout. SVG labels are checked as text, with independent visual styling. The repository adapter calls the installed `~/dev/vl-branding/checks/register-pass.sh`, whose qualified default is Codex `gpt-6.1-sol` at high effort through ChatGPT subscription authentication. The installed reader verifies authentication, provider, model, effort, and complete text coverage. Inherited provider/model/effort overrides are ignored by the adapter. Failed or incomplete reads block approval; provider failure cannot select another provider or API billing. Only matching installed-reader approvals can satisfy the gate.

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

The following opt-in stages use saved inputs without training or attack search:

```bash
just prepare-semantic-review
just build-review-tools
just audit-saved-candidates
just audit-detector-baselines
just summarize-original-coverage
just screen-review-sample
```

Preparation and original-coverage reduction load no models. Classifier and computational screens run serially on the local device; missing public screening weights are downloaded through the catalog. The private review tools are `data/08_reporting/private/semantic_audit/reviewer_1.html` and `reviewer_2.html`. They contain the same fixed blinded sample, separate local progress, resume imports, and separate exports. `just review-pairs reviewer_1 8765` and `just review-pairs reviewer_2 8766` serve the respective tools on loopback. See the [review guide](docs/validation.qmd) before importing independently completed exports.

## Staged review of target-changing candidates

```bash
just prepare-flip-review
just build-flip-review-tools
just review-flips reviewer_1
just review-flips reviewer_2
just import-flip-reviews /absolute/reviewer_1.csv /absolute/reviewer_2.csv
just flip-review-results
```

The fixed cohort and role-specific tools are private under `data/08_reporting/private/semantic_flip_audit/`. The verified browser route uses loopback addresses `http://127.0.0.1:8767/` and `http://127.0.0.1:8768/`. JSON exports restore progress for the matching role; CSV exports supply independent judgments for import. Explicit reuse of genuine old exports requires exact text and role matches, retains provenance, and creates a separate progress backup for each reviewer. Each reviewer may restore only their own matching backup. Existing broad-review tools and browser state remain separate.

The first-stage estimate retains its fixed 300-occurrence draw and repeated-pair multiplicities. It uses exact hypergeometric 95% intervals over the changed-trial population, divided by all 8,595 attempts. The source-success-gated outcome uses source success AND human validity on that same draw. Unresolved judgments widen intervals and can withhold point estimates. Full-review outputs report literal valid/invalid/unresolved counts and logical yield and original-union bounds. Preparation publishes `semantic_flip_review_preparation.json`; `semantic_flip_review_summary.json` remains absent until explicit finalization. Reporting accepts that absence and rejects finalized evidence with a different opaque preparation binding, population, or design.

## Implementation and provenance

The [pure core](src/transfer_risk/lib/) contains deterministic algorithms. [Catalog configuration](conf/base/catalog.yml) owns data, model sources, checkpoints, attack partitions, and report outputs. The dynamic attack graph expands by surrogate, recipe, and example shard, with resumption through `--only-missing-outputs`.

The saved run uses root seed `20260616` and the environment in `uv.lock`. Its implementation has two known defects: BiLSTM predictions and representations depend on batch padding, and fresh transformer-head initialization preceded the training seed. Main retains that historical model and attack code, with corrected saved-data reporting. The paper's measurements have not been regenerated with fixed training code.

The separate [`codex/methodology-recompute` branch](https://github.com/turn1a/safety-classifier-transfer-risk/tree/codex/methodology-recompute) corrects padding, seeds fresh transformer heads, and makes torch token limits explicit. It also implements a prospective prompt-injection protocol with grouped held-out partitions, controlled seed replicates, a separate frozen replication target, disjoint representation and calibration data, clean-input gates, immutable identities, offline attack assets, isolated output paths, selection-lock checks, and validity-aware comparison with random and source-only baselines. New catalog execution adapters are exercised with synthetic tests. Model/tokenizer payloads, transactional metadata, and completion hashes are persisted together and verified on recovery. Fresh gates validate resumed execution before inference. Classifiers receive registered lower-trust text only; diagnostics retain full token counts, token limits, truncation, and missing measurements. Criterion-level adjudication preserves unresolved evidence. Statistical inference conditions on the three realized training replicates. The optional application executor is disabled and unimplemented. The protocol still requires reviewed corpus data and realized pinned model/asset manifests before execution; no replacement experiment has run.

A rerun could change CKA ordering, group membership, source successes, target counts, and correlations; the direction is unknown. The historical token-limit change has no demonstrated effect because all saved limits were already 512. [Recomputation notes](docs/recomputation.md) distinguish measured local checks, branch corrections, and unavailable experimental evidence. Switching branches does not isolate ignored checkpoints or attack artifacts.

The study measures bounded classifier transfer behavior and does not certify robustness. [infra/README.md](infra/README.md) documents the completed cloud sweep and its execution controls.

## Prospective replacement experiment

The correction branch contains a separate [v2 protocol](docs/experiment-v2.qmd), a dedicated `experiment_v2` environment, fail-closed evidence validation, gated target preparation and source training, independent-probe CKA, complete capped attack accounting, target auditing, calibration-only selector freezing, and target-free analysis. It requires a new grouped pure prompt-injection corpus, an independently prepared target of another backbone, repeated same-task source training, fixed query budgets, and real human review. Null revisions and empty templates intentionally block the checked-in preflight. The historical default pipeline and results remain separate. No replacement experiment has run.

```bash
uv run kedro run --env experiment_v2 --pipeline experiment_v2_preflight
```

The [phase contract and current blockers](docs/experiment-v2.qmd) distinguish executable reducers, execution adapters, and unavailable evidence. Synthetic contract tests establish accounting behavior; they do not establish corrected experimental results.

## License

MIT. See [LICENSE](LICENSE).
