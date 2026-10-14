# Study strengthening implementation plan

> For agentic workers: use the dispatching-parallel-agents workflow for the independent analysis, annotation preparation, and prospective protocol tasks. The parent handles model inference, integration, reporting, and publication.

Goal: strengthen the historical study with saved-data analysis and bounded local inference, and prepare a separate scientifically redesigned experiment without executing training or attack generation.

Architecture: main preserves the completed experiment and adds explicit retrospective audit stages outside the default pipeline. The correction worktree contains the existing model fixes and a prospective protocol with isolated catalog outputs, evidence validation, and held-out analysis. Public outputs contain aggregates; prompt pairs and annotations remain private.

Tech stack: Python 3.13, uv, Kedro, NumPy, pandas, PyTorch on a single local device, Matplotlib, Quarto, and Typst.

Spec: the user's two-track instruction, AGENTS.md, and docs/recomputation.md.

## Global constraints

- Provision no cloud resources and run no new training or TextAttack searches.
- Derive sampling seeds from transfer_risk.lib.seeds.
- Route all experiment inputs and outputs through the Kedro catalog.
- Publish no original or perturbed prompts, private pair identifiers, or model checkpoints.
- Preserve historical model and attack behavior on main.
- Treat new analyses as post-hoc checks of a designed model pool.
- Keep mechanical text fidelity, human judgments, and application effectiveness distinct.
- Preserve pending human-review states. Do not substitute generated labels for independent human evidence.
- Use the repository's Codex register reviewer for both required passes before each commit.

## Task 1: Compare conditional transfer with procedure yield

Files: src/transfer_risk/lib/pool_sensitivity.py, tests/lib/test_pool_sensitivity.py, and manuscript reporting integration.

Interface: build_pool_sensitivity(cells, similarity_rows, attempts_per_cell, recipes, excluded_surrogates, selection_size) returns a JSON-compatible aggregate report. Inputs are the public target-audit cell table and per-surrogate similarity summary.

- Test complete cell grids, integer count bounds, undefined conditional denominators, equal attempted budgets, exact associations, and descriptive subset enumeration.
- Recompute ten-model and nine-transformer associations for conditional macro rates and verified flips per attempted trial.
- Report every leave-one-model-out correlation and the full distribution of equal-budget three-model choices.
- Add a catalog-owned table and figure. State that actual query expenditure was not retained.

## Task 2: Prepare semantic and intent review from saved pairs

Files: src/transfer_risk/lib/semantic_audit.py, src/transfer_risk/pipelines/semantic_audit/, conf/base/parameters_semantic_audit.yml, and focused tests.

Interface: prepare_semantic_audit(adversarial_examples, task_splits, semantic_params, seed) returns deduplicated private pairs, a blinded annotation sheet, private sample linkage, and public mechanical fidelity aggregates. A separate review pipeline consumes supplied annotations.

- Test global pair deduplication, stratum membership probabilities, outcome blinding, and source attribution after historical clipping.
- Test distinct reviewer identities, explicit independence attestation, disagreement handling, and incomplete evidence.
- Prepare a deterministic stratified review sample from all saved outcomes, including source failures and skips.
- Publish annotation coverage and pending status only. A sample cannot supply precise rates for every one of the fifty cells.

## Task 3: Audit all returned candidates on local classifiers

Files: src/transfer_risk/lib/retrospective_audit.py, src/transfer_risk/pipelines/retrospective_audit/, conf/base/parameters_retrospective_audit.yml, and focused tests.

Interface: run_retrospective_target(private_pairs, private_linkage, checkpoint_source, target_params, device_params) returns private pair predictions, aggregate cells, and a non-text model fingerprint context. It performs classifier inference only and queries no attack generator.

- Test all-attempt aggregation, source-success filtering, target self-exclusion, baseline accounting, positive-label mappings, and aggregate-only public schemas.
- Load models only from catalog-owned local checkpoint boundaries and operate serially.
- Compare the original target's source-success aggregates against the published audit before interpreting added candidates.
- Evaluate saved candidates against selected existing pretrained detectors. Label this retrospective cross-target evidence, with no independent attack regeneration or general selector validation.
- Record current checkpoint fingerprints and explain that they do not reconstruct missing historical fingerprints.

## Task 4: Prepare the prospective redesigned experiment

Files: correction-worktree v2 configuration, pure protocol/evidence/analysis modules, explicit Kedro stages, regression tests, and docs/experiment-v2.qmd.

- Specify prompt injection separately from jailbreaks, human-reviewed originals, group-aware held-out partitions, and a disjoint representation probe.
- Require immutable data/model/tokenizer revisions and checkpoint fingerprints, isolated outputs, explicit binary label maps, fixed query caps, target-role exclusions, and frozen selection rules.
- Compare selection with random and source-only baselines using held-out valid attack yield, with original-level unions and family/origin groups as the uncertainty units.
- Define an optional single-turn application study with clean/original/perturbed controls and objective-specific checks; missing original context is unassessable.
- Provide execution-stage contracts that distinguish implemented code from absent experimental inputs or orchestration. Run synthetic tests only.

## Task 5: Integrate, render, review, and publish

Files: conf/base/catalog.yml, src/transfer_risk/pipeline_registry.py, justfile, docs/paper.qmd, docs/index.qmd, docs/guide.qmd, docs/recomputation.md, README.md, CHANGELOG.md, SPEC.md, and AGENTS.md.

- Register all retrospective and review-preparation stages outside __default__.
- Run no-model reductions and bounded local classifier audits through the catalog.
- Update claims, rounded tables, figures, limitations, and links using the resulting artifacts.
- Run just check, both Codex register passes, Quarto HTML/PDF rendering, and visual layout verification.
- Commit and publish main only after those checks pass, then integrate main into the correction branch and verify its final diff and checks.
- Verify the published paper and report what evidence improved and what remains pending.
