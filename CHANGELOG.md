# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Saved-pool conditional versus fixed-trial yield analyses, complete three-source subset enumeration, model-omission sensitivities, and distinct-original union coverage.

- Serial local inference over all returned transformer candidates, including failed source searches, with a 45-cell historical reference check and aggregate common-source checkpoint comparison.

- Clean-input canonical-label confusion counts for the target and nine saved transformer checkpoints.

- A fixed 278-pair blinded review sample, separate offline human-review tools, progress and CSV resume support, strict export validation, and unresolved-evidence handling.

- Pinned local embedding and bidirectional NLI screens, private per-pair scores, public truncation coverage, and a validity guide. Computational scores are hidden from reviewers and supply no validity labels.

- A study guide with model cards, dataset links, metric definitions, and public artifact descriptions. Marked terms show definitions on hover or keyboard focus and link to the guide on touch devices. Figures open in a larger view.

- A PDF export of the same manuscript using Quarto's bundled Typst compiler, with no model execution or TeX installation.

- An explicit `paper_reporting` stage that generates manuscript table includes, an outcome diagram, an annotated cell heatmap, a probe comparison, a synthetic DBS selector diagram, and a nine-transformer sensitivity from public aggregates and configuration without model inference or attacks.

- Commit hooks requiring the checker and a Codex read-through for authored prose, including Quarto `.qmd`, plus a commit-message register check. The reader model is pinned in `.register-model`; failed and incomplete reads block approval. Diagram labels are checked independently of VL visual branding.

- Reporting tests for complete probe comparisons, configurable exclusion labels, reconciled transitions, rounded table values, and changed-during-review language subjects.

- `docs/recomputation.md` documents the historical implementation defects and the separate `codex/methodology-recompute` branch containing fixes that require regeneration of experimental results.

- A post-hoc frozen-target audit that deduplicates successful-source originals and perturbations, records aggregate target baselines and perturbed outcomes, and emits `target_audit_cells.csv`, `target_audit_sources.csv`, and `target_audit_summary.json` without raw prompts.

- An explicit audit finalization path that recomputes corrected DBS from saved CKA matrices without target inference, plus a target-free audit-report pipeline for the true-flip figures.

- A separate training-only CKA sensitivity workflow. It uses a dedicated training split, a deterministic 1,600-row balanced probe, serial MPS execution, and no attacks or training.

- True target injection-to-benign figures: `fig_true_flip_scatter.png` and `fig_true_flip_ablation.png`.

- Two redacted qualitative audit entries with metadata and audit notes only. Underlying prompt text remains private.

### Changed

- The paper distinguishes conditional transfer, source-success-gated trial yield, all-returned-candidate outcomes, and distinct-original coverage. Nine-transformer CKA associations are 0.800, 0.400, and 0.733 for the first three outcomes, respectively.

- The local all-candidate pass adds 405 target changes from failed source searches to 998 source-success flips, giving 1,403 changes in 8,595 transformer trials (16.3%). Semantic validity and application effectiveness remain unmeasured.

- Documentation describes the correction branch's prospective controlled experiment and runnable catalog adapters, together with reviewed-data and pinned-identity requirements. No replacement training or attack run is claimed.

- The paper now defines evaluator access, implemented attack constraints, clipping, denominators, aggregation, procedure yield, and the original implementation and provenance limits. It has complete model and method tables, numbered figures, discussion, and consistent display rounding; source JSON and CSV precision is preserved.

- The historical BiLSTM padding dependence is disclosed. Excluding its saved cells gives rho = 0.800 and exact p = 0.014 over nine transformers. Historical outcomes are retained.

- Main preserves the defective historical BiLSTM and fresh-head initialization used for the writeup. Masked padding, seeded fresh heads, and explicit torch token-limit handling are confined to `codex/methodology-recompute`, with synthetic regression tests and no replacement experiment results.

- The unused `cka.mode` setting is removed. Representation extraction remains batched, with CKA calculated over globally centered full-probe activations.

- Corrected DBS now uses Bresenham-centered diagonal boxes throughout reporting and audit finalization.

- The prior conditional target-benign metric, 1,398 / 4,362 = 32.0%, is retained as a historical conditional estimand. It includes 344 unchanged target-benign records.

- The primary case-study outcome is 1,054 verified target injection-to-benign outcomes among 4,009 successful source-attack records with target injection baselines, a rate of 26.3%.

- For the designed ten-surrogate pool, macro true target flip rate versus mean CKA has rho = 0.830 and enumerated two-sided p = 0.0047 under the exchangeability null. M1 minus M2 is 33.1 percentage points with p = 0.05. The three-versus-three design cannot satisfy the report's strict p < 0.05 rule.

- The known-source-excluded and training-only CKA checks are documented as post-hoc within-pool sensitivities. They do not establish record-level decontamination or generalization.

- README, Quarto pages, SPEC, contributor guidance, and infrastructure notes now distinguish original conditional output, audited target outcomes, source scope, provenance, and current figures.

## Historical implementation

- Built a Kedro project with catalog-owned data and models, dynamic surrogate and attack pipelines, pure CKA, DBS, seed, association, and ablation helpers, in-process TextAttack, MLflow tracking, a Quarto site, and cloud sweep recipes.
- Implemented the original seven-stage chain: data, models, similarity, attacks, transfer, risk, and reporting. Later audit and CKA sensitivity stages are explicit additions outside the default chain.
