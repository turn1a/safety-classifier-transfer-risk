# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A post-hoc frozen-target audit that deduplicates successful-source originals and perturbations, records aggregate target baselines and perturbed outcomes, and emits `target_audit_cells.csv`, `target_audit_sources.csv`, and `target_audit_summary.json` without raw prompts.
- An explicit audit finalization path that recomputes corrected DBS from saved CKA matrices without target inference, plus a target-free audit-report pipeline for the true-flip figures.
- A separate training-only CKA sensitivity workflow. It uses a dedicated training split, a deterministic 1,600-row balanced probe, serial MPS execution, and no attacks or training.
- True target injection-to-benign figures: `fig_true_flip_scatter.png` and `fig_true_flip_ablation.png`.
- Two redacted qualitative audit entries with metadata and audit notes only. Underlying prompt text remains private.

### Changed

- Corrected DBS now uses Bresenham-centered diagonal boxes throughout reporting and audit finalization.
- The prior conditional target-benign metric, 1,398 / 4,362, is retained as a historical conditional estimand rather than a target prediction-change rate.
- The primary case-study outcome is now 1,054 verified target injection-to-benign outcomes among 4,009 successful-source originals with target injection baselines, a rate of 0.2629.
- For the designed ten-surrogate pool, macro true target flip rate versus mean CKA has rho = 0.8303 and enumerated two-sided p = 0.00471 under the exchangeability null. M1 minus M2 is +33.0511 percentage points with p = 0.05 from the three-versus-three enumeration floor and outside the report's strict p < 0.05 rule.
- The known-source-excluded and training-only CKA checks are documented as post-hoc within-pool sensitivities. They do not establish record-level decontamination or generalization.
- README, Quarto pages, SPEC, contributor guidance, and infrastructure notes now distinguish original conditional output, audited target outcomes, source scope, provenance, and current figures.

## Historical implementation

- Built a Kedro project with catalog-owned data and models, dynamic surrogate and attack pipelines, pure CKA, DBS, seed, association, and ablation helpers, in-process TextAttack, MLflow tracking, a Quarto site, and cloud sweep recipes.
- Implemented the original seven-stage chain: data, models, similarity, attacks, transfer, risk, and reporting. Later audit and CKA sensitivity stages are explicit additions outside the default chain.
