# Original implementation protocol and audited results

This document preserves the original implementation protocol for `safety-classifier-transfer-risk` and records the audited behavior that later replaced incomplete outcome definitions. It is an implementation history, not a preregistration. The current research narrative is in [docs/paper.qmd](docs/paper.qmd); this file describes scope, data flow, measurement definitions, and deviations from the original plan.

## Purpose and boundary

The project measures whether adversarial text perturbations generated against surrogate classifiers can produce target injection-to-benign outcomes for a public prompt-injection detector. It uses CKA to rank surrogates, attacks those surrogates with TextAttack, audits the frozen target on original and perturbed texts, and compares outcomes with the similarity ordering.

This project measures bounded transfer behavior for a named target, designed surrogate pool, data mixture, and attack suite. It does not certify robustness.

The target is [`protectai/deberta-v3-base-prompt-injection-v2`](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2), a public detector whose model card defines 0 as benign and 1 as injection-detected. The project was archived and no longer maintained as of July 2026. It is retained as a fixed historical target.

Vassilev derives theoretical limits for finite security controls in [IEEE Security & Privacy](https://doi.org/10.1109/MSEC.2026.3678214). Cox and Bunzel provide the CKA-guided selection framework, and Klause and Bunzel provide DBS context.

## Current scope

The implemented v1 covers one frozen target, ten text-classification surrogates, five TextAttack recipes, CKA and corrected DBS similarity, target outcome auditing, a source-excluded sensitivity, and a training-only CKA sensitivity.

Deferred work remains outside v1:

- GCG and nanoGCG suffix attacks.
- LLM-judge targets.
- Multi-turn attacks.
- New target categories such as jailbreak, CBRNE, and toxicity.
- Agentic wrappers and service productization.

CSAM, election content, bias or fairness classifiers, profanity classifiers, and image modalities remain excluded.

## Canonical data and labels

The canonical corpus combines [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections), [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification), and [`Lakera/gandalf_ignore_instructions`](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions). It begins with 2,968 rows. Normalized exact-deduplication removes 21 rows, leaving 2,947: 1,910 canonical positives and 1,037 benign examples.

The positive label mixes direct prompt-injection and jailbreak-style examples. The post-deduplication source counts are 1,286 jackhhao, 999 Lakera, and 662 deepset. The source counts in target-audit artifacts instead count successful source-attack records. They are not raw test-prompt counts.

The test attack set contains 191 canonical positive prompts under this mixed label. The target card lists `jackhhao/jailbreak-classification` among its training sources, and this evaluation also uses that source. Record-level decontamination against the target's full training mixture is unavailable.

## Similarity and selection

The original CKA probe uses 2,000 balanced prompts sampled from canonical data before splitting. It uses CLS pooling and a 512-token window. Linear CKA compares every target-layer and surrogate-layer pair. The resulting matrix mean is `mean_cka`.

DBS uses the union of Bresenham-centered diagonal boxes over the rectangular layer-pair CKA matrix. A box of zero gives the strict Bresenham diagonal mean. A sufficiently wide box covers the full matrix. Reporting recomputes corrected DBS from private saved CKA matrices before export.

Original thresholds are quartiles of the ten-surrogate mean-CKA pool:

- r1 = 0.4497817638.
- r2 = 0.4134581293.
- M1: `deberta-base-ft-seed`, `deepset-deberta-injection`, `deberta-small-pi-v2`.
- M2: `llama-prompt-guard-22m`, `deberta-base-pi-v1`, `bilstm-attention`.

The thresholds are calibrated and evaluated on the same designed pool. They are a descriptive selection rule, not a holdout selection procedure.

## Source attacks and the original conditional outcome

Each of 50 surrogate-recipe cells starts with the 191 canonical positive test prompts. This gives 9,550 attempted source evaluations. TextAttack skips 950 baseline source misses. Of 8,600 eligible attempts, 4,362 source attacks succeed and 4,238 fail. The eligible source attack success rate is 50.7%.

Source success means the source surrogate changes from the canonical positive label to benign after perturbation. The initial transfer output reported:

`conditional target-benign rate = target_perturbed_benign / source_successful`

This is P[target benign on perturbed prompt | source success]. The pooled historical value is 1,398 / 4,362 = 0.3204951857. It remains valid as a conditional target-benign estimand, while it is not a verified target prediction change because the initial run did not retain target baselines.

The historical conditional CSV and figure remain published as secondary outputs:

- [docs/artifacts/master_results_table.csv](docs/artifacts/master_results_table.csv)
- [docs/artifacts/surrogate_summary.csv](docs/artifacts/surrogate_summary.csv)
- [docs/figures/fig_transfer_scatter.png](docs/figures/fig_transfer_scatter.png), titled and scoped as the historical conditional target-benign metric.

## Post-hoc target outcome audit

The explicit `audit` pipeline reuses successful source-attack records and runs frozen-target inference on deduplicated originals and perturbations. It evaluates 4,374 unique texts. It writes raw aggregate cells, raw aggregate sources, and non-text context. It neither runs attacks nor loads surrogate models.

The `audit_finalization` stage rebuilds corrected DBS and final audit summaries from saved aggregates and CKA matrices without target inference. The target-free `audit_reporting` stage publishes aggregate-only tables and figures. Every public audit artifact is catalog-owned and MLflow-wrapped. Raw prompts and perturbations are excluded from the public site.

The audited primary outcome is:

`verified target injection-to-benign rate = true_target_flips / target_original_injection`

This is the verified target injection-to-benign rate conditional on source success and a target injection baseline. It is still limited by the canonical mixed label and target-training overlap concern.

The published audit artifacts are:

- [docs/artifacts/target_audit_cells.csv](docs/artifacts/target_audit_cells.csv), 50 aggregate cells.
- [docs/artifacts/target_audit_sources.csv](docs/artifacts/target_audit_sources.csv), aggregate source rows.
- [docs/artifacts/target_audit_summary.json](docs/artifacts/target_audit_summary.json), counts, associations, selection summaries, and corrected DBS values.
- [docs/artifacts/qualitative_examples.json](docs/artifacts/qualitative_examples.json), two redacted qualitative audit entries with metadata and audit notes only.
- [docs/figures/fig_true_flip_scatter.png](docs/figures/fig_true_flip_scatter.png).
- [docs/figures/fig_true_flip_ablation.png](docs/figures/fig_true_flip_ablation.png).

The audit summary also contains the source-corrected, known-source-excluded sensitivity. The public source table exposes only aggregates needed to interpret that correction.

The public qualitative audit does not contain prompt text. It includes surrogate, recipe, edit count, categorical label, change summary, and audit note. One entry records an assessed lexical substitution as materially meaning-changing while the underlying prompts remain private.

## Outcome statistics

The full audit contains 4,362 successful source attacks. The target initially predicts injection for 4,009 and benign for 353. It predicts benign after perturbation for 1,398. The verified target injection-to-benign count is 1,054 / 4,009 = 0.2629084560.

The primary association uses one macro true target flip rate per surrogate. For the designed ten-surrogate pool, mean CKA versus macro true target flip rate has rho = 0.8303030303 and two-sided p = 0.0047106481 under exact enumeration and the exchangeability null. Corrected DBS versus the same outcome has rho = 0.5878787879 and p = 0.0806057099.

M1 has macro true target flip mean 0.4733655391 and M2 has 0.1428548539. The difference is 33.0510685 percentage points. The one-sided p is 0.05, equal to the 1-in-20 floor. The project decision rule requires strict p < 0.05, so this contrast does not satisfy that rule.

The p-values condition on exchangeability of surrogate outcome ranks under the null for this designed pool. The ten surrogates are non-random and not a demonstrated population sample. The results are within-pool descriptive evidence, not general predictor validation.

Tree regressors remain exploratory code. The project does not report random-forest feature importance or held-out predictor performance as evidence.

## Source-excluded post-hoc sensitivity

The known-source-excluded sensitivity removes every `jackhhao/jailbreak-classification` successful source-attack record before cell aggregation. It retains 3,384 source successes, 3,158 target baseline injections, and 982 verified target injection-to-benign outcomes, for a rate of 0.3109563015.

The source-success breakdown for the full audit is:

- Lakera: 2,851.
- deepset: 533.
- jackhhao: 978.

Under the post-hoc exclusion, mean CKA versus macro true target flip rate has rho = 0.9030303030 and p = 0.00080742945. M1 has mean 0.5338213245 and M2 0.1638472756, a 36.9974049 percentage-point difference with p = 0.05. This result shows that removing the known source did not weaken the observed within-pool association. It does not establish absence of target-training contamination.

## Training-only CKA sensitivity

`similarity_audit_prepare` materializes a dedicated training dataframe from the saved split. It contains 2,357 rows, 1,528 positive and 829 benign, with zero normalized text overlap with held-out validation and test splits.

`similarity_audit` creates a deterministic 1,600-row balanced training-only probe, 800 per label. It uses CLS pooling, max sequence length 512, CKA batch size 64, and DBS box 1. It is an explicit serial MPS run because target representations and an fp32 surrogate coexist in memory. It runs no attacks and performs no training.

The probe uses training rows already seen by local fine-tunes. It removes direct attack-test probe overlap while it does not provide external validation.

The training-only mean CKA rank versus original mean CKA has rho = 0.9878787879 and p = 5.5114638447971785e-06. Corrected DBS rank has rho = 1.0 and p = 5.511463844797178e-07. Training thresholds are r1 = 0.4512952903 and r2 = 0.4157880497. M1 and M2 membership each have Jaccard overlap 1.0 with the original selection.

Training-only mean CKA versus the full-cohort macro true target flip rate has rho = 0.8181818182 and p = 0.00581845238. Membership is unchanged, so the M1 versus M2 contrast remains +33.0511 percentage points with p = 0.05. These rank-stability statistics are post-hoc sensitivity evidence. They do not establish generalization.

The published sensitivity artifacts are:

- [docs/artifacts/training_probe_similarity.csv](docs/artifacts/training_probe_similarity.csv).
- [docs/artifacts/training_probe_similarity_sensitivity.json](docs/artifacts/training_probe_similarity_sensitivity.json).

## Architecture and artifact policy

The project separates pure algorithms from Kedro orchestration:

- `transfer_risk.lib` contains CKA, Bresenham DBS, seed derivation, association tests, selection ablation, audit aggregation, and public reporting helpers.
- Pipeline nodes load catalog inputs, perform model and target inference, and persist outputs.
- Dynamic models, similarity, and attacks fan out by configured surrogate and attack shard.
- Audit, audit finalization, audit reporting, training-split preparation, and training-only CKA sensitivity are explicit pipelines outside `__default__`.

All datasets, models, CKA matrices, checkpoints, attack partitions, audit aggregates, figures, and public artifacts are catalog-owned. Public files are wrapped in `MlflowArtifactDataset`. The reporting stages publish aggregate rows only where raw text would be unnecessary for evidence review.

The production attack sweep ran mostly on an ARM64 AWS `r8g.48xlarge` Graviton4 spot instance with 192 vCPUs. Final reductions, target auditing, and reporting ran locally. ARM64 ONNX Runtime failed fused transformer attention during production, so transformer victims used torch. Exported ONNX graphs remain parity checks.

The [results manifest](docs/artifacts/results_manifest.json) records `5af7330` as the cloud execution repo ref. Current source links describe the audited post-run reporting implementation and can postdate that run.

## Original plan history

The original five-phase implementation plan is retained here as history:

| Original phase             | Recorded outcome                                                                                                                                                                             |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scaffold                   | Kedro project, pure core, tests, uv workflow, and catalog structure implemented.                                                                                                             |
| Data and models            | Canonical corpus, ten-surrogate pool, local fine-tunes, pretrained detectors, and BiLSTM implemented.                                                                                        |
| Similarity and calibration | Layer-wise CKA, corrected Bresenham DBS, quartile thresholds, and M1/M2 selection implemented.                                                                                               |
| Attacks and transfer       | Five in-process TextAttack recipes, sharding, resume, and original conditional target-benign output implemented.                                                                             |
| Audit and reporting        | Target audit, audit finalization, audit reporting, source-excluded sensitivity, training-only CKA sensitivity, current figures, and public aggregate artifacts added after the original run. |

The original protocol expected a same-backbone CKA near 0.9 and initially treated a target benign prediction after source success as transfer. The completed audit found the same-backbone anchor at mean CKA 0.475 and replaced the primary outcome with verified target injection-to-benign rates. The original plan's intended predictor feature set and holdout evaluation were not implemented as research evidence.

## Reproducibility

One root seed derives component seeds through `SeedSequence`. `uv.lock` pins the environment. Kedro-MLflow records parameters, metrics, and catalog artifacts. The static Quarto build reads generated figures and public aggregate artifacts. It does not execute model, attack, cloud, or training work.

## References

- Cox, D. S. and Bunzel, N. (2025). *Quantifying the Risk of Transferred Black Box Attacks.* arXiv:2511.05102.
- Klause, G. and Bunzel, N. (2025). *The Relationship Between Network Similarity and Transferability of Adversarial Attacks.* arXiv:2501.18629.
- Vassilev, A. (2026). *Robust AI Security and Alignment: A Sisyphean Endeavor?* IEEE Security & Privacy, 24(3), 52-58. https://doi.org/10.1109/MSEC.2026.3678214
- Kornblith, S. et al. (2019). *Similarity of Neural Network Representations Revisited.* ICML.
- Morris, J. et al. (2020). *TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP.* EMNLP Demonstrations.
