# Implementation defects and deferred recomputation

The source models and attack candidates reported in the paper come from the completed sweep. That implementation has the defects listed below. Main adds corrected reductions and local inference over saved candidates and clean inputs; it has no replacement training or attack results. The separate [`codex/methodology-recompute` branch](https://github.com/turn1a/safety-classifier-transfer-risk/tree/codex/methodology-recompute) contains model corrections and a prospective controlled redesign, with synthetic regression tests. The direction and size of their experimental effects remain unmeasured.

## Findings and possible effects

| Finding                        | Historical behavior                                                                                                                                                          | Correction on `codex/methodology-recompute`                                                                                                                 | Possible effect of a rerun                                                                                                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BiLSTM padding                 | Padded positions pass through the bidirectional recurrence, attention, and representation means. Predictions and CKA can depend on batch companions.                         | Pack valid sequence lengths, mask attention and representation pooling, record the padding policy, and load old checkpoints with an explicit legacy policy. | BiLSTM weights, validation accuracy, predictions, CKA, source eligibility, successful perturbations, and target outcomes could change. Group thresholds and pool correlations could also change. |
| Fresh transformer-head seeding | Hub loading initializes task heads before the derived training seed is set. The root seed does not fully determine their initial weights.                                    | Reinitialize fresh task-head modules after seeding, preserving the pretrained backbone.                                                                     | The four local fine-tunes, their representations, similarity ordering, selection groups, attack successes, and target outcomes could change. The direction and magnitude are unknown.            |
| Torch attack token limits      | The victim wrapper follows the saved tokenizer limit instead of the configured attack limit. All saved transformer limits resolve to 512, matching this run's configuration. | Pass the configured maximum sequence length explicitly on both attack backends.                                                                             | No historical window mismatch was demonstrated. A future run with a different configured limit could otherwise attack different input prefixes.                                                  |

The nine-transformer sensitivity excludes the defective historical BiLSTM from saved outcomes. It retains 998 verified flips among 3,410 target-positive records, or 29.3%, with CKA rho = 0.800 and exact p = 0.014. This is a post-hoc dependence check. It neither repairs the BiLSTM nor estimates what a padding-corrected sweep would produce.

## Calculations refreshed on main

The audit finalization recomputes corrected Bresenham DBS from saved CKA matrices and recalculates target-baseline aggregates, macro rates, exact associations, source exclusion, and M1/M2 comparisons. The training-probe refresh reduces saved training-only matrices and recalculates thresholds, membership, and sensitivity statistics. Public reporting then creates rounded manuscript tables, the target transition diagram, the recipe heatmap, the paired probe plot, and the surrogate-excluded result.

```bash
just refresh-target-audit
just audit-report
just refresh-training-summary
just paper-report
just docs
```

These commands consume saved inputs and run no model inference, retraining, or attacks. The public-only `paper-report` and `docs` commands can be used without the private saved matrices. Recalculated historical totals remain 4,362 successful source records, 4,009 target-positive baselines, and 1,054 verified flips, or 26.3%. Display rounding preserves source JSON and CSV precision.

## Local evidence added without new attacks

The fixed-trial analysis gives CKA rho = 0.400 (p = 0.29) for source-success-gated yield across nine transformers, compared with conditional rho = 0.800 (p = 0.014). Local reevaluation of all returned candidates adds 405 changes from failed source searches to 998 source-success flips. This distinct post-hoc policy yields 1,403 changes in 8,595 trials (16.3%), with rho = 0.733 (p = 0.031). Every retained source-success reference cell reproduces its earlier counts. Current fingerprints identify the local pass. They do not establish the missing historical checkpoint identities.

All 84 three-transformer subsets can also be compared by the number of distinct target-positive originals affected. The CKA-selected subset covers 104 of 167 originals under the source-success gate and 120 under the all-candidate policy. Other observed selections cover more. These comparisons reuse the same evaluation inputs and supply no prospective selector validation or equal-query-cost claim.

Clean-input checks show different canonical-label behavior, including deepset flagging 60 of 104 canonically benign inputs. The common-source checkpoint comparison therefore cannot be read as a robustness ranking. A broad 278-pair independent human audit and two local computational screens are prepared. A separate staged review covers 1,318 target-changing exact pairs. It draws 300 of 1,403 changed trial occurrences before deduplicating to 291 initial display pairs, followed by 1,027 remaining pairs. Human judgments, valid-attack yield, and downstream effectiveness remain unmeasured. See [validity checks](validation.qmd) for the evidence and review tool.

```bash
just audit-saved-candidates
just audit-detector-baselines
just summarize-original-coverage
just prepare-semantic-review
just build-review-tools
just screen-review-sample
```

The inference commands run serially on the local device. Preparation and coverage reduction use saved artifacts without models. No command here launches cloud resources, training, or attack search.

## Prospective controlled redesign

The correction branch prepares a prompt-injection-specific study with reviewed original labels, group-aware source/original/attack-family splits, and disjoint representation-probe, calibration, and held-out evaluation data. It uses controlled same-task surrogate backbones with seed replicates and a separately prepared frozen RoBERTa replication target. The archived ProtectAI target remains a historical reference. Known shared-source exclusions and grouped splits constrain overlap; they cannot establish absence of unknown pretraining contamination.

Selection compares CKA with random source subsets and a source-only calibration baseline. The outcome counts each eligible original once after taking the union across selected sources and recipes, and averages seed replicates within original groups. Validity-aware held-out comparisons use paired group outcomes. Clean-input recall and hard-negative false-positive gates prevent treating a detector that rejects benign inputs as useful solely because its candidate-flip rate is low. Human evidence remains required for semantic-valid outcomes.

The branch includes catalog adapters for future target preparation, source materialization/training, CKA, bounded attack execution, complete candidate target auditing, and analysis. Evidence gates require immutable model/tokenizer identities, realized weight hashes, pinned offline auxiliary attack assets, code/lock/runtime provenance, isolated run paths, and calibration/selection locks. Saved run records retain the original execution and selection identities so old evaluation artifacts cannot acquire a new lock. Synthetic tests exercise those contracts; no model training or attack execution has occurred.

Model persistence uses a transactional bundle with `payload/`, `metadata.json`, and `complete.json`. Verified completed staging bundles recover without retraining; incomplete or corrupt bundles fail. Separate catalog metadata projections can be rebuilt from the verified bundle. Fresh validation gates run on resumed execution even when persistent outputs exist. Payload validation requires actual weights and tokenizer data, every indexed shard, and a supported index for sharded weights. Auxiliary files cannot substitute for those components.

Before target inference, candidate hashes, schemas, budgets, model identities, clean predictions, and consistent labels for identical content must pass. Classifiers receive registered lower-trust UTF-8 text only; trusted context supports human ground truth and optional application checks. Original/candidate diagnostics record token counts, registered limits, truncation, and missing measurements while retaining attempted denominators. An independent adjudicator can resolve uncertain or conflicting criteria; full review completion remains distinct from logical validity. Application controls bind exact inputs and execution artifacts, but the application executor remains disabled and unimplemented. Statistical inference conditions on the three realized training replicates.

Before execution, an operator must supply reviewed corpus groups and labels, resolved upstream revisions, realized model and auxiliary-asset manifests, and a sealed run registration. The template contains missing identities explicitly and must reject them. External datasets require review and cannot establish decontamination by their names alone. A separate optional single-turn application study requires complete task/trust-boundary context and clean/original/perturbed controls; no downstream success rate is claimed.

## Requirements before replacement results

A future rerun must use an isolated worktree and separate catalog output paths. Git branches do not isolate ignored checkpoints, CKA matrices, attack partitions, or audit files. Reusing `--only-missing-outputs` against the historical paths can skip corrected training or combine old and new outputs.

New-run provenance must record immutable model and dataset revisions, checkpoint fingerprints, the execution Git ref, and actual hardware. The original manifest describes the historical sweep and cannot identify a new execution. These requirements have not been satisfied retroactively by the code fixes.

The padding fix requires a newly trained BiLSTM, fresh original and training-probe representations, new attacks for its five recipe cells, and target auditing of those new successes. The head-seeding fix requires retraining the four local transformer surrogates and regenerating their similarity, attack, and target-audit stages. All affected thresholds, memberships, statistics, and public figures then need recalculation. Existing pretrained-surrogate outcomes may be retained only if their exact model identity, input ordering, recipe configuration, and shard seeds are unchanged and recorded. The paper must identify any mixed reuse explicitly.

No corrected experimental rate, improved validity claim, or reproducibility claim follows from the synthetic tests alone. The branch is prepared code; its effect on this study remains unmeasured. No cloud resource was provisioned for this review.

## Methodological limits that remain

The original three-versus-three group enumeration has only 20 allocations, giving a minimum p of 0.05. It cannot satisfy the project's strict p < 0.05 decision rule. The paper retains that rule and reports the contrast as exploratory; changing the threshold after observing the result would not validate selection.

The mixed injection/jailbreak label, known shared training source, success-conditioned audit, altered semantic constraints, missing immutable Hub revisions, and absent cross-stage target fingerprint remain limitations. Padding and seeding fixes do not resolve them. Semantic validity requires a defined adjudication protocol or a separately justified constraint evaluation; it cannot be inferred from a classifier prediction change. Independent targets and a redesigned selection evaluation require new evidence.
