# Implementation defects and deferred recomputation

The October 2026 review separates saved-data corrections from implementation fixes requiring new experiment outcomes. All paper measurements describe the historical execution. Main retains the model and attack implementation used for that execution, including the defects below. Its aggregate reductions, manuscript tables, figures, and statistical summaries have been checked and recalculated locally. The separate `codex/methodology-recompute` branch contains future-execution fixes with synthetic regression tests. No replacement training or attack sweep has been performed.

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

## Requirements before replacement results

A future rerun must use an isolated worktree and separate catalog output paths. Git branches do not isolate ignored checkpoints, CKA matrices, attack partitions, or audit files. Reusing `--only-missing-outputs` against the historical paths can skip corrected training or combine old and new outputs.

New-run provenance must record immutable model and dataset revisions, checkpoint fingerprints, the execution Git ref, and actual hardware. The original manifest describes the historical sweep and cannot identify a new execution. These requirements have not been satisfied retroactively by the code fixes.

The padding fix requires a newly trained BiLSTM, fresh original and training-probe representations, new attacks for its five recipe cells, and target auditing of those new successes. The head-seeding fix requires retraining the four local transformer surrogates and regenerating their similarity, attack, and target-audit stages. All affected thresholds, memberships, statistics, and public figures then need recalculation. Existing pretrained-surrogate outcomes may be retained only if their exact model identity, input ordering, recipe configuration, and shard seeds are unchanged and recorded. The paper must identify any mixed reuse explicitly.

No corrected experimental rate, improved validity claim, or reproducibility claim follows from the synthetic tests alone. The branch is prepared code; its effect on this study remains unmeasured. No cloud resource was provisioned for this review.

## Methodological limits that remain

The original three-versus-three group enumeration has only 20 allocations, giving a minimum p of 0.05. It cannot satisfy the project's strict p < 0.05 decision rule. The paper retains that rule and reports the contrast as exploratory; changing the threshold after observing the result would not validate selection.

The mixed injection/jailbreak label, known shared training source, success-conditioned audit, altered semantic constraints, missing immutable Hub revisions, and absent cross-stage target fingerprint remain limitations. Padding and seeding fixes do not resolve them. Semantic validity requires a defined adjudication protocol or a separately justified constraint evaluation; it cannot be inferred from a classifier prediction change. Independent targets and a redesigned selection evaluation require new evidence.
