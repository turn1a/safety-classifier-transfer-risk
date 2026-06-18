# Project Brief: Safety-Classifier Transferability Risk Mapper

**Working name:** `transfer-risk` (package), `safety-classifier-transfer-risk` (repo). Rename freely.

**Audience:** This is an engineering spec for Claude Code to scaffold and implement. It is not a research narrative. Build in the phase order given. Each phase has a Definition of Done; do not advance until it is met.

**One-line description:** A reproducible, automated pipeline that estimates how easily a text-based AI safety classifier (e.g. a prompt-injection detector) can be fooled by adversarial examples crafted on *other* models, by (1) measuring representational similarity between a target classifier and a pool of surrogate classifiers using Centered Kernel Alignment (CKA), (2) selecting surrogates that bracket the target in similarity, (3) attacking the surrogates, (4) measuring how often those attacks transfer to the target, and (5) fitting a regression that predicts transfer rate from similarity.

______________________________________________________________________

## 1. Intellectual framing (why this exists)

Three papers bracket this work. Read them before implementing; they are in `/refs`.

1. **Vassilev (2025), "Robust AI Security and Alignment: A Sisyphean Endeavor?"** (arXiv:2512.10100). Proves that *complete* guardrails are impossible: the set of adversarial prompts that evade any finite checker is infinite, so no classifier can ever cover it. **Implication for us:** the right goal is never "certify this filter is safe" — that is provably unattainable — but "quantify and compare how leaky a given filter is." This is the motivation for building a *measurement* tool rather than a *certification* tool.

1. **Cox & Bunzel (2025), "Quantifying the Risk of Transferred Black Box Attacks"** (arXiv:2511.05102). Proposes the constructive method: since you cannot map all adversarial subspaces, select surrogate models that are both highly similar AND highly dissimilar to the target (measured by CKA), attack them, measure transfer, and fit a regression-based risk estimator. **This is the method we reproduce.** Original paper is image-domain; we port it to text safety classifiers.

1. **Klause & Bunzel (2025), "The Relationship Between Network Similarity and Transferability of Adversarial Attacks"** (arXiv:2501.18629). The empirical predecessor that justifies Cox & Bunzel's thresholds and introduces Diagonal Box Similarity (DBS). Source of the CNN-derived thresholds we must recalibrate for text.

**The gap we fill:** No public, runnable implementation of the Cox & Bunzel method exists, and none has been applied to text safety classifiers. We build the first one. The contribution is *operationalization*: working code, concrete numbers against named targets, automated and repeatable.

**What this tool does and does not claim.** It quantifies and compares adversarial transfer risk across classifiers and predicts which surrogates yield successful transfer. It does **not** certify robustness — Vassilev (2025) proves that is impossible. State this boundary in the README.

______________________________________________________________________

## 2. Goals and non-goals

### v1 goals (this build)

- Reproduce the Cox & Bunzel 5-step pipeline end to end for **prompt-injection detection** classifiers.
- Primary target model: `protectai/deberta-v3-base-prompt-injection-v2`.
- Surrogate pool spanning high-to-low CKA similarity (DeBERTa family down to architecturally distinct encoders + a deliberate non-transformer outlier).
- Classification attacks via TextAttack (TextFooler, BERT-Attack, BAE, PWWS, DeepWordBug).
- CKA + DBS similarity computation on encoder hidden states, with empirically recalibrated thresholds `r1`, `r2`.
- Regression that predicts transfer success rate from similarity features.
- Headline result: high-CKA surrogates (M1) transfer at a higher rate than low-CKA surrogates (M2), tested by a one-sided permutation test on mean and max transfer rate.
- Fully automated, seeded, tested, and reproducible on an Apple M4 Pro (48 GB unified memory).

### v1 non-goals (explicitly deferred)

- GCG / nanoGCG adversarial-suffix attacks → scoped for the **LLM-judge tier** (Llama Guard, Granite Guardian, ShieldGemma) in **v2**. Leave clean extension points but do not implement in v1.
- Jailbreak, CBRNE, and toxicity target categories → **v2/v3**. Architecture must be category-agnostic so adding them is config, not rewrite.
- Multi-turn / Crescendo-style attacks → out of scope (we attack single-turn classification). Document this as a deliberate limitation.
- Productization / service / agentic orchestration → future. Build the deterministic core; do not wrap it in an agent.

### Excluded from the project entirely (do not build, even later)

- CSAM detection (no lawful public training data; ethical/legal landmines).
- Election-interference / political-content classifiers (contested ground truth).
- Bias/fairness-violation classifiers (normative, not a gateable classification task).
- Profanity detection (lookup-table problem, no interesting decision boundary).
- Any image-modality classification (scope is text-side only).

______________________________________________________________________

## 3. Core methodology (precise)

### 3.1 The pipeline

For a target classifier `T` and a pool of surrogate classifiers `S_1..S_k`, all trained on the same task (prompt-injection detection):

1. **Probe set.** Fix a deterministic sample of `N_probe` task inputs (mix of benign + injection prompts), tokenized to a fixed max length. Same inputs through every model.
1. **Similarity.** For each `S_i`, compute the layer-by-layer CKA matrix against `T` on the probe set, harvesting per-example hidden states. Reduce each matrix to two scalars: mean CKA (over all layer pairs) and DBS (diagonal-band average).
1. **Threshold calibration.** From the observed pairwise similarity distribution, set `r1` ≈ upper quartile, `r2` ≈ lower quartile. **Do not copy the paper's CNN-derived `r1≈0.55`, `r2≈0.35`** — recalibrate on text and record the chosen values + the data they came from.
1. **Surrogate selection.** `M1` = surrogates with similarity ≥ `r1` (high); `M2` = surrogates with similarity ≤ `r2` (low). Require `|M1| ≥ 1` and `|M2| ≥ 1`; target ≥ 3 each where the pool allows.
1. **Attack + transfer.** For each surrogate × attack recipe, generate adversarial examples on a fixed eval set the surrogate originally classified correctly. Feed every adversarial example to frozen `T`; transfer success rate = fraction that flip `T`'s prediction.
1. **Regression.** Fit transfer rate ~ (mean CKA, DBS, attack recipe, surrogate param count, shared-tokenizer flag). Use `DecisionTreeRegressor` (depth 6, matching Klause & Bunzel) and `RandomForestRegressor` as comparison.
1. **Ablation.** Compare high-CKA (`M1`) vs low-CKA (`M2`) surrogate selection with a one-sided permutation test on the difference in group-mean transfer rate, run on both the per-surrogate mean-across-recipes and max-across-recipes summaries. Enumerate the label assignments exactly when the group sizes are small (an exact permutation p-value); otherwise sample.

### 3.2 Linear CKA — reference implementation

Center activation matrices `X (n×p)` and `Y (n×q)`; CKA is a normalized squared inner product, invariant to orthogonal rotation and isotropic scaling, in `[0,1]`.

```python
import torch

def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    return (torch.linalg.multi_dot([H, K, H]) * L).sum() / (n - 1) ** 2

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.T, Y @ Y.T
    num = _hsic(K, L)
    den = torch.sqrt(_hsic(K, K) * _hsic(L, L))
    return float(num / den)
```

For large `N_probe`, use the **minibatch CKA** estimator (Nguyen, Raghu, Kornblith 2020): accumulate HSIC over batches and average. Prefer the `anatome` library or `RistoAle97/centered-kernel-alignment` for the production path; keep the snippet above as the reference implementation that the unit tests check against.

**Hidden-state extraction:** use the **last-layer CLS-token** embedding by default; support **mean-pooled last hidden state** as a config switch. Compute a full layer-by-layer matrix (register forward hooks on each transformer block), not just the final layer — Ding, Denain & Steinhardt (2021) showed final-layer CKA on BERT can be insensitive, so we report both the aggregate and the diagonal-band (DBS) score.

### 3.3 Diagonal Box Similarity (DBS)

Given the L×L layer-pair CKA matrix, DBS averages only cells within a square box of half-width `b` around each diagonal point (use the Bresenham line to walk the discrete diagonal; take the union of boxes; average unique cells). `b` is configurable. Unit-test invariants: `b=0` ⇒ DBS = mean of the strict diagonal; `b=L` ⇒ DBS = full-matrix mean.

______________________________________________________________________

## 4. Technical stack

| Concern                   | Choice                       | Notes                                            |
| ------------------------- | ---------------------------- | ------------------------------------------------ |
| Language                  | Python 3.13 (native arm64)   | newest stable minus one; MPS works               |
| Dependency mgmt           | **uv**                       | commit `uv.lock`; byte-deterministic re-runs     |
| Tensor / attacks          | **PyTorch w/ MPS backend**   | primary compute path on M4 Pro                   |
| Generative inference (v2) | **MLX**                      | only for LLM-judge tier later; not needed v1     |
| Models / data             | `transformers`, `datasets`   | HuggingFace                                      |
| Classification attacks    | **TextAttack** (turn1a fork) | in-process on transformers 5; all five recipes   |
| Suffix attacks (v2)       | **nanoGCG**                  | extension point only in v1                       |
| CKA                       | our own implementation       | anatome / torch-cka unmaintained or not on PyPI  |
| Regression                | `scikit-learn`               | DecisionTreeRegressor, RandomForestRegressor     |
| Statistics                | `scipy.stats`                | correlations, paired t-tests, bootstrap          |
| Experiment tracking       | **MLflow** (local)           | W&B optional via config flag                     |
| CLI                       | `typer`                      | one command per pipeline stage                   |
| Config                    | `pydantic-settings` + YAML   | one config tree per experiment                   |
| Plots                     | `matplotlib` (+ `seaborn`)   | three headline figures                           |
| Lint / type / hooks       | `ruff`, `mypy`, `pre-commit` | enforced in CI                                   |
| Tests                     | `pytest`                     | unit + integration                               |
| CI                        | GitHub Actions on `macos-14` | Apple-Silicon runners                            |
| Artifacts                 | `git-lfs`                    | model checkpoints; DVC is overkill at this scale |

______________________________________________________________________

## 5. Models for v1 (prompt-injection task)

All surrogates must be trained/fine-tuned on the **same task** (binary: injection vs benign) so transfer is meaningful. Some are available pre-fine-tuned (use directly); others you fine-tune yourself from a pre-trained backbone using the recipe in §7.

**Target (`T`):**

- `protectai/deberta-v3-base-prompt-injection-v2` — DeBERTa-v3-base, ~86M backbone (~184M with embeddings), binary.

**High-similarity pool candidates (`M1`)** — share DeBERTa lineage / tokenizer:

- `protectai/deberta-v3-base-prompt-injection` (v1)
- `protectai/deberta-v3-small-prompt-injection-v2`
- `deepset/deberta-v3-base-injection`
- `meta-llama/Llama-Prompt-Guard-2-86M` (mDeBERTa-base)
- a self-fine-tuned `microsoft/deberta-v3-base` with a different random seed

**Low-similarity pool candidates (`M2`)** — diverge in architecture / pretraining objective:

- `meta-llama/Llama-Prompt-Guard-2-22M` (DeBERTa-xsmall, ~22M)
- self-fine-tuned `bert-base-uncased`
- self-fine-tuned `roberta-base`
- self-fine-tuned `google/electra-small-discriminator`
- self-fine-tuned `xlnet-base-cased`
- **a non-transformer outlier**: BiLSTM-with-attention (or fastText) trained from scratch on the task — the deliberate floor of the similarity range

Final `M1`/`M2` membership is decided **empirically** from computed CKA, not from this list. The lists are the pool; the thresholds do the splitting. Expect text encoders to cluster higher than the CNN values in the paper (likely 0.5–0.9 for related ones); this is why recalibration is mandatory.

**Model registry requirement:** the surrogate layer must be **model-agnostic** — adding a new target or surrogate is a single config/registry entry taking any HF `text-classification` identifier or a local checkpoint path. Newly released guard models appear constantly; adding them must be one line.

______________________________________________________________________

## 6. Datasets for v1

Training / fine-tuning surrogates (prompt injection):

- `deepset/prompt-injections` (small, clean: ~660 train / ~110 test)
- `jackhhao/jailbreak-classification` (~1,306)
- `Lakera/gandalf_ignore_instructions` (~1,000)
- `xTRam1/safe-guard-prompt-injection`
- augmentation if needed: `hackaprompt/hackaprompt-dataset`

Evaluation:

- A held-out split from the above, **plus** the Lakera **PINT** benchmark harness (`lakeraai/pint-benchmark`) for an external reference. Note the official PINT held-out portion is not public; report on what is available and say so.

Build a single canonical, deduplicated, train/val/test-split task dataset behind a loader so every surrogate trains on identical data. Watch for known leakage/duplication across these public sets — dedupe across sources before splitting.

______________________________________________________________________

## 7. Surrogate fine-tuning recipe

For backbones without a usable pre-fine-tuned checkpoint:

- Head: `AutoModelForSequenceClassification`, `num_labels=2`.
- Optimizer AdamW, lr `2e-5`, batch size 32, 3 epochs, max seq len 256 (configurable).
- Device `mps`; mixed precision off by default on MPS for stability (config flag to enable).
- Seed everything from one root seed via `numpy.random.SeedSequence`; derive per-run seeds.
- Save checkpoint + a metadata JSON (backbone, dataset hash, seed, val accuracy).
- Sanity check: two seeds of the *same* backbone+recipe should land near ~0.9+ pairwise CKA — use this to validate the CKA implementation before trusting any cross-architecture number.

For the BiLSTM/fastText outlier: train from scratch on the same task dataset; ~20 min target on M4 Pro.

______________________________________________________________________

## 8. Attacks

### v1 — classification attacks (TextAttack)

Run each as a TextAttack recipe via the `Attacker` API, in-process in the main environment:

- **TextFooler** (Jin et al. 2020) — synonym swap, importance-ranked
- **BERT-Attack** (Li et al. 2020) — MLM-based contextual substitution
- **BAE** (Garg & Ramakrishnan 2020) — BERT-based insert/replace
- **PWWS** (Ren et al. 2019) — WordNet synonyms, saliency-weighted
- **DeepWordBug** (Gao et al. 2018) — character-level perturbations

Each adversarial example here = "a prompt that was clearly an injection but is now classified benign by the surrogate." Transfer = does it also fool `T`.

**Implementation note.** TextAttack is the only library that does word/char-level adversarial attacks on text *classifiers* (garak and PyRIT red-team generative LLMs; OpenAttack and TextFlint are abandoned). It is unmaintained for transformers 5, so attacks run against a minimal fork, [turn1a/TextAttack](https://github.com/turn1a/TextAttack), that makes the `flair` imports lazy and swaps the TensorFlow Universal Sentence Encoder for a torch-native sentence-transformers constraint (`semantic_encoder: use` restores the original, behind the `textattack[tensorflow]` extra). All five recipes run in-process; none are dropped for a missing dependency. The fork's two changes are small enough to upstream as PRs.

### Attack-coverage checklist (from Vassilev's failure taxonomy)

Make sure the recipe set spans these failure modes; note coverage in the report:

- obfuscation / char-level → DeepWordBug ✓
- synonym / lexical → TextFooler, BAE, PWWS ✓
- optimization-based → GCG (**v2**, not v1) — note the gap
- contextual framing & politeness/tone → **add at least one** such attack (a templated transformation) so v1 isn't purely lexical
- compositional ambiguity, ASCII-art, RAG-injection, Crescendo multi-turn → **out of scope v1**; list explicitly as deferred

### v2 extension point (do not implement now)

`nanoGCG` against the LLM-judge tier (Llama Guard 3-1B/8B, Granite Guardian 3.2-5B, ShieldGemma-2B/9B). Define the abstract `Attack` interface in v1 so GCG slots in as another implementation. Note: GCG needs gradients through the model w.r.t. input one-hots and is memory-heavy; on 48 GB it is feasible only up to ~1.5–3B params in full precision. Flag this in the v2 notes.

______________________________________________________________________

## 9. Repository structure

```
safety-classifier-transfer-risk/
  pyproject.toml
  uv.lock
  README.md
  CLAUDE.md                      # conventions (see §13)
  Makefile                       # make probe|cka|attack|transfer|regress|ablate|all
  .pre-commit-config.yaml
  .github/workflows/ci.yml
  refs/                          # the three papers (PDFs)
  src/transfer_risk/
    __init__.py
    config.py                    # pydantic-settings; loads YAML experiment configs
    seeds.py                     # SeedSequence-based deterministic seeding
    data/
      task_dataset.py            # canonical dedup + split loader (prompt-injection)
      probe_set.py               # fixed N_probe activation probe builder
    models/
      registry.py                # name/path -> loaded HF or local model; deterministic
      hooks.py                   # forward-hook hidden-state capture (per layer)
      finetune.py                # surrogate fine-tuning recipe (§7)
      lstm_outlier.py            # BiLSTM/fastText non-transformer surrogate
    similarity/
      linear_cka.py              # reference impl (test target) + minibatch path
      dbs.py                     # Bresenham diagonal-box similarity
      matrix.py                  # build layer-by-layer CKA matrix; reduce to scalars
      thresholds.py              # empirical r1/r2 calibration from distribution
    attacks/
      base.py                    # abstract Attack interface (GCG slots in later)
      textattack_runner.py       # wrap TextAttack recipes
      recipes.py                 # recipe-name -> recipe mapping
    transfer/
      evaluate.py                # run adv examples vs target; record flips
    risk/
      regressors.py              # DecisionTree + RandomForest
      ablation.py                # CKA-guided vs random; bootstrap + paired t-test
      bootstrap.py
    reporting/
      plots.py                   # 3 headline figures
      tables.py
    cli.py                       # typer commands, one per stage
  configs/
    v1_prompt_injection.yaml
  data/                          # raw + processed (git-lfs / .gitignored as appropriate)
  models/                        # local checkpoints (git-lfs)
  results/                       # csv/parquet + plots, per run id
  notebooks/                     # exploratory only; never the source of truth
  tests/
    unit/
      test_linear_cka.py
      test_dbs.py
      test_hooks.py
      test_thresholds.py
      test_textattack_runner.py
    integration/
      test_pipeline_smoke.py
```

______________________________________________________________________

## 10. Module specifications (Definition of Done per module)

- **`similarity/linear_cka.py`** — `linear_cka(X, Y) -> float` and a minibatch variant. DoD: passes invariance tests (self=1.0; orthogonal-rotation invariant; isotropic-scaling invariant); minibatch result ≈ full result within tolerance on a fixed fixture.
- **`similarity/dbs.py`** — `dbs(matrix, box) -> float`. DoD: `box=0` equals strict-diagonal mean; `box>=dim` equals full-matrix mean; both asserted in tests.
- **`similarity/matrix.py`** — builds L×L CKA matrix from captured per-layer hidden states for two models on the probe set; returns matrix + (mean CKA, DBS). DoD: deterministic given fixed probe set + seeds.
- **`similarity/thresholds.py`** — `calibrate(similarities) -> (r1, r2)` via quartiles; persists chosen values + provenance. DoD: returns `0 < r2 < r1 < 1`; provenance recorded.
- **`models/hooks.py`** — forward hooks capturing last hidden state per transformer block + CLS/mean pooling switch. DoD: two forward passes on frozen model with same seed produce identical captures.
- **`models/registry.py`** — validate the configured surrogate pool (unique names, known kinds) and pre-check HuggingFace auth for gated models; models themselves load through their `hub__{name}` / `target_model` catalog datasets, not here. DoD: adding a surrogate is one `parameters_models.yml` entry (`{name, kind}`) plus its `hub__{name}` catalog source — no pipeline code change (the dynamic pipelines generate its nodes).
- **`models/finetune.py`** — recipe in §7; writes checkpoint + metadata JSON. DoD: produces a working binary classifier; same-backbone different-seed pair lands ≳0.9 CKA.
- **`attacks/base.py`** — abstract `Attack` with `generate(model, examples) -> adversarial_examples`. DoD: TextAttack runner implements it; signature accommodates a future GCG implementation.
- **`attacks/textattack_runner.py`** — runs a named recipe over a fixed eval set the surrogate classifies correctly; saves JSONL (original, adversarial, perturbation stats, success). DoD: smoke test over 5 examples × 1 recipe yields valid schema.
- **`transfer/evaluate.py`** — feeds adversarial examples to frozen target; computes transfer success rate per (surrogate, recipe). DoD: rate ∈ [0,1]; deterministic.
- **`risk/regressors.py`** — fit/evaluate DecisionTree (depth 6) + RandomForest on the assembled table. DoD: reports R² / accuracy on held-out surrogates with fixed seed.
- **`risk/ablation.py`** — high-CKA (M1) vs low-CKA (M2) selection, one-sided permutation test on mean and max transfer rate (exact enumeration for small groups, else Monte Carlo). DoD: emits per-group transfer means, effect sizes (pp), and exact/empirical p-values.
- **`reporting/plots.py`** — three figures: (a) CKA similarity matrix heatmap, (b) transfer rate vs CKA scatter per (surrogate, recipe), (c) regression fit + ablation comparison. DoD: figures render headless in CI.

______________________________________________________________________

## 11. Experimental protocol (the `make all` sequence)

1. **Setup** — `uv sync`; verify `torch.backends.mps.is_available()`; set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` and `PYTORCH_ENABLE_MPS_FALLBACK=1`.
1. **Data** — build canonical deduped prompt-injection dataset + splits.
1. **Models** — collect pre-fine-tuned surrogates; fine-tune the rest; train the outlier. Persist checkpoints + metadata.
1. **Probe** — build the fixed `N_probe` (start 1,000–2,000) probe set.
1. **CKA** (`make cka`) — layer-wise CKA matrices for target vs each surrogate; reduce to mean CKA + DBS; heatmap. **Validate** with the same-backbone-different-seed sanity check.
1. **Calibrate** (`make thresholds`) — set `r1`, `r2` from the observed distribution; record provenance.
1. **Select** — split surrogates into `M1` / `M2`.
1. **Attack** (`make attack`) — each surrogate × recipe on a fixed eval set (start 500 prompts); save adversarial examples. Longest stage; design for overnight runs.
1. **Transfer** (`make transfer`) — evaluate every adversarial example against `T`; assemble the master results table (surrogate, recipe, mean CKA, DBS, params, shared-tokenizer, transfer rate).
1. **Regress** (`make regress`) — fit + evaluate both regressors.
1. **Ablate** (`make ablate`) — high-CKA M1 vs low-CKA M2; one-sided permutation test on mean and max transfer rate (exact for small groups); final plots.

**Headline success criterion:** high-CKA surrogates (M1) yield ≥ 5 percentage-point higher mean transfer rate than low-CKA surrogates (M2), one-sided permutation p < 0.05 (note the small-group p-value floor: three vs three gives a minimum of 1/20 = 0.05). If not met, the most likely bug is a too-homogeneous pool (the "low-similarity" surrogates aren't actually low) — add the non-transformer outlier and a fastText model and recheck before concluding the method doesn't port.

**Tripwires:**

- All transfer rates > ~80% regardless of surrogate → pool too homogeneous; add outliers.
- All transfer rates < ~20% → attack budget too tight; raise TextFooler query budget (e.g. 800 → 2,000).
- CKA scores all bunched in 0.4–0.6 → switch the selection signal to DBS with small box size to spread them.

______________________________________________________________________

## 12. Testing, verification, reproducibility

**Unit tests (must pass in CI):**

- CKA: self-similarity = 1.0; invariance under random orthogonal right-multiply; invariance under positive isotropic scaling.
- DBS: `box=0` ⇒ diagonal mean; `box≥dim` ⇒ full mean.
- Hooks: deterministic captures across repeated forward passes.
- Thresholds: `0 < r2 < r1 < 1`.

**Integration test:** full pipeline on a 5-prompt probe set with 2 surrogates and 1 recipe; assert output schema and that transfer rate ∈ [0,1].

**Verification beyond tests:** the M1-vs-M2 selection ablation (a one-sided permutation test on mean and max transfer rate) is the primary scientific check; report the per-group transfer means and the exact p-value, noting the small-group p-value floor (three vs three gives a minimum of 1/20 = 0.05).

**Reproducibility:** single root seed → `SeedSequence` → per-component seeds (Python `random`, NumPy, PyTorch, and `TEXTATTACK_RANDOM_SEED`). Commit `uv.lock`. Track every run in MLflow (params, metrics, artifacts, git SHA). `git-lfs` for the handful of checkpoints. Log per-example CKA, perturbation magnitude, attack runtime, and query counts — the regressor can only use features you logged.

**CI:** GitHub Actions on `macos-14`; run lint (`ruff`), type-check (`mypy`), and the unit + integration tests. Keep the integration test tiny so CI stays fast.

______________________________________________________________________

## 13. CLAUDE.md (create this file in the repo root)

```markdown
# Conventions for this repo

- Python 3.13. Manage deps with `uv`; never hand-edit installed packages.
  Always `uv add` and commit `uv.lock`.
- Primary compute device is Apple MPS. Write device-agnostic code:
  `device = "mps" if torch.backends.mps.is_available() else "cpu"`.
- Never use mixed precision on MPS unless a config flag explicitly enables it.
- Seed everything from one root seed via numpy SeedSequence (see src/transfer_risk/seeds.py).
  No unseeded randomness anywhere.
- The surrogate layer is model-agnostic: adding a surrogate = one `parameters_models.yml`
  entry plus its `hub__{name}` catalog source, never a code change in the pipeline (the
  models/similarity/attacks pipelines generate its nodes dynamically from config).
- Do NOT implement GCG/nanoGCG, jailbreak/CBRNE/toxicity targets, multi-turn attacks,
  or any agentic wrapper in v1. Leave extension points only.
- This tool measures and compares risk. It never certifies robustness
  (Vassilev 2025 proves certification is impossible). Keep that boundary in docs.
- Notebooks are exploratory only; the pipeline source of truth is src/ + Makefile.
- Every pipeline stage is a `make` target and a typer CLI command.
- Recalibrate CKA thresholds empirically; never hardcode the paper's CNN values.
- Tests must pass before a phase is considered done. Run `make test`.
```

______________________________________________________________________

## 14. Implementation phases (build in this order)

**Phase 0 — Scaffold.** Repo, `pyproject.toml`, `uv`, `CLAUDE.md`, `Makefile`, pre-commit, CI skeleton. Implement `similarity/linear_cka.py` + `similarity/dbs.py` + `seeds.py` and their unit tests. *DoD:* `make test` green; CI passes on macOS runner.

**Phase 1 — Data + models.** Canonical prompt-injection dataset + splits; model registry; hooks; fine-tuning recipe; train the missing surrogates + the outlier; build the probe set. *DoD:* all surrogates load and classify; same-backbone-different-seed CKA ≳0.9 (validates CKA impl).

**Phase 2 — Similarity + calibration.** Layer-wise CKA matrices target vs surrogates; reduce to mean CKA + DBS; heatmap; empirical `r1`/`r2`; `M1`/`M2` split. *DoD:* similarity matrix + chosen thresholds + provenance persisted; heatmap renders.

**Phase 3 — Attacks + transfer.** TextAttack runner over the recipe set; transfer evaluation; master results table. *DoD:* JSONL adversarial logs + results table with all features populated.

**Phase 4 — Regression + ablation + plots.** Both regressors; CKA-guided-vs-random ablation with bootstrap + paired t-test; three headline figures. *DoD:* headline success criterion evaluated; figures produced; MLflow run captured.

**Phase 5 — Packaging + write-up.** README (with the measurement-not-certification boundary, the three-paper framing, and OWASP LLM01 mapping); reproduce-from-scratch instructions; results summary. *DoD:* a fresh clone + `make all` reproduces the headline numbers.

______________________________________________________________________

## 15. Hardware notes (Apple M4 Pro, 48 GB)

- Unified memory: ~35 GB usable after OS/runtime. All v1 encoders (\<250M params) and the full surrogate set fit simultaneously; no offloading needed.
- Use PyTorch MPS for all v1 work. If you hit a missing MPS kernel, `PYTORCH_ENABLE_MPS_FALLBACK=1` routes it to CPU (slower; fine for overnight runs).
- MLX is **not** needed for v1 (it matters for the v2 generative LLM-judge tier).
- The attack sweep dominates wall-clock (TextFooler ~90 min per surrogate at 500 prompts). Plan for an overnight `make attack`. v1 end-to-end target: ~8–20 hours depending on pool size and attack-set size; shrink to ~8 by using 200 prompts and 3 recipes for the first green run, then scale up.

______________________________________________________________________

## 16. Future extensions (record, don't build)

- **v2:** add the LLM-judge tier (Llama Guard 3, Granite Guardian 3.2, ShieldGemma) as targets; implement the `Attack` interface for nanoGCG; CKA on generative-model assistant-prompt hidden states; MLX for judge inference.
- **v3:** add jailbreak, CBRNE (fine-tune a detector on HarmBench + WMDP-bio/chem/cyber vs benign MMLU-Pro technical questions), and toxicity (`unitary/toxic-bert`, `facebook/roberta-hate-speech-dynabench-r4-target`) target categories. Add PII, hallucination (Vectara HHEM, Patronus Lynx), and refusal-judge (HarmBench-Llama-2-13b-cls, StrongREJECT) categories for cross-domain validation.
- **Productization:** wrap the deterministic core as a pre-deployment risk-scoring service; map each category to its OWASP LLM Top-10 entry; position as complementary to the Cox & Bunzel framework. Agentic surrogate selection is a later layer over the deterministic tools, not a replacement.

______________________________________________________________________

## 17. References

- Cox, D. S. & Bunzel, N. (2025). *Quantifying the Risk of Transferred Black Box Attacks.* arXiv:2511.05102 — https://arxiv.org/abs/2511.05102
- Klause, G. & Bunzel, N. (2025). *The Relationship Between Network Similarity and Transferability of Adversarial Attacks.* arXiv:2501.18629 — https://arxiv.org/abs/2501.18629
- Vassilev, A. (2025). *Robust AI Security and Alignment: A Sisyphean Endeavor?* arXiv:2512.10100 — https://arxiv.org/abs/2512.10100
- Tramèr, F. et al. (2017). *The Space of Transferable Adversarial Examples.* arXiv:1704.03453 — https://arxiv.org/abs/1704.03453
- Kornblith, S. et al. (2019). *Similarity of Neural Network Representations Revisited.* arXiv:1905.00414 — https://arxiv.org/abs/1905.00414
- Nguyen, T., Raghu, M. & Kornblith, S. (2020). *Do Wide and Deep Networks Learn the Same Things?* arXiv:2010.15327 — https://arxiv.org/abs/2010.15327
- Ding, F., Denain, J-S. & Steinhardt, J. (2021). *Grounding Representation Similarity with Statistical Testing.* NeurIPS — https://proceedings.neurips.cc/paper/2021/file/0c0bf917c7942b5a08df71f9da626f97-Paper.pdf
- Jin, D. et al. (2020). *Is BERT Really Robust? (TextFooler).* arXiv:1907.11932 — https://arxiv.org/abs/1907.11932
- Li, L. et al. (2020). *BERT-ATTACK.* arXiv:2004.09984 — https://arxiv.org/abs/2004.09984
- Garg, S. & Ramakrishnan, G. (2020). *BAE.* EMNLP — https://aclanthology.org/2020.emnlp-main.500/
- Ren, S. et al. (2019). *PWWS.* ACL — https://aclanthology.org/P19-1103/
- Gao, J. et al. (2018). *DeepWordBug.* arXiv:1801.04354 — https://arxiv.org/abs/1801.04354
- Morris, J. et al. (2020). *TextAttack.* arXiv:2005.05909 — https://arxiv.org/abs/2005.05909
- Zou, A. et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned LLMs (GCG).* arXiv:2307.15043 — https://arxiv.org/abs/2307.15043

Tooling: TextAttack https://github.com/QData/TextAttack · anatome https://github.com/moskomule/anatome · centered-kernel-alignment https://github.com/RistoAle97/centered-kernel-alignment · nanoGCG https://github.com/GraySwanAI/nanoGCG · PINT benchmark https://github.com/lakeraai/pint-benchmark
