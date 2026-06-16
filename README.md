# safety-classifier-transfer-risk

Measure how easily a text-based AI safety classifier (here, a prompt-injection
detector) can be fooled by adversarial examples crafted on *other* models. The
pipeline ports the Cox & Bunzel (2025) transferability-risk method from the image
domain to text safety classifiers.

> Status: scaffold. The repository structure, tooling, CI, and Kedro pipeline
> graph are in place and green; the pipeline nodes and the pure algorithms in
> `transfer_risk.lib` are stubs that raise `NotImplementedError` until the next
> phases implement them. See [SPEC.md](SPEC.md) for the full method and
> [CHANGELOG.md](CHANGELOG.md) for current scope.

## What it measures, and what it does not

This is a measurement tool, not a certification tool. It quantifies and compares
how leaky a given filter is and predicts which surrogate models yield successful
transfer. It does **not** certify robustness: Vassilev (2025) proves complete
guardrails are impossible, because the set of adversarial prompts that evade any
finite checker is infinite. The right question is therefore "how leaky is this
filter, relative to others?", not "is this filter safe?".

## Method

For a target classifier `T` and a pool of surrogate classifiers trained on the
same task (prompt-injection detection):

1. Fix a deterministic probe set of benign and injection prompts.
1. Compute layer-by-layer CKA representational similarity between each surrogate
   and `T`, reduced to two scalars: mean CKA and Diagonal Box Similarity (DBS).
1. Calibrate thresholds `r1`/`r2` from the observed similarity distribution and
   split the pool into high-similarity (`M1`) and low-similarity (`M2`) sets.
1. Attack the surrogates with TextAttack recipes (TextFooler, BERT-Attack, BAE,
   PWWS, DeepWordBug).
1. Feed the adversarial examples to the frozen target and record the transfer
   success rate.
1. Fit a regression predicting transfer rate from similarity features, and run a
   CKA-guided-vs-random ablation with a bootstrap paired t-test.

The headline question: does CKA-guided surrogate selection beat random selection
on maximum transfer rate, with statistical significance?

## Framing

Three papers bracket the work (PDFs in [`refs/`](refs/)):

- Cox & Bunzel (2025), *Quantifying the Risk of Transferred Black Box Attacks*
  (arXiv:2511.05102) — the method reproduced here.
- Klause & Bunzel (2025), *The Relationship Between Network Similarity and
  Transferability* (arXiv:2501.18629) — the empirical grounding and DBS.
- Vassilev (2025), *Robust AI Security and Alignment: A Sisyphean Endeavor?*
  (arXiv:2512.10100) — why the goal is measurement, not certification.

The target is a prompt-injection detector, so the project measures the robustness
of a mitigation for **OWASP LLM01: Prompt Injection**.

## Architecture

The work is an artifact-heavy, multi-stage DAG, which is why it is built on
[Kedro](https://kedro.org): the Data Catalog manages intermediate artifacts,
kedro-viz renders the DAG, and kedro-mlflow tracks runs. The security-relevant
algorithms live in `transfer_risk.lib` as pure, deterministic, unit-tested
functions; the Kedro nodes are thin wrappers that read and write through the
catalog. This keeps the security content separable from the orchestration.

Pipelines (one per stage): `data`, `models`, `similarity`, `attacks`,
`transfer`, `risk`, `reporting`, plus a `smoke` wiring check. `kedro run` (no
argument) executes the full chain; `kedro viz` shows it.

```
src/transfer_risk/
├── lib/             # pure algorithms: cka, dbs, seeds, thresholds (unit-tested)
├── pipelines/       # one Kedro pipeline per stage (thin nodes over lib)
├── pipeline_registry.py
└── settings.py
conf/base/           # catalog.yml, parameters_*.yml, mlflow.yml
data/                # Kedro data layers (01_raw … 08_reporting; gitignored)
docs/                # Quarto site (the blog series)
tests/               # lib/ unit tests + pipeline registry test
refs/                # the three reference papers
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems). The
Quarto CLI is needed only to build the docs.

```bash
just install                      # uv sync + install pre-commit hooks
just check                        # ruff + mypy + pytest
just run --pipeline smoke         # run the wiring-check pipeline end to end
just viz                          # open the interactive DAG
just docs                         # render the Quarto site (needs `quarto`)
just mlflow-ui                    # browse tracked runs
```

`just run --pipeline smoke` writes `data/08_reporting/smoke_report.json` and logs
a run to the local MLflow store, confirming the catalog and tracking wiring. The
domain pipelines build and visualize, but their nodes raise `NotImplementedError`
until implemented.

## Reproducibility

A single root seed derives independent per-component seeds (Python, NumPy,
PyTorch, TextAttack). `uv.lock` pins the environment; runs are tracked in MLflow
(`sqlite:///mlflow.db`) with parameters and the git SHA. CI runs lint, type-check,
and tests on every push.

## Roadmap

The build order and per-phase Definition of Done are in [SPEC.md](SPEC.md) §14.
Briefly: data and models → similarity and calibration → attacks and transfer →
regression and ablation → packaging. Out of scope for v1: GCG/suffix attacks,
jailbreak/CBRNE/toxicity targets, multi-turn attacks, and any agentic wrapper.

## License

MIT — see [LICENSE](LICENSE).
