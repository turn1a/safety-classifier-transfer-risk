"""Nodes for the attacks pipeline (SPEC.md §8).

TextAttack runs in an isolated subenv (``scripts/run_textattack.py``) because its
``flair`` dependency imports a symbol removed in transformers 5.x. This node
orchestrates one subprocess per (selected surrogate x recipe) and collects the
adversarial examples.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
_RUNNER = Path("scripts/run_textattack.py")


def _subenv_command(
    subenv: dict[str, Any],
    kind: str,
    source: str,
    recipe: str,
    input_path: Path,
    output_path: Path,
    query_budget: int,
    seed: int,
) -> list[str]:
    """Build the ``uv run --no-project`` command that runs one attack in the subenv."""
    command = [shutil.which("uv") or "uv", "run", "--no-project", "--python", str(subenv["python"])]
    for package in subenv["packages"]:
        command += ["--with", str(package)]
    command += [
        "python",
        str(_RUNNER),
        "--kind",
        str(kind),
        "--source",
        str(source),
        "--recipe",
        str(recipe),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--query-budget",
        str(query_budget),
        "--seed",
        str(seed),
    ]
    return command


def run_attacks(
    splits: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run each recipe against every surrogate in the isolated subenv.

    The whole pool is attacked (not just the CKA-selected M1/M2) so the risk stage's
    ablation can compare the CKA-guided subset against random subsets drawn from all of
    them; attacking only the selected set would make that comparison degenerate.
    """
    eval_size = int(params["eval_set_size"])
    query_budget = int(params["query_budget"])
    recipes = params["recipes"]
    subenv = params["subenv"]
    targets = list(manifest.keys())
    test_df = splits["test"]
    injections = test_df.loc[test_df["label"] == 1, "text"].head(eval_size).tolist()
    examples = [{"text": text, "label": 1} for text in injections]
    adversarial: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "eval.jsonl"
        input_path.write_text("\n".join(json.dumps(example) for example in examples))
        for name in targets:
            entry = manifest[name]
            for recipe in recipes:
                output_path = Path(tmp) / f"{name}__{recipe}.jsonl"
                command = _subenv_command(
                    subenv,
                    entry["kind"],
                    entry["source"],
                    recipe,
                    input_path,
                    output_path,
                    query_budget,
                    seed,
                )
                logger.info("Attacking %s with %s over %d examples", name, recipe, len(examples))
                result = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
                if result.returncode != 0:
                    logger.error(
                        "subenv attack %s/%s failed:\n%s", name, recipe, result.stderr[-2000:]
                    )
                    msg = f"attack subprocess failed for {name}/{recipe}"
                    raise RuntimeError(msg)
                records = [
                    json.loads(line)
                    for line in output_path.read_text().splitlines()
                    if line.strip()
                ]
                for record in records:
                    record["surrogate"] = name
                    record["recipe"] = recipe
                adversarial[f"{name}__{recipe}"] = records
    return adversarial
