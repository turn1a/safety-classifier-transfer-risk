"""Serve the separate staged cohort or import explicit role-bound human exports."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro_datasets.pandas import CSVDataset

from transfer_risk.lib.semantic_flip_audit import validate_flip_cohort
from transfer_risk.lib.semantic_flip_reviews import merge_flip_exports, progress_for_role
from transfer_risk.lib.semantic_review_tool import REVIEWER_IDS, build_review_dataset
from transfer_risk.scripts.semantic_review_server import _serve

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kedro.io import CatalogProtocol


def default_port(reviewer: str) -> int:
    """Return the separate loopback origin for a fixed staged-review role."""
    if reviewer not in REVIEWER_IDS:
        msg = "reviewer must be reviewer_1 or reviewer_2"
        raise ValueError(msg)
    return {"reviewer_1": 8767, "reviewer_2": 8768}[reviewer]


def _rows(catalog: CatalogProtocol[Any, Any], name: str, path: Path | None) -> list[dict[str, Any]]:
    """Load only an explicitly supplied private export through the catalog boundary."""
    if path is None:
        return []
    catalog[name] = CSVDataset(
        filepath=path.resolve(), load_args={"keep_default_na": False, "dtype": str}
    )
    return cast("list[dict[str, Any]]", catalog.load(name).to_dict("records"))


def combine_flip_catalog_exports(
    catalog: CatalogProtocol[Any, Any], args: argparse.Namespace
) -> dict[str, int]:
    """Merge validated exports into existing annotations, refusing conflicts before saving."""
    cohort = catalog.load("semantic_flip_cohort")
    validate_flip_cohort(cohort)
    existing = (
        catalog.load("semantic_flip_annotations")
        if catalog.exists("semantic_flip_annotations")
        else pd.DataFrame()
    )
    first = _rows(catalog, "semantic_flip_export_reviewer_1", args.reviewer_1)
    second = _rows(catalog, "semantic_flip_export_reviewer_2", args.reviewer_2)
    reuse = args.reuse_older_1 is not None or args.reuse_older_2 is not None
    older = (
        build_review_dataset(
            catalog.load("semantic_audit_annotation_candidates").to_dict("records")
        )
        if reuse
        else None
    )
    reused_first = _rows(catalog, "semantic_audit_reviewer_1_export", args.reuse_older_1)
    reused_second = _rows(catalog, "semantic_audit_reviewer_2_export", args.reuse_older_2)
    merged = merge_flip_exports(
        cohort["review_dataset"],
        first,
        second,
        existing,
        reuse_dataset=older,
        reuse_first=reused_first,
        reuse_second=reused_second,
    )
    catalog.save("semantic_flip_annotations", merged)
    for reviewer in REVIEWER_IDS:
        catalog.save(
            f"semantic_flip_progress_{reviewer}",
            progress_for_role(cohort["review_dataset"], merged, reviewer),
        )
    return {
        "submitted_rows": len(merged),
        "complete_rows": int((merged["review_status"] == "complete").sum()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Load the new cohort through Kedro and serve or import explicit human evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-path", type=Path, default=Path.cwd())
    parser.add_argument("--env", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--reviewer", choices=REVIEWER_IDS, required=True)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--open", dest="open_browser", action="store_true")
    combine = commands.add_parser("combine")
    combine.add_argument("--reviewer-1", type=Path)
    combine.add_argument("--reviewer-2", type=Path)
    combine.add_argument("--reuse-older-1", type=Path)
    combine.add_argument("--reuse-older-2", type=Path)
    combine.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project_path = args.project_path.resolve()
    bootstrap_project(project_path)
    session = KedroSession.create(project_path=project_path, env=args.env, save_on_close=False)
    with session:
        context = session.load_context()
        if args.command == "serve":
            args.port = args.port if args.port is not None else default_port(args.reviewer)
            _serve(context.catalog.load(f"semantic_flip_tool_{args.reviewer}"), args)
        else:
            counts = combine_flip_catalog_exports(context.catalog, args)
            logging.getLogger(__name__).info(
                "Retained %d reviewer rows (%d complete)",
                counts["submitted_rows"],
                counts["complete_rows"],
            )
            if args.finalize:
                session.run(pipeline_name="semantic_flip_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
