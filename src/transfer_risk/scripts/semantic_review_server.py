"""Serve one private review document on loopback or combine catalog-owned human exports."""

from __future__ import annotations

import argparse
import logging
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro_datasets.pandas import CSVDataset

from transfer_risk.lib.semantic_review_tool import (
    REVIEWER_IDS,
    build_review_dataset,
    combine_reviewer_exports,
    review_tool_content_security_policy,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kedro.io import CatalogProtocol

logger = logging.getLogger(__name__)
_MAX_PORT = 65535
_DEFAULT_PORTS = {"reviewer_1": 8765, "reviewer_2": 8766}


def load_review_html(catalog: CatalogProtocol[Any, Any], reviewer_id: str) -> str:
    """Read only the generated HTML catalog entry for one fixed reviewer role.

    Args:
        catalog: Configured Kedro catalog with private generated tool entries.
        reviewer_id: User or partner's fixed reviewer identity.

    Returns:
        Exact generated private HTML string.

    Raises:
        ValueError: If the role is unknown or its catalog output is not HTML text.
    """
    if reviewer_id not in REVIEWER_IDS:
        msg = "reviewer must be reviewer_1 or reviewer_2"
        raise ValueError(msg)
    html = catalog.load(f"semantic_audit_review_tool_{reviewer_id}")
    if not isinstance(html, str) or not html.strip():
        msg = "the private reviewer catalog entry must contain generated HTML text"
        raise ValueError(msg)
    return html


def _handler_type(html: str) -> type[BaseHTTPRequestHandler]:
    """Create a handler that serves one in-memory document and never reads URL paths."""
    document = html.encode("utf-8")
    policy = review_tool_content_security_policy(html) + "; frame-ancestors 'none'"

    class ReviewHandler(BaseHTTPRequestHandler):
        """Serve only the generated document, without file or upload endpoints."""

        def _respond(self, status: int, payload: bytes, content_type: str) -> None:
            """Return a bounded response with privacy and browser-resource controls."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Security-Policy", policy)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def do_GET(self) -> None:
            """Serve the root document only for the loopback host and same-origin requests."""
            host = f"127.0.0.1:{cast('HTTPServer', self.server).server_port}"
            if self.headers.get("Host") != host or self.headers.get("Origin") not in {
                None,
                f"http://{host}",
            }:
                self._respond(403, b"Forbidden.", "text/plain; charset=utf-8")
            elif self.path != "/":
                self._respond(404, b"Not found.", "text/plain; charset=utf-8")
            else:
                self._respond(200, document, "text/html; charset=utf-8")

        def do_HEAD(self) -> None:
            """Return root-document headers without a body."""
            self.do_GET()

        def do_POST(self) -> None:
            """Reject uploads and API mutations without reading or saving the body."""
            self._respond(405, b"Method not allowed.", "text/plain; charset=utf-8")

        do_PUT = do_POST  # noqa: N815 - standard-library HTTP handler API
        do_PATCH = do_POST  # noqa: N815 - standard-library HTTP handler API
        do_DELETE = do_POST  # noqa: N815 - standard-library HTTP handler API
        do_OPTIONS = do_POST  # noqa: N815 - standard-library HTTP handler API
        do_TRACE = do_POST  # noqa: N815 - standard-library HTTP handler API

        def log_message(self, _format: str, *_args: Any) -> None:
            """Suppress request logging so paths or client-supplied data are not recorded."""

    return ReviewHandler


def build_review_server(html: str, *, port: int) -> HTTPServer:
    """Bind a single-document HTTP server to IPv4 loopback only.

    Args:
        html: Generated private tool document, already loaded through the catalog.
        port: Local port, or zero to request an available test port.

    Returns:
        Bound server; the caller controls serving, shutdown and socket cleanup.

    Raises:
        ValueError: If the port is not an integer in the allowed range.
    """
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= _MAX_PORT:
        msg = "port must be an integer from 0 to 65535"
        raise ValueError(msg)
    return HTTPServer(("127.0.0.1", port), _handler_type(html))


def combine_catalog_exports(
    catalog: CatalogProtocol[Any, Any], semantic_params: Mapping[str, Any]
) -> dict[str, int]:
    """Validate two private exports completely before replacing derived annotation input.

    Args:
        catalog: Catalog owning the saved sample, separate exports and annotation output.
        semantic_params: Confirmed reviewer identity configuration.

    Returns:
        Non-text import counts suitable for terminal status. Partial or uncertain reviews
        are preserved; consensus is computed by the separate semantic-audit review stage.

    Raises:
        ValueError: If identities, dataset types, sample membership, or export fields
            fail validation.
    """
    if semantic_params.get("reviewer_ids") != list(REVIEWER_IDS):
        msg = "semantic_audit.reviewer_ids must be [reviewer_1, reviewer_2]"
        raise ValueError(msg)
    frames = [
        catalog.load("semantic_audit_annotation_candidates"),
        catalog.load("semantic_audit_reviewer_1_export"),
        catalog.load("semantic_audit_reviewer_2_export"),
    ]
    if not all(isinstance(frame, pd.DataFrame) for frame in frames):
        msg = "the review sample and both private exports must be catalog dataframes"
        raise ValueError(msg)
    candidates, first, second = frames
    dataset = build_review_dataset(cast("list[dict[str, Any]]", candidates.to_dict("records")))
    annotations = combine_reviewer_exports(
        dataset,
        cast("list[dict[str, Any]]", first.to_dict("records")),
        cast("list[dict[str, Any]]", second.to_dict("records")),
    )
    catalog.save("semantic_audit_annotations", annotations)
    return {
        "selected_pairs": len(candidates),
        "submitted_rows": len(annotations),
        "complete_rows": int((annotations["review_status"] == "complete").sum()),
        "in_progress_rows": int((annotations["review_status"] == "in_progress").sum()),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse loopback serving and explicitly requested private-export import commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-path", type=Path, default=Path.cwd())
    parser.add_argument("--env", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve one fixed-reviewer tool on 127.0.0.1")
    serve.add_argument("--reviewer", choices=REVIEWER_IDS, required=True)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--open", dest="open_browser", action="store_true")
    combine = commands.add_parser(
        "combine", help="Validate separate catalog exports and save annotations"
    )
    combine.add_argument(
        "--reviewer-1", type=Path, help="Optional local CSV override through the catalog"
    )
    combine.add_argument(
        "--reviewer-2", type=Path, help="Optional local CSV override through the catalog"
    )
    combine.add_argument(
        "--finalize", action="store_true", help="Run semantic_audit_review after validation"
    )
    return parser.parse_args(argv)


def _serve(html: str, args: argparse.Namespace) -> None:
    """Serve one generated private tool until the user interrupts the process."""
    port = args.port if args.port is not None else _DEFAULT_PORTS[args.reviewer]
    with build_review_server(html, port=port) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        logger.info("Private %s tool: %s", args.reviewer, url)
        logger.info("Stop with Ctrl-C. Labels remain in your browser; export a JSON backup.")
        if args.open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Stopped the private review server.")


def main(argv: Sequence[str] | None = None) -> int:
    """Load private artifacts through Kedro, then serve or validate human exports.

    Args:
        argv: Optional command arguments, excluding the executable name.

    Returns:
        Zero after a server shutdown or successful validated import/finalization.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    project_path = args.project_path.resolve()
    bootstrap_project(project_path)
    session = KedroSession.create(project_path=project_path, env=args.env, save_on_close=False)
    with session:
        context = session.load_context()
        if args.command == "serve":
            _serve(load_review_html(context.catalog, args.reviewer), args)
        else:
            for reviewer, path in (
                (REVIEWER_IDS[0], args.reviewer_1),
                (REVIEWER_IDS[1], args.reviewer_2),
            ):
                if path is not None:
                    context.catalog[f"semantic_audit_{reviewer}_export"] = CSVDataset(
                        filepath=path.resolve(),
                        load_args={"keep_default_na": False, "dtype": str},
                    )
            counts = combine_catalog_exports(context.catalog, context.params["semantic_audit"])
            logger.info(
                "Saved %d reviewer rows (%d complete, %d in progress) for %d selected pairs",
                counts["submitted_rows"],
                counts["complete_rows"],
                counts["in_progress_rows"],
                counts["selected_pairs"],
            )
            if args.finalize:
                session.run(pipeline_name="semantic_audit_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
