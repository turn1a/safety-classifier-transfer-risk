"""Tests for static Kedro-viz path sanitization."""

from __future__ import annotations

from pathlib import Path

import pytest
from kedro.config import OmegaConfigLoader

from transfer_risk.scripts.build_viz import sanitize_viz_text, sanitize_viz_tree

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = Path("/Users/aholszewska/dev/risk-transfer")
_REMOTE_VIZ_SOURCES = (
    "raw_deepset",
    "raw_jackhhao",
    "raw_lakera",
    "target_model",
    "hub__deberta-base-pi-v1",
    "hub__deberta-small-pi-v2",
    "hub__deepset-deberta-injection",
    "hub__llama-prompt-guard-86m",
    "hub__llama-prompt-guard-22m",
    "hub__bert-base-ft",
    "hub__roberta-base-ft",
    "hub__electra-small-ft",
    "hub__deberta-base-ft-seed",
)


def _read_text_bundle(root: Path) -> str:
    """Return concatenated UTF-8 static-site files without inspecting binary assets."""
    contents: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            contents.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(contents)


@pytest.fixture
def static_viz_tree_with_stale_api_node(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create static output with stale and live extensionless API resources."""
    root = tmp_path / "pipeline-viz"
    api = root / "api" / "nodes"
    api.mkdir(parents=True)
    (root / "index.html").write_text(
        '<script src="/Users/aholszewska/dev/risk-transfer/site.js"></script>',
        encoding="utf-8",
    )
    stale_node = api / "stale-node"
    stale_node.write_text(
        '{"filepath": "/Users/legacy/reports/fig_cka_heatmap.png"}',
        encoding="utf-8",
    )
    live_node = api / "live-node"
    live_node.write_text(
        '{"filepath": "data/08_reporting/master_results_table.csv", '
        '"source": "file:///Users/live-user/current-run.json"}',
        encoding="utf-8",
    )
    return root, stale_node, live_node


def test_sanitize_viz_text_replaces_workspace_paths() -> None:
    """Replace literal local workspace paths in a generated API payload."""
    rendered = (
        '{"catalog": "/Users/aholszewska/dev/risk-transfer/conf/base/catalog.yml", '
        '"uri": "file:///Users/aholszewska/dev/risk-transfer/data"}'
    )

    sanitized = sanitize_viz_text(rendered, _WORKSPACE_ROOT)

    assert str(_WORKSPACE_ROOT) not in sanitized
    assert "/Users/aholszewska" not in sanitized
    assert "<workspace>" in sanitized


def test_sanitize_viz_tree_rewrites_stale_and_live_extensionless_api_nodes(
    static_viz_tree_with_stale_api_node: tuple[Path, Path, Path],
) -> None:
    """Sanitize all text resources while retaining every generated API node."""
    root, stale_node, live_node = static_viz_tree_with_stale_api_node

    changed = sanitize_viz_tree(root, _WORKSPACE_ROOT)

    assert changed == 3
    assert stale_node.is_file()
    assert live_node.is_file()
    assert "/Users/" not in _read_text_bundle(root)


def test_viz_build_cleans_all_static_output_before_building() -> None:
    """Remove both possible stale output directories before Kedro-viz builds."""
    recipe = (
        (_PROJECT_ROOT / "justfile")
        .read_text(encoding="utf-8")
        .split("viz-build:\n", maxsplit=1)[1]
        .split("\n\n", maxsplit=1)[0]
    )
    commands = [line.strip() for line in recipe.splitlines() if line.startswith("    ")]

    assert commands == [
        "rm -rf build docs/pipeline-viz",
        "KEDRO_ENV=viz uv run kedro viz build",
        "mv build docs/pipeline-viz",
        (
            "uv run python -m transfer_risk.scripts.build_viz docs/pipeline-viz "
            '--workspace-root "$PWD"'
        ),
    ]


def test_viz_environment_stubs_remote_catalog_sources() -> None:
    """Keep static graph creation independent of Hub datasets and model access."""
    loader = OmegaConfigLoader(
        conf_source=str(_PROJECT_ROOT / "conf"),
        base_env="base",
        default_run_env="viz",
    )
    catalog = loader["catalog"]

    assert {name: catalog[name]["type"] for name in _REMOTE_VIZ_SOURCES} == dict.fromkeys(
        _REMOTE_VIZ_SOURCES,
        "MemoryDataset",
    )


def test_checked_in_static_viz_bundle_is_current_and_sanitized() -> None:
    """Keep the published graph present, path-safe, and aligned with current reporting nodes."""
    root = _PROJECT_ROOT / "docs" / "pipeline-viz"
    bundle = _read_text_bundle(root)

    assert (root / "index.html").is_file()
    assert "/Users/aholszewska" not in bundle
    assert "build_qualitative_examples" in bundle
    assert "recompute_risk_master_dbs" in bundle
    assert "plot_selection_ablation" not in bundle
