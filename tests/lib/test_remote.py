"""Tests for the pure model-cache path helper."""

from pathlib import Path

from transfer_risk.lib.remote import model_cache_path


def test_deterministic_and_under_cache_root() -> None:
    first = model_cache_path("s3://bucket/data/06_models/foo", "/cache")
    second = model_cache_path("s3://bucket/data/06_models/foo", "/cache")
    assert first == second
    assert Path(first).parent == Path("/cache")
    assert Path(first).name.startswith("foo-")


def test_trailing_slash_ignored() -> None:
    assert model_cache_path("s3://bucket/x/foo/", "/c") == model_cache_path(
        "s3://bucket/x/foo", "/c"
    )


def test_same_basename_different_remote_does_not_collide() -> None:
    one = model_cache_path("s3://bucket-a/foo", "/c")
    two = model_cache_path("s3://bucket-b/foo", "/c")
    assert one != two
    assert Path(one).name.startswith("foo-")
    assert Path(two).name.startswith("foo-")
