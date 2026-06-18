"""Deterministic seeding (SPEC.md §12)."""

from __future__ import annotations

from transfer_risk.lib.seeds import derive_seeds


def test_same_root_seed_is_reproducible() -> None:
    assert derive_seeds(20260616) == derive_seeds(20260616)


def test_different_root_seeds_differ() -> None:
    assert derive_seeds(1) != derive_seeds(2)


def test_components_are_distinct() -> None:
    bundle = derive_seeds(20260616)
    assert len({bundle.python, bundle.numpy, bundle.torch, bundle.textattack}) == 4
