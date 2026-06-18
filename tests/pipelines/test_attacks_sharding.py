"""Unit tests for the attack-sweep example sharding (pure index logic, no textattack)."""

from transfer_risk.pipelines.attacks.nodes import _shard_indices


def test_shard_indices_tile_range_without_gaps_or_overlap() -> None:
    """Spans must tile ``range(n)`` exactly — ordered, no gaps, no overlaps."""
    for n, size in [(60, 4), (60, 7), (12, 4), (10, 1), (5, 10), (1, 1)]:
        spans = _shard_indices(n, size)
        covered = [i for start, stop in spans for i in range(start, stop)]
        assert covered == list(range(n)), (n, size, spans)
        assert all(stop - start <= max(1, size) for start, stop in spans)
        assert all(start < stop for start, stop in spans)


def test_shard_indices_empty() -> None:
    """No examples yields no spans."""
    assert _shard_indices(0, 4) == []


def test_shard_indices_coerces_nonpositive_size() -> None:
    """A zero or negative shard size is coerced to one example per shard."""
    assert _shard_indices(3, 0) == [(0, 1), (1, 2), (2, 3)]
