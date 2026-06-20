"""The RAM-bounded worker cap fits torch victims in memory (see transfer_risk.runner)."""

from transfer_risk.runner import ram_bounded_workers


def test_ram_bound_caps_workers_on_a_memory_light_box() -> None:
    """192 vCPU / 384 GB (c-family, 2 GB/vCPU): RAM, not the CPU count, is the limit."""
    assert ram_bounded_workers(192, 384.0, gb_per_worker=5.0) == 76


def test_ram_bound_uses_every_core_on_a_high_ram_box() -> None:
    """192 vCPU / 1536 GB (r-family, 8 GB/vCPU): the CPU count is the limit, every core used."""
    assert ram_bounded_workers(192, 1536.0, gb_per_worker=5.0) == 192


def test_ram_bound_never_drops_below_one() -> None:
    """Even a tiny box runs at least one worker."""
    assert ram_bounded_workers(8, 2.0, gb_per_worker=5.0) == 1
