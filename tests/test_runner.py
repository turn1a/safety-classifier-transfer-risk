"""The RAM-bounded worker cap fits torch victims in memory (see transfer_risk.runner)."""

from transfer_risk.runner import ram_bounded_workers


def test_ram_bound_caps_a_memory_light_box() -> None:
    """384 GB (the c-family, 2 GB/vCPU) is RAM-bound: far fewer workers than its 192 vCPUs."""
    assert ram_bounded_workers(192, 384.0) == 32  # floor(384 / 12)


def test_ram_bound_fits_more_on_the_high_ram_run_box() -> None:
    """1536 GB (r8g.48xlarge, the run box) fits ~128 workers — well above a memory-light box."""
    assert ram_bounded_workers(192, 1536.0) == 128  # floor(1536 / 12)


def test_ram_bound_is_cpu_bound_when_ram_is_ample() -> None:
    """With ample RAM the CPU count is the binding limit, so every core is used."""
    assert ram_bounded_workers(192, 100_000.0) == 192


def test_ram_bound_never_drops_below_one() -> None:
    """Even a tiny box runs at least one worker."""
    assert ram_bounded_workers(8, 2.0) == 1
