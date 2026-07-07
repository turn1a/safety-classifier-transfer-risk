"""Pure, deterministic algorithms — no I/O, no network, no Kedro.

This is the security-relevant core: CKA representational similarity
(:mod:`~transfer_risk.lib.cka`), Diagonal Box Similarity
(:mod:`~transfer_risk.lib.dbs`), deterministic seeding
(:mod:`~transfer_risk.lib.seeds`), and empirical threshold calibration
(:mod:`~transfer_risk.lib.thresholds`). These functions are unit-tested in
isolation and reused by the Kedro pipeline nodes.

See ``SPEC.md`` §3 for the specifications and the reference CKA implementation.
"""
