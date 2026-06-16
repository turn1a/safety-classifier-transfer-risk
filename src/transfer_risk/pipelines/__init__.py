"""Kedro modular pipelines for transfer_risk.

Each subpackage is one stage of the measurement chain and exposes
``create_pipeline``; ``kedro.framework.project.find_pipelines`` discovers them.
See ``SPEC.md`` §11 for the stage sequence (data -> models -> similarity ->
attacks -> transfer -> risk -> reporting), plus a ``smoke`` wiring check.
"""
