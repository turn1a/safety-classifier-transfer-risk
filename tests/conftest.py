"""Shared pytest configuration.

Configure the Kedro project once so ``find_pipelines`` can discover the pipeline
packages when the registry is exercised directly in tests.
"""

from kedro.framework.project import configure_project

configure_project("transfer_risk")
