"""transfer_risk — measure adversarial transferability risk for text safety classifiers.

This package ports the Cox & Bunzel (2025) transferability-risk method to text
safety classifiers (see ``SPEC.md``). It is organised as Kedro pipelines over a
Data Catalog; the security-relevant algorithms live in :mod:`transfer_risk.lib`.
"""

__version__ = "0.1.0"
