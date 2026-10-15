"""Behavioral tests for aggregate-only manuscript reporting."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from transfer_risk.lib.paper_reporting import (
    audit_transition_counts,
    exclude_surrogates,
    render_sensitivity_table,
    render_surrogate_table,
)

_ROOT = Path(__file__).resolve().parents[2]


def _saved_summary() -> dict:
    """Load the public, aggregate-only audit fixture."""
    return json.loads((_ROOT / "docs/artifacts/target_audit_summary.json").read_text())


def test_transition_counts_reconcile_original_and_perturbed_marginals() -> None:
    """The two target-benign numerators differ by unchanged benign records."""
    assert audit_transition_counts(_saved_summary()) == {
        "injection_to_injection": 2955,
        "injection_to_benign": 1054,
        "benign_to_injection": 9,
        "benign_to_benign": 344,
    }


def test_transition_counts_reject_impossible_joint_counts() -> None:
    """An inconsistent saved marginal cannot generate a plausible diagram."""
    summary = _saved_summary()
    summary["full_cohort"]["true_target_flips"] = 1400
    with pytest.raises(ValueError, match="inconsistent"):
        audit_transition_counts(summary)


def test_excluding_bilstm_recomputes_counts_and_exact_association() -> None:
    """The sensitivity omits the affected model without creating new outcomes."""
    summary = _saved_summary()
    before = deepcopy(summary)
    result = exclude_surrogates(summary, ["bilstm-attention"])
    assert result["n_surrogates"] == 9
    assert result["source_successful"] == 3704
    assert result["target_original_injection"] == 3410
    assert result["true_target_flips"] == 998
    assert result["true_target_flip_rate"] == pytest.approx(998 / 3410)
    assert result["mean_cka_association"]["rho"] == pytest.approx(0.8)
    assert result["mean_cka_association"]["two_sided_p"] == pytest.approx(5018 / 362880)
    assert result["mean_cka_association"]["exact"] is True
    assert summary == before


def test_excluding_an_unknown_surrogate_fails() -> None:
    """A misspelled exclusion cannot silently retain the affected model."""
    with pytest.raises(ValueError, match="unknown"):
        exclude_surrogates(_saved_summary(), ["missing-model"])


def test_sensitivity_label_names_the_configured_exclusion() -> None:
    """An alternate excluded model cannot be mislabeled as the BiLSTM."""
    summary = _saved_summary()
    training = json.loads(
        (_ROOT / "docs/artifacts/training_probe_similarity_sensitivity.json").read_text()
    )
    reduced = exclude_surrogates(summary, ["bert-base-ft"])
    table = render_sensitivity_table(summary, training, reduced)
    assert "`bert-base-ft` excluded, saved outcomes" in table
    assert "941 / 3,691" in table
    assert "BiLSTM excluded" not in table


def test_surrogate_table_uses_rounded_rates_and_excludes_private_fields() -> None:
    """Human-readable tables preserve counts and omit arbitrary private fields."""
    summary = _saved_summary()
    summary["full_cohort"]["surrogate_macro"]["rows"][0]["prompt"] = "PRIVATE_FIXTURE"
    table = render_surrogate_table(summary, 955)
    assert "55.7" in table
    assert "56.9" in table
    assert "140 / 246" in table
    assert "0.5567388639729065" not in table
    assert "PRIVATE_FIXTURE" not in table


def test_sensitivity_table_explains_null_rates_correlations_and_probabilities() -> None:
    """Undefined statistics remain visible when a saved cohort has no target injections."""
    summary = _saved_summary()
    training = json.loads(
        (_ROOT / "docs/artifacts/training_probe_similarity_sensitivity.json").read_text()
    )
    reduced = exclude_surrogates(summary, ["bilstm-attention"])
    reduced.update(true_target_flip_rate=None, target_original_injection=0, true_target_flips=0)
    reduced["mean_cka_association"].update(
        rho=None, two_sided_p=None, reason="constant input sequences"
    )
    rendered = render_sensitivity_table(summary, training, reduced)
    assert "Undefined (no original target injections)" in rendered
    assert "Undefined (constant input sequences)" in rendered


@pytest.mark.parametrize("outcome", [None, 0.0])
def test_surrogate_exclusion_keeps_undefined_constant_association(outcome: float | None) -> None:
    """A missing or constant macro outcome produces a reasoned null correlation."""
    summary = _saved_summary()
    for row in summary["full_cohort"]["surrogate_macro"]["rows"]:
        row["true_target_flip_rate_macro_mean"] = outcome
    reduced = exclude_surrogates(summary, ["bilstm-attention"])
    stats = reduced["mean_cka_association"]
    assert stats["rho"] is None
    assert stats["two_sided_p"] is None
    assert "defined" in stats["reason"] if outcome is None else "constant" in stats["reason"]
