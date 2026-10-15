import json
from pathlib import Path

from transfer_risk.lib.strengthening_tables import (
    render_baseline_table,
    render_coverage_table,
    render_objective_table,
    render_policy_table,
)

_ROOT = Path(__file__).resolve().parents[2] / "docs/artifacts"


def test_objective_table_keeps_undefined_associations_and_their_reasons() -> None:
    """Missing correlations cannot disappear from the candidate-policy comparison."""
    objectives = json.loads((_ROOT / "pool_objective_sensitivity.json").read_text())
    comparison = json.loads((_ROOT / "checkpoint_comparison.json").read_text())
    stats = objectives["pools"]["all_surrogates"]["associations"][
        "macro_conditional_verified_flip_rate"
    ]
    stats.update(rho=None, two_sided_p=None, reason="no defined outcomes")
    comparison["reference_pool"]["association"] = {
        "rho": None,
        "two_sided_p": None,
        "reason": "constant input sequences",
    }
    rendered = render_objective_table(objectives, comparison)
    assert "Undefined (no defined outcomes)" in rendered
    assert "All-returned-candidate trial yield | Undefined (constant input sequences)" in rendered


def test_baseline_table_explains_empty_probability_denominators() -> None:
    """A zero-size clean class is reported as undefined with the missing denominator."""
    rendered = render_baseline_table(
        {
            "reports": [
                {
                    "target_id": "empty",
                    "pooled": {
                        "canonical_positive_count": 0,
                        "canonical_positive_predicted_positive_count": 0,
                        "canonical_negative_count": 0,
                        "canonical_negative_predicted_positive_count": 0,
                        "canonical_positive_recall": None,
                        "canonical_negative_false_positive_rate": None,
                        "canonical_label_accuracy": None,
                    },
                }
            ]
        }
    )
    assert "Undefined (no canonical-positive inputs)" in rendered
    assert "Undefined (no canonical-benign inputs)" in rendered
    assert "Undefined (no clean inputs)" in rendered


def test_coverage_table_keeps_distinct_original_denominator() -> None:
    table = render_coverage_table(
        {
            "original_target_positive": 167,
            "selected": {"source_gated_originals": 104, "all_candidate_originals": 120},
            "comparisons": {
                "source_gated": {"best_originals": 118, "n_strictly_higher": 19},
                "all_candidate": {"best_originals": 127, "n_strictly_higher": 16},
            },
            "n_subsets": 84,
            "attempted_trials_per_subset": 2865,
        }
    )
    assert "104/167 (62.3%)" in table
    assert "120/167 (71.9%)" in table
    assert "2,865" in table


def test_baseline_table_rounds_display_without_mutating_evidence() -> None:
    report = {
        "reports": [
            {
                "target_id": "detector",
                "pooled": {
                    "canonical_positive_count": 191,
                    "canonical_positive_predicted_positive_count": 167,
                    "canonical_negative_count": 104,
                    "canonical_negative_predicted_positive_count": 0,
                    "canonical_positive_recall": 167 / 191,
                    "canonical_negative_false_positive_rate": 0.0,
                    "canonical_label_accuracy": 271 / 295,
                },
            }
        ]
    }
    rendered = render_baseline_table(report)
    assert "167/191 (87.4%)" in rendered
    assert "0/104 (0.0%)" in rendered
    assert "0.874345" not in rendered
    assert report["reports"][0]["pooled"]["canonical_positive_recall"] == 167 / 191


def test_policy_table_names_all_returned_candidates_and_attempted_budget() -> None:
    rendered = render_policy_table(
        {
            "common_surrogates": ["a", "b", "c", "d", "e", "f", "g"],
            "n_recipes": 5,
            "common_pool": [
                {
                    "target": "reference",
                    "attempted": 6685,
                    "original_target_positive": 167,
                    "original_target_benign": 24,
                    "source_gated_flips": 700,
                    "all_candidate_flips": 900,
                    "all_candidate_yield": 900 / 6685,
                }
            ],
        }
    )
    assert "6,685" in rendered
    assert "13.5%" in rendered
    assert "All returned" in rendered
    assert "167/191" in rendered
