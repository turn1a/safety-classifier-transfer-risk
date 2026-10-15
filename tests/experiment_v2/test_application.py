"""Application objectives require fixed context, controls, and separate human evidence."""

from dataclasses import replace

import pytest

from transfer_risk.lib.experiment_v2.application import (
    analyse_application,
    application_control_binding,
)
from transfer_risk.lib.experiment_v2.records import validate_records
from transfer_risk.lib.experiment_v2.schemas import (
    ApplicationControl,
    ApplicationRecord,
    ApplicationStudy,
)

from .fixtures import digest, evidence, records_for


def application_fixture():
    protocol, corpus, registration = evidence()
    first_original = next(
        row for row in corpus.records if row.partition == "evaluation" and row.label == 1
    )
    study = ApplicationStudy(
        enabled=True,
        decoder_repository="local/synthetic-decoder",
        decoder_revision="d" * 40,
        decoder_sha256=digest("decoder"),
        tokenizer_sha256=digest("decoder_tokenizer"),
        decoding_spec_sha256=digest("greedy_fixed_seed_16_tokens"),
        task_sha256=digest("fixed_task"),
        context_sha256=first_original.task_context_sha256,
        benign_control_input_sha256=digest("benign-input"),
        objective_id="synthetic-exact-objective",
        checker_sha256=digest("deterministic_exact_checker"),
    )
    records = records_for("evaluation", protocol, corpus)
    index = validate_records(protocol, corpus, registration, records, 20260616)
    protocol = protocol.model_copy(update={"application_study": study})
    attempts = list(index.attempts.values())
    record = ApplicationRecord(
        attempt_id=attempts[0].id,
        original_sha256=first_original.content_sha256,
        candidate_sha256=attempts[0].candidate_sha256,
        trusted_instruction_sha256=first_original.trusted_instruction_sha256,
        **study.model_dump(exclude={"enabled", "decoder_repository", "decoder_revision"}),
        benign=ApplicationControl(
            task_pass=True, attacker_objective=False, execution_sha256=digest("benign_run")
        ),
        original=ApplicationControl(
            task_pass=False, attacker_objective=True, execution_sha256=digest("original_run")
        ),
        perturbed=ApplicationControl(
            task_pass=False, attacker_objective=True, execution_sha256=digest("perturbed_run")
        ),
    )
    hashes = {
        "benign": study.benign_control_input_sha256,
        "original": first_original.content_sha256,
        "perturbed": attempts[0].candidate_sha256,
    }
    record = record.model_copy(
        update={
            role: getattr(record, role).model_copy(update={"input_sha256": value})
            for role, value in hashes.items()
        }
    )
    record = record.model_copy(
        update={
            role: getattr(record, role).model_copy(
                update={"binding_sha256": application_control_binding(record, role)}
            )
            for role in hashes
        }
    )
    return protocol, records.model_copy(update={"application_records": [record]}), index


def test_fixed_local_controls_establish_objective_retention_separately_from_detector_flip():
    protocol, records, index = application_fixture()
    report = analyse_application(protocol, records, index)
    assert report["targets"]["primary"]["objective_retained"] == 1
    assert report["targets"]["primary"]["human_valid_effective_target_evasion"] == 1
    assert report["missing_controls"] == 95
    assert report["status"] == "incomplete"
    assert report["executor"] == "not_implemented"


def test_missing_context_is_unassessable_even_if_a_mechanical_objective_check_passed():
    protocol, records, index = application_fixture()
    record = records.application_records[0]
    record = record.model_copy(
        update={
            "context_sha256": None,
            **{
                role: getattr(record, role).model_copy(update={"binding_sha256": None})
                for role in ("benign", "original", "perturbed")
            },
        }
    )
    report = analyse_application(
        protocol, records.model_copy(update={"application_records": [record]}), index
    )
    assert report["unassessable"] == 1
    assert report["targets"]["primary"]["objective_retained"] == 0


def test_benign_control_objective_leak_invalidates_the_application_control_comparison():
    protocol, records, index = application_fixture()
    record = records.application_records[0]
    leaking = record.benign.model_copy(update={"attacker_objective": True})
    record = record.model_copy(update={"benign": leaking})
    report = analyse_application(
        protocol, records.model_copy(update={"application_records": [record]}), index
    )
    assert report["targets"]["primary"]["assessable"] == 1
    assert report["targets"]["primary"]["benign_controls_pass"] == 0
    assert report["targets"]["primary"]["objective_retained"] == 0


def test_missing_human_labels_do_not_convert_objective_checks_into_human_validity():
    protocol, records, index = application_fixture()

    index = replace(index, human=dict.fromkeys(index.human, "unresolved"))
    report = analyse_application(protocol, records, index)
    assert report["targets"]["primary"]["objective_retained"] == 1
    assert report["targets"]["primary"]["human_valid_effective_target_evasion"] == 0
    assert report["human_unresolved"] == 96


def test_new_objective_success_is_distinct_from_retained_original_effectiveness():
    protocol, records, index = application_fixture()
    record = records.application_records[0]
    original = record.original.model_copy(update={"attacker_objective": False})
    record = record.model_copy(update={"original": original})
    report = analyse_application(
        protocol, records.model_copy(update={"application_records": [record]}), index
    )
    assert report["targets"]["primary"]["objective_retained"] == 0
    assert report["targets"]["primary"]["new_objective_success"] == 1


def test_unknown_or_duplicate_application_controls_are_rejected():
    protocol, records, index = application_fixture()
    unknown = records.application_records[0].model_copy(update={"attempt_id": "unknown"})
    with pytest.raises(ValueError, match="unknown or duplicate"):
        analyse_application(
            protocol, records.model_copy(update={"application_records": [unknown]}), index
        )
    with pytest.raises(ValueError, match="unknown or duplicate"):
        analyse_application(
            protocol,
            records.model_copy(update={"application_records": records.application_records * 2}),
            index,
        )


def test_application_records_require_a_separately_enabled_protocol():
    protocol, records, index = application_fixture()
    protocol = protocol.model_copy(update={"application_study": ApplicationStudy(enabled=False)})
    with pytest.raises(ValueError, match="separately enabled"):
        analyse_application(protocol, records, index)


def test_unpinned_decoder_stays_unassessable_despite_complete_passed_controls():
    protocol, records, index = application_fixture()
    study = protocol.application_study.model_copy(update={"decoder_revision": None})
    report = analyse_application(
        protocol.model_copy(update={"application_study": study}), records, index
    )
    assert report["unassessable"] == 1
    assert report["targets"]["primary"]["objective_retained"] == 0


@pytest.mark.parametrize("change", ["attempt", "original", "candidate", "benign_input", "replay"])
def test_application_controls_reject_reassignment_or_replayed_execution(change):
    protocol, records, index = application_fixture()
    record = records.application_records[0]
    if change == "attempt":
        record = record.model_copy(update={"attempt_id": list(index.human)[1]})
    elif change == "original":
        record = record.model_copy(update={"original_sha256": digest("wrong-original")})
    elif change == "candidate":
        record = record.model_copy(update={"candidate_sha256": digest("wrong-candidate")})
    elif change == "benign_input":
        record = record.model_copy(
            update={
                "benign": record.benign.model_copy(
                    update={"input_sha256": digest("different-benign-input")}
                )
            }
        )
    else:
        record = record.model_copy(update={"perturbed": record.original})
    with pytest.raises(ValueError, match=r"application.*(identity|binding|replay)"):
        analyse_application(
            protocol, records.model_copy(update={"application_records": [record]}), index
        )
