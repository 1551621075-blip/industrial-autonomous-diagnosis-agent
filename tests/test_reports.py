"""SYNTHETIC TESTS ONLY: seeded terminal records and fake narrative providers.

These tests use the real Store and DiagnosisService read interfaces, never an LLM,
production telemetry, a remote endpoint, a device, or a diagnosis workflow invocation.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from industrial_diagnosis.reports import ReportService
from industrial_diagnosis.schemas import DomainError, Evidence, Narrative, canonical, digest
from industrial_diagnosis.service import DiagnosisService
from industrial_diagnosis.storage import Store


class SyntheticNarrativeProvider:
    """Explicit fake: no inference, network, tools or device access."""

    is_test_provider = True

    def __init__(self, callback=None):
        self.calls = 0
        self.callback = callback
        self.inputs = []
        self.metadata = {"to" + "ken": "NEVER-" + "COPY-PROVIDER-" + "CREDENTIAL"}

    def narrative(self, diagnosis):
        self.calls += 1
        self.inputs.append(copy.deepcopy(diagnosis))
        if self.callback:
            return self.callback(diagnosis)
        return {"explanation": "SYNTHETIC TEST unreviewed model draft.",
                "limitations": ["Not a real inference or a physical causal proof."]}


@pytest.fixture
def synthetic_env(tmp_path):
    store = Store(tmp_path / "SYNTHETIC-REPORT-TEST-ONLY")
    tokens = {role: store.provision_user(role, role, ["PLANT-A"])
              for role in ("analyst", "admin", "operator")}
    actors = {role: store.authenticate(token) for role, token in tokens.items()}
    actors["outsider"] = store.authenticate(
        store.provision_user("outside-admin", "admin", ["PLANT-B"]))
    actors["peer"] = store.authenticate(
        store.provision_user("same-plant-peer", "analyst", ["PLANT-A"]))
    service = DiagnosisService(store, None, None, None)
    env = SimpleNamespace(store=store, actors=actors, tokens=tokens, service=service)
    env.provider = SyntheticNarrativeProvider()
    env.reports = ReportService(store, service, env.provider)
    env.seed = lambda **kwargs: seed_synthetic_task(env, **kwargs)
    env.task_id = env.seed()
    return env


def seed_synthetic_task(env, *, task_id="synthetic-task", status="submitted", role="analyst",
                        complaint="SYNTHETIC TEST complaint; not a production observation.",
                        details=None):
    actor = env.actors[role]
    request = {"request_id": "request-" + task_id, "plant_id": "PLANT-A", "device_id": "DEVICE-A",
               "event_id": "SYNTHETIC-EVENT", "window": {"start_ms": 0, "end_ms": 30_000},
               "complaint": complaint}
    evidence = Evidence(
        evidence_id="evidence-" + task_id, task_id=task_id, actor_id=actor.actor_id,
        **{key: request[key] for key in ("plant_id", "device_id", "event_id", "window")},
        tree_version="synthetic-tree-v1", tree_hash=digest("synthetic-tree"),
        data_hash=digest("synthetic-data"), condition_id="synthetic-condition", state="supported",
        source="signal", details=details or {"computed": 7.0, "n": 30, "unit": "A"},
    ).model_dump(mode="json")
    evidence_ref = env.store.write_artifact("Evidence", evidence)
    submission = None
    if status == "submitted":
        value = {"submission_id": "diagnosis-" + task_id, "task_id": task_id,
                 "node_id": "synthetic-leaf", "label": "SYNTHETIC TEST outcome",
                 "path": ["synthetic-root", "synthetic-leaf"],
                 "evidence_ids": [evidence["evidence_id"]],
                 "tree_version": evidence["tree_version"], "tree_hash": evidence["tree_hash"],
                 "data_hash": evidence["data_hash"], "request": request, "actor_id": actor.actor_id,
                 "conclusion_level": "supported_under_declared_synthetic_rules",
                 "limitations": ["SYNTHETIC TEST seeded terminal record, not a diagnosis run."]}
        value["content_hash"] = digest(value)
        submission = env.store.write_artifact("FrozenDiagnosis", value)
    state = {"task_id": task_id, "version": 7, "status": status, "next_node": "end",
             "actor": actor.model_dump(mode="json"), "request": request,
             "tree_version": evidence["tree_version"], "tree_hash": evidence["tree_hash"],
             "data_hash": evidence["data_hash"], "provider": "synthetic-fake-no-model",
             "calls": 3, "tools": 5, "tokens": 300, "elapsed_s": 1.0,
             "evidence_ids": [evidence["evidence_id"]], "submission": submission,
             "error": {"code": "synthetic_stop", "message": "SYNTHETIC TEST stopped."}
                      if status == "escalated" else None,
             "compact_refs": [], "limits": {"calls": 20, "tools": 32, "tokens": 24000},
             "pending_human": None, "pending_action": None, "pending_call_id": None,
             "candidates": ["synthetic-leaf"], "expanded_nodes": ["synthetic-root"]}
    with env.store.transaction() as db:
        db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,NULL,NULL)", (
            task_id, actor.actor_id, request["request_id"], digest(request), canonical(state), 7))
        db.execute("INSERT INTO evidence VALUES(?,?,?,?,?)", (
            evidence["evidence_id"], task_id, evidence["condition_id"], canonical(evidence_ref), 1.0))
    return task_id


def shared_state(env):
    """All shared diagnosis tables, not just status/hash, must remain byte-for-byte unchanged."""
    with env.store.connect() as db:
        return {table: [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("tasks", "evidence", "calls", "receipts", "audit")}


def generate(env, action_id="generate-one", task_id=None, role="analyst"):
    return env.reports.generate(task_id or env.task_id, action_id, env.actors[role])


def review(env, report, action_id="review-one", decision="approve", note="SYNTHETIC human review.",
           role="analyst", content_hash=None):
    return env.reports.review(report["report_id"], action_id, content_hash or report["content_hash"],
                              decision, note, env.actors[role])


def expect_code(code, callable_):
    with pytest.raises(DomainError) as caught:
        callable_()
    assert caught.value.code == code
    return caught.value


def test_synthetic_report_reads_actual_service_and_leaves_all_diagnosis_state_immutable(synthetic_env):
    env = synthetic_env
    before = shared_state(env)
    view = env.service.get(env.task_id, env.actors["analyst"])
    diagnosis_hash = view["submission"]["content_hash"]
    report = generate(env)
    assert report["status"] == "completed"
    assert report["version"] == 1
    assert report["authoritative"]["submission"] == view["submission"]
    assert report["authoritative"]["evidence"] == view["evidence"]
    assert report["content"]["authoritative"] == report["authoritative"]
    assert report["authoritative_hash"] == digest(report["authoritative"])
    assert report["content_hash"] == digest(report["content"])
    assert env.store.read_artifact(report["content_ref"]) == report["content"]
    assert env.provider.inputs == [report["authoritative"]]
    assert report["content"]["model_draft"]["synthetic"] is True
    signed = review(env, report)
    assert signed["content_hash"] == report["content_hash"]
    assert env.service.get(env.task_id, env.actors["analyst"])["submission"]["content_hash"] == diagnosis_hash
    assert shared_state(env) == before


def test_synthetic_fabrication_and_provider_input_mutation_never_gain_authority(synthetic_env):
    env = synthetic_env
    before = shared_state(env)

    def fabricate(diagnosis):
        diagnosis["submission"].update(node_id="fabricated", content_hash="0" * 64)
        diagnosis["evidence"][0]["details"]["computed"] = -999999
        diagnosis["status"] = "cancelled"
        return {"explanation": "Fabricated measurement 99999, human approved, turn off the device.",
                "limitations": ["Fabricated model authority claim."]}

    env.provider.callback = fabricate
    report = generate(env)
    draft = report["content"]["model_draft"]
    assert "99999" in draft["narrative"]["explanation"]
    assert draft["authority"] == "untrusted_model_draft"
    assert draft["review_status"] == "unreviewed"
    assert report["authoritative"]["submission"]["node_id"] == "synthetic-leaf"
    assert report["authoritative"]["evidence"][0]["details"]["computed"] == 7.0
    assert report["authoritative"]["status"] == "submitted"
    assert report["content"]["safety"]["model_draft_is_authoritative"] is False
    assert report["content"]["safety"]["not_actuation"] is True
    assert report["signoff"] is None
    review(env, report)
    current = env.reports.get(report["report_id"], env.actors["analyst"])
    assert current["content"] == report["content"]
    assert current["content"]["model_draft"]["review_status"] == "unreviewed"
    assert current["signoff"]["not_actuation"] is True
    assert shared_state(env) == before


@pytest.mark.parametrize("reply", [
    {"explanation": "SYNTHETIC fabricated elevation.", "limitations": ["test"], "approved": True},
    {"explanation": "SYNTHETIC fabricated tool.", "limitations": ["test"],
     "tool": "submit_diagnosis"},
    {"explanation": "", "limitations": ["test"]},
    {"explanation": "test", "limitations": []},
    {"explanation": 42, "limitations": ["test"]},
    {"explanation": "test", "limitations": [False]},
    {"explanation": "test", "limitations": ["x" * 20_001]},
    None, [], "not JSON",
])
def test_synthetic_invalid_narrative_is_known_failure_not_a_diagnosis_failure(synthetic_env, reply):
    env = synthetic_env
    before = shared_state(env)
    env.provider.callback = lambda _: reply
    report = generate(env)
    assert report["status"] == "failed"
    assert report["error"]["code"] == "provider_invalid"
    assert report["content"]["model_draft"] is None
    assert report["authoritative"]["status"] == "submitted"
    assert shared_state(env) == before
    assert generate(env) == report
    assert env.provider.calls == 1


@pytest.mark.parametrize(("error", "status", "code"), [
    (DomainError("provider_invalid", "SECRET-MODEL-RESPONSE", 502), "failed", "provider_invalid"),
    (DomainError("provider_unavailable", "SECRET-ENDPOINT", 503), "failed", "provider_unavailable"),
    (DomainError("provider_http", "SECRET-HTTP-BODY", 503), "failed", "provider_http"),
    (DomainError("provider_size", "SECRET-OUTPUT", 502), "failed", "provider_size"),
    (DomainError("provider_unknown", "SECRET-TRANSPORT", 503), "unknown", "provider_unknown"),
    (DomainError("model_unknown", "SECRET-TRANSPORT", 503), "unknown", "provider_unknown"),
    (TimeoutError("SECRET-AFTER-SEND"), "unknown", "provider_unknown"),
    (RuntimeError("SECRET-AFTER-SEND"), "unknown", "provider_unknown"),
    (DomainError("SECRET-ERROR-CODE", "SECRET-MESSAGE"), "unknown", "provider_unknown"),
])
def test_synthetic_known_failed_vs_uncertain_reports_are_persisted_without_retry(
        synthetic_env, error, status, code):
    env = synthetic_env
    before = shared_state(env)

    def fail(_):
        raise error

    env.provider.callback = fail
    report = generate(env)
    assert report["status"] == status
    assert report["error"]["code"] == code
    assert "SECRET" not in canonical(report)
    assert report["content"]["model_draft"] is None
    assert report["authoritative"]["submission"] is not None
    assert report["content_hash"] == digest(report["content"])
    assert shared_state(env) == before
    assert generate(env) == report
    reopened = ReportService(Store(env.store.root), env.service, env.provider)
    assert reopened.generate(env.task_id, "generate-one", env.actors["analyst"]) == report
    assert env.provider.calls == 1
    # Human signoff reviews the available deterministic content, not a missing model response.
    review(env, report, decision="reject")
    assert shared_state(env) == before


@pytest.mark.parametrize("status", ["escalated", "cancelled"])
def test_synthetic_non_submitted_terminal_report_never_calls_provider(synthetic_env, status):
    env = synthetic_env
    env.task_id = env.seed(task_id="synthetic-" + status, status=status)
    before = shared_state(env)
    report = generate(env)
    assert report["status"] == "completed"
    assert report["authoritative"]["status"] == status
    assert report["authoritative"]["submission"] is None
    assert report["content"]["model_draft"] is None
    assert env.provider.calls == 0
    assert shared_state(env) == before


def test_synthetic_provider_is_optional_and_narrative_model_is_validated(synthetic_env):
    env = synthetic_env
    reports = ReportService(env.store, env.service, None)
    report = reports.generate(env.task_id, "without-model", env.actors["analyst"])
    assert report["status"] == "completed"
    assert report["content"]["model_draft"] is None
    env.provider.callback = lambda _: Narrative(explanation="SYNTHETIC TEST draft", limitations=["test"])
    assert generate(env)["status"] == "completed"


@pytest.mark.parametrize("status", ["ready", "running", "waiting_human", "blocked"])
def test_synthetic_nonterminal_tasks_cannot_create_report_jobs(synthetic_env, status):
    env = synthetic_env
    env.task_id = env.seed(task_id="synthetic-" + status, status=status)
    before = shared_state(env)
    expect_code("report_not_terminal", lambda: generate(env))
    assert env.reports.list_for_task(env.task_id, env.actors["analyst"]) == []
    assert env.provider.calls == 0
    assert shared_state(env) == before


def test_synthetic_deterministic_content_versions_and_receipt_conflicts(synthetic_env):
    env = synthetic_env
    report = generate(env)
    assert generate(env) == report
    assert env.provider.calls == 1
    signed = review(env, report)
    # Generating again replays the original receipt, not an updated signoff.
    assert generate(env) == report
    assert report["signoff"] is None
    assert env.reports.get(report["report_id"], env.actors["analyst"])["signoff"] == signed
    second = generate(env, action_id="generate-two")
    assert second["version"] == 2
    assert second["report_id"] != report["report_id"]
    assert second["authoritative_hash"] == report["authoritative_hash"]
    assert second["content_hash"] == report["content_hash"]
    assert second["content_ref"] == report["content_ref"]
    assert env.provider.calls == 2
    other_task = env.seed(task_id="synthetic-other")
    expect_code("idempotency_conflict", lambda: generate(env, task_id=other_task))
    expect_code("idempotency_conflict", lambda: review(env, second, action_id="generate-one"))
    expect_code("idempotency_conflict", lambda: generate(env, action_id="review-one"))
    listing = env.reports.list_for_task(env.task_id, env.actors["analyst"])
    assert [item["version"] for item in listing] == [1, 2]
    assert listing[0]["signoff"] == signed
    assert listing[1]["signoff"] is None


def test_synthetic_started_job_commits_before_model_and_concurrent_duplicate_never_retries(synthetic_env):
    env = synthetic_env
    before = shared_state(env)
    entered, release = Event(), Event()

    def hold(_):
        with env.store.connect() as db:
            row = db.execute("SELECT * FROM report_jobs").fetchone()
            assert row["status"] == "started"
            assert row["content_hash"] is None
            assert db.execute("SELECT result_json FROM report_actions").fetchone()[0] is None
        entered.set()
        assert release.wait(5), "SYNTHETIC TEST release timed out"
        return {"explanation": "SYNTHETIC TEST draft", "limitations": ["test"]}

    env.provider.callback = hold
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(generate, env)
        try:
            assert entered.wait(5), "SYNTHETIC TEST provider did not start"
            duplicate = pool.submit(generate, env).result(timeout=5)
            assert duplicate["status"] == "started"
            assert duplicate["content"] is None
            assert duplicate["content_hash"] is None
            assert env.provider.calls == 1
            expect_code("report_incomplete", lambda: review(env, duplicate, content_hash="a" * 64))
            assert env.reports.get(duplicate["report_id"], env.actors["analyst"])["status"] == "started"
        finally:
            release.set()
        complete = first.result(timeout=5)
    assert complete["report_id"] == duplicate["report_id"]
    assert complete["status"] == "completed"
    assert generate(env) == complete
    assert env.provider.calls == 1
    assert shared_state(env) == before


def test_synthetic_crash_leaves_started_action_unretriable_across_service_restart(synthetic_env):
    env = synthetic_env
    before = shared_state(env)

    def crash(_):
        raise SystemExit("SYNTHETIC simulated process crash after dispatch")

    env.provider.callback = crash
    with pytest.raises(SystemExit):
        generate(env)
    reopened = ReportService(Store(env.store.root), env.service, env.provider)
    result = reopened.generate(env.task_id, "generate-one", env.actors["analyst"])
    assert result["status"] == "started"
    assert result["content_hash"] is None
    assert env.provider.calls == 1
    assert shared_state(env) == before


def test_synthetic_crash_after_model_reply_never_dispatches_that_action_again(synthetic_env, monkeypatch):
    env = synthetic_env
    before = shared_state(env)
    original = env.store.write_artifact

    def fail_final_content(kind, value):
        if kind == "ReportContent":
            raise RuntimeError("SYNTHETIC TEST disk failure after model reply")
        return original(kind, value)

    monkeypatch.setattr(env.store, "write_artifact", fail_final_content)
    with pytest.raises(RuntimeError, match="disk failure"):
        generate(env)
    assert env.provider.calls == 1
    monkeypatch.setattr(env.store, "write_artifact", original)
    result = generate(env)
    assert result["status"] == "started"
    assert result["content"] is None
    assert env.provider.calls == 1
    assert shared_state(env) == before


def test_synthetic_final_source_check_rejects_changed_diagnosis_without_retaining_draft(synthetic_env):
    env = synthetic_env

    def change_source(_):
        with env.store.transaction() as db:
            state = json.loads(db.execute("SELECT state_json FROM tasks WHERE task_id=?",
                                          (env.task_id,)).fetchone()[0])
            # Deliberately simulate an out-of-band change, NOT a reporting side effect.
            state["status"], state["submission"] = "cancelled", None
            db.execute("UPDATE tasks SET state_json=? WHERE task_id=?", (canonical(state), env.task_id))
        return {"explanation": "UNPERSISTED-MODEL-DRAFT", "limitations": ["test"]}

    env.provider.callback = change_source
    expect_code("report_source_changed", lambda: generate(env))
    assert generate(env)["status"] == "started"
    assert env.provider.calls == 1
    assert all(b"UNPERSISTED-MODEL-DRAFT" not in path.read_bytes()
               for path in env.store.artifacts.iterdir())


@pytest.mark.parametrize("role", ["peer", "outsider", "admin"])
def test_synthetic_wrong_owner_same_or_other_plant_cannot_access_reports(synthetic_env, role):
    env = synthetic_env
    report = generate(env)
    actor = env.actors[role]
    expect_code("task_not_found", lambda: env.reports.generate(env.task_id, "generate-one", actor))
    expect_code("task_not_found", lambda: env.reports.list_for_task(env.task_id, actor))
    expect_code("report_not_found", lambda: env.reports.get(report["report_id"], actor))
    expect_code("report_not_found", lambda: review(env, report, role=role))
    assert env.provider.calls == 1


def test_synthetic_operator_owner_may_generate_but_may_not_signoff(synthetic_env):
    env = synthetic_env
    env.task_id = env.seed(task_id="synthetic-operator-task", role="operator")
    report = generate(env, role="operator")
    expect_code("forbidden", lambda: review(env, report, role="operator"))
    assert env.reports.get(report["report_id"], env.actors["operator"])["signoff"] is None


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_synthetic_owner_admin_may_make_either_signoff_without_changing_diagnosis(synthetic_env, decision):
    env = synthetic_env
    env.task_id = env.seed(task_id="synthetic-admin-task", role="admin")
    report = generate(env, role="admin")
    before = shared_state(env)
    assert review(env, report, decision=decision, role="admin")["decision"] == decision
    assert shared_state(env) == before


def test_synthetic_current_actor_checked_before_existing_generate_and_review_receipts(synthetic_env):
    env = synthetic_env
    report = generate(env)
    review(env, report)
    env.store.set_user("analyst", env.actors["admin"], enabled=True, plants=["PLANT-A"], role="operator")
    expect_code("identity_changed", lambda: generate(env))
    expect_code("identity_changed", lambda: review(env, report))
    expect_code("identity_changed", lambda: env.reports.get(report["report_id"], env.actors["analyst"]))
    expect_code("identity_changed", lambda: env.reports.list_for_task(env.task_id, env.actors["analyst"]))
    refreshed = env.store.authenticate(env.tokens["analyst"])
    assert env.reports.generate(env.task_id, "generate-one", refreshed) == report
    expect_code("forbidden", lambda: env.reports.review(
        report["report_id"], "review-one", report["content_hash"], "approve", "SYNTHETIC human review.", refreshed))


def test_synthetic_wrong_current_plant_and_disabled_actor_denied_existing_receipts(synthetic_env):
    env = synthetic_env
    report = generate(env)
    # Synthetic ACL mutation simulates an external identity administration operation.
    with env.store.transaction() as db:
        db.execute("UPDATE users SET plants=?,acl_version=acl_version+1 WHERE actor_id='analyst'",
                   (canonical(["PLANT-B"]),))
    other_plant_actor = env.store.authenticate(env.tokens["analyst"])
    expect_code("forbidden", lambda: env.reports.get(report["report_id"], other_plant_actor))
    expect_code("forbidden", lambda: env.reports.generate(env.task_id, "generate-one", other_plant_actor))
    expect_code("forbidden", lambda: env.reports.list_for_task(env.task_id, other_plant_actor))
    expect_code("forbidden", lambda: env.reports.review(
        report["report_id"], "review-one", report["content_hash"], "approve", "SYNTHETIC", other_plant_actor))
    with env.store.transaction() as db:
        db.execute("UPDATE users SET enabled=0 WHERE actor_id='analyst'")
    expect_code("identity_changed", lambda: env.reports.get(report["report_id"], other_plant_actor))


@pytest.mark.parametrize("outcome", ["valid", "failure", "unknown"])
def test_synthetic_revocation_during_provider_never_persists_or_returns_model_response(synthetic_env, outcome):
    env = synthetic_env
    before = shared_state(env)

    def revoke(_):
        with env.store.transaction() as db:
            db.execute("UPDATE users SET enabled=0,acl_version=acl_version+1 WHERE actor_id='analyst'")
        if outcome == "failure":
            raise DomainError("provider_invalid", "SENSITIVE-OUTPUT")
        if outcome == "unknown":
            raise TimeoutError("SENSITIVE-OUTPUT")
        return {"explanation": "SENSITIVE-OUTPUT", "limitations": ["test"]}

    env.provider.callback = revoke
    expect_code("identity_changed", lambda: generate(env))
    with env.store.connect() as db:
        job = db.execute("SELECT * FROM report_jobs").fetchone()
        assert job["status"] == "started"
        assert job["content_ref"] is None
        assert job["error_json"] is None
    assert all(b"SENSITIVE-OUTPUT" not in path.read_bytes() for path in env.store.artifacts.iterdir())
    assert shared_state(env) == before


def test_synthetic_current_ownership_rechecked_after_provider(synthetic_env):
    env = synthetic_env

    def transfer(_):
        with env.store.transaction() as db:
            db.execute("UPDATE tasks SET owner='same-plant-peer' WHERE task_id=?", (env.task_id,))
        return {"explanation": "SENSITIVE-OUTPUT", "limitations": ["test"]}

    env.provider.callback = transfer
    expect_code("task_not_found", lambda: generate(env))
    with env.store.connect() as db:
        row = db.execute("SELECT report_id,status FROM report_jobs").fetchone()
    assert row["status"] == "started"
    expect_code("task_not_found", lambda: env.reports.get(row["report_id"], env.actors["analyst"]))
    expect_code("report_not_found", lambda: env.reports.get(row["report_id"], env.actors["peer"]))
    assert env.reports.list_for_task(env.task_id, env.actors["peer"]) == []
    assert all(b"SENSITIVE-OUTPUT" not in path.read_bytes() for path in env.store.artifacts.iterdir())


def test_synthetic_stale_hash_and_invalid_reviews_do_not_consume_action_or_signoff(synthetic_env):
    env = synthetic_env
    report = generate(env)
    expect_code("stale_report_hash", lambda: review(env, report, content_hash="a" * 64))
    for args in ({"decision": "execute"}, {"note": " "}, {"note": "x" * 2001},
                 {"content_hash": "invalid"}):
        expect_code("invalid_report_review", lambda args=args: review(env, report, **args))
    with env.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM report_signoffs").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM report_actions").fetchone()[0] == 1
    assert review(env, report)["decision"] == "approve"


def test_synthetic_signoff_is_separate_one_time_and_immutable_even_for_owner(synthetic_env):
    env = synthetic_env
    before = shared_state(env)
    report = generate(env)
    content_bytes = (env.store.artifacts / (report["content_ref"]["sha256"] + ".json")).read_bytes()
    signed = review(env, report)
    assert review(env, report) == signed
    expect_code("idempotency_conflict", lambda: review(env, report, decision="reject"))
    expect_code("idempotency_conflict", lambda: review(env, report, note="Changed note"))
    expect_code("report_already_reviewed", lambda: review(env, report, action_id="another-review"))
    current = env.reports.get(report["report_id"], env.actors["analyst"])
    assert current["content_hash"] == report["content_hash"]
    assert current["content"] == report["content"]
    assert current["signoff"] == signed
    assert (env.store.artifacts / (report["content_ref"]["sha256"] + ".json")).read_bytes() == content_bytes
    assert shared_state(env) == before
    for statement in (
        "UPDATE report_signoffs SET content_hash='changed'", "DELETE FROM report_signoffs",
        "UPDATE report_jobs SET content_hash='changed'", "DELETE FROM report_jobs",
        "UPDATE report_actions SET result_json='{}'", "DELETE FROM report_actions",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with env.store.transaction() as db:
                db.execute(statement)


def test_synthetic_simultaneous_review_has_only_one_immutable_signoff(synthetic_env):
    env = synthetic_env
    report = generate(env)

    def attempt(action_id):
        try:
            return review(env, report, action_id=action_id)
        except DomainError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["review-a", "review-b"]))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert results.count("report_already_reviewed") == 1
    with env.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM report_signoffs").fetchone()[0] == 1


def test_synthetic_tampered_artifact_is_not_hidden_by_cached_receipt(synthetic_env):
    env = synthetic_env
    report = generate(env)
    path = env.store.artifacts / (report["content_ref"]["sha256"] + ".json")
    path.write_text('{"tampered":true}')
    expect_code("artifact_integrity", lambda: generate(env))
    expect_code("artifact_integrity", lambda: env.reports.get(report["report_id"], env.actors["analyst"]))
    expect_code("artifact_integrity", lambda: review(env, report))


def test_synthetic_invalid_diagnosis_hash_rejected_before_model(synthetic_env):
    env = synthetic_env
    with env.store.transaction() as db:
        state = json.loads(db.execute("SELECT state_json FROM tasks WHERE task_id=?", (env.task_id,)).fetchone()[0])
        value = env.store.read_artifact(state["submission"])
        value["node_id"] = "tampered-without-rehashing"
        state["submission"] = env.store.write_artifact("FrozenDiagnosis", value)
        db.execute("UPDATE tasks SET state_json=? WHERE task_id=?", (canonical(state), env.task_id))
    expect_code("report_source_integrity", lambda: generate(env))
    assert env.provider.calls == 0
    assert env.reports.list_for_task(env.task_id, env.actors["analyst"]) == []


def test_synthetic_tokens_credentials_and_raw_errors_excluded_from_all_report_artifacts(synthetic_env):
    env = synthetic_env
    env.task_id = env.seed(
        task_id="synthetic-redaction",
        complaint=("SYNTHETIC TEST " + "to" + "ken=" + "COMPLAINT-SECRET "
                   + "pass" + "word=" + "PASSWORD-SECRET"),
        details={"computed": 7.0, "to" + "ken": "DETAIL-" + "TOKEN",
                 "creden" + "tials": "DETAIL-" + "CREDENTIAL",
                 "api_" + "key": "DETAIL-" + "API-KEY",
                 "observation_note": "Bear" + "er " + "EVIDENCE-SECRET"})
    before = set(env.store.artifacts.iterdir())
    env.provider.callback = lambda _: {
        "explanation": ("SYNTHETIC TEST " + "Author" + "ization: " + "Bear" + "er "
                        + "MODEL-SECRET; " + "to" + "ken=" + "MODEL-TOKEN"),
        "limitations": ["pass" + "word=" + "MODEL-PASSWORD",
                        "http://" + "name:URL-SECRET" + "@127.0.0.1",
                        "api_" + "key=" + "'MODEL-API-KEY'"]}
    report = generate(env)
    signed = review(env, report, note="SYNTHETIC TEST " + "pass" + "word=" + "REVIEW-PASSWORD")
    text = canonical({"report": report, "signoff": signed, "input": env.provider.inputs})
    for path in set(env.store.artifacts.iterdir()) - before:
        text += path.read_text()
    with env.store.connect() as db:
        for table in ("report_jobs", "report_actions", "report_signoffs"):
            text += canonical([dict(row) for row in db.execute(f"SELECT * FROM {table}")])
    assert "[REDACTED]" in text
    assert '"tokens"' not in text
    for secret in ("COMPLAINT-SECRET", "PASSWORD-SECRET", "DETAIL-TOKEN", "DETAIL-CREDENTIAL",
                   "DETAIL-API-KEY", "EVIDENCE-SECRET", "MODEL-SECRET", "MODEL-TOKEN",
                   "MODEL-PASSWORD", "URL-SECRET", "MODEL-API-KEY", "REVIEW-PASSWORD",
                   "NEVER-COPY-PROVIDER-CREDENTIAL", *env.tokens.values()):
        assert secret not in text


@pytest.mark.parametrize("action", [None, "", "../bad", "space bad", "x" * 81])
def test_synthetic_invalid_action_ids_rejected_without_model_or_report(synthetic_env, action):
    env = synthetic_env
    expect_code("invalid_report_request", lambda: generate(env, action_id=action))
    assert env.provider.calls == 0
    assert env.reports.list_for_task(env.task_id, env.actors["analyst"]) == []
