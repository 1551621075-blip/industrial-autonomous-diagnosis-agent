"""Durable recovery/fencing tests; injected faults never substitute for real persistence."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_service import (
    ScriptedProvider,
    TestPublishedKnowledge,
    all_signals,
    make_lab,
    prepare,
    reply,
)

import industrial_diagnosis.service as service_module
from industrial_diagnosis.provider import SyntheticProvider
from industrial_diagnosis.schemas import DomainError, HumanAnswer, ReconcileRequest
from industrial_diagnosis.service import DiagnosisService
from industrial_diagnosis.storage import Store, now
from industrial_diagnosis.telemetry import TelemetryStore


class InjectedCrash(BaseException):
    """Model an abrupt process interruption, not a recoverable domain exception."""


def restart(lab, provider=None, **limits):
    """Reopen the actual database/artifacts with a new service and graph instance."""
    store = Store(lab.store.root)
    telemetry = TelemetryStore(store)
    knowledge = TestPublishedKnowledge(store, lab.knowledge.published["draft"])
    knowledge.published = copy.deepcopy(lab.knowledge.published)
    service = DiagnosisService(store, knowledge, telemetry,
                               provider if provider is not None else lab.service.provider, **limits)
    lab.store, lab.telemetry, lab.knowledge, lab.service = store, telemetry, knowledge, service
    return service


def reconcile_no_result(lab, view, action_id="reconcile-no-result"):
    return lab.service.reconcile(view["task_id"], ReconcileRequest(
        action_id=action_id, call_id=view["pending_call_id"], no_result=True, tokens=0,
        note="TEST owner-admin attests no result and exactly zero tokens consumed.",
    ), lab.actor)


def test_unknown_model_outcome_never_automatically_resends_across_restart(tmp_path):
    provider = ScriptedProvider([DomainError("provider_unknown", "Ambiguous inference transport", 503)])
    lab = make_lab(tmp_path, provider)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "blocked" and view["error"]["code"] == "provider_unknown"
    call_id = view["pending_call_id"]
    assert call_id and len(provider.contexts) == view["calls"] == 1
    assert lab.service.calls(task_id, lab.actor)[0]["status"] == "unknown"
    replacement = ScriptedProvider([reply("intent_classify")])
    restart(lab, replacement)
    for _ in range(3):
        view = lab.service.run(task_id, lab.actor)
        assert view["status"] == "blocked" and view["error"]["code"] == "model_unknown"
        assert view["calls"] == 1 and view["pending_call_id"] == call_id
    assert replacement.contexts == []


def test_reconciliation_requires_current_owner_admin_and_is_idempotent(tmp_path):
    provider = ScriptedProvider([DomainError("provider_unknown", "Uncertain inference")])
    lab = make_lab(tmp_path, provider, max_calls=1)
    view = lab.service.run(lab.create()["task_id"], lab.actor)
    command = ReconcileRequest(action_id="reconcile-once", call_id=view["pending_call_id"],
        reply=reply("intent_classify").action, tokens=71,
        note="TEST owner-admin recovered this exact action and token usage.")
    with pytest.raises(DomainError) as error:
        lab.service.reconcile(view["task_id"], command, lab.admin)
    assert error.value.code == "task_not_found"
    result = lab.service.reconcile(view["task_id"], command, lab.actor)
    assert result["status"] == "ready" and result["calls"] == 1 and result["tokens"] == 71
    assert result["pending_call_id"] is None
    assert lab.service.reconcile(view["task_id"], command, lab.actor) == result
    with pytest.raises(DomainError) as error:
        lab.service.reconcile(view["task_id"], command.model_copy(update={"tokens": 72}), lab.actor)
    assert error.value.code == "idempotency_conflict"
    result = lab.service.run(view["task_id"], lab.actor)
    assert result["status"] == "escalated" and result["error"]["code"] == "budget_exhausted"
    assert result["calls"] == result["tools"] == len(provider.contexts) == 1
    assert result["tokens"] == 71
    assert lab.service.calls(view["task_id"], lab.actor)[0]["status"] == "reconciled"
    assert lab.state(view["task_id"])["intent"]["kind"] == "conveyor_diagnosis"


def test_operator_owner_cannot_reconcile_even_own_unknown_call(tmp_path):
    lab = make_lab(tmp_path, ScriptedProvider([DomainError("provider_unknown", "Uncertain")]), role="operator")
    view = lab.service.run(lab.create()["task_id"], lab.actor)
    with pytest.raises(DomainError) as error:
        reconcile_no_result(lab, view)
    assert error.value.code == "forbidden"
    assert lab.service.get(view["task_id"], lab.actor)["pending_call_id"] == view["pending_call_id"]


def test_confirmed_no_result_retains_call_budget_and_audits_attestation(tmp_path):
    lab = make_lab(tmp_path, ScriptedProvider([DomainError("provider_unknown", "Uncertain")]), max_calls=1)
    view = lab.service.run(lab.create()["task_id"], lab.actor)
    reconciled = reconcile_no_result(lab, view)
    assert reconciled["calls"] == 1 and reconciled["tokens"] == reconciled["tools"] == 0
    assert reconciled["pending_call_id"] is None
    result = lab.service.run(view["task_id"], lab.actor)
    assert result["status"] == "escalated" and result["calls"] == 1
    assert lab.service.calls(view["task_id"], lab.actor)[0]["status"] == "confirmed_no_result"
    event = next(row for row in lab.service.audit(view["task_id"], lab.actor)
                 if row["event"] == "model_reconciled")
    assert json.loads(event["detail_json"])["human_attestation_not_external_proof"] is True


def test_crash_after_call_reservation_requires_reconcile_without_resend(tmp_path):
    provider = ScriptedProvider([InjectedCrash("exit after request reservation")])
    lab = make_lab(tmp_path, provider, max_calls=1)
    task_id = lab.create()["task_id"]
    with pytest.raises(InjectedCrash):
        lab.service.run(task_id, lab.actor)
    state = lab.state(task_id)
    assert state["calls"] == 1 and state["pending_call_id"] is not None
    assert lab.service.calls(task_id, lab.actor)[0]["status"] == "started"
    replacement = ScriptedProvider([reply("intent_classify")])
    restart(lab, replacement)
    view = lab.service.run(task_id, lab.actor)
    assert view["error"]["code"] == "model_unknown" and not replacement.contexts
    reconcile_no_result(lab, view)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["calls"] == 1
    assert not replacement.contexts


@pytest.mark.parametrize("boundary", ["reply_committed", "tool_committed"])
def test_committed_reply_or_tool_survives_crash_without_duplicate_execution(tmp_path, monkeypatch, boundary):
    provider = ScriptedProvider([reply("intent_classify", tokens=93)])
    lab = make_lab(tmp_path, provider, max_calls=1)
    task_id = lab.create()["task_id"]
    save = lab.service._save
    tripped = False

    def crash_at_boundary(state):
        nonlocal tripped
        matches = (state["pending_action"] is not None if boundary == "reply_committed"
                   else state["intent"] is not None and state["pending_action"] is None)
        if matches and not tripped:
            tripped = True
            raise InjectedCrash(boundary)
        return save(state)

    monkeypatch.setattr(lab.service, "_save", crash_at_boundary)
    with pytest.raises(InjectedCrash):
        lab.service.run(task_id, lab.actor)
    durable = lab.state(task_id)
    assert durable["calls"] == 1 and durable["tokens"] == 93
    assert durable["pending_call_id"] is None
    assert durable["tools"] == (0 if boundary == "reply_committed" else 1)
    replacement = ScriptedProvider([reply("intent_classify")])
    restart(lab, replacement, max_calls=20)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == "budget_exhausted"
    assert view["calls"] == view["tools"] == 1 and view["tokens"] == 93
    assert view["limits"]["calls"] == 1 and not replacement.contexts
    completed = [row for row in lab.service.audit(task_id, lab.actor) if row["event"] == "tool_completed"]
    assert len(completed) == 1


def test_crash_before_model_call_keeps_zero_calls_and_can_resume(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, ScriptedProvider([reply("intent_classify")]), max_calls=1)
    task_id = lab.create()["task_id"]
    monkeypatch.setattr(lab.service, "_agent", lambda state: (_ for _ in ()).throw(InjectedCrash()))
    with pytest.raises(InjectedCrash):
        lab.service.run(task_id, lab.actor)
    assert lab.state(task_id)["calls"] == 0
    assert lab.service.calls(task_id, lab.actor) == []
    replacement = ScriptedProvider([reply("intent_classify")])
    restart(lab, replacement)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["calls"] == view["tools"] == 1
    assert len(replacement.contexts) == 1


def test_active_lease_rejects_concurrent_worker_and_cancel(tmp_path):
    lab = make_lab(tmp_path)
    task_id = lab.create()["task_id"]
    with lab.store.transaction() as db:
        db.execute("UPDATE tasks SET lease_owner=?,lease_until=? WHERE task_id=?",
                   ("worker-live", now() + 60, task_id))
    for operation in (lambda: lab.service.run(task_id, lab.actor),
                      lambda: lab.service.cancel(task_id, "cancel-busy", lab.actor)):
        with pytest.raises(DomainError) as error:
            operation()
        assert error.value.code == "task_busy"
    assert lab.state(task_id)["calls"] == 0


@pytest.mark.parametrize("fence", ["version", "worker"])
def test_stale_worker_is_fenced_before_state_write(tmp_path, fence):
    lab = make_lab(tmp_path)
    task_id = lab.create()["task_id"]
    stale = lab.state(task_id)
    stale["worker"] = None
    with lab.store.transaction() as db:
        if fence == "version":
            db.execute("UPDATE tasks SET version=version+1 WHERE task_id=?", (task_id,))
        else:
            db.execute("UPDATE tasks SET lease_owner='worker-new' WHERE task_id=?", (task_id,))
    before = lab.state(task_id)
    stale["status"] = "submitted"
    with pytest.raises(DomainError) as error:
        lab.service._save(stale)
    assert error.value.code == "worker_fenced"
    assert lab.state(task_id) == before


def test_takeover_during_inference_fences_old_reply_and_preserves_new_lease(tmp_path):
    provider = ScriptedProvider([])
    lab = make_lab(tmp_path, provider)
    task_id = lab.create()["task_id"]

    def takeover(context):
        with lab.store.transaction() as db:
            db.execute("UPDATE tasks SET version=version+1,lease_owner=?,lease_until=? WHERE task_id=?",
                       ("worker-replacement", now() + 100, task_id))
        return reply("intent_classify")

    provider.steps.append(takeover)
    with pytest.raises(DomainError) as error:
        lab.service.run(task_id, lab.actor)
    assert error.value.code == "worker_fenced"
    with lab.store.connect() as db:
        task = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        call = db.execute("SELECT * FROM calls WHERE task_id=?", (task_id,)).fetchone()
    assert task["lease_owner"] == "worker-replacement"
    assert call["status"] == "started" and call["response_ref"] is None
    state = json.loads(task["state_json"])
    assert state["pending_action"] is None and state["tools"] == state["tokens"] == 0


def test_expired_lease_allows_new_worker_and_charges_interrupted_elapsed_time(tmp_path):
    lab = make_lab(tmp_path, ScriptedProvider([reply("intent_classify")]), max_seconds=5)
    task_id = lab.create()["task_id"]
    lab.update_state(task_id, worker="worker-crashed", status="running", node_started_at=now() - 10)
    with lab.store.transaction() as db:
        db.execute("UPDATE tasks SET lease_owner=?,lease_until=? WHERE task_id=?",
                   ("worker-crashed", now() - 1, task_id))
    restart(lab, max_seconds=300)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == "budget_exhausted"
    assert view["elapsed_s"] >= 10 and view["calls"] == view["tools"] == 0
    assert view["limits"]["seconds"] == 5
    with lab.store.connect() as db:
        row = db.execute("SELECT lease_owner,lease_until FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    assert row["lease_owner"] is None and row["lease_until"] is None


@pytest.mark.parametrize(("counter", "limit_name", "maximum"), [
    ("calls", "max_calls", 2), ("tools", "max_tools", 2),
    ("tokens", "max_tokens", 100), ("elapsed_s", "max_seconds", 5),
])
def test_frozen_budgets_cannot_reset_or_expand_on_restart(tmp_path, counter, limit_name, maximum):
    provider = ScriptedProvider([reply("intent_classify")])
    lab = make_lab(tmp_path, provider, **{limit_name: maximum})
    task_id = lab.create()["task_id"]
    original = lab.service.get(task_id, lab.actor)["limits"]
    lab.update_state(task_id, **{counter: maximum})
    restart(lab, **{limit_name: maximum * 100})
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == "budget_exhausted"
    assert view["limits"] == original and view[counter] >= maximum
    assert not provider.contexts


def test_reply_over_token_budget_is_charged_but_cannot_execute_tool(tmp_path):
    lab = make_lab(tmp_path, ScriptedProvider([reply("intent_classify", tokens=101)]), max_tokens=100)
    view = lab.service.run(lab.create()["task_id"], lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == "budget_exhausted"
    assert view["tokens"] == 101 and view["calls"] == 1 and view["tools"] == 0
    assert lab.state(view["task_id"])["intent"] is None


def test_compaction_archives_observations_without_resetting_budgets_or_evidence(tmp_path):
    steps = [reply("intent_classify"), reply("expand_tree_node", {"node_id": "root"}),
             reply("time_series_verify", {"condition_ids": ["drive_enabled"]})]
    steps.extend(reply("kg_query_candidates", {"query": f"conveyor query {index}"}) for index in range(7))
    provider = ScriptedProvider(steps)
    lab = make_lab(tmp_path, provider, max_calls=10, max_tokens=24000)
    task_id = lab.create()["task_id"]
    initial = lab.state(task_id)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["calls"] == view["tools"] == 10
    assert view["tokens"] == 1280 and view["compaction_count"] >= 1
    state = lab.state(task_id)
    archives = [lab.store.read_artifact(ref) for ref in state["compact_refs"]]
    complete = [item for archive in archives for item in archive] + state["observations"]
    assert len(complete) == 10 and complete[0]["tool"] == "intent_classify"
    assert len(view["evidence"]) == 1 and view["evidence"][0]["condition_id"] == "drive_enabled"
    for key in ("tree_ref", "tree_hash", "tree_version", "data_ref", "data_hash", "limits"):
        assert state[key] == initial[key]
    projected = provider.contexts[-1]["evidence"]
    assert len(projected) == len(view["evidence"])
    for compact, full in zip(projected, view["evidence"], strict=True):
        for key in ("evidence_id", "condition_id", "state", "source"):
            assert compact[key] == full[key]
        assert compact["details"]["computed"] == full["details"]["computed"]
    assert provider.contexts[-1]["remaining"] == {"calls": 1, "tools": 23, "tokens": 22848}
    restart(lab, max_calls=100)
    assert lab.service.run(task_id, lab.actor) == view


def test_human_pause_resume_retains_usage_and_frozen_limits(tmp_path):
    lab = make_lab(tmp_path, SyntheticProvider())
    view = lab.service.run(lab.create("jam")["task_id"], lab.actor)
    assert view["status"] == "waiting_human"
    before = {key: view[key] for key in ("calls", "tools", "tokens", "elapsed_s", "limits")}
    restart(lab, SyntheticProvider(), max_calls=100, max_tools=200, max_tokens=100000, max_seconds=500)
    answered = lab.answer(view)
    for key in before:
        assert answered[key] == before[key]
    resumed = lab.service.run(view["task_id"], lab.actor)
    for key in ("calls", "tools", "tokens", "elapsed_s"):
        assert resumed[key] >= before[key]
    assert resumed["limits"] == before["limits"]


@pytest.mark.parametrize(("drift", "code"), [
    ("version", "knowledge_changed"), ("tree_hash", "knowledge_changed"),
    ("tree_content", "knowledge_integrity"), ("data_snapshot", "snapshot_changed"),
    ("data_hash", "snapshot_changed"), ("model", "snapshot_changed"),
    ("artifact_bytes", "artifact_integrity"),
])
def test_snapshot_drift_blocks_resume_before_model_or_tools(tmp_path, drift, code):
    provider = ScriptedProvider([reply("intent_classify")])
    lab = make_lab(tmp_path, provider)
    task_id = lab.create()["task_id"]
    state = lab.state(task_id)
    if drift == "version":
        lab.knowledge.published["version_id"] = "test-publication-v2"
    elif drift == "tree_hash":
        lab.knowledge.published["content_hash"] = "0" * 64
    elif drift == "tree_content":
        lab.knowledge.published["draft"]["title"] += " altered"
    elif drift == "data_snapshot":
        snapshot = lab.store.read_artifact(state["data_ref"])
        snapshot["title"] += " altered"
        lab.update_state(task_id, data_ref=lab.store.write_artifact("TelemetrySnapshot", snapshot))
    elif drift == "data_hash":
        lab.update_state(task_id, data_hash="0" * 64)
    elif drift == "model":
        provider.fingerprint = "service-scripted-TEST-v2"
    else:
        artifact = lab.store.artifacts / (state["data_ref"]["sha256"] + ".json")
        artifact.write_bytes(b"corrupted test artifact")
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "blocked" and view["error"]["code"] == code
    assert view["calls"] == view["tools"] == 0 and not provider.contexts
    assert view["submission"] is None


def test_snapshot_change_during_inference_leaves_reserved_call_unexecuted(tmp_path):
    provider = ScriptedProvider([])
    lab = make_lab(tmp_path, provider)

    def drift(context):
        lab.knowledge.published["version_id"] = "test-new-publication"
        return reply("intent_classify")

    provider.steps.append(drift)
    view = lab.service.run(lab.create()["task_id"], lab.actor)
    assert view["status"] == "blocked" and view["error"]["code"] == "knowledge_changed"
    assert view["calls"] == 1 and view["tools"] == 0 and view["pending_call_id"] is not None
    assert lab.state(view["task_id"])["pending_action"] is None
    assert lab.service.calls(view["task_id"], lab.actor)[0]["status"] in {"started", "unknown"}


@pytest.mark.parametrize("change", ["snapshot", "acl"])
def test_human_answer_revalidates_snapshot_and_current_actor(tmp_path, change):
    lab = make_lab(tmp_path)
    view = lab.service.run(lab.create("jam")["task_id"], lab.actor)
    assert view["status"] == "waiting_human"
    before = lab.state(view["task_id"])
    if change == "snapshot":
        lab.knowledge.published["version_id"] = "test-new-publication"
        code = "knowledge_changed"
    else:
        lab.store.set_user(lab.actor.actor_id, lab.admin, enabled=True, plants=["PLANT-B"], role="admin")
        code = "identity_changed"
    with pytest.raises(DomainError) as error:
        lab.answer(view)
    assert error.value.code == code
    assert lab.state(view["task_id"]) == before


def test_reconciliation_rechecks_revoked_owner_authorization(tmp_path):
    lab = make_lab(tmp_path, ScriptedProvider([DomainError("provider_unknown", "Uncertain")]))
    view = lab.service.run(lab.create()["task_id"], lab.actor)
    lab.store.set_user(lab.actor.actor_id, lab.admin, enabled=True, plants=["PLANT-B"], role="admin")
    with pytest.raises(DomainError) as error:
        reconcile_no_result(lab, view)
    assert error.value.code == "identity_changed"
    assert lab.state(view["task_id"])["pending_call_id"] == view["pending_call_id"]


def test_transaction_rollback_removes_partial_evidence_rows_and_task_references(tmp_path, monkeypatch):
    lab = make_lab(tmp_path)
    task_id = prepare(lab, check_all=False)
    original = lab.service._record_evidence
    recorded = []

    def fail_after_second(db, state, condition, outcome):
        result = original(db, state, condition, outcome)
        recorded.append(result["evidence_id"])
        if len(recorded) == 2:
            raise DomainError("test_transaction_abort", "Injected failure after second evidence insert")
        return result

    monkeypatch.setattr(lab.service, "_record_evidence", fail_after_second)
    before = lab.state(task_id)
    state = lab.tool(task_id, "time_series_verify", {"condition_ids": all_signals(lab)})
    assert state["observations"][-1]["error"] == "test_transaction_abort"
    assert state["evidence_ids"] == before["evidence_ids"] == []
    assert state["tools"] == before["tools"] + 1  # Rejected work is still budgeted.
    with lab.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence WHERE task_id=?", (task_id,)).fetchone()[0] == 0
    assert lab.service.get(task_id, lab.actor)["evidence"] == []
    monkeypatch.setattr(lab.service, "_record_evidence", original)
    lab.legal_tool(task_id, "time_series_verify", {"condition_ids": all_signals(lab)})
    evidence = lab.service.get(task_id, lab.actor)["evidence"]
    assert len(evidence) == len(all_signals(lab))
    assert not set(recorded) & {item["evidence_id"] for item in evidence}


def test_crash_in_evidence_transaction_restores_checkpoint_and_retries_tool_once(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, ScriptedProvider([]), max_calls=1)
    task_id = prepare(lab, check_all=False)
    lab.update_state(task_id, calls=1, pending_action=reply("time_series_verify", {
        "condition_ids": all_signals(lab),
    }).action.model_dump(mode="json"))
    original = lab.service._record_evidence

    def crash_after_insert(db, state, condition, outcome):
        original(db, state, condition, outcome)
        raise InjectedCrash("uncommitted evidence")

    monkeypatch.setattr(lab.service, "_record_evidence", crash_after_insert)
    with pytest.raises(InjectedCrash):
        lab.service.run(task_id, lab.actor)
    state = lab.state(task_id)
    assert state["pending_action"] is not None and state["evidence_ids"] == []
    with lab.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence WHERE task_id=?", (task_id,)).fetchone()[0] == 0
    restart(lab)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["calls"] == 1
    assert len(view["evidence"]) == len(all_signals(lab))
    assert len({item["evidence_id"] for item in view["evidence"]}) == len(all_signals(lab))


def test_answer_receipt_and_evidence_rollback_together_then_retry_once(tmp_path, monkeypatch):
    lab = make_lab(tmp_path)
    view = lab.service.run(lab.create("jam")["task_id"], lab.actor)
    assert view["status"] == "waiting_human"
    before = lab.state(view["task_id"])
    command = HumanAnswer(action_id="answer-atomic", question_id=view["pending_human"]["question_id"],
                          answer=True, observation_note="TEST safe observation from earlier inspection.")
    audit = lab.store.audit

    def crash_on_audit(db, actor_id, event, detail, task_id=None):
        if event == "human_observation":
            raise InjectedCrash("before answer transaction commit")
        return audit(db, actor_id, event, detail, task_id)

    monkeypatch.setattr(lab.store, "audit", crash_on_audit)
    with pytest.raises(InjectedCrash):
        lab.service.answer(view["task_id"], command, lab.actor)
    assert lab.state(view["task_id"]) == before
    with lab.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence WHERE task_id=?", (view["task_id"],)).fetchone()[0] == len(view["evidence"])
        assert db.execute("SELECT COUNT(*) FROM receipts WHERE action_id=?", (command.action_id,)).fetchone()[0] == 0
    restart(lab)
    answered = lab.service.answer(view["task_id"], command, lab.actor)
    assert lab.service.answer(view["task_id"], command, lab.actor) == answered
    assert sum(item["source"] == "human" for item in answered["evidence"]) == 1


def test_crash_after_submission_commit_returns_same_terminal_result_on_restart(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, ScriptedProvider([]))
    task_id = prepare(lab)
    evidence = lab.service.get(task_id, lab.actor)["evidence"]
    lab.update_state(task_id, pending_action=reply("submit_diagnosis", {
        "node_id": "normal", "path": ["root", "normal"],
        "evidence_ids": [item["evidence_id"] for item in evidence],
    }).action.model_dump(mode="json"))
    save = lab.service._save

    def crash_after_commit(state):
        if state["status"] == "submitted":
            raise InjectedCrash("terminal submission committed before graph checkpoint")
        return save(state)

    monkeypatch.setattr(lab.service, "_save", crash_after_commit)
    with pytest.raises(InjectedCrash):
        lab.service.run(task_id, lab.actor)
    committed = lab.service.get(task_id, lab.actor)
    assert committed["status"] == "submitted" and committed["submission"]["node_id"] == "normal"
    restart(lab)
    assert lab.service.run(task_id, lab.actor) == committed
    assert lab.service.calls(task_id, lab.actor) == []
    audit = lab.service.audit(task_id, lab.actor)
    assert sum(row["event"] == "diagnosis_submitted" for row in audit) == 1


def test_submission_artifact_is_not_referenced_if_transaction_rolls_back(tmp_path, monkeypatch):
    lab = make_lab(tmp_path)
    task_id = prepare(lab)
    evidence = lab.service.get(task_id, lab.actor)["evidence"]
    args = {"node_id": "normal", "path": ["root", "normal"],
            "evidence_ids": [item["evidence_id"] for item in evidence]}
    audit = lab.store.audit

    def reject_submission(db, actor_id, event, detail, task_id=None):
        if event == "diagnosis_submitted":
            raise DomainError("test_submission_abort", "Abort after artifact creation before commit")
        return audit(db, actor_id, event, detail, task_id)

    monkeypatch.setattr(lab.store, "audit", reject_submission)
    state = lab.tool(task_id, "submit_diagnosis", args)
    assert state["submission"] is None and state["status"] != "submitted"
    assert state["observations"][-1]["error"] == "test_submission_abort"
    assert not any(row["event"] == "diagnosis_submitted" for row in lab.service.audit(task_id, lab.actor))
    assert lab.service.get(task_id, lab.actor)["evidence"] == evidence
    monkeypatch.setattr(lab.store, "audit", audit)
    assert lab.legal_tool(task_id, "submit_diagnosis", args)["status"] == "submitted"
    assert sum(row["event"] == "diagnosis_submitted" for row in lab.service.audit(task_id, lab.actor)) == 1


def test_transaction_snapshot_recheck_stops_mid_tool_acl_change(tmp_path, monkeypatch):
    lab = make_lab(tmp_path)
    task_id = prepare(lab, check_all=False)
    snapshot_valid = lab.service._snapshot_valid
    changed = False

    def revoke_before_transaction(state, db=None):
        nonlocal changed
        if not changed and db is None:
            changed = True
            lab.store.set_user(lab.actor.actor_id, lab.admin, enabled=True, plants=["PLANT-B"], role="admin")
        return snapshot_valid(state, db)

    monkeypatch.setattr(lab.service, "_snapshot_valid", revoke_before_transaction)
    lab.update_state(task_id, pending_action=reply("time_series_verify", {
        "condition_ids": all_signals(lab),
    }).action.model_dump(mode="json"))
    with pytest.raises(DomainError) as error:
        lab.service.run(task_id, lab.actor)
    assert error.value.code == "identity_changed"
    assert changed and lab.state(task_id)["evidence_ids"] == []


def test_elapsed_budget_exhausted_by_compaction_blocks_next_model_call(tmp_path, monkeypatch):
    provider = ScriptedProvider([reply("intent_classify")])
    lab = make_lab(tmp_path, provider, max_seconds=5)
    task_id = lab.create()["task_id"]
    lab.update_state(task_id, observations=[{"tool": "test-history"}] * 9)
    compact = lab.service._compact

    def slow_compaction(state):
        compact(state)
        # Charge deterministic elapsed work instead of sleeping in a recovery test.
        state["elapsed_s"] += 6.0

    monkeypatch.setattr(lab.service, "_compact", slow_compaction)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == "budget_exhausted"
    assert view["compaction_count"] == 1 and view["elapsed_s"] >= 6.0
    assert view["calls"] == 0 and provider.contexts == []


@pytest.mark.parametrize(("node", "boundary"), [
    ("agent", "node_preflight"),
    ("agent", "context"),
    ("agent", "context_artifact"),
    ("agent", "transaction_acquisition"),
    ("tools", "node_preflight"),
    ("tools", "transaction_acquisition"),
    ("tools", "transaction_preflight"),
])
def test_current_node_preflight_elapsed_prevents_dispatch(tmp_path, monkeypatch, node, boundary):
    """Current-node time, including locked preflight, counts before any dispatch."""
    provider = ScriptedProvider([reply("intent_classify")])
    lab = make_lab(tmp_path, provider, max_seconds=5)
    task_id = prepare(lab, check_all=False) if node == "tools" else lab.create()["task_id"]
    state = lab.state(task_id)
    state["worker"] = None
    if node == "tools":
        state["pending_action"] = reply("time_series_verify", {
            "condition_ids": ["drive_enabled"],
        }).action.model_dump(mode="json")
    before_tools = state["tools"]
    clock = {"monotonic": 1000.0, "armed": False, "delayed": False}
    monkeypatch.setattr(service_module, "time", SimpleNamespace(monotonic=lambda: clock["monotonic"]))

    def delay():
        assert not clock["delayed"]
        clock["monotonic"] += 6.0
        clock["delayed"] = True

    snapshot_valid = lab.service._snapshot_valid

    def preflight(state, db=None):
        result = snapshot_valid(state, db)
        if boundary == "node_preflight" and db is None:
            delay()
        elif boundary == "transaction_preflight" and db is not None:
            delay()
        if node == "tools" and db is None:
            clock["armed"] = True
        return result

    monkeypatch.setattr(lab.service, "_snapshot_valid", preflight)
    context = lab.service._context

    def build_context(state):
        result = context(state)
        if boundary == "context":
            delay()
        clock["armed"] = True
        return result

    monkeypatch.setattr(lab.service, "_context", build_context)
    write_artifact = lab.store.write_artifact

    def write_context(kind, value):
        result = write_artifact(kind, value)
        if boundary == "context_artifact" and kind == "ModelContext":
            delay()
        return result

    monkeypatch.setattr(lab.store, "write_artifact", write_context)
    transaction = lab.store.transaction

    @contextmanager
    def delayed_acquisition():
        with transaction() as db:
            if boundary == "transaction_acquisition" and clock["armed"] and not clock["delayed"]:
                delay()
            yield db

    monkeypatch.setattr(lab.store, "transaction", delayed_acquisition)
    result = lab.service._wrap(node)(state)
    assert clock["delayed"], "The requested dispatch boundary was not exercised"
    assert result["error"] and result["error"]["code"] == "budget_exhausted"
    assert result["next_node"] in {"escalate", "end"}
    assert result["elapsed_s"] == pytest.approx(6.0)
    assert result["calls"] == 0 and result["tools"] == before_tools
    assert result["evidence_ids"] == [] and result["submission"] is None
    assert result["pending_action"] is None and result["pending_call_id"] is None
    assert provider.contexts == [] and lab.service.calls(task_id, lab.actor) == []


@pytest.mark.parametrize("operation", ["answer", "reconcile", "cancel"])
def test_owner_command_detaches_expired_worker_without_renewing_lease(tmp_path, operation):
    provider = ScriptedProvider([DomainError("provider_unknown", "Uncertain before worker exit")])
    lab = make_lab(tmp_path, provider)
    if operation == "answer":
        task_id = prepare(lab, "jam", check_all=False)
        lab.legal_tool(task_id, "request_human_input", {
            "condition_id": "obstruction_confirmed", "question": "Was obstruction observed?",
        })
        view = lab.service.get(task_id, lab.actor)
        command = HumanAnswer(action_id="answer-expired-owner", question_id=view["pending_human"]["question_id"],
                              answer=True, observation_note="TEST safe observation obtained previously.")
        def execute():
            return lab.service.answer(task_id, command, lab.actor)
    elif operation == "reconcile":
        view = lab.service.run(lab.create()["task_id"], lab.actor)
        task_id = view["task_id"]
        command = ReconcileRequest(action_id="reconcile-expired-owner", call_id=view["pending_call_id"],
                                   no_result=True, tokens=0,
                                   note="TEST confirmed no result and zero tokens after crashed worker.")
        def execute():
            return lab.service.reconcile(task_id, command, lab.actor)
    else:
        task_id = lab.create()["task_id"]
        def execute():
            return lab.service.cancel(task_id, "cancel-expired-owner", lab.actor)
    lab.update_state(task_id, worker="worker-crashed")
    with lab.store.transaction() as db:
        db.execute("UPDATE tasks SET lease_owner=?,lease_until=? WHERE task_id=?",
                   ("worker-crashed", now() - 10, task_id))
    stale = lab.state(task_id)
    result = execute()
    assert result["status"] == ("cancelled" if operation == "cancel" else "ready")
    assert execute() == result  # Receipt replay must not recreate the dead lease.
    with lab.store.connect() as db:
        row = db.execute("SELECT lease_owner,lease_until FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    assert row["lease_owner"] is None and row["lease_until"] is None
    current = lab.state(task_id)
    assert current["worker"] is None
    # Even a worker that learned the latest version no longer owns this task.
    stale["version"] = current["version"]
    with pytest.raises(DomainError) as error:
        lab.service._save(stale)
    assert error.value.code == "worker_fenced"
    resumed = lab.service.run(task_id, lab.actor)
    if operation == "cancel":
        assert resumed["status"] == "cancelled"
    else:
        assert resumed["status"] != "ready"


@pytest.mark.parametrize("rejection", ["arguments", "tool"])
def test_stale_worker_rejection_audit_is_fenced_before_commit(tmp_path, monkeypatch, rejection):
    lab = make_lab(tmp_path)
    task_id = lab.create()["task_id"]
    state = lab.state(task_id)
    state["worker"] = None
    state["pending_action"] = reply("time_series_verify", {
        "condition_ids": ["drive_enabled"], **({"invented": True} if rejection == "arguments" else {}),
    }).action.model_dump(mode="json")
    transaction = lab.store.transaction
    acquire_count = 0
    rejection_acquisition = 1 if rejection == "arguments" else 2

    @contextmanager
    def take_over_before_rejection_transaction():
        nonlocal acquire_count
        acquire_count += 1
        if acquire_count == rejection_acquisition:
            # A new worker wins after the failed tool transaction has rolled back,
            # but before the old worker records the rejection in its new transaction.
            with transaction() as db:
                db.execute("UPDATE tasks SET version=version+1,lease_owner=?,lease_until=? WHERE task_id=?",
                           ("worker-replacement", now() + 100, task_id))
        with transaction() as db:
            yield db

    monkeypatch.setattr(lab.store, "transaction", take_over_before_rejection_transaction)
    before = lab.state(task_id)
    with pytest.raises(DomainError) as error:
        lab.service._tools(state)
    assert error.value.code == "worker_fenced"
    assert acquire_count >= rejection_acquisition
    assert lab.state(task_id) == before
    with lab.store.connect() as db:
        audit_count = db.execute("SELECT COUNT(*) FROM audit WHERE task_id=? AND event='tool_rejected'",
                                 (task_id,)).fetchone()[0]
        row = db.execute("SELECT lease_owner,version FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    assert audit_count == 0, "A fenced worker must not leave a committed rejection audit"
    assert row["lease_owner"] == "worker-replacement" and row["version"] == before["version"] + 1


def test_rejected_tool_audit_and_budget_checkpoint_commit_together(tmp_path):
    lab = make_lab(tmp_path)
    task_id = lab.create()["task_id"]
    lab.update_state(task_id, pending_action=reply("time_series_verify", {
        "condition_ids": ["drive_enabled"],
    }).action.model_dump(mode="json"))
    state = lab.state(task_id)
    state["worker"] = None
    lab.service._tools(state)
    durable = lab.state(task_id)
    assert durable["tools"] == durable["guard_failures"] == 1
    assert durable["pending_action"] is None
    assert durable["observations"][-1] == {"tool": "time_series_verify", "error": "intent_required"}
    assert sum(row["event"] == "tool_rejected" for row in lab.service.audit(task_id, lab.actor)) == 1


def test_elapsed_during_call_reservation_prevents_provider_dispatch(tmp_path, monkeypatch):
    provider = ScriptedProvider([reply("intent_classify")])
    lab = make_lab(tmp_path, provider, max_seconds=5)
    task_id = lab.create()["task_id"]
    state = lab.state(task_id)
    state["worker"] = None
    clock = {"value": 1000.0, "delayed": False}
    monkeypatch.setattr(service_module, "time", SimpleNamespace(monotonic=lambda: clock["value"]))
    persist = lab.service._persist

    def slow_reservation(db, state):
        result = persist(db, state)
        if state["pending_call_id"] is not None and not clock["delayed"]:
            clock["value"] += 6.0
            clock["delayed"] = True
        return result

    monkeypatch.setattr(lab.service, "_persist", slow_reservation)
    result = lab.service._wrap("agent")(state)
    assert clock["delayed"] and provider.contexts == []
    assert result["error"]["code"] == "budget_exhausted" and result["next_node"] == "escalate"
    assert result["pending_call_id"] is None and result["pending_action"] is None
    assert result["calls"] == 1 and result["tools"] == 0
    call = lab.service.calls(task_id, lab.actor)[0]
    assert call["status"] == "failed" and call["error_code"] == "budget_exhausted"


@pytest.mark.parametrize("boundary", ["evidence_validation", "submission_artifact"])
def test_current_submission_node_time_exhaustion_rolls_back_terminal_result(tmp_path, monkeypatch, boundary):
    lab = make_lab(tmp_path, max_seconds=5)
    task_id = prepare(lab)
    view = lab.service.get(task_id, lab.actor)
    state = lab.state(task_id)
    state["worker"] = None
    state["pending_action"] = reply("submit_diagnosis", {
        "node_id": "normal", "path": ["root", "normal"],
        "evidence_ids": [item["evidence_id"] for item in view["evidence"]],
    }).action.model_dump(mode="json")
    before_tools = state["tools"]
    clock = {"value": 1000.0, "delayed": False}
    monkeypatch.setattr(service_module, "time", SimpleNamespace(monotonic=lambda: clock["value"]))
    evidence = lab.service._evidence
    write_artifact = lab.store.write_artifact

    def slow_evidence(state, db=None):
        result = evidence(state, db)
        if boundary == "evidence_validation" and db is not None and not clock["delayed"]:
            clock["value"] += 6.0
            clock["delayed"] = True
        return result

    def slow_artifact(kind, value):
        result = write_artifact(kind, value)
        if boundary == "submission_artifact" and kind == "FrozenDiagnosis":
            clock["value"] += 6.0
            clock["delayed"] = True
        return result

    monkeypatch.setattr(lab.service, "_evidence", slow_evidence)
    monkeypatch.setattr(lab.store, "write_artifact", slow_artifact)
    result = lab.service._wrap("tools")(state)
    assert clock["delayed"]
    assert result["error"]["code"] == "budget_exhausted" and result["next_node"] == "escalate"
    assert result["submission"] is None and result["status"] != "submitted"
    assert result["tools"] == before_tools
    assert lab.service.get(task_id, lab.actor)["evidence"] == view["evidence"]
    assert not any(row["event"] == "diagnosis_submitted" for row in lab.service.audit(task_id, lab.actor))


def usage_error(code="provider_invalid", *, reported_tokens=None, usage_unknown=False):
    """Explicit provider metadata injection; exercise the service's real durable ledger."""
    error = DomainError(code, "TEST unusable model output with usage metadata", 502)
    if reported_tokens is not None:
        error.reported_tokens = reported_tokens
    error.usage_unknown = usage_unknown
    return error


@pytest.mark.parametrize("failure", ["reported_exception", "invalid_reply"])
def test_usage_ledger_sixteen_invalid_4000_token_replies_block_seventeenth_dispatch(tmp_path, failure):
    steps = [(usage_error(reported_tokens=4000) if failure == "reported_exception" else
              {"action": {"tool": "invented_tool", "arguments": {}, "reason": "TEST invalid action"},
               "tokens": 4000}) for _ in range(16)]
    provider = ScriptedProvider([*steps, reply("intent_classify")])
    lab = make_lab(tmp_path, provider, max_tokens=64000, max_calls=20)
    task_id = lab.create()["task_id"]
    for expected_calls in range(1, 17):
        view = lab.service.run(task_id, lab.actor)
        assert view["status"] == "blocked" and view["error"]["code"] == "provider_invalid"
        assert view["calls"] == expected_calls and view["tokens"] == expected_calls * 4000
        assert view["tools"] == 0 and view["pending_call_id"] is None
        ledger = lab.service.calls(task_id, lab.actor)
        assert len(ledger) == expected_calls
        assert all(call["status"] == "failed" and call["tokens"] == 4000 for call in ledger)
        assert sum(call["tokens"] for call in ledger) == view["tokens"]
        if expected_calls % 4 == 0:
            restart(lab, provider, max_tokens=100000, max_calls=100)
    for _ in range(2):
        view = lab.service.run(task_id, lab.actor)
        assert view["status"] == "escalated" and view["error"]["code"] == "budget_exhausted"
        assert view["calls"] == 16 and view["tokens"] == view["limits"]["tokens"] == 64000
    assert len(provider.contexts) == 16 and len(provider.steps) == 1
    assert len(lab.service.calls(task_id, lab.actor)) == 16


@pytest.mark.parametrize(("code", "reported_tokens", "usage_unknown"), [
    ("provider_invalid", None, True),
    ("provider_invalid", 4000, True),
    ("provider_unknown", 4000, False),
])
def test_usage_unknown_metadata_persists_and_prevents_resend(tmp_path, code, reported_tokens, usage_unknown):
    provider = ScriptedProvider([usage_error(code, reported_tokens=reported_tokens, usage_unknown=usage_unknown),
                                 reply("intent_classify")])
    lab = make_lab(tmp_path, provider)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "blocked" and view["error"]["code"] == code
    assert view["calls"] == 1 and view["tokens"] == (reported_tokens or 0)
    call_id = view["pending_call_id"]
    assert call_id is not None
    ledger = lab.service.calls(task_id, lab.actor)
    assert len(ledger) == 1 and ledger[0]["status"] == "unknown"
    assert ledger[0]["error_code"] == code and ledger[0]["tokens"] == reported_tokens
    restart(lab, provider)
    for _ in range(2):
        view = lab.service.run(task_id, lab.actor)
        assert view["status"] == "blocked" and view["error"]["code"] == "model_unknown"
        assert view["pending_call_id"] == call_id and view["tokens"] == (reported_tokens or 0)
    assert len(provider.contexts) == 1 and len(provider.steps) == 1


@pytest.mark.parametrize("failure", ["reported_exception", "invalid_reply"])
@pytest.mark.parametrize("invalid_usage", [None, True, -1, 100001, "4000", 4000.0])
def test_usage_unknown_invalid_token_metadata_never_enables_automatic_resend(
        tmp_path, failure, invalid_usage):
    if failure == "reported_exception":
        step = usage_error()
        step.reported_tokens = invalid_usage
    else:
        step = {"action": {"tool": "invented_tool", "arguments": {}, "reason": "TEST invalid action"}}
        if invalid_usage is not None:
            step["tokens"] = invalid_usage
    provider = ScriptedProvider([step, reply("intent_classify")])
    lab = make_lab(tmp_path, provider)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "blocked" and view["error"]["code"] == "provider_invalid"
    assert view["calls"] == 1 and view["tokens"] == view["tools"] == 0
    call_id = view["pending_call_id"]
    assert call_id is not None
    call = lab.service.calls(task_id, lab.actor)[0]
    assert call["status"] == "unknown" and call["tokens"] is None
    restart(lab, provider)
    view = lab.service.run(task_id, lab.actor)
    assert view["pending_call_id"] == call_id and view["error"]["code"] == "model_unknown"
    assert view["tokens"] == 0 and len(provider.contexts) == 1 and len(provider.steps) == 1


@pytest.mark.parametrize("failure", ["reported_exception", "invalid_reply"])
def test_usage_ledger_known_zero_is_not_conflated_with_unknown_usage(tmp_path, failure):
    step = usage_error(reported_tokens=0, usage_unknown=True) if failure == "reported_exception" else {
        "action": {"tool": "invented_tool", "arguments": {}, "reason": "TEST invalid action"}, "tokens": 0,
    }
    provider = ScriptedProvider([step, reply("intent_classify")])
    lab = make_lab(tmp_path, provider, max_calls=1)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    assert view["tokens"] == 0
    call = lab.service.calls(task_id, lab.actor)[0]
    assert call["tokens"] == 0
    if failure == "reported_exception":
        assert call["status"] == "unknown" and view["pending_call_id"] == call["call_id"]
        command = ReconcileRequest(action_id="known-zero-usage", call_id=call["call_id"], no_result=True,
            note="TEST no result; retain the provider's already recorded zero-token usage.")
        result = lab.service.reconcile(task_id, command, lab.actor)
        assert result["tokens"] == 0
        event = next(row for row in lab.service.audit(task_id, lab.actor) if row["event"] == "model_reconciled")
        detail = json.loads(event["detail_json"])
        assert detail["tokens"] == 0 and detail["conservative_usage"] is False
    else:
        assert call["status"] == "failed" and view["pending_call_id"] is None
    ended = lab.service.run(task_id, lab.actor)
    assert ended["status"] == "escalated" and ended["tokens"] == 0 and ended["calls"] == 1
    assert len(provider.contexts) == 1


def test_usage_ledger_omitted_unknown_no_result_conservatively_exhausts_frozen_budget(tmp_path):
    provider = ScriptedProvider([reply("intent_classify", tokens=137),
        usage_error(usage_unknown=True), reply("expand_tree_node", {"node_id": "root"})])
    lab = make_lab(tmp_path, provider, max_tokens=1000)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    assert view["calls"] == 2 and view["tokens"] == 137 and view["pending_call_id"] is not None
    command = ReconcileRequest(action_id="unknown-usage-conservative", call_id=view["pending_call_id"],
        no_result=True, note="TEST no usable result; actual consumed tokens cannot be established.")
    result = lab.service.reconcile(task_id, command, lab.actor)
    assert result["status"] == "ready" and result["tokens"] == 1000
    ledger = lab.service.calls(task_id, lab.actor)
    assert [call["tokens"] for call in ledger] == [137, 863]
    assert ledger[-1]["status"] == "confirmed_no_result"
    assert sum(call["tokens"] for call in ledger) == result["tokens"]
    event = next(row for row in lab.service.audit(task_id, lab.actor) if row["event"] == "model_reconciled")
    detail = json.loads(event["detail_json"])
    assert detail["tokens"] == 863 and detail["conservative_usage"] is True
    assert detail["human_attestation_not_external_proof"] is True
    assert lab.service.reconcile(task_id, command, lab.actor) == result
    with pytest.raises(DomainError) as error:
        lab.service.reconcile(task_id, command.model_copy(update={"tokens": 0}), lab.actor)
    assert error.value.code == "idempotency_conflict"
    restart(lab, provider, max_tokens=100000)
    assert lab.service.reconcile(task_id, command, lab.actor) == result
    ended = lab.service.run(task_id, lab.actor)
    assert ended["status"] == "escalated" and ended["error"]["code"] == "budget_exhausted"
    assert ended["tokens"] == ended["limits"]["tokens"] == 1000 and ended["calls"] == 2
    assert len(provider.contexts) == 2


def test_usage_ledger_explicit_zero_no_result_is_attested_and_can_resume(tmp_path):
    provider = ScriptedProvider([reply("intent_classify", tokens=137), usage_error(usage_unknown=True),
                                 reply("expand_tree_node", {"node_id": "root"}, tokens=128)])
    lab = make_lab(tmp_path, provider, max_calls=3, max_tokens=1000)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    result = reconcile_no_result(lab, view, "attested-zero-usage")
    assert result["tokens"] == 137 and result["calls"] == 2
    assert [call["tokens"] for call in lab.service.calls(task_id, lab.actor)] == [137, 0]
    event = next(row for row in lab.service.audit(task_id, lab.actor) if row["event"] == "model_reconciled")
    detail = json.loads(event["detail_json"])
    assert detail["tokens"] == 0 and detail["conservative_usage"] is False
    assert detail["human_attestation_not_external_proof"] is True
    assert "zero tokens" in detail["note"]
    ended = lab.service.run(task_id, lab.actor)
    assert ended["status"] == "escalated" and ended["calls"] == 3 and ended["tokens"] == 265
    assert ended["expanded_nodes"] == ["root"] and len(provider.contexts) == 3
    assert sum(call["tokens"] for call in lab.service.calls(task_id, lab.actor)) == ended["tokens"]


@pytest.mark.parametrize(("outcome", "attested_usage", "expected_usage"), [
    ("no_result", None, 400),
    ("no_result", 400, 400),
    ("no_result", 650, 650),
    ("reply", 400, 400),
    ("reply", 650, 650),
])
def test_usage_ledger_reconcile_only_adds_delta_and_receipt_never_double_charges(
        tmp_path, outcome, attested_usage, expected_usage):
    provider = ScriptedProvider([reply("intent_classify", tokens=137),
                                 usage_error(reported_tokens=400, usage_unknown=True)])
    lab = make_lab(tmp_path, provider, max_calls=2)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    assert view["tokens"] == 537 and view["pending_call_id"] is not None
    command = ReconcileRequest(action_id="usage-delta", call_id=view["pending_call_id"],
        no_result=outcome == "no_result", tokens=attested_usage,
        reply=reply("expand_tree_node", {"node_id": "root"}).action if outcome == "reply" else None,
        note="TEST owner-admin attests the total usage for this one reserved call.")
    result = lab.service.reconcile(task_id, command, lab.actor)
    assert result["tokens"] == 137 + expected_usage and result["calls"] == 2
    ledger = lab.service.calls(task_id, lab.actor)
    assert [call["tokens"] for call in ledger] == [137, expected_usage]
    assert sum(call["tokens"] for call in ledger) == result["tokens"]
    assert ledger[-1]["status"] == ("reconciled" if outcome == "reply" else "confirmed_no_result")
    after = lab.state(task_id)
    assert lab.service.reconcile(task_id, command, lab.actor) == result
    assert lab.state(task_id) == after
    with pytest.raises(DomainError) as error:
        lab.service.reconcile(task_id, command.model_copy(update={"tokens": expected_usage + 1}), lab.actor)
    assert error.value.code == "idempotency_conflict"
    restart(lab, provider)
    assert lab.service.reconcile(task_id, command, lab.actor) == result
    ended = lab.service.run(task_id, lab.actor)
    assert ended["tokens"] == 137 + expected_usage and ended["calls"] == 2
    assert ended["tools"] == (2 if outcome == "reply" else 1)
    assert len(provider.contexts) == 2
    events = [row for row in lab.service.audit(task_id, lab.actor) if row["event"] == "model_reconciled"]
    assert len(events) == 1
    assert json.loads(events[0]["detail_json"])["conservative_usage"] is False


@pytest.mark.parametrize(("outcome", "tokens"), [("no_result", 0), ("no_result", 399), ("reply", 399)])
def test_usage_ledger_reconcile_cannot_reduce_reported_usage(tmp_path, outcome, tokens):
    provider = ScriptedProvider([usage_error(reported_tokens=400, usage_unknown=True)])
    lab = make_lab(tmp_path, provider)
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    command = ReconcileRequest(action_id="usage-reduction", call_id=view["pending_call_id"],
        no_result=outcome == "no_result", tokens=tokens,
        reply=reply("intent_classify").action if outcome == "reply" else None,
        note="TEST attempt to reduce already recorded provider usage must fail.")
    before = lab.state(task_id)
    before_calls = lab.service.calls(task_id, lab.actor)
    with pytest.raises(DomainError) as error:
        lab.service.reconcile(task_id, command, lab.actor)
    assert error.value.code == "usage_conflict"
    assert lab.state(task_id) == before and lab.service.calls(task_id, lab.actor) == before_calls
    with lab.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM receipts WHERE action_id=?", (command.action_id,)).fetchone()[0] == 0
    assert not any(row["event"] == "model_reconciled" for row in lab.service.audit(task_id, lab.actor))


def test_usage_ledger_reconcile_delta_receipt_and_audit_roll_back_together(tmp_path, monkeypatch):
    lab = make_lab(tmp_path, ScriptedProvider([usage_error(reported_tokens=400, usage_unknown=True)]))
    task_id = lab.create()["task_id"]
    view = lab.service.run(task_id, lab.actor)
    command = ReconcileRequest(action_id="usage-atomic", call_id=view["pending_call_id"],
        no_result=True, tokens=650, note="TEST total usage attestation is atomic with reconciliation.")
    before = lab.state(task_id)
    calls_before = lab.service.calls(task_id, lab.actor)
    audit = lab.store.audit

    def crash_after_receipt(db, actor_id, event, detail, task_id=None):
        if event == "model_reconciled":
            raise InjectedCrash("reconcile before transaction commit")
        return audit(db, actor_id, event, detail, task_id)

    monkeypatch.setattr(lab.store, "audit", crash_after_receipt)
    with pytest.raises(InjectedCrash):
        lab.service.reconcile(task_id, command, lab.actor)
    assert lab.state(task_id) == before and lab.service.calls(task_id, lab.actor) == calls_before
    with lab.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM receipts WHERE action_id=?", (command.action_id,)).fetchone()[0] == 0
    restart(lab)
    result = lab.service.reconcile(task_id, command, lab.actor)
    assert result["tokens"] == 650
    assert lab.service.reconcile(task_id, command, lab.actor) == result
    assert lab.service.calls(task_id, lab.actor)[0]["tokens"] == 650
    assert sum(row["event"] == "model_reconciled" for row in lab.service.audit(task_id, lab.actor)) == 1


def test_compaction_node_runs_before_resumed_agent_without_erasing_pending_evidence(tmp_path):
    lab = make_lab(tmp_path, ScriptedProvider([reply("intent_classify")]), max_calls=1)
    task_id = lab.create()["task_id"]
    history = [{"tool": "test-history", "sequence": index} for index in range(9)]
    lab.update_state(task_id, observations=history, tokens=20, tools=2, elapsed_s=1.0)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["compaction_count"] == 1
    assert view["calls"] == 1 and view["tools"] == 3 and view["tokens"] == 148
    assert view["elapsed_s"] >= 1.0
    state = lab.state(task_id)
    archived = lab.store.read_artifact(state["compact_refs"][0])
    assert archived + state["observations"][:-1] == history
