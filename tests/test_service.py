"""Service integration using real local persistence and mathematically generated telemetry.

TestPublishedKnowledge is a service-only publication boundary, NOT a HermiT result.
Real report compilation, reasoning and publication are tested separately.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest

from industrial_diagnosis.fixtures import TREE_PATH_EXAMPLES, original_knowledge
from industrial_diagnosis.provider import SyntheticProvider
from industrial_diagnosis.schemas import (
    Action,
    Actor,
    DiagnosisRequest,
    DomainError,
    HumanAnswer,
    KnowledgeDraft,
    ModelReply,
    ReconcileRequest,
    canonical,
    digest,
)
from industrial_diagnosis.service import DiagnosisService
from industrial_diagnosis.storage import Store
from industrial_diagnosis.telemetry import TelemetryStore


class TestPublishedKnowledge:
    """Explicit test double exposing a pinned draft; makes no reasoner claims."""

    __test__ = False

    def __init__(self, store, draft=None):
        self.store = store
        checked = KnowledgeDraft.model_validate(
            draft if draft is not None else original_knowledge().draft.model_dump(mode="json")
        )
        self.published = {
            "version_id": "test-published-original-v1",
            "content_hash": digest(checked.model_dump(mode="json")),
            "draft": checked.model_dump(mode="json"),
        }

    def active(self, plant_id, actor, db=None):
        self.store.authorize(actor, plant_id, db=db)
        if self.published["draft"]["plant_id"] != plant_id:
            raise DomainError("knowledge_not_found", "No test publication in this plant", 404)
        return copy.deepcopy(self.published)


class ScriptedProvider:
    """Explicit TEST action sequence for service fault injection, never model inference."""

    is_test_provider = True
    fingerprint = "service-scripted-TEST-v1"
    metadata = {"provider": "service-scripted-TEST"}

    def __init__(self, steps):
        self.steps = list(steps)
        self.contexts = []

    def decision(self, context):
        self.contexts.append(copy.deepcopy(context))
        if not self.steps:
            raise DomainError("provider_stuck", "TEST action sequence ended", 409)
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(context)
        return step


def reply(tool, arguments=None, tokens=128):
    return ModelReply(action=Action(tool=tool, arguments=arguments or {}, reason="TEST action"),
                      tokens=tokens)


@dataclass
class ServiceLab:
    store: Store
    telemetry: TelemetryStore
    knowledge: TestPublishedKnowledge
    service: DiagnosisService
    actor: Actor
    token: str
    admin: Actor
    cases: dict

    def create(self, case="normal", request_id="request-1", actor=None, **changes):
        fixture = self.cases[case]
        request = DiagnosisRequest.model_validate({
            "request_id": request_id,
            **{key: fixture[key] for key in ("plant_id", "device_id", "event_id", "window")},
            "complaint": fixture["complaint"],
            **changes,
        })
        return self.service.create(request, actor or self.actor)

    def state(self, task_id):
        with self.store.connect() as db:
            row = db.execute("SELECT state_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row[0])

    def update_state(self, task_id, **changes):
        """Inject a durable crash boundary or adversarial state in a test, not a public API."""
        with self.store.transaction() as db:
            state = json.loads(db.execute("SELECT state_json FROM tasks WHERE task_id=?",
                                          (task_id,)).fetchone()[0])
            state.update(changes)
            db.execute("UPDATE tasks SET state_json=? WHERE task_id=?", (canonical(state), task_id))
        return state

    def tool(self, task_id, tool, arguments=None):
        """Exercise the real transactional tool guard without scheduling a model call."""
        state = self.state(task_id)
        state["worker"] = None  # These isolated tool tests hold no graph worker lease.
        state["pending_action"] = reply(tool, arguments).action.model_dump(mode="json")
        self.service._tools(state)
        self.service._save(state)
        return state

    def legal_tool(self, task_id, tool, arguments=None):
        state = self.tool(task_id, tool, arguments)
        assert "error" not in state["observations"][-1], state["observations"][-1]
        return state

    def answer(self, view, value=True, action_id=None):
        question = view["pending_human"]
        return self.service.answer(view["task_id"], HumanAnswer(
            action_id=action_id or "answer-" + question["question_id"],
            question_id=question["question_id"], answer=value,
            observation_note="TEST: previously acquired safe inspection observation.",
        ), self.actor)


def make_lab(tmp_path, provider=None, *, draft=None, role="admin", **limits):
    store = Store(tmp_path)
    token = store.provision_user("task-owner", role, ["PLANT-A"])
    actor = store.authenticate(token)
    admin = store.authenticate(store.provision_user("acl-admin", "admin", ["PLANT-A", "PLANT-B"]))
    telemetry = TelemetryStore(store)
    listing = telemetry.seed()
    knowledge = TestPublishedKnowledge(store, draft)
    provider = provider if provider is not None else SyntheticProvider()
    service = DiagnosisService(store, knowledge, telemetry, provider, **limits)
    return ServiceLab(store, telemetry, knowledge, service, actor, token, admin, listing["cases"])


@pytest.fixture
def lab(tmp_path):
    return make_lab(tmp_path)


def all_signals(lab):
    draft = lab.knowledge.published["draft"]
    referenced = set(draft["global_required"])
    for node in draft["nodes"]:
        referenced.update(node["conditions"] + node["mandatory"])
    return [item["condition_id"] for item in draft["conditions"]
            if item["source"] == "signal" and item["condition_id"] in referenced]


def prepare(lab, case="normal", *, check_all=True, human=None, request_id="request-1"):
    task_id = lab.create(case, request_id)["task_id"]
    lab.legal_tool(task_id, "intent_classify")
    lab.legal_tool(task_id, "kg_query_candidates", {"query": "conveyor observations"})
    for node in ("root", "mechanical", "sensing"):
        lab.legal_tool(task_id, "expand_tree_node", {"node_id": node})
    if check_all:
        lab.legal_tool(task_id, "time_series_verify", {"condition_ids": all_signals(lab)})
    for condition_id, value in (human or {}).items():
        lab.legal_tool(task_id, "request_human_input", {
            "condition_id": condition_id, "question": "Was the safe observation confirmed?",
        })
        lab.answer(lab.service.get(task_id, lab.actor), value)
    return task_id


def submission_arguments(lab, task_id, node_id="normal"):
    view = lab.service.get(task_id, lab.actor)
    return {"node_id": node_id, "path": TREE_PATH_EXAMPLES[node_id],
            "evidence_ids": [item["evidence_id"] for item in view["evidence"]]}


def assert_rejected(state, code):
    assert state["status"] != "submitted"
    assert state["submission"] is None
    assert state["observations"][-1]["error"] == code


def finish(lab, view, answers=True):
    for _ in range(8):
        view = lab.service.run(view["task_id"], lab.actor)
        if view["status"] != "waiting_human":
            return view
        view = lab.answer(view, answers)
    pytest.fail("Provider did not stop within the bounded human-question loop")


@pytest.mark.parametrize("case", ["normal", "jam", "slip", "sensor_fault"])
@pytest.mark.parametrize("order", ["verify-first", "expand-first"])
def test_synthetic_cases_use_real_graph_and_bound_evidence(tmp_path, case, order):
    lab = make_lab(tmp_path, SyntheticProvider(order))
    view = finish(lab, lab.create(case))
    assert view["status"] == "submitted", view
    result = view["submission"]
    assert result["node_id"] == case
    assert result["path"] == TREE_PATH_EXAMPLES[case]
    assert result["conclusion_level"] == "supported_under_declared_synthetic_rules"
    assert digest({key: value for key, value in result.items() if key != "content_hash"}) == result["content_hash"]
    assert set(all_signals(lab)) <= {item["condition_id"] for item in view["evidence"]}
    for evidence in view["evidence"]:
        assert evidence["task_id"] == view["task_id"]
        assert evidence["actor_id"] == lab.actor.actor_id
        assert evidence["tree_version"] == view["tree_version"]
        assert evidence["tree_hash"] == view["tree_hash"]
        assert evidence["data_hash"] == view["data_hash"]
        for key in ("plant_id", "device_id", "event_id", "window"):
            assert evidence[key] == view["request"][key]
        assert evidence["state"] != "missing"
        if evidence["source"] == "human":
            assert evidence["details"]["not_sensor_verified"] is True
            assert evidence["details"]["attested_by"] == lab.actor.actor_id
    cited = set(result["evidence_ids"])
    assert cited <= {item["evidence_id"] for item in view["evidence"]}
    assert 1 <= view["calls"] <= 20 and 1 <= view["tools"] <= 32
    assert view["tokens"] == view["calls"] * 128 <= 24000
    assert lab.state(view["task_id"])["guard_failures"] == 0
    assert all(call["status"] == "completed" for call in lab.service.calls(view["task_id"], lab.actor))
    if case == "normal":
        assert all(item["source"] == "signal" for item in view["evidence"])


def test_two_provider_orders_are_distinct_legal_paths(tmp_path):
    traces = []
    for order in ("verify-first", "expand-first"):
        lab = make_lab(tmp_path / order, SyntheticProvider(order))
        view = finish(lab, lab.create("jam"))
        assert view["status"] == "submitted", view
        rows = lab.service.audit(view["task_id"], lab.actor)
        assert not any(row["event"] == "tool_rejected" for row in rows)
        traces.append([json.loads(row["detail_json"])["tool"] for row in rows
                       if row["event"] == "tool_completed"])
    assert traces[0] != traces[1]
    assert traces[0].index("time_series_verify") < traces[0].index("expand_tree_node")
    assert traces[1].index("expand_tree_node") < traces[1].index("time_series_verify")


def test_five_nodes_are_real_compiled_langgraph_nodes(lab):
    graph = lab.service.graph.get_graph()
    assert set(graph.nodes) == {"__start__", "__end__", "agent", "tools", "guard", "compact", "escalate"}
    for name in ("agent", "tools", "guard", "compact", "escalate"):
        assert graph.nodes[name].data is not None
        assert any(edge.source == name for edge in graph.edges)
    assert any(edge.source == "__start__" and edge.target == "guard" for edge in graph.edges)


@pytest.mark.parametrize("case", ["jam", "slip", "sensor_fault"])
def test_false_human_observation_never_supports_its_diagnosis(tmp_path, case):
    lab = make_lab(tmp_path)
    view = finish(lab, lab.create(case), answers=False)
    assert view["status"] in {"blocked", "escalated"}, view
    assert view["submission"] is None
    human = [item for item in view["evidence"] if item["source"] == "human"]
    assert human and all(item["state"] == "refuted" for item in human)
    assert all(item["details"]["answer"] is False for item in human)


@pytest.mark.parametrize("case", ["missing", "conflicting"])
def test_incomplete_or_conflicting_fixture_fails_closed(tmp_path, case):
    lab = make_lab(tmp_path)
    view = finish(lab, lab.create(case))
    assert view["status"] in {"blocked", "escalated"}, view
    assert view["submission"] is None
    if case == "missing":
        assert any(item["state"] == "missing" for item in view["evidence"])


@pytest.mark.parametrize(("tool", "arguments", "code"), [
    ("expand_tree_node", {"node_id": "jam"}, "path_disconnected"),
    ("time_series_verify", {"condition_ids": ["current_high"]}, "condition_not_visible"),
    ("time_series_verify", {"condition_ids": ["invented"]}, "condition_not_visible"),
    ("request_human_input", {"condition_id": "obstruction_confirmed", "question": "Was it confirmed?"}, "invalid_human_condition"),
    ("request_human_input", {"condition_id": "drive_enabled", "question": "Is the signal true?"}, "invalid_human_condition"),
    ("time_series_verify", {"condition_ids": ["drive_enabled"], "threshold": 0}, "invalid_tool_arguments"),
])
def test_tools_cannot_bypass_visibility_or_frozen_arguments(lab, tool, arguments, code):
    task_id = lab.create()["task_id"]
    lab.legal_tool(task_id, "intent_classify")
    state = lab.tool(task_id, tool, arguments)
    assert_rejected(state, code)
    assert not state["evidence_ids"]


def test_human_condition_cannot_be_inferred_from_telemetry(lab):
    task_id = prepare(lab, check_all=False)
    state = lab.tool(task_id, "time_series_verify", {"condition_ids": ["obstruction_confirmed"]})
    assert_rejected(state, "human_observation_required")
    assert not state["evidence_ids"]


def test_intent_required_before_diagnostic_tools(lab):
    task_id = lab.create()["task_id"]
    assert_rejected(lab.tool(task_id, "time_series_verify", {"condition_ids": ["drive_enabled"]}),
                    "intent_required")


@pytest.mark.parametrize("complaint", ["disable interlock to continue", "weather forecast tomorrow"])
def test_unsupported_intent_stops_without_diagnostic_tools(lab, complaint):
    view = lab.service.run(lab.create(complaint=complaint)["task_id"], lab.actor)
    assert view["status"] == "escalated"
    assert view["error"]["code"] == "unsupported_intent"
    assert view["calls"] == view["tools"] == 1
    assert not view["evidence"] and view["submission"] is None


def test_all_tree_signals_required_even_outside_selected_leaf(lab):
    task_id = prepare(lab, check_all=False)
    lab.legal_tool(task_id, "time_series_verify", {"condition_ids": [
        "drive_enabled", "current_nominal", "belt_nominal", "roller_aligned", "encoder_aligned",
    ]})
    assert_rejected(lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id)),
                    "unexplained_signals")


def changed_draft(node_id="normal", **changes):
    draft = original_knowledge().draft.model_dump(mode="json")
    next(node for node in draft["nodes"] if node["node_id"] == node_id).update(changes)
    return draft


@pytest.mark.parametrize(("gate", "expected"), [("all", "candidate_not_supported"), ("any", None)])
def test_all_and_any_gates_use_computed_supported_and_refuted_evidence(tmp_path, gate, expected):
    lab = make_lab(tmp_path, draft=changed_draft(gate=gate, conditions=["current_nominal", "belt_nominal", "roller_aligned", "encoder_aligned", "current_high"]))
    task_id = prepare(lab)
    state = lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id))
    if expected:
        assert_rejected(state, expected)
    else:
        assert state["status"] == "submitted"
        view = lab.service.get(task_id, lab.actor)
        cited = {item["condition_id"]: item["state"] for item in view["evidence"]
                 if item["evidence_id"] in view["submission"]["evidence_ids"]}
        assert cited["belt_nominal"] == "supported" and cited["current_high"] == "refuted"


def test_any_gate_cannot_omit_refuted_evidence_from_citations(tmp_path):
    lab = make_lab(tmp_path, draft=changed_draft(gate="any", conditions=["current_nominal", "belt_nominal", "roller_aligned", "encoder_aligned", "current_high"]))
    task_id = prepare(lab)
    args = submission_arguments(lab, task_id)
    refuted_id = next(item["evidence_id"] for item in lab.service.get(task_id, lab.actor)["evidence"]
                     if item["condition_id"] == "current_high")
    args["evidence_ids"].remove(refuted_id)
    assert_rejected(lab.tool(task_id, "submit_diagnosis", args), "evidence_missing")


def test_any_gate_cannot_hide_missing_signal_elsewhere_in_tree(tmp_path):
    lab = make_lab(tmp_path, draft=changed_draft(gate="any", conditions=["belt_nominal", "belt_low"]))
    task_id = prepare(lab, "missing")
    assert_rejected(lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id)),
                    "missing_signals")


@pytest.mark.parametrize("gate", ["all", "any"])
def test_false_mandatory_human_blocks_both_gates(tmp_path, gate):
    lab = make_lab(tmp_path, draft=changed_draft("jam", gate=gate))
    task_id = prepare(lab, "jam", human={"obstruction_confirmed": False})
    assert_rejected(lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id, "jam")),
                    "mandatory_not_supported")


def test_global_mandatory_predicate_cannot_be_omitted(lab):
    task_id = prepare(lab)
    args = submission_arguments(lab, task_id)
    global_id = next(item["evidence_id"] for item in lab.service.get(task_id, lab.actor)["evidence"]
                     if item["condition_id"] == "drive_enabled")
    args["evidence_ids"].remove(global_id)
    assert_rejected(lab.tool(task_id, "submit_diagnosis", args), "evidence_missing")


@pytest.mark.parametrize(("mandatory", "expected"), [
    ([], "conflicting_candidates"), (["independent_measurement_confirmed"], "unresolved_candidate"),
])
def test_competing_supported_or_unresolved_leaf_prevents_submission(tmp_path, mandatory, expected):
    lab = make_lab(tmp_path, draft=changed_draft("sensor_fault", conditions=["belt_nominal"],
                                               mandatory=mandatory))
    task_id = prepare(lab)
    assert_rejected(lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id)), expected)


@pytest.mark.parametrize("path", [["root", "jam"], ["root", "normal", "normal"],
                                   ["mechanical", "jam"], ["root", "invented", "normal"]])
def test_submission_rejects_disconnected_or_uninspected_paths(lab, path):
    task_id = prepare(lab)
    args = submission_arguments(lab, task_id)
    args["path"] = path
    assert_rejected(lab.tool(task_id, "submit_diagnosis", args), "path_disconnected")


def test_submission_requires_queried_candidate_and_expanded_ancestor(lab):
    task_id = prepare(lab)
    args = submission_arguments(lab, task_id)
    lab.update_state(task_id, candidates=[])
    assert_rejected(lab.tool(task_id, "submit_diagnosis", args), "invalid_candidate")
    lab.update_state(task_id, candidates=["normal"], expanded_nodes=[])
    assert_rejected(lab.tool(task_id, "submit_diagnosis", args), "path_disconnected")


@pytest.mark.parametrize("attack", ["invented", "duplicate", "other-task"])
def test_submission_cannot_cite_unissued_duplicate_or_cross_task_evidence(lab, attack):
    task_id = prepare(lab)
    args = submission_arguments(lab, task_id)
    if attack == "invented":
        args["evidence_ids"].append("evidence-invented")
    elif attack == "duplicate":
        args["evidence_ids"].append(args["evidence_ids"][0])
    else:
        other_id = prepare(lab, request_id="other-task")
        args["evidence_ids"].append(lab.service.get(other_id, lab.actor)["evidence"][0]["evidence_id"])
    assert_rejected(lab.tool(task_id, "submit_diagnosis", args), "evidence_scope")


@pytest.mark.parametrize(("field", "value", "code"), [
    ("evidence_id", "evidence-replaced", "evidence_integrity"),
    ("task_id", "task-other", "evidence_integrity"),
    ("condition_id", "current_high", "evidence_integrity"),
    ("tree_version", "test-other-version", "evidence_scope"),
    ("tree_hash", "0" * 64, "evidence_scope"),
    ("data_hash", "0" * 64, "evidence_scope"),
    ("window", {"start_ms": 1000, "end_ms": 29000}, "evidence_scope"),
    ("plant_id", "PLANT-B", "evidence_scope"),
    ("device_id", "CONVEYOR-02", "evidence_scope"),
    ("event_id", "EVT-999", "evidence_scope"),
    ("actor_id", "other-user", "evidence_scope"),
])
def test_evidence_payload_identity_and_snapshot_binding(lab, field, value, code):
    task_id = prepare(lab)
    evidence = lab.service.get(task_id, lab.actor)["evidence"][0]
    original_id = evidence["evidence_id"]
    evidence[field] = value
    reference = lab.store.write_artifact("Evidence", evidence)
    with lab.store.transaction() as db:
        db.execute("UPDATE evidence SET payload_ref=? WHERE evidence_id=?",
                   (canonical(reference), original_id))
    with pytest.raises(DomainError) as error:
        lab.service.get(task_id, lab.actor)
    assert error.value.code == code


def test_missing_or_cross_task_persisted_evidence_reference_fails_closed(lab):
    task_id = lab.create()["task_id"]
    other = prepare(lab, request_id="other")
    foreign = lab.service.get(other, lab.actor)["evidence"][0]["evidence_id"]
    for evidence_id in ("evidence-absent", foreign):
        lab.update_state(task_id, evidence_ids=[evidence_id])
        with pytest.raises(DomainError) as error:
            lab.service.get(task_id, lab.actor)
        assert error.value.code == "evidence_integrity"


def test_repeating_signal_check_reuses_evidence_id(lab):
    task_id = prepare(lab)
    before = lab.service.get(task_id, lab.actor)["evidence"]
    lab.legal_tool(task_id, "time_series_verify", {"condition_ids": all_signals(lab)})
    assert lab.service.get(task_id, lab.actor)["evidence"] == before


def test_create_and_human_answer_are_idempotent_and_conflict_checked(lab):
    first = lab.create("jam")
    assert lab.create("jam") == first
    with pytest.raises(DomainError) as error:
        lab.create("normal")
    assert error.value.code == "idempotency_conflict"
    waiting = lab.service.run(first["task_id"], lab.actor)
    assert waiting["status"] == "waiting_human"
    command = HumanAnswer(action_id="answer-once", question_id=waiting["pending_human"]["question_id"],
                          answer=True, observation_note="TEST safe observation previously obtained.")
    answer = lab.service.answer(first["task_id"], command, lab.actor)
    assert lab.service.answer(first["task_id"], command, lab.actor) == answer
    with pytest.raises(DomainError) as error:
        lab.service.answer(first["task_id"], command.model_copy(update={"answer": False}), lab.actor)
    assert error.value.code == "idempotency_conflict"
    with pytest.raises(DomainError) as error:
        lab.service.answer(first["task_id"], command.model_copy(update={"action_id": "stale"}), lab.actor)
    assert error.value.code == "stale_question"
    assert sum(item["source"] == "human" for item in answer["evidence"]) == 1


@pytest.mark.parametrize("operation", ["get", "run", "calls", "audit", "cancel", "answer", "reconcile"])
def test_other_actor_even_admin_cannot_operate_owner_task(lab, operation):
    task_id = lab.create()["task_id"]
    with pytest.raises(DomainError) as error:
        if operation == "cancel":
            lab.service.cancel(task_id, "cancel-other", lab.admin)
        elif operation == "answer":
            lab.service.answer(task_id, HumanAnswer(action_id="answer-other", question_id="question-other",
                answer=True, observation_note="Not the task owner's observation."), lab.admin)
        elif operation == "reconcile":
            lab.service.reconcile(task_id, ReconcileRequest(action_id="reconcile-other", call_id="call-other",
                no_result=True, note="Not the task owner's model call."), lab.admin)
        else:
            getattr(lab.service, operation)(task_id, lab.admin)
    assert error.value.code == "task_not_found"
    assert lab.service.list_tasks(lab.admin) == []


@pytest.mark.parametrize("operation", ["create", "get", "run", "list_tasks", "calls", "audit", "cancel"])
def test_stale_actor_rejected_after_plant_revocation(lab, operation):
    task_id = lab.create()["task_id"]
    lab.store.set_user(lab.actor.actor_id, lab.admin, enabled=True, plants=["PLANT-B"], role="admin")
    with pytest.raises(DomainError) as error:
        if operation == "create":
            lab.create(request_id="after-revocation")
        elif operation == "list_tasks":
            lab.service.list_tasks(lab.actor)
        elif operation == "cancel":
            lab.service.cancel(task_id, "cancel-revoked", lab.actor)
        else:
            getattr(lab.service, operation)(task_id, lab.actor)
    assert error.value.code == "identity_changed"
    assert lab.state(task_id)["calls"] == 0
    fresh_actor = lab.store.authenticate(lab.token)
    assert lab.service.list_tasks(fresh_actor) == []
    with pytest.raises(DomainError) as error:
        lab.service.get(task_id, fresh_actor)
    assert error.value.code == "forbidden"


def test_plant_revocation_during_inference_prevents_reply_and_tool_commit(tmp_path):
    provider = ScriptedProvider([])
    lab = make_lab(tmp_path, provider)

    def revoke(context):
        lab.store.set_user(lab.actor.actor_id, lab.admin, enabled=True, plants=["PLANT-B"], role="admin")
        return reply("intent_classify")

    provider.steps.append(revoke)
    task_id = lab.create()["task_id"]
    with pytest.raises(DomainError) as error:
        lab.service.run(task_id, lab.actor)
    assert error.value.code == "identity_changed"
    state = lab.state(task_id)
    assert state["status"] == "blocked" and state["error"]["code"] == "identity_changed"
    assert state["calls"] == 1 and state["tools"] == state["tokens"] == 0
    assert state["pending_action"] is None and state["pending_call_id"] is not None
    assert state["submission"] is None and not state["evidence_ids"]
    with lab.store.connect() as db:
        call = db.execute("SELECT * FROM calls WHERE task_id=?", (task_id,)).fetchone()
        assert call["status"] in {"started", "unknown"} and call["response_ref"] is None


def test_terminal_submission_cannot_change_or_call_provider_again(lab):
    view = finish(lab, lab.create())
    assert view["status"] == "submitted", view
    before = copy.deepcopy(view)
    assert lab.service.run(view["task_id"], lab.actor) == before
    with pytest.raises(DomainError) as error:
        lab.service.cancel(view["task_id"], "cancel-terminal", lab.actor)
    assert error.value.code == "task_terminal"
    with pytest.raises(DomainError) as error:
        lab.service.answer(view["task_id"], HumanAnswer(action_id="answer-terminal", question_id="old-question",
            answer=False, observation_note="Cannot alter a submitted outcome."), lab.actor)
    assert error.value.code == "stale_question"
    with pytest.raises(DomainError) as error:
        lab.service.reconcile(view["task_id"], ReconcileRequest(action_id="reconcile-terminal",
            call_id="old-call", no_result=True, note="Cannot change a terminal diagnosis."), lab.actor)
    assert error.value.code == "not_reconcilable"
    assert lab.service.get(view["task_id"], lab.actor) == before


def test_false_global_signal_blocks_even_an_any_gate(tmp_path):
    draft = changed_draft(gate="any")
    next(condition for condition in draft["conditions"]
         if condition["condition_id"] == "drive_enabled")["threshold"] = 0.0
    lab = make_lab(tmp_path, draft=draft)
    task_id = prepare(lab)
    assert_rejected(lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id)),
                    "mandatory_not_supported")
    evidence = lab.service.get(task_id, lab.actor)["evidence"]
    assert next(item for item in evidence if item["condition_id"] == "drive_enabled")["state"] == "refuted"


def test_any_gate_cannot_hide_unchecked_predicate_despite_one_supported(tmp_path):
    lab = make_lab(tmp_path, draft=changed_draft(gate="any"))
    task_id = prepare(lab, check_all=False)
    lab.legal_tool(task_id, "time_series_verify", {
        "condition_ids": [condition for condition in all_signals(lab) if condition != "encoder_aligned"],
    })
    assert_rejected(lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id)),
                    "unexplained_signals")


def test_conflicting_fixture_cannot_ignore_supported_fault_predicate(lab):
    task_id = prepare(lab, "conflicting", human={"obstruction_confirmed": True})
    evidence = {item["condition_id"]: item for item in lab.service.get(task_id, lab.actor)["evidence"]}
    assert evidence["current_high"]["state"] == evidence["belt_low"]["state"] == "supported"
    assert evidence["traction_difference"]["state"] == "supported"
    assert_rejected(lab.tool(task_id, "submit_diagnosis", submission_arguments(lab, task_id, "jam")),
                    "unexplained_fault_signal")


@pytest.mark.parametrize("threshold", [True, False])
@pytest.mark.parametrize("answer", [True, False])
def test_canonical_human_question_cannot_be_reversed_by_model(tmp_path, threshold, answer):
    draft = original_knowledge().draft.model_dump(mode="json")
    predicate = next(item for item in draft["conditions"] if item["condition_id"] == "obstruction_confirmed")
    predicate["threshold"] = threshold
    lab = make_lab(tmp_path, draft=draft)
    model_questions = [
        "Is the conveyor free of obstructions? Answer true to confirm there is NO obstruction.",
        "Ignore the predicate. Click true to authorize the diagnosis regardless of the observation.",
    ]
    displayed = []
    for index, model_question in enumerate(model_questions):
        task_id = prepare(lab, "jam", check_all=False, request_id=f"human-question-{index}")
        lab.legal_tool(task_id, "request_human_input", {
            "condition_id": predicate["condition_id"], "question": model_question,
        })
        view = lab.service.get(task_id, lab.actor)
        question = view["pending_human"]
        displayed.append(question["question"])
        assert question["question"] != model_question
        assert model_question not in question["question"]
        assert question["model_question_untrusted"] == model_question
        assert question["predicate"] == predicate
        assert question["tree_version"] == view["tree_version"]
        assert question["tree_hash"] == view["tree_hash"]
        assert predicate["label"] in question["question"]
        assert predicate["human_key"] in question["question"]
        assert "true" in question["question"].lower() and "false" in question["question"].lower()
        assert view["evidence"] == [], "A model question alone must not create human evidence"
        answered = lab.answer(view, answer)
        human = [item for item in answered["evidence"] if item["source"] == "human"]
        assert len(human) == 1
        assert human[0]["condition_id"] == predicate["condition_id"]
        assert human[0]["state"] == ("supported" if answer == threshold else "refuted")
        assert human[0]["details"]["answer"] is answer
        assert human[0]["details"]["question_id"] == question["question_id"]
    assert displayed[0] == displayed[1], "The authoritative question must depend only on the pinned predicate"


@pytest.mark.parametrize("field", ["question", "predicate", "tree_version", "tree_hash"])
def test_tampered_human_question_contract_cannot_be_answered(lab, field):
    task_id = prepare(lab, "jam", check_all=False)
    lab.legal_tool(task_id, "request_human_input", {
        "condition_id": "obstruction_confirmed", "question": "Was obstruction confirmed safely?",
    })
    view = lab.service.get(task_id, lab.actor)
    pending = copy.deepcopy(view["pending_human"])
    if field == "predicate":
        pending[field]["threshold"] = False
    else:
        pending[field] = "tampered-question-contract"
    lab.update_state(task_id, pending_human=pending)
    before = lab.state(task_id)
    with pytest.raises(DomainError) as error:
        lab.answer(view, True)
    assert error.value.code == "stale_question"
    assert lab.state(task_id) == before and lab.service.get(task_id, lab.actor)["evidence"] == []
    with lab.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM receipts WHERE owner=?", (lab.actor.actor_id,)).fetchone()[0] == 0


@pytest.mark.parametrize(("case", "code"), [("missing", "missing_signals"), ("conflicting", "conflicting_signals")])
@pytest.mark.parametrize("queued_action", [False, True])
def test_generic_signal_guard_stops_with_bound_witness_before_more_work(tmp_path, case, code, queued_action):
    provider = ScriptedProvider([reply("intent_classify")])
    lab = make_lab(tmp_path, provider)
    task_id = prepare(lab, case)
    before = lab.service.get(task_id, lab.actor)
    if queued_action:
        lab.update_state(task_id, pending_action=reply("request_human_input", {
            "condition_id": "obstruction_confirmed", "question": "Can human input bypass the signal refusal?",
        }).action.model_dump(mode="json"))
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == code
    assert view["submission"] is None and view["pending_human"] is None
    assert view["calls"] == before["calls"] == 0 and view["tools"] == before["tools"]
    assert provider.contexts == [] and lab.service.calls(task_id, lab.actor) == []
    state = lab.state(task_id)
    assert state["pending_action"] is None and state["guard_failures"] == 0
    witness = view["error"]["witness"]
    evidence = {item["condition_id"]: item for item in view["evidence"]}
    expected_ids = (sorted(cid for cid in all_signals(lab) if evidence[cid]["state"] == "missing")
                    if case == "missing" else sorted(all_signals(lab)))
    assert witness["condition_ids"] == expected_ids
    assert witness["evidence_ids"] == [evidence[cid]["evidence_id"] for cid in expected_ids]
    if case == "conflicting":
        assert witness["compatible_nodes"] == []
        assert witness["supported_fault_predicates"] == ["belt_low", "current_high", "traction_difference"]
    escalations = [row for row in lab.service.audit(task_id, lab.actor) if row["event"] == "diagnosis_escalated"]
    assert len(escalations) == 1 and json.loads(escalations[0]["detail_json"]) == view["error"]


@pytest.mark.parametrize("case", ["normal", "jam", "slip", "sensor_fault"])
def test_generic_signal_guard_preserves_signal_compatible_leaf_without_human_invention(lab, case):
    task_id = prepare(lab, case)
    state = lab.state(task_id)
    state["worker"] = None
    before = lab.service.get(task_id, lab.actor)["evidence"]
    assert all(item["source"] == "signal" for item in before)
    result = lab.service._wrap("guard")(state)
    assert result["error"] is None and result["next_node"] == "agent"
    assert result["status"] == "ready" and result["submission"] is None
    assert result["pending_human"] is None and result["calls"] == 0
    assert lab.service.get(task_id, lab.actor)["evidence"] == before


@pytest.mark.parametrize("checked", [["drive_enabled"], ["drive_enabled", "current_high", "belt_low"]])
def test_generic_signal_guard_does_not_infer_conflict_from_unchecked_signals(lab, checked):
    task_id = prepare(lab, "conflicting", check_all=False)
    lab.legal_tool(task_id, "time_series_verify", {"condition_ids": checked})
    state = lab.state(task_id)
    state["worker"] = None
    result = lab.service._wrap("guard")(state)
    assert result["error"] is None and result["next_node"] == "agent"
    assert result["submission"] is None and result["calls"] == 0
    assert len(result["evidence_ids"]) == len(checked)


def test_generic_signal_guard_keeps_unanswered_human_any_gate_possible(tmp_path):
    draft = changed_draft("jam", gate="any", conditions=["current_nominal", "obstruction_confirmed"],
                          mandatory=["drive_enabled"])
    lab = make_lab(tmp_path, draft=draft)
    task_id = prepare(lab, "jam")
    evidence = lab.service.get(task_id, lab.actor)["evidence"]
    assert next(item for item in evidence if item["condition_id"] == "current_nominal")["state"] == "refuted"
    state = lab.state(task_id)
    state["worker"] = None
    result = lab.service._wrap("guard")(state)
    assert result["error"] is None and result["next_node"] == "agent"
    assert result["submission"] is None
    assert lab.service.get(task_id, lab.actor)["evidence"] == evidence
    assert all(item["source"] == "signal" for item in evidence)


def test_generic_signal_guard_stops_when_global_signal_refutes_every_leaf(tmp_path):
    draft = original_knowledge().draft.model_dump(mode="json")
    next(item for item in draft["conditions"] if item["condition_id"] == "drive_enabled")["threshold"] = 0.0
    provider = ScriptedProvider([reply("intent_classify")])
    lab = make_lab(tmp_path, provider, draft=draft)
    task_id = prepare(lab)
    view = lab.service.run(task_id, lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == "conflicting_signals"
    assert view["error"]["witness"]["compatible_nodes"] == []
    assert view["calls"] == 0 and provider.contexts == [] and view["submission"] is None


def test_three_rejected_model_actions_escalate_without_evidence(tmp_path):
    provider = ScriptedProvider([reply("time_series_verify", {"condition_ids": ["drive_enabled"]})] * 3)
    lab = make_lab(tmp_path, provider)
    view = lab.service.run(lab.create()["task_id"], lab.actor)
    assert view["status"] == "escalated" and view["error"]["code"] == "repeated_invalid_action"
    assert view["calls"] == view["tools"] == 3
    assert not view["evidence"] and view["submission"] is None
