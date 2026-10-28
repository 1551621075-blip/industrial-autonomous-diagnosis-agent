from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from industrial_diagnosis.fixtures import original_knowledge
from industrial_diagnosis.schemas import (
    Condition,
    DiagnosisRequest,
    DomainError,
    HumanAnswer,
    KnowledgeBuildRequest,
    ReconcileRequest,
    SignalPoint,
    Window,
    canonical,
    digest,
)
from industrial_diagnosis.storage import Store


def test_artifact_atomic_content_and_concurrent_writers(tmp_path):
    store = Store(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(lambda _: store.write_artifact("Example", {"a": [1, 2, 3]}), range(32)))
    assert all(ref == refs[0] for ref in refs)
    assert store.read_artifact(refs[0]) == {"a": [1, 2, 3]}
    assert not list(store.artifacts.glob(".pending-*"))
    assert (store.artifacts / (refs[0]["sha256"] + ".json")).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("damage", ["bytes", "hash", "kind", "symlink", "content", "missing"])
def test_artifact_integrity(tmp_path, damage):
    store = Store(tmp_path)
    ref = store.write_artifact("Example", {"value": 1})
    path = store.artifacts / (ref["sha256"] + ".json")
    if damage == "bytes":
        ref["bytes"] += 1
    elif damage == "hash":
        ref["sha256"] = "../bad"
    elif damage == "kind":
        ref["kind"] = "Other"
    elif damage == "symlink":
        copy_path = tmp_path / "own-test-copy.json"
        copy_path.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy_path)
    elif damage == "content":
        path.write_text("changed")
    else:
        path.unlink()
    with pytest.raises(DomainError, match="Artifact|artifact"):
        store.read_artifact(ref)


def test_artifact_refuses_oversize_and_collision(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(DomainError) as exc:
        store.write_artifact("Large", "a" * 4_000_000)
    assert exc.value.code == "artifact_too_large"
    ref = store.write_artifact("Object", {"key": 1})
    (store.artifacts / (ref["sha256"] + ".json")).write_text("changed")
    with pytest.raises(DomainError) as exc:
        store.write_artifact("Object", {"key": 1})
    assert exc.value.code == "artifact_integrity"


def test_transaction_rolls_back(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(RuntimeError), store.transaction() as db:
        store.audit(db, "actor", "event", {})
        raise RuntimeError("simulated crash")
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 0


def test_auth_current_acl_and_scoped_admin(tmp_path):
    store = Store(tmp_path)
    token = store.provision_user("admin", "admin", ["PLANT-A"])
    admin = store.authenticate(token)
    other_token = store.provision_user("other", "operator", ["PLANT-B"])
    target_token = store.provision_user("target", "analyst", ["PLANT-A"])
    target = store.authenticate(target_token)
    with store.connect() as db:
        assert token not in json.dumps([dict(row) for row in db.execute("SELECT * FROM users")])
    with pytest.raises(DomainError) as exc:
        store.set_user("other", admin, enabled=False, plants=["PLANT-B"], role="operator")
    assert exc.value.code == "forbidden"
    store.set_user("target", admin, enabled=True, plants=["PLANT-A"], role="operator")
    with pytest.raises(DomainError) as exc:
        store.current_actor(target)
    assert exc.value.code == "identity_changed"
    assert store.authenticate(target_token).role == "operator"
    with pytest.raises(DomainError):
        store.authorize(store.authenticate(other_token), "PLANT-A")
    store.set_user("target", admin, enabled=False, plants=["PLANT-A"], role="operator")
    with pytest.raises(DomainError):
        store.authenticate(target_token)


@pytest.mark.parametrize("token", [None, "", "short", "x" * 201, "wrong" * 8])
def test_auth_denies_invalid_tokens(tmp_path, token):
    with pytest.raises(DomainError) as exc:
        Store(tmp_path).authenticate(token)
    assert exc.value.code == "unauthorized"


def test_idempotent_receipt_owner_and_content(tmp_path):
    store = Store(tmp_path)
    actor = store.authenticate(store.provision_user("admin", "admin", ["PLANT-A"]))
    with store.transaction() as db:
        assert store.receipt(db, actor, "action", {"a": 1}) is None
        store.save_receipt(db, actor, "action", {"a": 1}, {"result": 2})
        assert store.receipt(db, actor, "action", {"a": 1}) == {"result": 2}
        with pytest.raises(DomainError) as exc:
            store.receipt(db, actor, "action", {"a": 2})
        assert exc.value.code == "idempotency_conflict"


@pytest.mark.parametrize("start,end", [(True, 1000), ("0", 1000), (0, "30000"), (0, 0), (10, 5), (0, 3600001), (-1, 500)])
def test_window_strict_and_bounded(start, end):
    with pytest.raises(ValidationError):
        Window(start_ms=start, end_ms=end)


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_human_boolean_requires_real_boolean(value):
    with pytest.raises(ValidationError):
        HumanAnswer(action_id="answer", question_id="question", answer=value,
                    observation_note="Original safe synthetic observation")


def test_schema_extra_nonfinite_and_hash_stability():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        canonical({"value": float("nan")})
    with pytest.raises(ValidationError):
        SignalPoint(timestamp_ms=0, value=float("inf"))
    with pytest.raises(ValidationError):
        DiagnosisRequest(request_id="id", plant_id="PLANT-A", device_id="D", event_id="E",
                         window={"start_ms": 0, "end_ms": 1000}, complaint="conveyor", tool="unexpected")


@pytest.mark.parametrize("damage", ["cycle", "disconnected", "unknown_condition", "branch_conditions", "duplicate_node", "multiple_parent", "root_leaf", "report_ids"])
def test_knowledge_tree_structural_rejection(damage):
    value = original_knowledge().model_dump(mode="json")
    tree = value["draft"]
    if damage == "cycle":
        tree["nodes"][1]["children"].append("root")
    elif damage == "disconnected":
        tree["nodes"][0]["children"].remove("sensing")
    elif damage == "unknown_condition":
        tree["global_required"] = ["missing"]
    elif damage == "branch_conditions":
        tree["nodes"][0]["conditions"] = ["drive_enabled"]
    elif damage == "duplicate_node":
        tree["nodes"].append(copy.deepcopy(tree["nodes"][-1]))
    elif damage == "multiple_parent":
        tree["nodes"][0]["children"].append("jam")
    elif damage == "root_leaf":
        tree["root_id"] = "jam"
    else:
        value["reports"].pop()
    with pytest.raises(ValidationError):
        KnowledgeBuildRequest.model_validate(value)


@pytest.mark.parametrize("extra", [{"signal": None}, {"threshold": True}, {"upper": 3.0}, {"statistic": "difference_mean"}, {"human_key": "obs"}])
def test_signal_condition_contract(extra):
    value = {"condition_id": "c", "label": "Motor current", "source": "signal", "signal": "motor_current",
             "operator": "gt", "threshold": 8.0, "unit": "A", **extra}
    with pytest.raises(ValidationError):
        Condition.model_validate(value)


@pytest.mark.parametrize("no_result,reply,tokens", [(False, None, None), (True, None, -1), (True, {"tool": "intent_classify", "arguments": {}, "reason": "test"}, 2), (False, {"tool": "intent_classify", "arguments": {}, "reason": "test"}, None), (False, {"tool": "intent_classify", "arguments": {}, "reason": "test"}, 0)])
def test_reconcile_requires_exactly_one_outcome(no_result, reply, tokens):
    with pytest.raises(ValidationError):
        ReconcileRequest(action_id="action", call_id="call", no_result=no_result,
                         reply=reply, tokens=tokens, note="Reviewed original local test record")


@pytest.mark.parametrize("tokens", [None, 0, 4000, 100000])
def test_reconcile_no_result_distinguishes_unknown_and_attested_usage(tokens):
    request = ReconcileRequest(action_id="action", call_id="call", no_result=True,
                               tokens=tokens, note="Explicit synthetic usage attestation")
    assert request.tokens == tokens


@pytest.mark.parametrize("tokens", [True, "0", 0.0, -1, 100001])
def test_reconcile_rejects_invalid_attested_usage(tokens):
    with pytest.raises(ValidationError):
        ReconcileRequest(action_id="action", call_id="call", no_result=True,
                         tokens=tokens, note="Explicit synthetic usage attestation")
