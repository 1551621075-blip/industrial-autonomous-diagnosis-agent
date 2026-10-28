from __future__ import annotations

import copy
import json
import re
import time
from contextlib import nullcontext
from contextvars import ContextVar
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import ValidationError

from .schemas import (
    ARGUMENT_SCHEMAS,
    Action,
    Actor,
    DiagnosisRequest,
    DomainError,
    Evidence,
    HumanAnswer,
    KnowledgeDraft,
    ModelReply,
    ReconcileRequest,
    SubmitArgs,
    canonical,
    digest,
)
from .storage import Store, new_id, now

NODES = ["agent", "tools", "guard", "compact", "escalate"]
TERMINAL = {"submitted", "escalated", "cancelled"}
MAX_CALLS, MAX_TOOLS, MAX_TOKENS, MAX_SECONDS = 20, 32, 64000, 300
LEASE_SECONDS = 600
_NODE_CLOCK: ContextVar[float | None] = ContextVar("diagnosis_node_clock", default=None)


class GraphState(TypedDict, total=False):
    task_id: str
    worker: str
    version: int
    status: str
    next_node: str
    actor: dict
    request: dict
    tree_ref: dict
    tree_version: str
    tree_hash: str
    data_ref: dict
    data_hash: str
    provider: str
    calls: int
    tools: int
    tokens: int
    elapsed_s: float
    observations: list[dict]
    evidence_ids: list[str]
    expanded_nodes: list[str]
    candidates: list[str]
    intent: dict | None
    pending_action: dict | None
    pending_call_id: str | None
    pending_human: dict | None
    submission: dict | None
    error: dict | None
    guard_failures: int
    compact_refs: list[dict]
    limits: dict
    node_started_at: float | None


class DiagnosisService:
    def __init__(self, store: Store, knowledge, telemetry, provider, *, max_calls=MAX_CALLS,
                 max_tools=MAX_TOOLS, max_tokens=MAX_TOKENS, max_seconds=MAX_SECONDS):
        self.store, self.knowledge, self.telemetry, self.provider = store, knowledge, telemetry, provider
        self.limits = {"calls": max_calls, "tools": max_tools, "tokens": max_tokens, "seconds": max_seconds}
        if any(type(value) not in (int, float) or value <= 0 for value in self.limits.values()):
            raise ValueError("Budgets must be positive")
        graph = StateGraph(GraphState)
        for name in NODES:
            graph.add_node(name, self._wrap(name))
        graph.add_edge(START, "guard")
        for name in NODES:
            graph.add_conditional_edges(name, self._route, {n: n for n in NODES} | {"end": END})
        self.graph = graph.compile()

    @staticmethod
    def _route(state):
        return state.get("next_node", "end")

    def _owned(self, task_id: str, actor: Actor, db=None):
        if db is None:
            with self.store.connect() as connection:
                return self._owned(task_id, actor, connection)
        self.store.current_actor(actor, db)
        row = db.execute("SELECT * FROM tasks WHERE task_id=? AND owner=?", (task_id, actor.actor_id)).fetchone()
        if row is None:
            raise DomainError("task_not_found", "Task unavailable", 404)
        state = json.loads(row["state_json"])
        self.store.authorize(actor, state["request"]["plant_id"], db=db)
        return row, state

    def _persist(self, db, state):
        row = db.execute("SELECT version,lease_owner FROM tasks WHERE task_id=?", (state["task_id"],)).fetchone()
        if row is None or row["version"] != state["version"] or row["lease_owner"] != state.get("worker"):
            raise DomainError("worker_fenced", "Stale worker cannot modify task", 409)
        state["version"] += 1
        db.execute("UPDATE tasks SET state_json=?,version=?,lease_until=? WHERE task_id=?",
                   (canonical(state), state["version"], now() + LEASE_SECONDS if state.get("worker") else None, state["task_id"]))

    def _save(self, state):
        # A worker may persist its blocked status after revocation, never another tool result.
        with self.store.transaction() as db:
            self._persist(db, state)

    def _snapshot_valid(self, state, db=None):
        actor = Actor.model_validate(state["actor"])
        self.store.current_actor(actor, db)
        request = DiagnosisRequest.model_validate(state["request"])
        tree = self.knowledge.active(request.plant_id, actor, db=db)
        if tree["version_id"] != state["tree_version"] or tree["content_hash"] != state["tree_hash"]:
            raise DomainError("knowledge_changed", "Active knowledge changed; create a new task", 409)
        data = self.telemetry.event(request.plant_id, request.device_id, request.event_id, request.window, actor)
        if data["data_hash"] != state["data_hash"] or self.provider.fingerprint != state["provider"]:
            raise DomainError("snapshot_changed", "Data or model changed; create a new task", 409)
        if digest(self.store.read_artifact(state["tree_ref"])) != digest(tree["draft"]):
            raise DomainError("knowledge_integrity", "Pinned knowledge differs from published version", 409)
        stored = self.store.read_artifact(state["data_ref"])
        if stored != data:
            raise DomainError("snapshot_changed", "Pinned data does not match current source", 409)

    def create(self, request: DiagnosisRequest, actor: Actor):
        self.store.authorize(actor, request.plant_id)
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE owner=? AND request_id=?",
                             (actor.actor_id, request.request_id)).fetchone()
        if row:
            if row["request_hash"] != digest(request.model_dump(mode="json")):
                raise DomainError("idempotency_conflict", "Request ID reused with different content", 409)
            return self.get(row["task_id"], actor)
        tree = self.knowledge.active(request.plant_id, actor)
        draft = KnowledgeDraft.model_validate(tree["draft"])
        data = self.telemetry.event(request.plant_id, request.device_id, request.event_id, request.window, actor)
        state: GraphState = {
            "task_id": new_id("task"), "version": 0, "worker": "", "status": "ready", "next_node": "guard",
            "actor": actor.model_dump(mode="json"), "request": request.model_dump(mode="json"),
            "tree_ref": self.store.write_artifact("KnowledgeSnapshot", draft.model_dump(mode="json")),
            "tree_version": tree["version_id"], "tree_hash": tree["content_hash"],
            "data_ref": self.store.write_artifact("TelemetrySnapshot", data), "data_hash": data["data_hash"],
            "provider": self.provider.fingerprint, "calls": 0, "tools": 0, "tokens": 0, "elapsed_s": 0.0,
            "observations": [], "evidence_ids": [], "expanded_nodes": [], "candidates": [], "intent": None,
            "pending_action": None, "pending_call_id": None, "pending_human": None, "submission": None,
            "error": None, "guard_failures": 0, "compact_refs": [],
            "limits": self.limits.copy(), "node_started_at": None,
        }
        with self.store.transaction() as db:
            self.store.current_actor(actor, db)
            existing = db.execute("SELECT * FROM tasks WHERE owner=? AND request_id=?",
                                  (actor.actor_id, request.request_id)).fetchone()
            if existing:
                if existing["request_hash"] != digest(state["request"]):
                    raise DomainError("idempotency_conflict", "Request ID collision", 409)
                return self._view(json.loads(existing["state_json"]), actor)
            db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,0,NULL,NULL)", (
                state["task_id"], actor.actor_id, request.request_id, digest(state["request"]), canonical(state)))
            self.store.audit(db, actor.actor_id, "task_created", {"tree_version": state["tree_version"]}, state["task_id"])
        return self.get(state["task_id"], actor)

    def run(self, task_id: str, actor: Actor):
        worker = new_id("worker")
        with self.store.transaction() as db:
            row, state = self._owned(task_id, actor, db)
            if state["status"] in TERMINAL | {"waiting_human"}:
                return self._view(state, actor)
            if row["lease_owner"] and row["lease_until"] > now():
                raise DomainError("task_busy", "Task already has an active worker", 409)
            if state["actor"] != actor.model_dump(mode="json"):
                raise DomainError("identity_changed", "Task authorization snapshot changed", 403)
            if state["node_started_at"] is not None:
                # Charge an interrupted node conservatively; resuming never resets the budget.
                state["elapsed_s"] += max(0, now() - state["node_started_at"])
                state["node_started_at"] = None
            state["worker"] = worker
            state["status"] = "running"
            db.execute("UPDATE tasks SET lease_owner=? WHERE task_id=?", (worker, task_id))
            self._persist(db, state)
        try:
            session = getattr(self.provider, "session", None)
            with tracing_context(enabled=False), session() if callable(session) else nullcontext():
                self.graph.invoke(state, {"recursion_limit": 180, "callbacks": []})
        finally:
            with self.store.transaction() as db:
                db.execute("UPDATE tasks SET lease_owner=NULL,lease_until=NULL WHERE task_id=? AND lease_owner=?",
                           (task_id, worker))
        return self.get(task_id, actor)

    @staticmethod
    def _elapsed(state):
        started = _NODE_CLOCK.get()
        return state["elapsed_s"] + (max(0.0, time.monotonic() - started) if started is not None else 0.0)

    def _require_time(self, state):
        if self._elapsed(state) >= state["limits"]["seconds"]:
            raise DomainError("budget_exhausted", "Cumulative time exhausted before dispatch or submission", 409)

    def _wrap(self, node):
        def invoke(incoming):
            state = copy.deepcopy(incoming)
            if state["status"] in TERMINAL:
                state["next_node"] = "end"
                return state
            started = time.monotonic()
            clock_token = _NODE_CLOCK.set(started)
            try:
                state["node_started_at"] = now()
                self._save(state)
                try:
                    if node not in {"escalate"}:
                        self._snapshot_valid(state)
                    getattr(self, "_" + node)(state)
                except DomainError as error:
                    if error.code == "worker_fenced":
                        raise
                    state["error"] = {"code": error.code, "message": error.message}
                    if error.code == "budget_exhausted":
                        state["pending_action"] = None
                        state["next_node"] = "escalate"
                    else:
                        state["status"], state["next_node"] = "blocked", "end"
                finally:
                    state["elapsed_s"] += max(0.0, time.monotonic() - started)
                    state["node_started_at"] = None
                self._save(state)
                return state
            finally:
                _NODE_CLOCK.reset(clock_token)
        return invoke

    def _guard(self, state):
        if state["status"] in TERMINAL | {"waiting_human"}:
            state["next_node"] = "end"
            return
        if state.get("pending_call_id"):
            with self.store.connect() as db:
                row = db.execute("SELECT * FROM calls WHERE call_id=?", (state["pending_call_id"],)).fetchone()
            if row is None or row["status"] in {"started", "unknown"}:
                raise DomainError("model_unknown", "Uncertain model call requires owner-admin reconciliation", 409)
        refusal = self._signal_refusal(state)
        if refusal is not None:
            state["error"] = refusal
            state["pending_action"] = None
            state["next_node"] = "escalate"
            return
        if state["pending_action"] is not None:
            state["next_node"] = "tools"
            return
        if (state["calls"] >= state["limits"]["calls"] or state["tools"] >= state["limits"]["tools"]
                or state["tokens"] >= state["limits"]["tokens"] or self._elapsed(state) >= state["limits"]["seconds"]):
            state["error"] = {"code": "budget_exhausted", "message": "Diagnosis budget exhausted without valid submission"}
            state["next_node"] = "escalate"
        elif state["guard_failures"] >= 3:
            state["error"] = {"code": "repeated_invalid_action", "message": "Repeated invalid actions require human review"}
            state["next_node"] = "escalate"
        else:
            state["next_node"] = "compact" if len(state["observations"]) > 6 else "agent"

    def _signal_refusal(self, state):
        tree = KnowledgeDraft.model_validate(self.store.read_artifact(state["tree_ref"]))
        conditions = {c.condition_id: c for c in tree.conditions}
        referenced = set(tree.global_required)
        normal = set(tree.global_required)
        faults = set()
        leaves = [node for node in tree.nodes if node.kind != "branch"]
        for leaf in leaves:
            predicates = set(leaf.conditions + leaf.mandatory)
            referenced.update(predicates)
            (normal if leaf.kind == "normal" else faults).update(predicates)
        signals = {cid for cid in referenced if conditions[cid].source == "signal"}
        issued = {e["condition_id"]: e for e in self._evidence(state)}
        missing = [issued[cid] for cid in sorted(signals & issued.keys()) if issued[cid]["state"] == "missing"]
        if missing:
            return {"code": "missing_signals", "message": "Pinned signal evidence is incomplete; human answers cannot replace it",
                    "witness": {"condition_ids": [e["condition_id"] for e in missing],
                                "evidence_ids": [e["evidence_id"] for e in missing]}}
        if not signals <= issued.keys():
            return None
        positives = {cid for cid in (faults - normal) & signals if issued[cid]["state"] == "supported"}
        for leaf in leaves:
            required = set(tree.global_required + leaf.mandatory + leaf.conditions)
            mandatory = set(tree.global_required + leaf.mandatory) & signals
            if any(issued[cid]["state"] != "supported" for cid in mandatory):
                continue
            # Unanswered human predicates remain possible; signal-only stopping never invents their truth.
            gate = [conditions[cid].source == "human" or issued[cid]["state"] == "supported"
                    for cid in leaf.conditions]
            if (all(gate) if leaf.gate == "all" else any(gate)) and positives <= required:
                return None
        return {"code": "conflicting_signals", "message": "No declared leaf can explain the complete signal evidence",
                "witness": {"condition_ids": sorted(signals),
                            "evidence_ids": [issued[cid]["evidence_id"] for cid in sorted(signals)],
                            "supported_fault_predicates": sorted(positives), "compatible_nodes": []}}

    def _context(self, state):
        draft = self.store.read_artifact(state["tree_ref"])
        tree = {key: draft[key] for key in ("root_id", "nodes", "conditions", "global_required")}
        tree["conditions"] = [{key: value for key, value in c.items()
            if key not in {"min_samples", "max_gap_ms", "max_age_ms"}} for c in tree["conditions"]]
        evidence = [{**{key: item[key] for key in ("evidence_id", "condition_id", "state", "source")},
                     "details": {key: value for key, value in item["details"].items()
                                 if key in {"computed", "n", "reason", "answer"}}}
                    for item in self._evidence(state)]
        observations = [{key: value for key, value in item.items() if key in {"tool", "error"}}
                        for item in state["observations"]]
        return {"request": state["request"], "tree": tree,
                "intent": state["intent"], "candidates": state["candidates"],
                "expanded_nodes": state["expanded_nodes"], "evidence": evidence,
                "observations": observations, "pending_human": state["pending_human"],
                "remaining": {name: state["limits"][name] - state[name] for name in ("calls", "tools", "tokens")}}

    def _agent(self, state):
        if (state["calls"] >= state["limits"]["calls"] or state["tools"] >= state["limits"]["tools"]
                or state["tokens"] >= state["limits"]["tokens"] or self._elapsed(state) >= state["limits"]["seconds"]):
            state["error"] = {"code": "budget_exhausted", "message": "Model call blocked by cumulative budget"}
            state["next_node"] = "escalate"
            return
        context = self._context(state)
        call_id = new_id("call")
        context_ref = self.store.write_artifact("ModelContext", context)
        with self.store.transaction() as db:
            self.store.current_actor(Actor.model_validate(state["actor"]), db)
            self._require_time(state)
            state["calls"] += 1
            state["pending_call_id"] = call_id
            db.execute("INSERT INTO calls VALUES(?,?,?,'started',?,NULL,NULL,NULL,?)", (
                call_id, state["task_id"], "decision", canonical(context_ref), now()))
            self._persist(db, state)
        received_tokens = None
        try:
            self._require_time(state)
            reply = self.provider.decision(context)
            received_tokens = reply.tokens if isinstance(reply, ModelReply) else reply.get("tokens") if isinstance(reply, dict) else None
            reply = ModelReply.model_validate(reply.model_dump() if isinstance(reply, ModelReply) else reply)
        except (DomainError, ValidationError) as exc:
            code = exc.code if isinstance(exc, DomainError) else "provider_invalid"
            usage = getattr(exc, "reported_tokens", received_tokens)
            valid_usage = type(usage) is int and 0 <= usage <= 100000
            uncertain_usage = bool(getattr(exc, "usage_unknown", False)) or (
                (isinstance(exc, ValidationError) or hasattr(exc, "reported_tokens")) and not valid_usage
            )
            known = code not in {"provider_unknown", "model_unknown"} and not uncertain_usage
            with self.store.transaction() as db:
                db.execute("UPDATE calls SET status=?,tokens=?,error_code=? WHERE call_id=?",
                           ("failed" if known else "unknown", usage if valid_usage else None, code, call_id))
                if valid_usage:
                    state["tokens"] += usage
                if known:
                    state["pending_call_id"] = None
                self._persist(db, state)
            raise DomainError(code, "Local model did not return a usable action", 502) from None
        self._snapshot_valid(state)
        with self.store.transaction() as db:
            self.store.current_actor(Actor.model_validate(state["actor"]), db)
            response_ref = self.store.write_artifact("ModelReply", reply.model_dump(mode="json"))
            db.execute("UPDATE calls SET status='completed',response_ref=?,tokens=? WHERE call_id=?",
                       (canonical(response_ref), reply.tokens, call_id))
            state["tokens"] += reply.tokens
            state["pending_call_id"] = None
            state["pending_action"] = reply.action.model_dump(mode="json")
            state["next_node"] = "tools"
            self._persist(db, state)

    def _evidence(self, state, db=None):
        if db is None:
            with self.store.connect() as connection:
                return self._evidence(state, connection)
        result = []
        for eid in state["evidence_ids"]:
            row = db.execute("SELECT * FROM evidence WHERE evidence_id=? AND task_id=?", (eid, state["task_id"])).fetchone()
            if row is None:
                raise DomainError("evidence_integrity", "Task evidence missing", 409)
            item = Evidence.model_validate(self.store.read_artifact(json.loads(row["payload_ref"])))
            if item.evidence_id != eid or item.task_id != state["task_id"] or item.condition_id != row["condition_id"]:
                raise DomainError("evidence_integrity", "Evidence identity mismatch", 409)
            request = state["request"]
            if (item.tree_version != state["tree_version"] or item.tree_hash != state["tree_hash"]
                    or item.data_hash != state["data_hash"] or item.window.model_dump(mode="json") != request["window"]
                    or any(getattr(item, key) != request[key] for key in ("plant_id", "device_id", "event_id"))
                    or item.actor_id != state["actor"]["actor_id"]):
                raise DomainError("evidence_scope", "Evidence does not belong to this task snapshot", 409)
            result.append(item.model_dump(mode="json"))
        return result

    def _record_evidence(self, db, state, condition, outcome):
        item = Evidence(evidence_id=new_id("evidence"), task_id=state["task_id"],
            **{key: state["request"][key] for key in ("plant_id", "device_id", "event_id", "window")},
            tree_version=state["tree_version"], tree_hash=state["tree_hash"], data_hash=state["data_hash"],
            condition_id=condition.condition_id, source=condition.source, state=outcome["state"],
            details=outcome["details"], actor_id=state["actor"]["actor_id"])
        reference = self.store.write_artifact("Evidence", item.model_dump(mode="json"))
        db.execute("INSERT INTO evidence VALUES(?,?,?,?,?)", (item.evidence_id, state["task_id"],
            item.condition_id, canonical(reference), now()))
        state["evidence_ids"].append(item.evidence_id)
        return item.model_dump(mode="json")

    def _tools(self, state):
        action = Action.model_validate(state["pending_action"])
        if (state["tools"] >= state["limits"]["tools"] or state["tokens"] > state["limits"]["tokens"]
                or self._elapsed(state) >= state["limits"]["seconds"]):
            state["error"] = {"code": "budget_exhausted", "message": "Tool execution blocked by budget"}
            state["pending_action"] = None
            state["next_node"] = "escalate"
            return
        original = copy.deepcopy(state)
        try:
            args = ARGUMENT_SCHEMAS[action.tool].model_validate(action.arguments)
            tree = KnowledgeDraft.model_validate(self.store.read_artifact(state["tree_ref"]))
            with self.store.transaction() as db:
                self.store.current_actor(Actor.model_validate(state["actor"]), db)
                self._snapshot_valid(state, db)
                self._require_time(state)
                state["tools"] += 1
                result = self._execute(db, state, tree, action.tool, args)
                state["observations"].append({"tool": action.tool, "result": result})
                state["pending_action"] = None
                state["next_node"] = "guard"
                self.store.audit(db, state["actor"]["actor_id"], "tool_completed", {
                    "tool": action.tool, "arguments_hash": digest(action.arguments)}, state["task_id"])
                self._persist(db, state)
        except (ValidationError, DomainError) as exc:
            state.clear()
            state.update(original)
            if isinstance(exc, DomainError) and exc.code in {
                "worker_fenced", "identity_changed", "artifact_integrity", "knowledge_changed", "snapshot_changed",
                "knowledge_integrity", "forbidden", "evidence_integrity", "telemetry_integrity", "budget_exhausted"
            }:
                raise
            state["tools"] += 1
            code = exc.code if isinstance(exc, DomainError) else "invalid_tool_arguments"
            state["guard_failures"] += 1
            state["pending_action"] = None
            state["observations"].append({"tool": action.tool, "error": code})
            state["next_node"] = "guard"
            with self.store.transaction() as db:
                self.store.current_actor(Actor.model_validate(state["actor"]), db)
                self._persist(db, state)
                self.store.audit(db, state["actor"]["actor_id"], "tool_rejected", {"tool": action.tool, "code": code}, state["task_id"])

    def _execute(self, db, state, tree, tool, args):
        nodes = {n.node_id: n for n in tree.nodes}
        conditions = {c.condition_id: c for c in tree.conditions}
        if tool == "intent_classify":
            text = state["request"]["complaint"].lower()
            destructive = any(term in text for term in ("关闭联锁", "绕过联锁", "disable interlock", "write plc"))
            unrelated = any(term in text for term in ("天气预报", "weather forecast", "股票预测"))
            state["intent"] = {"kind": "unsupported" if destructive or unrelated else "conveyor_diagnosis",
                "method": "bounded_local_request_classifier", "not_a_safety_proof": True}
            if destructive or unrelated:
                state["status"] = "escalated"
                state["error"] = {"code": "unsupported_intent", "message": "Request outside read-only conveyor diagnosis"}
            return state["intent"]
        if not state["intent"] or state["intent"]["kind"] != "conveyor_diagnosis":
            raise DomainError("intent_required", "Classify task intent before diagnostic tools")
        if tool == "kg_query_candidates":
            tokens = set(re.findall(r"[a-z]+|[一-鿿]", args.query.lower()))
            ranked = []
            for node in tree.nodes:
                if node.kind == "branch":
                    continue
                target = set(re.findall(r"[a-z]+|[一-鿿]", (node.label + " " + " ".join(node.terms)).lower()))
                ranked.append((len(tokens & target), node.node_id))
            ranked.sort(key=lambda pair: (-pair[0], pair[1]))
            state["candidates"] = [nid for _, nid in ranked[:8]]
            return {"candidates": [{"node_id": nid, "lexical_score": score} for score, nid in ranked[:8]],
                    "note": "Ranking is a knowledge hypothesis, never a verified diagnosis"}
        if tool == "expand_tree_node":
            if args.node_id not in nodes:
                raise DomainError("unknown_node", "Node not in pinned tree")
            if args.node_id != tree.root_id and not any(
                    args.node_id in nodes[nid].children for nid in state["expanded_nodes"]):
                raise DomainError("path_disconnected", "Expand parent before child")
            if args.node_id not in state["expanded_nodes"]:
                state["expanded_nodes"].append(args.node_id)
            node = nodes[args.node_id]
            return {"node": node.model_dump(mode="json"), "children": [nodes[c].model_dump(mode="json") for c in node.children]}
        visible = set(tree.global_required)
        for nid in state["expanded_nodes"]:
            visible.update(nodes[nid].conditions + nodes[nid].mandatory)
            for child in nodes[nid].children:
                visible.update(nodes[child].conditions + nodes[child].mandatory)
        if tool == "time_series_verify":
            if any(cid not in conditions or cid not in visible for cid in args.condition_ids):
                raise DomainError("condition_not_visible", "Expand the owning branch before checking its conditions")
            if any(conditions[cid].source != "signal" for cid in args.condition_ids):
                raise DomainError("human_observation_required", "Human predicates cannot be inferred from signals")
            snapshot = self.store.read_artifact(state["data_ref"])
            prior = {item["condition_id"]: item for item in self._evidence(state, db)}
            results = []
            for cid in args.condition_ids:
                results.append(prior[cid] if cid in prior else self._record_evidence(db, state, conditions[cid],
                    self.telemetry.verify(snapshot, conditions[cid])))
            return {"evidence": results}
        if tool == "request_human_input":
            if args.condition_id not in visible or args.condition_id not in conditions or conditions[args.condition_id].source != "human":
                raise DomainError("invalid_human_condition", "Only visible declared human conditions can be requested")
            if any(e["condition_id"] == args.condition_id for e in self._evidence(state, db)):
                raise DomainError("observation_already_recorded", "Human observation is frozen for this event")
            condition = conditions[args.condition_id]
            state["pending_human"] = {"question_id": new_id("question"), "condition_id": args.condition_id,
                "question": self._human_question(condition), "predicate": condition.model_dump(mode="json"),
                "tree_version": state["tree_version"], "tree_hash": state["tree_hash"],
                "model_question_untrusted": args.question,
                "instruction": "回答的是上述已审核观察事实的真假，不是模型改写的问题，也不是是否满足诊断。仅补充已安全取得的观察；不能确认时不要猜测，可取消或升级。"}
            state["status"] = "waiting_human"
            return state["pending_human"]
        if tool == "submit_diagnosis":
            return self._submit(db, state, tree, args)
        raise DomainError("unknown_tool", "Tool unavailable")

    def _submit(self, db, state, tree, args: SubmitArgs):
        nodes = {n.node_id: n for n in tree.nodes}
        if args.node_id not in nodes or nodes[args.node_id].kind == "branch" or args.node_id not in state["candidates"]:
            raise DomainError("invalid_candidate", "Submit a queried leaf candidate")
        if (args.path[0] != tree.root_id or args.path[-1] != args.node_id or len(set(args.path)) != len(args.path)
                or any(n not in nodes for n in args.path)
                or any(b not in nodes[a].children for a, b in zip(args.path, args.path[1:], strict=False))
                or any(n not in state["expanded_nodes"] for n in args.path[:-1])):
            raise DomainError("path_disconnected", "Submission path must be continuous and inspected")
        evidence = {e["evidence_id"]: e for e in self._evidence(state, db)}
        if len(set(args.evidence_ids)) != len(args.evidence_ids) or any(eid not in evidence for eid in args.evidence_ids):
            raise DomainError("evidence_scope", "Only current task evidence may be submitted")
        bound = {evidence[eid]["condition_id"]: evidence[eid] for eid in args.evidence_ids}
        all_evidence = {e["condition_id"]: e for e in evidence.values()}
        condition_map = {c.condition_id: c for c in tree.conditions}
        referenced = set(tree.global_required)
        for leaf in tree.nodes:
            referenced.update(leaf.conditions + leaf.mandatory)
        signals = {cid for cid in referenced if condition_map[cid].source == "signal"}
        if not signals <= set(all_evidence):
            raise DomainError("unexplained_signals", "Check all declared signal predicates before submission")
        if any(all_evidence[cid]["state"] == "missing" for cid in signals):
            raise DomainError("missing_signals", "Incomplete signal coverage cannot support a final diagnosis")
        node = nodes[args.node_id]
        required = set(tree.global_required + node.mandatory + node.conditions)
        if not required <= set(bound):
            raise DomainError("evidence_missing", "All global, mandatory and leaf conditions need explicit evidence")
        normal_predicates = set(tree.global_required)
        fault_predicates = set()
        for leaf in tree.nodes:
            if leaf.kind == "normal":
                normal_predicates.update(leaf.conditions + leaf.mandatory)
            elif leaf.kind == "diagnosis":
                fault_predicates.update(leaf.conditions + leaf.mandatory)
        unexplained = (fault_predicates - normal_predicates - required) & signals
        if any(all_evidence[c]["state"] == "supported" for c in unexplained):
            raise DomainError("unexplained_fault_signal", "Supported fault-specific predicates lie outside the chosen path")
        mandatory = tree.global_required + node.mandatory
        if any(bound[c]["state"] != "supported" for c in mandatory):
            raise DomainError("mandatory_not_supported", "Mandatory predicates lack support")
        states = [bound[c]["state"] for c in node.conditions]
        if "missing" in states or not (all(s == "supported" for s in states) if node.gate == "all"
                                      else any(s == "supported" for s in states)):
            raise DomainError("candidate_not_supported", "Declared condition gate is not satisfied")
        competing, unresolved = [], []
        for other in tree.nodes:
            if other.kind == "branch" or other.node_id == node.node_id:
                continue
            other_mandatory = tree.global_required + other.mandatory
            mandatory_refuted = any(all_evidence.get(c, {}).get("state") == "refuted" for c in other_mandatory)
            flags = [all_evidence.get(c, {}).get("state", "missing") for c in other.conditions]
            gate_refuted = ("refuted" in flags) if other.gate == "all" else all(f == "refuted" for f in flags)
            if mandatory_refuted or gate_refuted:
                continue
            mandatory_ok = all(all_evidence.get(c, {}).get("state") == "supported" for c in other_mandatory)
            gate_ok = all(f == "supported" for f in flags) if other.gate == "all" else any(f == "supported" for f in flags)
            if mandatory_ok and gate_ok and "missing" not in flags:
                competing.append(other.node_id)
            else:
                unresolved.append(other.node_id)
        if competing:
            raise DomainError("conflicting_candidates", "Multiple fully supported candidates require human review")
        if unresolved:
            raise DomainError("unresolved_candidate", "Competing path still needs declared observations")
        submission = {"submission_id": new_id("diagnosis"), "task_id": state["task_id"],
            "node_id": node.node_id, "label": node.label, "path": args.path, "evidence_ids": args.evidence_ids,
            "tree_version": state["tree_version"], "tree_hash": state["tree_hash"], "data_hash": state["data_hash"],
            "request": state["request"], "actor_id": state["actor"]["actor_id"],
            "conclusion_level": "supported_under_declared_synthetic_rules",
            "limitations": ["规则满足不证明现实物理因果。", "未验证的其他路径不等于已排除。", "不执行维修或设备控制。"]}
        self._require_time(state)
        submission["content_hash"] = digest(submission)
        state["submission"] = self.store.write_artifact("FrozenDiagnosis", submission)
        self._require_time(state)
        state["status"] = "submitted"
        self.store.audit(db, state["actor"]["actor_id"], "diagnosis_submitted", {
            "submission_id": submission["submission_id"], "content_hash": submission["content_hash"]}, state["task_id"])
        return submission

    def _compact(self, state):
        archived = state["observations"][:-4]
        state["compact_refs"].append(self.store.write_artifact("CompactedObservations", archived))
        state["observations"] = state["observations"][-4:]
        state["next_node"] = "agent"

    def _escalate(self, state):
        state["status"], state["next_node"] = "escalated", "end"
        state["pending_action"] = None
        with self.store.transaction() as db:
            self._persist(db, state)
            self.store.audit(db, state["actor"]["actor_id"], "diagnosis_escalated", state["error"] or {}, state["task_id"])

    @staticmethod
    def _human_question(condition):
        return (f"请确认已审核观察事实：{condition.label}\n"
                f"字段 {condition.human_key}：选择 true 表示此事实为真，false 表示此事实为假。\n"
                f"程序将与声明阈值 {str(condition.threshold).lower()} 比较；不要把‘否’改答为‘是’来满足规则。")

    def answer(self, task_id: str, request: HumanAnswer, actor: Actor):
        command = {"task_id": task_id, "type": "answer", **request.model_dump(mode="json")}
        with self.store.transaction() as db:
            row, state = self._owned(task_id, actor, db)
            saved = self.store.receipt(db, actor, request.action_id, command)
            if saved is not None:
                return saved
            if row["lease_owner"] and row["lease_until"] > now():
                raise DomainError("task_busy", "Task has active worker", 409)
            if state["actor"] != actor.model_dump(mode="json"):
                raise DomainError("identity_changed", "Task authorization changed", 403)
            self._snapshot_valid(state, db)
            question = state["pending_human"]
            if state["status"] != "waiting_human" or not question or question["question_id"] != request.question_id:
                raise DomainError("stale_question", "Human question is not current", 409)
            tree = KnowledgeDraft.model_validate(self.store.read_artifact(state["tree_ref"]))
            condition = next(c for c in tree.conditions if c.condition_id == question["condition_id"])
            if (question.get("predicate") != condition.model_dump(mode="json")
                    or question.get("question") != self._human_question(condition)
                    or question.get("tree_version") != state["tree_version"]
                    or question.get("tree_hash") != state["tree_hash"]):
                raise DomainError("stale_question", "Human question does not match the frozen observation contract", 409)
            evidence = self._record_evidence(db, state, condition, {
                "state": "supported" if request.answer == condition.threshold else "refuted",
                "details": {"answer": request.answer, "observation_note": request.observation_note,
                            "question_id": question["question_id"], "question": question["question"],
                            "predicate": question["predicate"], "attested_by": actor.actor_id,
                            "not_sensor_verified": True}})
            state["observations"].append({"tool": "human_answer", "result": evidence})
            state["pending_human"], state["status"], state["next_node"] = None, "ready", "guard"
            db.execute("UPDATE tasks SET lease_owner=NULL,lease_until=NULL WHERE task_id=?", (task_id,))
            state["worker"] = None
            self._persist(db, state)
            result = self._view(state, actor, db)
            self.store.save_receipt(db, actor, request.action_id, command, result)
            self.store.audit(db, actor.actor_id, "human_observation", {"evidence_id": evidence["evidence_id"]}, task_id)
            return result

    def reconcile(self, task_id: str, request: ReconcileRequest, actor: Actor):
        if actor.role != "admin":
            raise DomainError("forbidden", "Owner must also be administrator to reconcile", 403)
        command = {"task_id": task_id, "type": "reconcile", **request.model_dump(mode="json")}
        with self.store.transaction() as db:
            row, state = self._owned(task_id, actor, db)
            saved = self.store.receipt(db, actor, request.action_id, command)
            if saved is not None:
                return saved
            if row["lease_owner"] and row["lease_until"] > now():
                raise DomainError("task_busy", "Task has active worker", 409)
            if state["status"] in TERMINAL or state["pending_call_id"] != request.call_id:
                raise DomainError("not_reconcilable", "No matching uncertain model call", 409)
            call = db.execute("SELECT * FROM calls WHERE call_id=? AND task_id=?", (request.call_id, task_id)).fetchone()
            if call is None or call["status"] not in {"started", "unknown"}:
                raise DomainError("not_reconcilable", "Call outcome is already known", 409)
            recorded_usage = call["tokens"]
            if request.tokens is not None and recorded_usage is not None and request.tokens < recorded_usage:
                raise DomainError("usage_conflict", "Reconciliation cannot reduce already recorded token usage", 409)
            conservative_usage = request.tokens is None and recorded_usage is None
            usage = request.tokens if request.tokens is not None else recorded_usage
            if usage is None:
                usage = max(0, state["limits"]["tokens"] - state["tokens"])
            state["tokens"] += max(0, usage - (recorded_usage or 0))
            if request.reply is not None:
                ARGUMENT_SCHEMAS[request.reply.tool].model_validate(request.reply.arguments)
                ref = self.store.write_artifact("ModelReply", {"action": request.reply.model_dump(mode="json"), "tokens": usage})
                db.execute("UPDATE calls SET status='reconciled',response_ref=?,tokens=? WHERE call_id=?",
                           (canonical(ref), usage, request.call_id))
                state["pending_action"] = request.reply.model_dump(mode="json")
            else:
                db.execute("UPDATE calls SET status='confirmed_no_result',tokens=? WHERE call_id=?", (usage, request.call_id))
            state["pending_call_id"] = None
            state["status"], state["next_node"], state["error"] = "ready", "guard", None
            db.execute("UPDATE tasks SET lease_owner=NULL,lease_until=NULL WHERE task_id=?", (task_id,))
            state["worker"] = None
            self._persist(db, state)
            result = self._view(state, actor, db)
            self.store.save_receipt(db, actor, request.action_id, command, result)
            self.store.audit(db, actor.actor_id, "model_reconciled", {"call_id": request.call_id,
                "no_result": request.no_result, "tokens": usage, "conservative_usage": conservative_usage,
                "note": request.note, "human_attestation_not_external_proof": True}, task_id)
            return result

    def cancel(self, task_id: str, action_id: str, actor: Actor):
        command = {"task_id": task_id, "type": "cancel"}
        with self.store.transaction() as db:
            row, state = self._owned(task_id, actor, db)
            saved = self.store.receipt(db, actor, action_id, command)
            if saved is not None:
                return saved
            if row["lease_owner"] and row["lease_until"] > now():
                raise DomainError("task_busy", "Cannot cancel in-flight worker", 409)
            if state["status"] in TERMINAL:
                raise DomainError("task_terminal", "Frozen terminal outcome cannot change", 409)
            state["status"], state["next_node"] = "cancelled", "end"
            state["pending_human"] = None
            db.execute("UPDATE tasks SET lease_owner=NULL,lease_until=NULL WHERE task_id=?", (task_id,))
            state["worker"] = None
            self._persist(db, state)
            result = self._view(state, actor, db)
            self.store.save_receipt(db, actor, action_id, command, result)
            self.store.audit(db, actor.actor_id, "task_cancelled", {}, task_id)
            return result

    def _view(self, state, actor, db=None):
        value = {key: copy.deepcopy(state.get(key)) for key in (
            "task_id", "status", "request", "tree_version", "tree_hash", "data_hash", "provider", "calls", "tools",
            "tokens", "elapsed_s", "pending_human", "error", "candidates", "expanded_nodes", "pending_call_id")}
        value["evidence"] = self._evidence(state, db)
        value["submission"] = self.store.read_artifact(state["submission"]) if state["submission"] else None
        value["compaction_count"] = len(state["compact_refs"])
        value["limits"] = state["limits"].copy()
        return value

    def get(self, task_id: str, actor: Actor):
        _, state = self._owned(task_id, actor)
        return self._view(state, actor)

    def list_tasks(self, actor: Actor):
        self.store.current_actor(actor)
        with self.store.connect() as db:
            rows = db.execute("SELECT state_json FROM tasks WHERE owner=? ORDER BY rowid DESC LIMIT 100", (actor.actor_id,)).fetchall()
        return [self._view(s, actor) for row in rows if (s := json.loads(row[0]))["request"]["plant_id"] in actor.plants]

    def calls(self, task_id: str, actor: Actor):
        self._owned(task_id, actor)
        with self.store.connect() as db:
            return [dict(row) for row in db.execute("SELECT call_id,kind,status,tokens,error_code,created_at FROM calls WHERE task_id=? ORDER BY rowid", (task_id,))]

    def audit(self, task_id: str, actor: Actor):
        self._owned(task_id, actor)
        with self.store.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM audit WHERE task_id=? ORDER BY id", (task_id,))]
