from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class DomainError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Actor(StrictModel):
    actor_id: Identifier
    role: Literal["operator", "analyst", "admin"]
    plants: list[Identifier]
    acl_version: int = Field(ge=1)


class Window(StrictModel):
    start_ms: int = Field(ge=0, le=2**53 - 1)
    end_ms: int = Field(gt=0, le=2**53 - 1)

    @model_validator(mode="after")
    def valid_window(self):
        if not 0 < self.end_ms - self.start_ms <= 3_600_000:
            raise ValueError("Window must be nonempty and at most one hour")
        return self


class DiagnosisRequest(StrictModel):
    request_id: Identifier
    plant_id: Identifier
    device_id: Identifier
    event_id: Identifier
    window: Window
    complaint: str = Field(min_length=2, max_length=2000)


class Condition(StrictModel):
    condition_id: Identifier
    label: str = Field(min_length=2, max_length=200)
    source: Literal["signal", "human"]
    signal: Literal["motor_current", "belt_speed", "roller_speed", "encoder_speed", "drive_enabled"] | None = None
    statistic: Literal["mean", "max", "min", "difference_mean"] = "mean"
    other_signal: Literal["belt_speed", "roller_speed", "encoder_speed"] | None = None
    operator: Literal["gt", "lt", "between", "eq"]
    threshold: float | bool
    upper: float | None = None
    unit: str = Field(min_length=1, max_length=20)
    min_samples: int = Field(default=20, ge=2, le=1000)
    max_gap_ms: int = Field(default=1500, ge=1, le=60000)
    max_age_ms: int = Field(default=1500, ge=0, le=60000)
    human_key: Identifier | None = None

    @model_validator(mode="after")
    def source_contract(self):
        if self.source == "signal":
            if self.signal is None or self.human_key is not None or isinstance(self.threshold, bool):
                raise ValueError("Signal condition requires a numeric threshold and signal")
            if (self.statistic == "difference_mean") != (self.other_signal is not None):
                raise ValueError("Difference statistic requires exactly one other signal")
        elif (self.human_key is None or self.signal is not None or self.other_signal is not None
              or self.operator != "eq" or not isinstance(self.threshold, bool)):
            raise ValueError("Human condition requires a boolean equality observation")
        if self.operator == "between" and (self.upper is None or self.upper < self.threshold):
            raise ValueError("Between requires ordered bounds")
        if self.operator != "between" and self.upper is not None:
            raise ValueError("Upper bound only belongs to between")
        return self


class TreeNode(StrictModel):
    node_id: Identifier
    label: str = Field(min_length=2, max_length=200)
    kind: Literal["branch", "diagnosis", "normal"]
    children: list[Identifier] = Field(default_factory=list, max_length=12)
    gate: Literal["all", "any"] = "all"
    conditions: list[Identifier] = Field(default_factory=list, max_length=12)
    mandatory: list[Identifier] = Field(default_factory=list, max_length=12)
    terms: list[str] = Field(default_factory=list, max_length=20)


class KnowledgeDraft(StrictModel):
    tree_id: Identifier
    plant_id: Identifier
    title: str = Field(min_length=2, max_length=200)
    root_id: Identifier
    nodes: list[TreeNode] = Field(min_length=2, max_length=64)
    conditions: list[Condition] = Field(min_length=1, max_length=64)
    global_required: list[Identifier] = Field(default_factory=list, max_length=12)
    source_report_ids: list[Identifier] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def valid_tree(self):
        nodes = {n.node_id: n for n in self.nodes}
        conditions = {c.condition_id: c for c in self.conditions}
        if len(nodes) != len(self.nodes) or len(conditions) != len(self.conditions):
            raise ValueError("Duplicate node or condition IDs")
        if self.root_id not in nodes or nodes[self.root_id].kind != "branch":
            raise ValueError("Root must be a branch")
        parents: dict[str, str] = {}
        for node in self.nodes:
            if len(set(node.children)) != len(node.children):
                raise ValueError("Duplicate children")
            if len(set(node.conditions)) != len(node.conditions) or len(set(node.mandatory)) != len(node.mandatory):
                raise ValueError("Duplicate condition binding")
            if any(c not in conditions for c in node.conditions + node.mandatory):
                raise ValueError("Unknown condition")
            if node.kind == "branch" and (not node.children or node.conditions or node.mandatory):
                raise ValueError("Branch requires children and has no diagnosis predicates")
            if node.kind != "branch" and (node.children or not node.conditions):
                raise ValueError("Diagnosis leaf requires predicates and no children")
            for child in node.children:
                if child not in nodes or child in parents or child == self.root_id:
                    raise ValueError("Invalid child or multiple parents")
                parents[child] = node.node_id
        seen: set[str] = set()

        def visit(nid):
            if nid in seen:
                raise ValueError("Cycle")
            seen.add(nid)
            for child in nodes[nid].children:
                visit(child)

        visit(self.root_id)
        if seen != set(nodes) or any(c not in conditions for c in self.global_required):
            raise ValueError("Disconnected tree or unknown global condition")
        if len(set(self.global_required)) != len(self.global_required):
            raise ValueError("Duplicate global condition")
        return self


class ReportSource(StrictModel):
    report_id: Identifier
    title: str = Field(min_length=2, max_length=200)
    text: str = Field(min_length=40, max_length=20000)
    synthetic: Literal[True] = True


class KnowledgeBuildRequest(StrictModel):
    request_id: Identifier
    draft: KnowledgeDraft
    reports: list[ReportSource] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def reports_match(self):
        ids = [r.report_id for r in self.reports]
        if len(set(ids)) != len(ids) or set(ids) != set(self.draft.source_report_ids):
            raise ValueError("Draft source report IDs must match supplied reports")
        return self


class PublishRequest(StrictModel):
    action_id: Identifier
    version_id: Identifier
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_note: str = Field(min_length=10, max_length=2000)


class Action(StrictModel):
    tool: Literal["intent_classify", "kg_query_candidates", "time_series_verify", "expand_tree_node",
                  "request_human_input", "submit_diagnosis"]
    arguments: dict[str, Any]
    reason: str = Field(min_length=1, max_length=500)


class IntentArgs(StrictModel):
    pass


class CandidatesArgs(StrictModel):
    query: str = Field(min_length=1, max_length=500)


class VerifyArgs(StrictModel):
    condition_ids: list[Identifier] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def unique_ids(self):
        if len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("Duplicate condition IDs")
        return self


class ExpandArgs(StrictModel):
    node_id: Identifier


class HumanArgs(StrictModel):
    condition_id: Identifier
    question: str = Field(min_length=4, max_length=500)


class SubmitArgs(StrictModel):
    node_id: Identifier
    path: list[Identifier] = Field(min_length=2, max_length=16)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=64)


ARGUMENT_SCHEMAS = {
    "intent_classify": IntentArgs, "kg_query_candidates": CandidatesArgs,
    "time_series_verify": VerifyArgs, "expand_tree_node": ExpandArgs,
    "request_human_input": HumanArgs, "submit_diagnosis": SubmitArgs,
}


class HumanAnswer(StrictModel):
    action_id: Identifier
    question_id: Identifier
    answer: bool
    observation_note: str = Field(min_length=10, max_length=1000)


class ReconcileRequest(StrictModel):
    action_id: Identifier
    call_id: Identifier
    no_result: bool = False
    reply: Action | None = None
    tokens: int | None = Field(default=None, ge=0, le=100000)
    note: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def one_outcome(self):
        if self.no_result == (self.reply is not None):
            raise ValueError("Supply exactly one confirmed no result or a reply")
        if self.reply is not None and (self.tokens is None or self.tokens == 0):
            raise ValueError("A reconciled reply requires positive token usage")
        return self


class Evidence(StrictModel):
    evidence_id: Identifier
    task_id: Identifier
    plant_id: Identifier
    device_id: Identifier
    event_id: Identifier
    window: Window
    tree_version: Identifier
    tree_hash: str
    data_hash: str
    condition_id: Identifier
    state: Literal["supported", "refuted", "missing"]
    source: Literal["signal", "human"]
    details: dict[str, Any]
    actor_id: Identifier


class ModelReply(StrictModel):
    action: Action
    tokens: int = Field(gt=0, le=100000)


class Narrative(StrictModel):
    explanation: str = Field(min_length=1, max_length=1200)
    limitations: list[str] = Field(min_length=1, max_length=6)


class SignalPoint(StrictModel):
    timestamp_ms: int = Field(ge=0, le=2**53 - 1)
    value: float

    @model_validator(mode="after")
    def finite(self):
        if not math.isfinite(self.value):
            raise ValueError("Nonfinite signal")
        return self
