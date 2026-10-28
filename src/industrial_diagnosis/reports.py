"""Independent post-END reports; neither drafts nor human signoffs actuate a device.

Only this module's tables and content-addressed artifacts are written. Diagnosis state,
evidence, model-call accounting and diagnosis receipts are never updated by reporting.
A durable ``started`` job may represent a live or crashed call: it is never retried.
"""
from __future__ import annotations

import copy
import json
import re

from pydantic import ValidationError

from .schemas import Actor, DomainError, Narrative, canonical, digest
from .storage import Store, new_id, now

_TERMINAL = {"submitted", "escalated", "cancelled"}
_KNOWN_FAILURES = {
    "provider_invalid", "provider_size", "provider_http", "provider_unavailable",
    "model_context", "model_endpoint", "model_identity", "runtime_config",
    "runtime_integrity", "runtime_request",
}
_SENSITIVE_KEY = re.compile(
    r"token|secret|password|passwd|credential|authorization|api.?key|cookie|endpoint|generation_url",
    re.IGNORECASE,
)
_CREDENTIAL_TEXT = re.compile(
    r"\b(?:access[_ -]?token|refresh[_ -]?token|token|password|passwd|secret|credentials?"
    r"|authorization|api[_ -]?key|cookie)\b[\"']?\s*[:=]\s*"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;&]+)", re.IGNORECASE,
)
_SAFETY = {
    "read_only": True,
    "not_actuation": True,
    "model_draft_is_authoritative": False,
    "signoff_scope": "report_content_only_not_device_authorization",
    "notice": "Only the recorded program outcome and evidence are authoritative. "
              "A model draft may contain false claims, remains untrusted and unreviewed in this "
              "artifact, and is never evidence. Signoff is recorded separately and cannot change "
              "the diagnosis or authorize device control. Credential-like text is redacted; "
              "the submission content_hash identifies the original frozen diagnosis.",
}


def _redact(value):
    """Omit credential/usage fields and redact recognizable credentials in free text.

    No actor credentials, provider metadata, model envelopes, raw exception messages or
    accounting fields are copied into reports in the first place.
    """
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()
                if not _SENSITIVE_KEY.search(key)}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
                       "[REDACTED]", value, flags=re.DOTALL)
        value = re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/@]+@", r"\1[REDACTED]@", value)
        value = re.sub(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]+", "[REDACTED]", value)
        value = _CREDENTIAL_TEXT.sub("[REDACTED]", value)
        value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{16}|"
                       r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b",
                       "[REDACTED]", value)
    return value


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value) is None:
        raise DomainError("invalid_report_request", "A valid report/task/action identifier is required", 422)


class ReportService:
    """Owner-scoped immutable reports with separate, one-time owner analyst/admin signoff.

    ``provider=None`` disables the optional narrative. Only submitted tasks are sent to
    ``provider.narrative``; it receives a detached, sanitized authoritative snapshot.
    generate() replays its original completed receipt (not subsequent signoffs). Use
    get()/list_for_task() for the current, separately attached signoff. A started job has
    no final content/hash and cannot be reviewed. Failed/unknown jobs retain a usable
    deterministic report, with the narrative error recorded outside its content.
    """

    def __init__(self, store: Store, diagnosis_service, provider=None):
        self.store, self.diagnosis_service, self.provider = store, diagnosis_service, provider
        with store.transaction() as db:
            statements = (
                """CREATE TABLE IF NOT EXISTS report_jobs(
                    report_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                    owner TEXT NOT NULL, plant_id TEXT NOT NULL, action_id TEXT NOT NULL,
                    version INTEGER NOT NULL, status TEXT NOT NULL
                      CHECK(status IN ('started','completed','failed','unknown')),
                    authoritative_ref TEXT NOT NULL, authoritative_hash TEXT NOT NULL,
                    content_ref TEXT, content_hash TEXT, error_json TEXT,
                    created_at REAL NOT NULL, completed_at REAL,
                    UNIQUE(owner,action_id), UNIQUE(task_id,version),
                    CHECK((status='started' AND content_ref IS NULL AND content_hash IS NULL
                           AND completed_at IS NULL AND error_json IS NULL)
                       OR (status!='started' AND content_ref IS NOT NULL
                           AND content_hash IS NOT NULL AND completed_at IS NOT NULL)))""",
                """CREATE TABLE IF NOT EXISTS report_actions(
                    owner TEXT NOT NULL, action_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                    report_id TEXT NOT NULL REFERENCES report_jobs(report_id), result_json TEXT,
                    PRIMARY KEY(owner,action_id))""",
                """CREATE TABLE IF NOT EXISTS report_signoffs(
                    report_id TEXT PRIMARY KEY REFERENCES report_jobs(report_id),
                    content_hash TEXT NOT NULL, signoff_ref TEXT NOT NULL)""",
                """CREATE TRIGGER IF NOT EXISTS report_jobs_final_immutable
                    BEFORE UPDATE ON report_jobs WHEN OLD.status!='started'
                    BEGIN SELECT RAISE(ABORT,'Final report is immutable'); END""",
                """CREATE TRIGGER IF NOT EXISTS report_jobs_source_immutable
                    BEFORE UPDATE OF report_id,task_id,owner,plant_id,action_id,version,
                        authoritative_ref,authoritative_hash,created_at ON report_jobs
                    BEGIN SELECT RAISE(ABORT,'Report source is immutable'); END""",
                """CREATE TRIGGER IF NOT EXISTS report_actions_receipt_immutable
                    BEFORE UPDATE ON report_actions WHEN OLD.result_json IS NOT NULL
                    BEGIN SELECT RAISE(ABORT,'Report receipt is immutable'); END""",
                """CREATE TRIGGER IF NOT EXISTS report_actions_request_immutable
                    BEFORE UPDATE OF owner,action_id,request_hash,report_id ON report_actions
                    BEGIN SELECT RAISE(ABORT,'Report action is immutable'); END""",
                """CREATE TRIGGER IF NOT EXISTS report_signoffs_immutable
                    BEFORE UPDATE ON report_signoffs
                    BEGIN SELECT RAISE(ABORT,'Report signoff is immutable'); END""",
            )
            for statement in statements:
                db.execute(statement)
            for table in ("report_jobs", "report_actions", "report_signoffs"):
                db.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN SELECT RAISE(ABORT,'Report history is immutable'); END""")

    def _owned_task(self, task_id, actor, db, *, review=False):
        self.store.current_actor(actor, db)
        row = db.execute("SELECT owner,state_json FROM tasks WHERE task_id=? AND owner=?",
                         (task_id, actor.actor_id)).fetchone()
        if row is None:
            raise DomainError("task_not_found", "Task unavailable", 404)
        state = json.loads(row["state_json"])
        self.store.authorize(actor, state["request"]["plant_id"],
                             roles={"analyst", "admin"} if review else None, db=db)
        return state

    def _owned_report(self, report_id, actor, db, *, review=False):
        self.store.current_actor(actor, db)
        row = db.execute("SELECT * FROM report_jobs WHERE report_id=? AND owner=?",
                         (report_id, actor.actor_id)).fetchone()
        if row is None:
            raise DomainError("report_not_found", "Report unavailable", 404)
        state = self._owned_task(row["task_id"], actor, db, review=review)
        if state["request"]["plant_id"] != row["plant_id"]:
            raise DomainError("forbidden", "Report no longer belongs to the current task scope", 403)
        self.store.authorize(actor, row["plant_id"], db=db)
        return row

    def _authoritative(self, task_id, actor, state):
        view = self.diagnosis_service.get(task_id, actor)
        if view["status"] not in _TERMINAL:
            raise DomainError("report_not_terminal", "Reporting requires a terminal diagnosis task", 409)
        if (view["task_id"] != task_id or view["status"] != state["status"]
                or view["request"] != state["request"]):
            raise DomainError("report_source_changed", "Current task view does not match its source", 409)
        submission = view["submission"]
        if view["status"] == "submitted":
            if (not isinstance(submission, dict) or submission.get("task_id") != task_id
                    or submission.get("actor_id") != actor.actor_id
                    or submission.get("request") != view["request"]
                    or submission.get("content_hash") != digest(
                        {key: value for key, value in submission.items() if key != "content_hash"})):
                raise DomainError("report_source_integrity", "Frozen diagnosis identity/hash mismatch", 409)
            evidence_ids = {item["evidence_id"] for item in view["evidence"]}
            if not set(submission["evidence_ids"]) <= evidence_ids:
                raise DomainError("report_source_integrity", "Frozen diagnosis evidence is unavailable", 409)
        elif submission is not None:
            raise DomainError("report_source_integrity", "Non-submitted task has a diagnosis", 409)
        # Deliberately omit provider configuration, tokens, budgets, pending actions and traces.
        authoritative = {key: copy.deepcopy(view[key]) for key in (
            "task_id", "status", "request", "tree_version", "tree_hash", "data_hash",
            "submission", "evidence",
        )}
        authoritative["evidence"].sort(key=lambda item: item["evidence_id"])
        authoritative["error"] = ({"code": view["error"]["code"]} if view.get("error") else None)
        return _redact(authoritative)

    @staticmethod
    def _action(db, actor, action_id, request):
        row = db.execute("SELECT * FROM report_actions WHERE owner=? AND action_id=?",
                         (actor.actor_id, action_id)).fetchone()
        if row is not None and row["request_hash"] != digest(request):
            raise DomainError("idempotency_conflict", "Report action ID reused with different content", 409)
        return row

    def _render(self, row, db, *, signoff=True):
        authoritative = self.store.read_artifact(json.loads(row["authoritative_ref"]))
        if digest(authoritative) != row["authoritative_hash"]:
            raise DomainError("report_integrity", "Authoritative report hash mismatch", 409)
        reference = json.loads(row["content_ref"]) if row["content_ref"] else None
        content = self.store.read_artifact(reference) if reference else None
        if content is not None and (digest(content) != row["content_hash"]
                                    or content["authoritative"] != authoritative):
            raise DomainError("report_integrity", "Report content hash/source mismatch", 409)
        review = None
        if signoff:
            signed = db.execute("SELECT * FROM report_signoffs WHERE report_id=?",
                                (row["report_id"],)).fetchone()
            if signed is not None:
                review = self.store.read_artifact(json.loads(signed["signoff_ref"]))
                if (signed["content_hash"] != row["content_hash"]
                        or review["content_hash"] != row["content_hash"]
                        or review["report_id"] != row["report_id"]):
                    raise DomainError("report_integrity", "Report signoff hash mismatch", 409)
        return {
            "report_id": row["report_id"], "task_id": row["task_id"], "action_id": row["action_id"],
            "version": row["version"], "status": row["status"], "authoritative": authoritative,
            "authoritative_hash": row["authoritative_hash"], "content": content,
            "content_hash": row["content_hash"], "content_ref": reference,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "created_at": row["created_at"], "completed_at": row["completed_at"], "signoff": review,
        }

    def generate(self, task_id: str, action_id: str, actor: Actor) -> dict:
        _identifier(task_id)
        _identifier(action_id)
        request = {"type": "generate_report", "task_id": task_id}
        with self.store.transaction() as db:
            state = self._owned_task(task_id, actor, db)
            action = self._action(db, actor, action_id, request)
            if action is not None:
                row = self._owned_report(action["report_id"], actor, db)
                result = self._render(row, db, signoff=False)
                return json.loads(action["result_json"]) if action["result_json"] else result
            authoritative = self._authoritative(task_id, actor, state)
            # Recheck after reading the public service interface, before writing any artifact.
            self._owned_task(task_id, actor, db)
            report_id, created_at = new_id("report"), now()
            version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM report_jobs WHERE task_id=?",
                                 (task_id,)).fetchone()[0]
            ref = self.store.write_artifact("ReportAuthoritative", authoritative)
            db.execute("""INSERT INTO report_jobs(report_id,task_id,owner,plant_id,action_id,version,
                status,authoritative_ref,authoritative_hash,created_at)
                VALUES(?,?,?,?,?,?,'started',?,?,?)""", (
                    report_id, task_id, actor.actor_id, state["request"]["plant_id"], action_id,
                    version, canonical(ref), digest(authoritative), created_at))
            db.execute("INSERT INTO report_actions VALUES(?,?,?,?,NULL)", (
                actor.actor_id, action_id, digest(request), report_id))
        # The started job/action commit precedes the model call. Never hold a DB lock in inference.
        draft, error, status = None, None, "completed"
        if authoritative["status"] == "submitted" and self.provider is not None:
            with self.store.transaction() as db:
                self._owned_report(report_id, actor, db)
            try:
                reply = self.provider.narrative(copy.deepcopy(authoritative))
            except Exception as exc:
                # Unexpected transport/process exceptions are ambiguous, not safe-to-retry failures.
                code = exc.code if isinstance(exc, DomainError) else "provider_unknown"
                if isinstance(exc, ValidationError):
                    code = "provider_invalid"
                if code not in _KNOWN_FAILURES:
                    code = "provider_unknown"
                status = "failed" if code in _KNOWN_FAILURES else "unknown"
                error = {"code": code, "message": "Optional narrative failed; no automatic retry."
                         if status == "failed" else
                         "Optional narrative outcome is uncertain; no automatic retry or recovery "
                         "under this action ID. The deterministic report remains available."}
            else:
                try:
                    value = reply.model_dump() if isinstance(reply, Narrative) else reply
                    value = Narrative.model_validate(value, strict=True).model_dump(mode="json")
                    # Narrative.limitations has no per-string length bound; impose a byte limit.
                    if len(canonical(value).encode()) > 20_000:
                        raise ValueError("Narrative too large")
                    value = Narrative.model_validate(_redact(value), strict=True).model_dump(mode="json")
                    draft = {"authority": "untrusted_model_draft", "review_status": "unreviewed",
                             "narrative": value,
                             "synthetic": getattr(self.provider, "is_test_provider", False) is True}
                except (ValidationError, ValueError, TypeError, RecursionError):
                    status = "failed"
                    error = {"code": "provider_invalid", "message": "Optional narrative is invalid; "
                             "no draft retained and no automatic retry."}
        # BaseException/process death deliberately leaves started, including after model completion.
        with self.store.transaction() as db:
            row = self._owned_report(report_id, actor, db)
            state = self._owned_task(task_id, actor, db)
            current = self._authoritative(task_id, actor, state)
            if current != authoritative:
                raise DomainError("report_source_changed", "Diagnosis changed during reporting", 409)
            self._owned_report(report_id, actor, db)
            if row["status"] != "started":
                raise DomainError("report_finalized", "Report job is already finalized", 409)
            content = {"schema_version": 1, "authoritative": authoritative,
                       "model_draft": draft, "safety": copy.deepcopy(_SAFETY)}
            reference = self.store.write_artifact("ReportContent", content)
            db.execute("""UPDATE report_jobs SET status=?,content_ref=?,content_hash=?,
                error_json=?,completed_at=? WHERE report_id=? AND status='started'""", (
                    status, canonical(reference), digest(content),
                    canonical(error) if error else None, now(), report_id))
            row = self._owned_report(report_id, actor, db)
            result = self._render(row, db, signoff=False)
            db.execute("UPDATE report_actions SET result_json=? WHERE owner=? AND action_id=?",
                       (canonical(result), actor.actor_id, action_id))
            return result

    def review(self, report_id: str, action_id: str, content_hash: str, decision: str,
               note: str, actor: Actor) -> dict:
        _identifier(report_id)
        _identifier(action_id)
        if (not isinstance(content_hash, str) or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
                or decision not in ("approve", "reject") or not isinstance(note, str)
                or not 1 <= len(note) <= 2000 or not note.strip()):
            raise DomainError("invalid_report_review", "Review requires an exact content hash, "
                              "approve/reject decision and a nonempty note (up to 2000 characters)", 422)
        request = {"type": "review_report", "report_id": report_id, "content_hash": content_hash,
                   "decision": decision, "note": note}
        with self.store.transaction() as db:
            row = self._owned_report(report_id, actor, db, review=True)
            action = self._action(db, actor, action_id, request)
            report = self._render(row, db)
            if action is not None:
                return json.loads(action["result_json"])
            if report["content_hash"] is None:
                raise DomainError("report_incomplete", "A started job has no final content to review", 409)
            if report["content_hash"] != content_hash:
                raise DomainError("stale_report_hash", "Review hash does not match immutable report content", 409)
            if report["signoff"] is not None:
                raise DomainError("report_already_reviewed", "Report signoff is immutable", 409)
            signoff = {"signoff_id": new_id("signoff"), "report_id": report_id,
                       "content_hash": content_hash, "decision": decision, "note": _redact(note),
                       "actor_id": actor.actor_id, "role": actor.role, "acl_version": actor.acl_version,
                       "created_at": now(), "scope": "report_content_only", "not_actuation": True}
            self._owned_report(report_id, actor, db, review=True)
            ref = self.store.write_artifact("ReportSignoff", signoff)
            db.execute("INSERT INTO report_signoffs VALUES(?,?,?)",
                       (report_id, content_hash, canonical(ref)))
            db.execute("INSERT INTO report_actions VALUES(?,?,?,?,?)", (
                actor.actor_id, action_id, digest(request), report_id, canonical(signoff)))
            return signoff

    def get(self, report_id: str, actor: Actor) -> dict:
        _identifier(report_id)
        # A short transaction also prevents ACL/ownership changes during artifact loading.
        with self.store.transaction() as db:
            row = self._owned_report(report_id, actor, db)
            result = self._render(row, db)
            self._owned_report(report_id, actor, db)
            return result

    def list_for_task(self, task_id: str, actor: Actor) -> list[dict]:
        _identifier(task_id)
        with self.store.transaction() as db:
            self._owned_task(task_id, actor, db)
            rows = db.execute("SELECT report_id FROM report_jobs WHERE task_id=? AND owner=? "
                              "ORDER BY version", (task_id, actor.actor_id)).fetchall()
            result = [self._render(self._owned_report(row["report_id"], actor, db), db) for row in rows]
            self._owned_task(task_id, actor, db)
            return result
