from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .schemas import Actor, DomainError, canonical, digest


def new_id(prefix: str) -> str:
    return prefix + "-" + uuid.uuid4().hex


def now() -> float:
    return time.time()


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True, mode=0o700)
        self.database = self.root / "industrial.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users(
                  actor_id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, role TEXT NOT NULL,
                  plants TEXT NOT NULL, acl_version INTEGER NOT NULL, enabled INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(
                  task_id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_id TEXT NOT NULL,
                  request_hash TEXT NOT NULL, state_json TEXT NOT NULL, version INTEGER NOT NULL,
                  lease_owner TEXT, lease_until REAL, UNIQUE(owner,request_id));
                CREATE TABLE IF NOT EXISTS calls(
                  call_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, kind TEXT NOT NULL,
                  status TEXT NOT NULL, context_ref TEXT NOT NULL, response_ref TEXT,
                  tokens INTEGER, error_code TEXT, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS receipts(
                  owner TEXT NOT NULL, action_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                  result_json TEXT NOT NULL, PRIMARY KEY(owner,action_id));
                CREATE TABLE IF NOT EXISTS evidence(
                  evidence_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, condition_id TEXT NOT NULL,
                  payload_ref TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS audit(
                  id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, actor_id TEXT NOT NULL,
                  event TEXT NOT NULL, detail_json TEXT NOT NULL, created_at REAL NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def provision_user(self, actor_id: str, role: str, plants: list[str]) -> str:
        actor = Actor(actor_id=actor_id, role=role, plants=plants, acl_version=1)
        if not plants or len(set(plants)) != len(plants):
            raise DomainError("invalid_user", "At least one unique plant is required")
        token = secrets.token_urlsafe(32)
        with self.transaction() as db:
            try:
                db.execute("INSERT INTO users VALUES(?,?,?,?,1,1)", (
                    actor.actor_id, hashlib.sha256(token.encode()).hexdigest(), role, canonical(plants)))
            except sqlite3.IntegrityError:
                raise DomainError("user_exists", "User already exists", 409) from None
        return token

    @staticmethod
    def _actor(row):
        return Actor(actor_id=row["actor_id"], role=row["role"], plants=json.loads(row["plants"]),
                     acl_version=row["acl_version"])

    def authenticate(self, token: str) -> Actor:
        if not isinstance(token, str) or not 20 <= len(token) <= 200:
            raise DomainError("unauthorized", "Authentication required", 401)
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE token_hash=? AND enabled=1", (
                hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if row is None:
            raise DomainError("unauthorized", "Authentication required", 401)
        return self._actor(row)

    def current_actor(self, actor: Actor, db=None) -> Actor:
        if db is None:
            with self.connect() as connection:
                return self.current_actor(actor, connection)
        row = db.execute("SELECT * FROM users WHERE actor_id=? AND enabled=1", (actor.actor_id,)).fetchone()
        if row is None or self._actor(row) != actor:
            raise DomainError("identity_changed", "Current identity or permissions changed", 403)
        return actor

    def authorize(self, actor: Actor, plant: str, roles=None, db=None) -> Actor:
        actor = self.current_actor(actor, db)
        if plant not in actor.plants or (roles is not None and actor.role not in roles):
            raise DomainError("forbidden", "Action outside current authorization", 403)
        return actor

    def set_user(self, target: str, actor: Actor, *, enabled: bool, plants: list[str], role: str):
        self.current_actor(actor)
        if actor.role != "admin":
            raise DomainError("forbidden", "Administrator required", 403)
        checked = Actor(actor_id=target, role=role, plants=plants, acl_version=1)
        with self.transaction() as db:
            self.current_actor(actor, db)
            row = db.execute("SELECT * FROM users WHERE actor_id=?", (target,)).fetchone()
            if row is None or not set(json.loads(row["plants"])) <= set(actor.plants) or not set(plants) <= set(actor.plants):
                raise DomainError("forbidden", "User outside administrator plants", 403)
            db.execute("UPDATE users SET enabled=?,role=?,plants=?,acl_version=acl_version+1 WHERE actor_id=?",
                       (int(enabled), checked.role, canonical(plants), target))
            self.audit(db, actor.actor_id, "user_permissions", {"target": target, "enabled": enabled})

    def write_artifact(self, kind: str, value) -> dict:
        payload = {"kind": kind, "value": value}
        content = canonical(payload).encode()
        if len(content) > 4_000_000:
            raise DomainError("artifact_too_large", "Artifact exceeds local size limit", 413)
        sha = hashlib.sha256(content).hexdigest()
        path = self.artifacts / (sha + ".json")
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise DomainError("artifact_integrity", "Artifact content collision or unsafe file", 409)
        else:
            fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.artifacts)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                        raise DomainError("artifact_integrity", "Concurrent artifact collision", 409) from None
                directory = os.open(self.artifacts, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                os.unlink(temporary)
        return {"sha256": sha, "kind": kind, "bytes": len(content)}

    def read_artifact(self, reference: dict):
        sha = reference.get("sha256", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise DomainError("artifact_integrity", "Invalid artifact reference", 409)
        path = self.artifacts / (sha + ".json")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
            raise DomainError("artifact_integrity", "Artifact is unavailable", 409)
        content = path.read_bytes()
        if len(content) != reference.get("bytes") or hashlib.sha256(content).hexdigest() != sha:
            raise DomainError("artifact_integrity", "Artifact hash mismatch", 409)
        value = json.loads(content)
        if value["kind"] != reference.get("kind"):
            raise DomainError("artifact_integrity", "Artifact type mismatch", 409)
        return value["value"]

    @staticmethod
    def audit(db, actor_id: str, event: str, detail: dict, task_id: str | None = None):
        db.execute("INSERT INTO audit(task_id,actor_id,event,detail_json,created_at) VALUES(?,?,?,?,?)",
                   (task_id, actor_id, event, canonical(detail), now()))

    def receipt(self, db, actor: Actor, action_id: str, request: dict):
        row = db.execute("SELECT * FROM receipts WHERE owner=? AND action_id=?",
                         (actor.actor_id, action_id)).fetchone()
        if row is None:
            return None
        if row["request_hash"] != digest(request):
            raise DomainError("idempotency_conflict", "Action ID reused with different content", 409)
        return json.loads(row["result_json"])

    def save_receipt(self, db, actor: Actor, action_id: str, request: dict, result: dict):
        db.execute("INSERT INTO receipts VALUES(?,?,?,?)", (
            actor.actor_id, action_id, digest(request), canonical(result)))
