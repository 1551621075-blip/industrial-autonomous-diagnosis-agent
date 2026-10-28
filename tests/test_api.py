"""In-process HTTP tests; no server, model process, or OWL reasoner is started.

Only diagnosis publication is replaced by the explicit TestPublishedKnowledge boundary.
The API, persistence, telemetry, graph/tool execution and report interface are real.
Real knowledge reasoning/publication is covered separately, not claimed by these tests.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_service import TestPublishedKnowledge

from industrial_diagnosis.api import MAX_BODY, create_app
from industrial_diagnosis.fixtures import TREE_PATH_EXAMPLES, original_knowledge, original_reports
from industrial_diagnosis.provider import SyntheticProvider
from industrial_diagnosis.schemas import DomainError, digest


@pytest.fixture
def api_lab(tmp_path):
    app = create_app(tmp_path / "api-test-only", SyntheticProvider())
    store = app.state.store
    tokens = {
        "owner": store.provision_user("api-owner", "analyst", ["PLANT-A"]),
        "peer": store.provision_user("api-peer", "analyst", ["PLANT-A"]),
        "admin": store.provision_user("api-admin", "admin", ["PLANT-A", "PLANT-B"]),
        "outside": store.provision_user("api-outside", "admin", ["PLANT-B"]),
        "operator": store.provision_user("api-operator", "operator", ["PLANT-A"]),
    }
    listing = app.state.telemetry.seed()
    # Keep app.state.knowledge real for extraction/listing; only task publication is fake.
    app.state.service.knowledge = TestPublishedKnowledge(store)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(app=app, client=client, store=store, tokens=tokens, listing=listing,
                              headers={"Authorization": "Bearer " + tokens["owner"]})


def authorization(lab, identity):
    return {"Authorization": "Bearer " + lab.tokens[identity]}


def request_body(lab, case="normal", request_id="api-request"):
    fixture = lab.listing["cases"][case]
    return {"request_id": request_id,
            **{key: fixture[key] for key in (
                "plant_id", "device_id", "event_id", "window", "complaint",
            )}}


def create_task(lab, case="normal", request_id="api-request", identity="owner"):
    response = lab.client.post("/api/tasks", json=request_body(lab, case, request_id),
                               headers=authorization(lab, identity))
    assert response.status_code == 200, response.text
    return response.json()


def run_task(lab, task_id, identity="owner"):
    response = lab.client.post(f"/api/tasks/{task_id}/run", json={},
                               headers=authorization(lab, identity))
    assert response.status_code == 200, response.text
    return response.json()


def assert_security_headers(response):
    assert response.headers.get("cache-control") == "no-store"
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("referrer-policy") == "no-referrer"
    assert response.headers.get("x-frame-options") == "DENY"
    policy = response.headers.get("content-security-policy", "")
    for directive in ("default-src 'self'", "script-src 'self'", "style-src 'self'",
                      "connect-src 'self'", "object-src 'none'", "frame-ancestors 'none'",
                      "base-uri 'none'", "form-action 'self'"):
        assert directive in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "set-cookie" not in response.headers


def assert_no_credentials(lab, response):
    for token in lab.tokens.values():
        assert token not in response.text
        assert hashlib.sha256(token.encode()).hexdigest() not in response.text
    assert "traceback" not in response.text.lower()


PROTECTED_ROUTES = [
    ("GET", "/api/me", None),
    ("GET", "/api/cases", None),
    ("GET", "/api/knowledge", None),
    ("GET", "/api/knowledge/version-absent", None),
    ("POST", "/api/knowledge/extract", {}),
    ("POST", "/api/knowledge/build", {}),
    ("POST", "/api/knowledge/publish", {}),
    ("GET", "/api/tasks", None),
    ("POST", "/api/tasks", {}),
    ("GET", "/api/tasks/task-absent", None),
    ("POST", "/api/tasks/task-absent/run", {}),
    ("POST", "/api/tasks/task-absent/answer", {}),
    ("POST", "/api/tasks/task-absent/reconcile", {}),
    ("POST", "/api/tasks/task-absent/cancel", {}),
    ("GET", "/api/tasks/task-absent/audit", None),
    ("GET", "/api/tasks/task-absent/reports", None),
    ("POST", "/api/tasks/task-absent/reports", {}),
    ("GET", "/api/reports/report-absent", None),
    ("POST", "/api/reports/report-absent/review", {}),
]


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED_ROUTES)
def test_every_private_route_requires_bearer_authentication(api_lab, method, path, body):
    response = api_lab.client.request(method, path, json=body)
    assert response.status_code == 401, response.text
    assert response.json()["error"] == "unauthorized"
    assert_no_credentials(api_lab, response)
    assert_security_headers(response)
    with api_lab.store.connect() as db:
        assert db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("header", [
    "", "Basic DO-NOT-ECHO-CREDENTIAL", "Bearer wrong", "Bearer " + "x" * 201,
    "bearer " + "DO-NOT-ECHO-CREDENTIAL", "Bearer " + "DO-NOT-ECHO-INVALID-" + str(1234567890),
])
def test_invalid_bearer_never_echoes_supplied_credentials(api_lab, header):
    response = api_lab.client.get("/api/me", headers={"Authorization": header})
    assert response.status_code == 401
    assert "DO-NOT-ECHO" not in response.text
    assert_no_credentials(api_lab, response)


def test_login_is_bearer_identity_lookup_not_a_cookie_session(api_lab):
    response = api_lab.client.get("/api/me", headers=api_lab.headers)
    assert response.status_code == 200
    assert response.json() == {"actor_id": "api-owner", "role": "analyst",
                               "plants": ["PLANT-A"], "acl_version": 1}
    assert_no_credentials(api_lab, response)
    assert_security_headers(response)
    assert api_lab.client.get("/api/me").status_code == 401
    response = api_lab.client.get("/api/me", params={"token": api_lab.tokens["owner"]},
                                  headers={"Cookie": "token=" + api_lab.tokens["owner"]})
    assert response.status_code == 401
    assert_no_credentials(api_lab, response)


def test_disabled_user_cannot_reuse_a_previously_valid_bearer(api_lab):
    admin = api_lab.store.authenticate(api_lab.tokens["admin"])
    api_lab.store.set_user("api-owner", admin, enabled=False, plants=["PLANT-A"], role="analyst")
    response = api_lab.client.get("/api/me", headers=api_lab.headers)
    assert response.status_code == 401
    assert_no_credentials(api_lab, response)


@pytest.mark.parametrize("path", ["/", "/health", "/static/app.js", "/static/style.css",
                                  "/static/unknown.js", "/docs", "/openapi.json"])
def test_public_resources_and_not_found_responses_have_security_headers(api_lab, path):
    response = api_lab.client.get(path)
    assert response.status_code == (404 if path in {
        "/static/unknown.js", "/docs", "/openapi.json",
    } else 200)
    assert_security_headers(response)
    assert_no_credentials(api_lab, response)


def test_health_explicitly_discloses_synthetic_provider_and_no_device_control(api_lab):
    assert api_lab.client.get("/health").json() == {
        "status": "ok", "synthetic_only": True, "device_control": False,
        "provider_mode": "test-only",
    }


@pytest.mark.parametrize("host", ["localhost", "localhost:8003", "127.0.0.1:8003",
                                  "127.0.0.2", "[::1]:8003"])
def test_loopback_hosts_with_matching_origin_are_allowed(api_lab, host):
    response = api_lab.client.get("/health", headers={"Host": host, "Origin": "http://" + host})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("host", ["attacker.invalid", "0.0.0.0:8003", "192.168.1.2",
                                  "localhost.attacker.invalid", "[::]"])
def test_nonloopback_host_is_rejected(api_lab, host):
    response = api_lab.client.get("/health", headers={"Host": host})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_host"
    assert_security_headers(response)


@pytest.mark.parametrize("origin", [
    "null", "http://attacker.invalid", "http://localhost", "http://testserver:8003",
    "https://testserver", "http://testserver/", "http://testserver?token=DO-NOT-ECHO",
    "http://testserver#DO-NOT-ECHO", "http://[::1",
    "http://" + "user" + ":" + "DO-NOT-ECHO" + "@testserver",
])
def test_foreign_or_malformed_origin_is_denied_without_echo(api_lab, origin):
    response = api_lab.client.get("/api/me", headers={**api_lab.headers, "Origin": origin})
    assert response.status_code == 403, response.text
    assert response.json()["error"] == "cross_origin_denied"
    assert "DO-NOT-ECHO" not in response.text
    assert_no_credentials(api_lab, response)
    assert_security_headers(response)


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded",
                                         "multipart/form-data", "application/jsonp"])
def test_mutations_require_json_media_type(api_lab, content_type):
    headers = dict(api_lab.headers)
    if content_type:
        headers["Content-Type"] = content_type
    response = api_lab.client.post("/api/tasks", content=b"{}", headers=headers)
    assert response.status_code == 415
    assert response.json()["error"] == "json_required"
    assert_security_headers(response)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH"])
def test_streamed_body_limit_cannot_be_bypassed_by_missing_content_length(api_lab, method):
    response = api_lab.client.request(method, "/api/tasks", headers={
        **api_lab.headers, "Content-Type": "application/json",
    }, content=iter([b"{" + b" " * 127_999, b" " * 128_000, b"}"]))
    assert response.status_code == 413, response.text
    assert response.json()["error"] == "body_too_large"
    assert_security_headers(response)
    with api_lab.store.connect() as db:
        assert db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_body_limit_accepts_exact_limit_and_rejects_one_more_byte(api_lab):
    raw = json.dumps(request_body(api_lab)).encode()
    padded = raw + b" " * (MAX_BODY - len(raw))
    headers = {**api_lab.headers, "Content-Type": "application/json; charset=utf-8"}
    response = api_lab.client.post("/api/tasks", content=padded, headers=headers)
    assert response.status_code == 200, response.text
    too_large = api_lab.client.post("/api/tasks", content=padded + b" ", headers=headers)
    assert too_large.status_code == 413
    assert too_large.json()["error"] == "body_too_large"


@pytest.mark.parametrize("payload", [
    b'{"token":' + json.dumps("DO-NOT-ECHO-INVALID-JSON").encode(), b'[]', b'null',
])
def test_json_validation_errors_do_not_echo_input_or_parser_details(api_lab, payload):
    response = api_lab.client.post("/api/tasks", content=payload, headers={
        **api_lab.headers, "Content-Type": "application/json",
    })
    assert response.status_code == 422
    assert set(response.json()) == {"error", "message"}
    assert response.json()["error"] == "invalid_request"
    assert isinstance(response.json()["message"], str) and response.json()["message"]
    assert "DO-NOT-ECHO" not in response.text
    assert_security_headers(response)


@pytest.mark.parametrize("payload", [
    pytest.param(b"not-json", id="malformed"),
    pytest.param(b"", id="empty"),
    pytest.param(b"[]", id="array"),
    pytest.param(b"null", id="null"),
    pytest.param(b"true", id="boolean"),
    pytest.param(b"42", id="number"),
    pytest.param(b'"DO-NOT-ECHO-root-string"', id="string"),
    pytest.param(b'{"key":1,"key":2}', id="duplicate-key"),
    pytest.param(b'{"nested":{"key":1,"key":2}}', id="nested-duplicate"),
    pytest.param(b'{"key":1,"\\u006bey":2}', id="escaped-duplicate"),
    pytest.param(b'{"unused":NaN}', id="nan"),
    pytest.param(b'{"unused":Infinity}', id="infinity"),
    pytest.param(b'{"unused":-Infinity}', id="negative-infinity"),
    pytest.param(b'{"unused":1e309}', id="overflow-infinity"),
    pytest.param(b'{"unused":-1e309}', id="negative-overflow-infinity"),
    pytest.param(b'{"unused":"\xff"}', id="invalid-utf8"),
    pytest.param(b'{"nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}", id="too-deep"),
])
def test_run_endpoint_requires_valid_json_even_without_body_fields(api_lab, payload):
    task = create_task(api_lab)
    response = api_lab.client.post(f"/api/tasks/{task['task_id']}/run", content=payload, headers={
        **api_lab.headers, "Content-Type": "application/json",
    })
    assert response.status_code == 422, response.text[:300]
    assert set(response.json()) == {"error", "message"}
    assert response.json()["error"] == "invalid_request"
    assert "DO-NOT-ECHO" not in response.text
    assert_no_credentials(api_lab, response)
    assert_security_headers(response)
    untouched = api_lab.client.get(f"/api/tasks/{task['task_id']}", headers=api_lab.headers).json()
    assert untouched == task
    assert untouched["calls"] == untouched["tools"] == 0


def test_unknown_request_fields_are_rejected_without_echo(api_lab):
    response = api_lab.client.post("/api/tasks", headers=api_lab.headers, json={
        **request_body(api_lab), "authorization": api_lab.tokens["owner"],
        "error": "/private/DO-NOT-ECHO-TRACEBACK",
    })
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert "DO-NOT-ECHO" not in response.text
    assert_no_credentials(api_lab, response)


class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.resources = []
        self.inline_handlers = []
        self.inline_scripts = []
        self._script = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        self.inline_handlers.extend(key for key in values if key.startswith("on"))
        if tag == "script":
            self._script = True
            self.resources.append((values.get("src"), "application/javascript"))
        elif tag == "link" and values.get("rel") == "stylesheet":
            self.resources.append((values.get("href"), "text/css"))

    def handle_endtag(self, tag):
        if tag == "script":
            self._script = False

    def handle_data(self, data):
        if self._script and data.strip():
            self.inline_scripts.append(data)


def test_workbench_resource_references_dom_ids_and_token_handling_are_valid(api_lab):
    page = api_lab.client.get("/")
    assert page.headers["content-type"].startswith("text/html")
    parser = ResourceParser()
    parser.feed(page.text)
    assert len(parser.ids) == len(set(parser.ids))
    assert not parser.inline_handlers and not parser.inline_scripts
    assert set(parser.resources) == {
        ("/static/app.js", "application/javascript"), ("/static/style.css", "text/css"),
    }
    for path, media_type in parser.resources:
        response = api_lab.client.get(path)
        assert response.status_code == 200 and response.text.strip()
        assert response.headers["content-type"].startswith(media_type)
    js = api_lab.client.get("/static/app.js").text
    referenced = set(re.findall(r'\b(?:el|show|on|choices)\("([\w-]+)"', js))
    assert referenced <= set(parser.ids)
    assert {"login", "token", "logout", "answer", "signoff"} <= set(parser.ids)
    for forbidden in ("localStorage", "sessionStorage", "indexedDB", "document.cookie", "innerHTML"):
        assert forbidden not in js
    assert "textContent" in js
    assert 'credentials: "omit"' in js
    assert not re.search(r"https?://", js)


def test_workbench_javascript_parses_without_execution_or_server(api_lab):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Optional Node.js syntax checker is not installed")
    result = subprocess.run([node, "--check", "-"],
                            input=api_lab.client.get("/static/app.js").text,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", ["/static/api.py", "/static/industrial.sqlite3",
                                  "/static/%2e%2e%2fapi.py", "/static/.env"])
def test_static_resources_are_allowlisted_not_arbitrary_files(api_lab, path):
    response = api_lab.client.get(path)
    assert response.status_code == 404
    assert "def create_app" not in response.text
    assert_no_credentials(api_lab, response)


def test_cases_are_plant_scoped_and_reading_does_not_reseed(api_lab):
    before = api_lab.listing
    first = api_lab.client.get("/api/cases", headers=api_lab.headers)
    assert first.status_code == 200, first.text
    cases = first.json()
    assert isinstance(cases, list) and cases
    assert {row["plant_id"] for row in cases} == {"PLANT-A"}
    assert {row["case_label"] for row in cases} >= set(before["cases"])
    assert api_lab.client.get("/api/cases", headers=api_lab.headers).json() == cases
    other = api_lab.client.get("/api/cases", headers=authorization(api_lab, "outside"))
    assert other.status_code == 200
    assert other.json() and {row["plant_id"] for row in other.json()} == {"PLANT-B"}
    assert_no_credentials(api_lab, first)


def test_original_report_extraction_is_real_but_does_not_build_or_publish(api_lab):
    request = {"request_id": "api-extraction", "reports": [
        report.model_dump(mode="json") for report in original_reports()
    ]}
    response = api_lab.client.post("/api/knowledge/extract", headers=api_lab.headers, json=request)
    assert response.status_code == 200, response.text
    expected = original_knowledge().model_dump(mode="json")
    expected["request_id"] = request["request_id"]
    assert response.json() == expected
    assert api_lab.client.get("/api/knowledge", headers=api_lab.headers).json() == []


@pytest.mark.parametrize("identity", ["operator", "outside"])
def test_knowledge_extraction_requires_author_role_and_plant(api_lab, identity):
    response = api_lab.client.post("/api/knowledge/extract", headers=authorization(api_lab, identity),
                                  json={"request_id": "extract-denied", "reports": [
                                      report.model_dump(mode="json") for report in original_reports()
                                  ]})
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


@pytest.mark.parametrize("case", ["normal", "jam", "slip", "sensor_fault"])
def test_actual_http_tool_flow_pauses_for_human_and_submits_bound_evidence(api_lab, case):
    task = create_task(api_lab, case)
    assert task["calls"] == task["tools"] == 0
    task_id = task["task_id"]
    human_questions = 0
    for _ in range(8):
        task = run_task(api_lab, task_id)
        if task["status"] != "waiting_human":
            break
        human_questions += 1
        question = task["pending_human"]
        assert task["submission"] is None
        assert not any(row["condition_id"] == question["condition_id"] for row in task["evidence"])
        answer = {"action_id": "answer-" + question["question_id"],
                  "question_id": question["question_id"], "answer": True,
                  "observation_note": "TEST ONLY: safe observation was already independently obtained."}
        response = api_lab.client.post(f"/api/tasks/{task_id}/answer", json=answer,
                                       headers=api_lab.headers)
        assert response.status_code == 200, response.text
        assert api_lab.client.post(f"/api/tasks/{task_id}/answer", json=answer,
                                    headers=api_lab.headers).json() == response.json()
    assert task["status"] == "submitted", task
    assert human_questions == (0 if case == "normal" else 1)
    assert task["submission"]["node_id"] == case
    assert task["submission"]["path"] == TREE_PATH_EXAMPLES[case]
    assert task["submission"]["conclusion_level"] == "supported_under_declared_synthetic_rules"
    assert task["calls"] > 0 and task["tools"] > 0 and task["tokens"] == task["calls"] * 128
    for evidence in task["evidence"]:
        assert evidence["task_id"] == task_id
        assert evidence["actor_id"] == "api-owner"
        assert evidence["tree_hash"] == task["tree_hash"]
        assert evidence["data_hash"] == task["data_hash"]
        assert evidence["plant_id"] == "PLANT-A"
        if evidence["source"] == "human":
            assert evidence["details"]["not_sensor_verified"] is True
    assert run_task(api_lab, task_id) == task
    audit = api_lab.client.get(f"/api/tasks/{task_id}/audit", headers=api_lab.headers)
    assert audit.status_code == 200
    trace = audit.json()
    assert trace["events"] and len(trace["model_calls"]) == task["calls"]
    tools = {json.loads(event["detail_json"])["tool"] for event in trace["events"]
             if event["event"] == "tool_completed"}
    assert {"intent_classify", "kg_query_candidates", "expand_tree_node", "time_series_verify",
            "submit_diagnosis"} <= tools
    if human_questions:
        assert "request_human_input" in tools
    assert_no_credentials(api_lab, audit)


@pytest.mark.parametrize("invalid", ["true", "false", "yes", 1, 0, 1.0, None, [], {}])
def test_human_answer_requires_json_boolean_without_consuming_question(api_lab, invalid):
    task = run_task(api_lab, create_task(api_lab, "jam")["task_id"])
    assert task["status"] == "waiting_human"
    question = task["pending_human"]
    response = api_lab.client.post(f"/api/tasks/{task['task_id']}/answer", headers=api_lab.headers,
                                  json={"action_id": "answer-invalid", "answer": invalid,
                                        "question_id": question["question_id"],
                                        "observation_note": "TEST observation must be a strict boolean."})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert api_lab.client.get(f"/api/tasks/{task['task_id']}", headers=api_lab.headers).json() == task


def test_false_human_answer_is_preserved_and_never_supports_diagnosis(api_lab):
    task = run_task(api_lab, create_task(api_lab, "jam")["task_id"])
    assert task["status"] == "waiting_human"
    response = api_lab.client.post(f"/api/tasks/{task['task_id']}/answer", headers=api_lab.headers,
                                  json={"action_id": "answer-false", "answer": False,
                                        "question_id": task["pending_human"]["question_id"],
                                        "observation_note": "TEST obstruction was not observed."})
    assert response.status_code == 200, response.text
    human = [item for item in response.json()["evidence"] if item["source"] == "human"]
    assert len(human) == 1
    assert human[0]["details"]["answer"] is False and human[0]["state"] == "refuted"
    final = run_task(api_lab, task["task_id"])
    assert final["status"] in {"blocked", "escalated"} and final["submission"] is None


@pytest.mark.parametrize("invalid", ["true", "false", 1, 0, None])
def test_reconcile_requires_strict_boolean_before_task_lookup(api_lab, invalid):
    response = api_lab.client.post("/api/tasks/task-absent/reconcile", headers=api_lab.headers,
                                  json={"action_id": "reconcile-invalid", "call_id": "call-absent",
                                        "no_result": invalid, "note": "TEST explicit reconciliation."})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


RECONCILED_ACTION = {"tool": "intent_classify", "arguments": {}, "reason": "TEST recorded action"}


@pytest.mark.parametrize("outcome", [
    pytest.param({"no_result": True}, id="no-result-unknown-usage"),
    pytest.param({"no_result": True, "tokens": None}, id="no-result-null-usage"),
    pytest.param({"no_result": True, "tokens": 0}, id="no-result-confirmed-zero"),
    pytest.param({"no_result": True, "tokens": 123}, id="no-result-confirmed-positive"),
    pytest.param({"no_result": True, "tokens": 100000}, id="no-result-maximum-usage"),
    pytest.param({"reply": RECONCILED_ACTION, "tokens": 1}, id="reply-positive-usage"),
    pytest.param({"reply": RECONCILED_ACTION, "tokens": 100000}, id="reply-maximum-usage"),
])
def test_reconcile_usage_contract_accepts_valid_outcome_before_lookup(api_lab, outcome):
    response = api_lab.client.post("/api/tasks/task-absent/reconcile",
                                  headers=authorization(api_lab, "admin"), json={
                                      "action_id": "reconcile-valid", "call_id": "call-absent",
                                      "note": "TEST explicit recorded outcome and optional usage.",
                                      **outcome,
                                  })
    # A valid request reaches the real admin ownership lookup, not schema rejection.
    assert response.status_code == 404, response.text
    assert response.json()["error"] == "task_not_found"
    assert_security_headers(response)
    assert_no_credentials(api_lab, response)


@pytest.mark.parametrize("outcome", [
    pytest.param({}, id="missing-outcome"),
    pytest.param({"no_result": False}, id="false-without-reply"),
    pytest.param({"no_result": False, "tokens": 0}, id="false-zero-without-reply"),
    pytest.param({"no_result": True, "reply": RECONCILED_ACTION, "tokens": 1},
                 id="contradictory-outcomes"),
    pytest.param({"reply": RECONCILED_ACTION}, id="reply-unknown-usage"),
    pytest.param({"reply": RECONCILED_ACTION, "tokens": None}, id="reply-null-usage"),
    pytest.param({"reply": RECONCILED_ACTION, "tokens": 0}, id="reply-zero-usage"),
    pytest.param({"no_result": True, "tokens": -1}, id="negative-usage"),
    pytest.param({"no_result": True, "tokens": 100001}, id="excessive-usage"),
    pytest.param({"no_result": True, "tokens": "0"}, id="string-usage"),
    pytest.param({"no_result": True, "tokens": 0.0}, id="float-usage"),
    pytest.param({"no_result": True, "tokens": False}, id="false-usage"),
    pytest.param({"no_result": True, "tokens": True}, id="true-usage"),
])
def test_reconcile_usage_contract_rejects_invalid_or_missing_outcome(api_lab, outcome):
    response = api_lab.client.post("/api/tasks/task-absent/reconcile",
                                  headers=authorization(api_lab, "admin"), json={
                                      "action_id": "reconcile-invalid", "call_id": "call-absent",
                                      "note": "TEST missing or invalid outcome and token usage.",
                                      **outcome,
                                  })
    assert response.status_code == 422, response.text
    assert response.json()["error"] == "invalid_request"
    assert_security_headers(response)
    assert_no_credentials(api_lab, response)


@pytest.mark.parametrize("usage", [
    pytest.param({}, id="unknown-usage-exhausts-remaining-budget"),
    pytest.param({"tokens": 0}, id="attested-zero-allows-explicit-continuation"),
    pytest.param({"tokens": 123}, id="attested-positive-usage-is-charged"),
])
def test_http_no_result_reconciliation_accounts_usage_without_automatic_retry(
    api_lab, monkeypatch, usage,
):
    provider = api_lab.app.state.service.provider
    original_decision = provider.decision
    attempted_calls = []

    def unknown(_context):
        attempted_calls.append("unknown")
        raise DomainError("provider_unknown", "TEST uncertain local invocation", 503)

    monkeypatch.setattr(provider, "decision", unknown)
    task_id = create_task(api_lab, identity="admin")["task_id"]
    task = run_task(api_lab, task_id, identity="admin")
    assert task["status"] == "blocked" and task["pending_call_id"]
    assert task["calls"] == 1 and task["tools"] == task["tokens"] == 0
    call_id = task["pending_call_id"]
    action = {"action_id": "reconcile-usage", "call_id": call_id, "no_result": True,
              "note": "TEST human attestation from synthetic invocation records.", **usage}
    response = api_lab.client.post(f"/api/tasks/{task_id}/reconcile",
                                  headers=authorization(api_lab, "admin"), json=action)
    assert response.status_code == 200, response.text
    reconciled = response.json()
    charged = usage.get("tokens", task["limits"]["tokens"])
    assert reconciled["status"] == "ready" and reconciled["pending_call_id"] is None
    assert reconciled["tokens"] == charged and reconciled["calls"] == 1
    assert reconciled["tools"] == 0 and reconciled["submission"] is None
    assert attempted_calls == ["unknown"]
    replay = api_lab.client.post(f"/api/tasks/{task_id}/reconcile",
                                headers=authorization(api_lab, "admin"), json=action)
    assert replay.status_code == 200 and replay.json() == reconciled
    audit = api_lab.client.get(f"/api/tasks/{task_id}/audit",
                              headers=authorization(api_lab, "admin")).json()
    recorded = next(call for call in audit["model_calls"] if call["call_id"] == call_id)
    assert recorded["status"] == "confirmed_no_result" and recorded["tokens"] == charged
    detail = next(json.loads(event["detail_json"]) for event in audit["events"]
                  if event["event"] == "model_reconciled")
    assert detail["conservative_usage"] is ("tokens" not in usage)
    assert detail["tokens"] == charged
    assert detail["human_attestation_not_external_proof"] is True
    assert_security_headers(response)
    assert_no_credentials(api_lab, response)

    def counted(context):
        attempted_calls.append("synthetic")
        return original_decision(context)

    monkeypatch.setattr(provider, "decision", counted)
    final = run_task(api_lab, task_id, identity="admin")
    if "tokens" not in usage:
        assert final["status"] == "escalated" and final["submission"] is None
        assert final["error"]["code"] == "budget_exhausted"
        assert attempted_calls == ["unknown"] and final["calls"] == 1
        assert final["tokens"] == final["limits"]["tokens"]
    else:
        assert final["status"] == "submitted", final
        assert len(attempted_calls) == final["calls"] > 1
        assert final["tokens"] == charged + (final["calls"] - 1) * 128


TASK_OPERATIONS = [
    ("GET", "", None), ("GET", "/audit", None), ("POST", "/run", {}),
    ("POST", "/cancel", {"action_id": "deny-cancel"}),
    ("POST", "/answer", {"action_id": "deny-answer", "question_id": "question-absent",
                          "answer": True, "observation_note": "TEST unauthorized observation."}),
    ("POST", "/reconcile", {"action_id": "deny-reconcile", "call_id": "call-absent",
                             "no_result": True, "note": "TEST unauthorized reconciliation."}),
    ("GET", "/reports", None), ("POST", "/reports", {"action_id": "deny-report"}),
]


@pytest.mark.parametrize("identity", ["peer", "admin", "outside"])
@pytest.mark.parametrize(("method", "suffix", "body"), TASK_OPERATIONS)
def test_task_owner_isolation_including_same_plant_admin(api_lab, identity, method, suffix, body):
    task = create_task(api_lab)
    response = api_lab.client.request(method, f"/api/tasks/{task['task_id']}{suffix}", json=body,
                                      headers=authorization(api_lab, identity))
    expected = (403, "forbidden") if identity == "peer" and suffix == "/reconcile" else (
        404, "task_not_found",
    )  # Reconciliation checks administrator role before looking up ownership.
    assert response.status_code == expected[0], response.text
    assert response.json()["error"] == expected[1]
    assert task["task_id"] not in response.text
    assert api_lab.client.get("/api/tasks", headers=authorization(api_lab, identity)).json() == []
    assert api_lab.client.get(f"/api/tasks/{task['task_id']}", headers=api_lab.headers).json() == task
    assert_no_credentials(api_lab, response)


def test_task_creation_cannot_cross_plant_authorization(api_lab):
    foreign = api_lab.listing["isolation_cases"]["other_plant"]
    response = api_lab.client.post("/api/tasks", headers=api_lab.headers, json={
        "request_id": "deny-plant", **{key: foreign[key] for key in (
            "plant_id", "device_id", "event_id", "window", "complaint",
        )},
    })
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"
    assert api_lab.client.get("/api/tasks", headers=api_lab.headers).json() == []


def test_current_plant_revocation_blocks_owned_task_on_next_http_request(api_lab):
    task = create_task(api_lab)
    admin = api_lab.store.authenticate(api_lab.tokens["admin"])
    api_lab.store.set_user("api-owner", admin, enabled=True, plants=["PLANT-B"], role="analyst")
    assert api_lab.client.get("/api/tasks", headers=api_lab.headers).json() == []
    response = api_lab.client.get(f"/api/tasks/{task['task_id']}", headers=api_lab.headers)
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_api_report_generation_review_and_read_use_present_reports_interface(api_lab):
    task = run_task(api_lab, create_task(api_lab)["task_id"])
    assert task["status"] == "submitted"
    task_id = task["task_id"]
    action = {"action_id": "generate-report"}
    response = api_lab.client.post(f"/api/tasks/{task_id}/reports", json=action, headers=api_lab.headers)
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["status"] == "completed" and report["signoff"] is None
    assert report["content_hash"] == digest(report["content"])
    assert report["authoritative"]["submission"] == task["submission"]
    assert report["content"]["model_draft"]["synthetic"] is True
    assert api_lab.client.post(f"/api/tasks/{task_id}/reports", json=action,
                               headers=api_lab.headers).json() == report
    for identity in ("peer", "admin", "outside"):
        hidden = api_lab.client.get(f"/api/reports/{report['report_id']}",
                                    headers=authorization(api_lab, identity))
        assert hidden.status_code == 404 and hidden.json()["error"] == "report_not_found"
        denied = api_lab.client.post(f"/api/reports/{report['report_id']}/review", json={
            "action_id": "foreign-review", "content_hash": report["content_hash"],
            "decision": "approve", "note": "TEST not the owner of this report.",
        }, headers=authorization(api_lab, identity))
        assert denied.status_code == 404 and denied.json()["error"] == "report_not_found"
    review = {"action_id": "review-report", "content_hash": report["content_hash"],
              "decision": "approve", "note": "TEST manual review of synthetic evidence only."}
    stale = api_lab.client.post(f"/api/reports/{report['report_id']}/review", headers=api_lab.headers,
                                json={**review, "content_hash": "0" * 64})
    assert stale.status_code == 409 and stale.json()["error"] == "stale_report_hash"
    signed = api_lab.client.post(f"/api/reports/{report['report_id']}/review", json=review,
                                 headers=api_lab.headers)
    assert signed.status_code == 200, signed.text
    signoff = signed.json()
    assert signoff["content_hash"] == report["content_hash"]
    assert signoff["actor_id"] == "api-owner" and signoff["not_actuation"] is True
    assert api_lab.client.post(f"/api/reports/{report['report_id']}/review", json=review,
                               headers=api_lab.headers).json() == signoff
    current = api_lab.client.get(f"/api/reports/{report['report_id']}", headers=api_lab.headers).json()
    assert current == {**report, "signoff": signoff}
    assert api_lab.client.get(f"/api/tasks/{task_id}/reports", headers=api_lab.headers).json() == [current]
    assert api_lab.client.get(f"/api/tasks/{task_id}", headers=api_lab.headers).json() == task
    assert_no_credentials(api_lab, response)
    assert_security_headers(response)


def test_report_generation_requires_a_terminal_task(api_lab):
    task = create_task(api_lab)
    response = api_lab.client.post(f"/api/tasks/{task['task_id']}/reports", headers=api_lab.headers,
                                  json={"action_id": "premature-report"})
    assert response.status_code == 409
    assert response.json()["error"] == "report_not_terminal"
    assert api_lab.client.get(f"/api/tasks/{task['task_id']}/reports", headers=api_lab.headers).json() == []


def test_provider_error_details_are_not_echoed_in_task_or_audit(api_lab, monkeypatch):
    marker = "DO-NOT-ECHO-PRIVATE-PROVIDER-ERROR " + api_lab.tokens["owner"]

    def fail(_context):
        raise DomainError("provider_invalid", marker, 502)

    monkeypatch.setattr(api_lab.app.state.service.provider, "decision", fail)
    task = run_task(api_lab, create_task(api_lab)["task_id"])
    assert task["status"] == "blocked" and task["error"]["code"] == "provider_invalid"
    for suffix in ("", "/audit"):
        response = api_lab.client.get(f"/api/tasks/{task['task_id']}{suffix}", headers=api_lab.headers)
        assert response.status_code == 200
        assert "DO-NOT-ECHO" not in response.text
        assert_no_credentials(api_lab, response)


def test_unexpected_server_errors_never_echo_exception_or_bearer(api_lab, monkeypatch):
    def fail(_actor):
        raise RuntimeError("DO-NOT-ECHO-INTERNAL-EXCEPTION " + api_lab.tokens["owner"])

    monkeypatch.setattr(api_lab.app.state.service, "list_tasks", fail)
    response = api_lab.client.get("/api/tasks", headers=api_lab.headers)
    assert response.status_code == 500
    assert "DO-NOT-ECHO" not in response.text
    assert_no_credentials(api_lab, response)
    assert_security_headers(response)
