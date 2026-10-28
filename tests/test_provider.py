"""Self-contained provider tests: fake HTTP only, no actual inference or fixture oracle."""
from __future__ import annotations

import copy
import json

import httpx
import pytest

from industrial_diagnosis import provider as module
from industrial_diagnosis.provider import (
    PROTOCOL,
    LocalProvider,
    ProviderFailure,
    SyntheticProvider,
    action_response_schema,
    local_url,
)
from industrial_diagnosis.schemas import ARGUMENT_SCHEMAS, DomainError, digest


def envelope(action=None, tokens=73):
    action = action or {"tool": "intent_classify", "arguments": {}, "reason": "Check intent"}
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(action)}}],
            "usage": {"total_tokens": tokens}}


@pytest.mark.parametrize(("url", "expected"), [
    ("http://127.0.0.1:18080/", "http://127.0.0.1:18080"),
    ("http://127.1.2.3/v1/", "http://127.1.2.3/v1"),
    ("http://[::1]:18080", "http://[::1]:18080"),
    ("http://[0:0:0:0:0:0:0:1]:80/", "http://[::1]"),
])
def test_numeric_loopback_only(url, expected):
    assert local_url(url) == expected


@pytest.mark.parametrize("url", [
    "https://127.0.0.1", "http://localhost", "http://localhost.localdomain", "http://example.com",
    "http://0.0.0.0", "http://192.168.1.2", "http://[::]", "http://[::ffff:127.0.0.1]",
    "http://127.0.0.1@evil.invalid", "http://" + "user:password" + "@127.0.0.1", "http://@127.0.0.1",
    "http://127.0.0.1?", "http://127.0.0.1#", "http://127.0.0.1?q=secret",
    "http://127.0.0.1#fragment", "http://127.0.0.1:99999", "http://127.0.0.1:0",
    "http://127.0.0.1:bad", "http://127.0.0.1:", "http://127.0.0.1\n",
    " http://127.0.0.1", "http://127.0.0.1\\evil", "http://[::1%25lo0]", "http://127.1",
    "http://2130706433", "http://0177.0.0.1", "http://0x7f000001", "http://[::1", None, 123,
    "http://127.0.0.1/../remote", "http://127.0.0.1//example.com", "http://127.0.0.1/a:b",
])
def test_reject_nonlocal_or_indirect_urls(url):
    with pytest.raises(DomainError) as error:
        LocalProvider(url)
    assert error.value.code == "model_endpoint"


@pytest.mark.parametrize("model", [None, "", " ", 123, "secret\n", "a" * 201])
def test_explicit_model_identity(model):
    with pytest.raises(DomainError) as error:
        LocalProvider("http://127.0.0.1", model)
    assert error.value.code == "model_identity"


def test_fingerprint_is_stable_protocol_tagged_sha():
    first = LocalProvider("http://127.0.0.1:80/", "local-qwen")
    second = LocalProvider("http://127.0.0.1", "local-qwen")
    assert first.fingerprint == second.fingerprint == "local-" + digest(first.metadata)
    assert first.metadata["protocol"] == PROTOCOL
    assert len(first.fingerprint.removeprefix("local-")) == 64
    assert first.fingerprint != LocalProvider("http://127.0.0.1", "other").fingerprint
    assert first.fingerprint != LocalProvider("http://127.0.0.2", "local-qwen").fingerprint


def test_schema_exact_six_tool_specific_variants():
    schema = action_response_schema()
    variants = schema["oneOf"]
    assert len(variants) == 6
    assert {item["properties"]["tool"]["enum"][0] for item in variants} == {
        "intent_classify", "kg_query_candidates", "time_series_verify", "expand_tree_node",
        "request_human_input", "submit_diagnosis",
    }
    for item in variants:
        assert item["additionalProperties"] is False
        assert set(item["required"]) == {"tool", "arguments", "reason"}
        tool = item["properties"]["tool"]["enum"][0]
        assert item["properties"]["arguments"] == ARGUMENT_SCHEMAS[tool].model_json_schema()
        assert item["properties"]["arguments"]["additionalProperties"] is False


@pytest.mark.parametrize(("tool", "arguments"), [
    ("intent_classify", {}),
    ("kg_query_candidates", {"query": "Conveyor stopped"}),
    ("time_series_verify", {"condition_ids": ["current-high"]}),
    ("expand_tree_node", {"node_id": "root"}),
    ("request_human_input", {"condition_id": "obstruction", "question": "Is there obstruction?"}),
    ("submit_diagnosis", {"node_id": "jam", "path": ["root", "jam"], "evidence_ids": ["e1"]}),
])
def test_decision_all_tools_are_model_chosen(monkeypatch, tool, arguments):
    client = LocalProvider("http://127.0.0.1:18080")
    captured = []

    def post(url, payload):
        captured.append((url, payload))
        return envelope({"tool": tool, "arguments": arguments, "reason": "Next observed step"})

    monkeypatch.setattr(client, "_post", post)
    context = {"observations": [{"tool": "test", "result": {"state": "missing"}}],
               "tree": {"nodes": [{"node_id": "other", "label": "All knowledge labels visible"}]}}
    before = copy.deepcopy(context)
    reply = client.decision(context)
    assert reply.action.tool == tool
    assert reply.action.arguments == arguments
    assert reply.tokens == 73 and type(reply.tokens) is int
    assert context == before
    assert len(captured) == 1
    assert captured[0][0] == "http://127.0.0.1:18080/v1/chat/completions"
    payload = captured[0][1]
    assert json.loads(payload["messages"][1]["content"]) == context
    assert payload["response_format"]["json_schema"]["schema"] == action_response_schema()
    assert "frozen by the program" in payload["messages"][0]["content"]
    assert "not a fixed pipeline" in payload["messages"][0]["content"]


@pytest.mark.parametrize("action", [
    {"tool": "unknown", "arguments": {}, "reason": "No"},
    {"tool": "intent_classify", "arguments": {"query": "smuggled"}, "reason": "No"},
    {"tool": "expand_tree_node", "arguments": {}, "reason": "No"},
    {"tool": "expand_tree_node", "arguments": {"node_id": 123}, "reason": "No"},
    {"tool": "time_series_verify", "arguments": {"condition_ids": ["c", "c"]}, "reason": "No"},
    {"tool": "time_series_verify", "arguments": {"condition_ids": []}, "reason": "No"},
    {"tool": "time_series_verify", "arguments": {"condition_ids": ["../x"]}, "reason": "No"},
    {"tool": "request_human_input", "arguments": {"condition_id": "c", "question": "No",
                                                 "answer": True}, "reason": "No"},
    {"tool": "intent_classify", "arguments": {}, "reason": ""},
    {"tool": "intent_classify", "arguments": {}, "reason": False},
    {"tool": "intent_classify", "arguments": {}, "reason": "No", "extra": 1},
])
def test_independent_action_and_argument_validation(monkeypatch, action):
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: envelope(action))
    with pytest.raises(DomainError) as error:
        client.decision({})
    assert error.value.code == "provider_invalid"
    assert isinstance(error.value, ProviderFailure)
    assert error.value.reported_tokens == 73 and error.value.usage_unknown is False


@pytest.mark.parametrize("result", [
    None, [], {}, {"choices": []}, {"choices": {}}, {"choices": [None]},
    {"choices": [{"message": None}]},
    {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": {}}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": None},
])
def test_malformed_envelopes_fail_cleanly(monkeypatch, result):
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: result)
    with pytest.raises(DomainError) as error:
        client.decision({})
    assert error.value.code == "provider_invalid"


@pytest.mark.parametrize("tokens", [True, False, 0, -1, 1.5, "73", None, 100001])
def test_token_usage_must_be_positive_integer_not_bool(monkeypatch, tokens):
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: envelope(tokens=tokens))
    with pytest.raises(DomainError) as error:
        client.decision({})
    assert error.value.code == "provider_invalid"


@pytest.mark.parametrize("change", ["extra-choice", "no-usage", "no-finish", "refusal", "tool_calls",
                                     "duplicate", "nan", "markdown"])
def test_unexpected_choice_content_is_not_silently_repaired(monkeypatch, change):
    result = envelope()
    choice = result["choices"][0]
    if change == "extra-choice":
        result["choices"].append(copy.deepcopy(choice))
    elif change == "no-usage":
        result.pop("usage")
    elif change == "no-finish":
        choice.pop("finish_reason")
    elif change in {"refusal", "tool_calls"}:
        choice["message"][change] = "rejected"
    else:
        choice["message"]["content"] = {
            "duplicate": '{"tool":"intent_classify","tool":"intent_classify",'
                         '"arguments":{},"reason":"why"}',
            "nan": '{"tool":"intent_classify","arguments":{},"reason":NaN}',
            "markdown": '```json\n{"tool":"intent_classify","arguments":{},"reason":"why"}\n```',
        }[change]
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: result)
    with pytest.raises(DomainError) as error:
        client.decision({})
    assert error.value.code == "provider_invalid"


def fake_http(monkeypatch, handler):
    original = httpx.Client
    configurations = []

    def client(**kwargs):
        configurations.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(module.httpx, "Client", client)
    return configurations


@pytest.mark.parametrize(("change", "code", "expected_tokens"), [
    ("invalid-tool", "provider_invalid", 4000),
    ("invalid-arguments", "provider_invalid", 4000),
    ("truncated", "provider_invalid", 4000),
    ("refusal", "provider_invalid", 4000),
    ("extra-choice", "provider_invalid", 4000),
    ("malformed-action", "provider_invalid", 4000),
    ("duplicate-action-key", "provider_invalid", 4000),
    ("missing-usage", "provider_invalid", None),
    ("zero-usage", "provider_invalid", 0),
    ("http-rejected", "provider_http", 4000),
    ("http-ambiguous", "provider_unknown", 4000),
    ("http-rejected-zero", "provider_http", 0),
    ("http-rejected-unknown", "provider_http", None),
    ("invalid-json", "provider_invalid", None),
    ("oversize", "provider_size", None),
])
def test_http_failure_preserves_reported_usage_without_retry(monkeypatch, change, code, expected_tokens):
    result = envelope(tokens=4000)
    status = 200
    if change == "invalid-tool":
        result = envelope({"tool": "invented", "arguments": {}, "reason": "TEST invalid"}, 4000)
    elif change == "invalid-arguments":
        result = envelope({"tool": "intent_classify", "arguments": {"answer": True},
                           "reason": "TEST invalid arguments"}, 4000)
    elif change == "truncated":
        result["choices"][0]["finish_reason"] = "length"
    elif change == "refusal":
        result["choices"][0]["message"]["refusal"] = "TEST refused"
    elif change == "extra-choice":
        result["choices"].append(copy.deepcopy(result["choices"][0]))
    elif change in {"malformed-action", "duplicate-action-key"}:
        result["choices"][0]["message"]["content"] = ("{broken" if change == "malformed-action" else
            '{"tool":"intent_classify","tool":"intent_classify","arguments":{},"reason":"TEST"}')
    elif change == "missing-usage":
        result.pop("usage")
    elif change == "zero-usage":
        result["usage"]["total_tokens"] = 0
    elif change.startswith("http-"):
        status = 500 if change == "http-ambiguous" else 422
        result = {"error": {"message": "TEST rejected or uncertain"},
                  "usage": {"total_tokens": expected_tokens}}
    if change == "oversize":
        monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 20)
    requests = []

    def handler(request):
        requests.append(request)
        if change == "invalid-json":
            return httpx.Response(200, content=b"{broken")
        return httpx.Response(status, json=result)

    fake_http(monkeypatch, handler)
    with pytest.raises(ProviderFailure) as error:
        LocalProvider("http://127.0.0.1").decision({})
    assert error.value.code == code
    assert error.value.reported_tokens == expected_tokens
    assert error.value.usage_unknown is (expected_tokens is None)
    assert len(requests) == 1 and requests[0].method == "POST"


@pytest.mark.parametrize("tokens", [True, False, -1, 1.5, "73", None, 100001])
def test_http_invalid_usage_never_becomes_a_trusted_zero(monkeypatch, tokens):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=envelope(tokens=tokens))

    fake_http(monkeypatch, handler)
    with pytest.raises(ProviderFailure) as error:
        LocalProvider("http://127.0.0.1").decision({})
    assert error.value.code == "provider_invalid"
    assert error.value.reported_tokens is None and error.value.usage_unknown is True
    assert len(requests) == 1


def test_http_transport_is_bounded_local_and_no_env(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=envelope())

    options = fake_http(monkeypatch, handler)
    result = LocalProvider("http://127.0.0.1:18080").decision({})
    assert result.tokens == 73
    assert len(requests) == 1 and requests[0].method == "POST"
    assert options[0]["trust_env"] is False and options[0]["follow_redirects"] is False
    assert options[0]["timeout"].connect == 5
    assert options[0]["timeout"].read == module.REQUEST_TIMEOUT


@pytest.mark.parametrize(("status", "code"), [
    (400, "provider_http"), (401, "provider_http"), (422, "provider_http"),
    (429, "provider_http"), (302, "provider_unknown"), (500, "provider_unknown"),
])
def test_http_errors_and_redirects_never_retry(monkeypatch, status, code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://outside.invalid"})

    fake_http(monkeypatch, handler)
    with pytest.raises(DomainError) as error:
        LocalProvider("http://127.0.0.1").decision({})
    assert error.value.code == code
    assert len(requests) == 1


@pytest.mark.parametrize(("exception", "code"), [
    (httpx.ConnectError, "provider_unavailable"), (httpx.ConnectTimeout, "provider_unavailable"),
    (httpx.PoolTimeout, "provider_unavailable"),
    (httpx.ReadTimeout, "provider_unknown"), (httpx.ReadError, "provider_unknown"),
    (httpx.WriteTimeout, "provider_unknown"), (httpx.WriteError, "provider_unknown"),
    (httpx.RemoteProtocolError, "provider_unknown"),
])
def test_transport_unknown_is_distinct_from_no_connection(monkeypatch, exception, code):
    requests = []

    def handler(request):
        requests.append(request)
        raise exception("controlled transport failure", request=request)

    fake_http(monkeypatch, handler)
    with pytest.raises(DomainError) as error:
        LocalProvider("http://127.0.0.1").decision({})
    assert error.value.code == code and len(requests) == 1
    assert isinstance(error.value, ProviderFailure)
    assert error.value.reported_tokens == (0 if code == "provider_unavailable" else None)
    assert error.value.usage_unknown is (code == "provider_unknown")


@pytest.mark.parametrize("content", [b"[]", b"null", b"{not json}", b"\xff", b'{"a":1,"a":2}',
                                    b'{"a":Infinity}'])
def test_response_json_rejected_cleanly(monkeypatch, content):
    fake_http(monkeypatch, lambda _: httpx.Response(200, content=content))
    with pytest.raises(DomainError) as error:
        LocalProvider("http://127.0.0.1").decision({})
    assert error.value.code == "provider_invalid"


def test_response_byte_limit(monkeypatch):
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 10)
    fake_http(monkeypatch, lambda _: httpx.Response(200, content=b" " * 11))
    with pytest.raises(DomainError) as error:
        LocalProvider("http://127.0.0.1").decision({})
    assert error.value.code == "provider_size"


def test_stream_elapsed_deadline_is_unknown(monkeypatch):
    clock = iter([0, module.REQUEST_TIMEOUT + 1])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    fake_http(monkeypatch, lambda _: httpx.Response(200, json=envelope()))
    with pytest.raises(DomainError) as error:
        LocalProvider("http://127.0.0.1").decision({})
    assert error.value.code == "provider_unknown"


@pytest.mark.parametrize("context", [[], {"bad": float("nan")}, {"large": "a" * 200001}])
def test_bad_context_fails_before_http(monkeypatch, context):
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: pytest.fail("Network must not be reached"))
    with pytest.raises(DomainError) as error:
        client.decision(context)
    assert error.value.code == "model_context"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False


def test_invalid_dispatch_endpoint_reports_zero_without_http(monkeypatch):
    fake_http(monkeypatch, lambda _: pytest.fail("Rejected endpoint cannot dispatch HTTP"))
    with pytest.raises(ProviderFailure) as error:
        LocalProvider._post("https://outside.invalid/v1/chat/completions", {})
    assert error.value.code == "model_endpoint"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False


def assert_decision_projection(projected, original):
    """Only node lexical terms and top-level null condition fields may be omitted."""
    assert projected.keys() == original.keys()
    assert {key: value for key, value in projected.items() if key != "tree"} == {
        key: value for key, value in original.items() if key != "tree"
    }
    assert projected["tree"].keys() == original["tree"].keys()
    assert {key: value for key, value in projected["tree"].items() if key not in {"nodes", "conditions"}} == {
        key: value for key, value in original["tree"].items() if key not in {"nodes", "conditions"}
    }
    for actual, full in zip(projected["tree"]["nodes"], original["tree"]["nodes"], strict=True):
        assert actual == {key: value for key, value in full.items() if key != "terms"}
    for actual, full in zip(projected["tree"]["conditions"], original["tree"]["conditions"], strict=True):
        assert actual == {key: value for key, value in full.items() if value is not None}


def test_immutable_request_tree_serialized_before_mutable_state(monkeypatch):
    client = LocalProvider("http://127.0.0.1")
    payloads = []

    def post(_url, payload):
        payloads.append(payload)
        return envelope()

    monkeypatch.setattr(client, "_post", post)
    context = public_context()
    context["tree"]["nodes"][0]["terms"] = ["TEST lexical index only"]
    context["tree"]["conditions"][0]["upper"] = None
    original = copy.deepcopy(context)
    client.decision(context)
    context["candidates"] = ["normal"]
    context["evidence"] = [{"condition_id": "enabled", "evidence_id": "e1", "state": "supported"}]
    client.decision(context)
    first, second = [item["messages"][1]["content"] for item in payloads]
    marker = ',"evidence":'
    assert first.split(marker)[0] == second.split(marker)[0]
    assert list(json.loads(first))[:2] == ["request", "tree"]
    assert_decision_projection({key: value for key, value in json.loads(first).items()
                                if key != "available_actions"}, original)
    assert all(item["cache_prompt"] is True for item in payloads)
    assert all(variant["properties"]["reason"]["maxLength"] == 96
               for variant in action_response_schema()["oneOf"])
    assert "Candidates present: do not query again" in payloads[0]["messages"][0]["content"]


def test_late_candidate_query_keeps_evidence_in_cached_prefix(monkeypatch):
    client = LocalProvider("http://127.0.0.1")
    payloads = []

    def post(_url, payload):
        payloads.append(payload)
        return envelope()

    monkeypatch.setattr(client, "_post", post)
    context = public_context()
    context["evidence"] = [{"condition_id": "enabled", "evidence_id": "e1", "state": "supported"}]
    client.decision(context)
    context["candidates"] = ["normal"]
    client.decision(context)
    first, second = [item["messages"][1]["content"] for item in payloads]
    assert first.split(',"candidates":')[0] == second.split(',"candidates":')[0]
    assert first.index('"evidence":') < first.index('"candidates":')


def test_legal_menu_preserves_expand_verify_choice_and_issued_ids():
    context = public_context()
    assert module.available_actions(context) == {"intent_classify": {}}
    context["intent"] = {"kind": "conveyor_diagnosis"}
    menu = module.available_actions(context)
    assert set(menu) == {"kg_query_candidates", "expand_tree_node", "time_series_verify"}
    assert menu["time_series_verify"]["condition_ids"] == ["enabled"]
    assert menu["expand_tree_node"]["node_id"] == ["root"]
    context["candidates"] = ["jam", "normal"]
    assert "kg_query_candidates" not in module.available_actions(context)
    context["expanded_nodes"] = ["root", "drive"]
    context["evidence"] = [{"condition_id": "enabled", "evidence_id": "e1", "state": "supported"}]
    menu = module.available_actions(context)
    assert "enabled" not in menu["time_series_verify"]["condition_ids"]
    assert "obstruction" not in menu["time_series_verify"]["condition_ids"]
    assert "request_human_input" not in menu  # Unknown signals must not trigger irrelevant questions.


def decision_projection_context(stage):
    """Public provider context with lexical index and complete non-null predicates."""
    context = complete_local_context() if stage == "submit" else public_context()
    for node in context["tree"]["nodes"]:
        node["terms"] = ["TEST lexical index", node["label"]]
        node["optional_annotation"] = None  # Null node fields must not be stripped.
    for condition in context["tree"]["conditions"]:
        if condition["source"] == "human":
            condition.update(operator="eq", threshold=False, human_key="physical_obstruction",
                             signal=None, other_signal=None, statistic=None, unit=None, upper=None)
        else:
            condition.update(operator="between", threshold=0.0, upper=4.5, unit="TEST unit",
                             signal=condition["condition_id"], other_signal=None, statistic="mean",
                             min_samples=3, max_gap_ms=150, max_age_ms=250, human_key=None)
        condition["annotation"] = {"terms": ["TEST nested semantics retained"], "value": None}
    if stage != "classify":
        context["intent"] = {"kind": "conveyor_diagnosis"}
    if stage == "human":
        context.update(candidates=["jam"], expanded_nodes=["root", "drive"])
        context["evidence"] = [
            {"evidence_id": "e-" + item["condition_id"], "condition_id": item["condition_id"],
             "source": "signal", "state": "supported" if item["condition_id"] in {
                 "enabled", "high-current", "slow-belt"} else "refuted", "details": {}}
            for item in context["tree"]["conditions"] if item["source"] == "signal"
        ]
    for item in context["evidence"]:
        item["details"].update(computed=0.0, answer=False, other_signal=None,
                               terms=["TEST evidence terms retained"])
    context["request"]["terms"] = ["TEST complaint index retained outside nodes"]
    context["request"]["annotation"] = None
    context["terms"] = ["TEST top-level terms retained"]
    context["observations"].append({"tool": "TEST", "result": {"value": None, "answer": False}})
    return context


@pytest.mark.parametrize("stage", ["classify", "investigate", "human", "submit"])
def test_decision_projection_omits_only_index_and_null_conditions_without_mutation(monkeypatch, stage):
    context = decision_projection_context(stage)
    original = copy.deepcopy(context)
    context_hash = digest(context)
    menu = module.available_actions(context)
    schema = module.decision_response_schema(context)
    if stage == "classify":
        action = {"tool": "intent_classify", "arguments": {}, "reason": "TEST classify"}
    elif stage == "investigate":
        action = {"tool": "time_series_verify", "arguments": {
            "condition_ids": menu["time_series_verify"]["condition_ids"]}, "reason": "TEST verify"}
    elif stage == "human":
        action = {"tool": "request_human_input", "arguments": {
            "condition_id": "obstruction", "question": "TEST safe recorded observation?"},
            "reason": "TEST request declared observation"}
    else:
        action = {"tool": "submit_diagnosis", "arguments": module._submission_arguments(context, "normal"),
                  "reason": "TEST exact issued path and citations"}
    menu_inputs, schema_inputs, requests = [], [], []
    real_menu, real_schema = module.available_actions, module.decision_response_schema

    def menu_from_full(full):
        menu_inputs.append(copy.deepcopy(full))
        return real_menu(full)

    def schema_from_full(full):
        schema_inputs.append(copy.deepcopy(full))
        return real_schema(full)

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=envelope(action))

    monkeypatch.setattr(module, "available_actions", menu_from_full)
    monkeypatch.setattr(module, "decision_response_schema", schema_from_full)
    configurations = fake_http(monkeypatch, handler)
    client = LocalProvider("http://127.0.0.1")
    result = client.decision(context)
    assert result.action.model_dump() == action
    assert context == original and digest(context) == context_hash
    assert menu_inputs and schema_inputs and all(item == original for item in menu_inputs + schema_inputs)
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    projected = json.loads(payload["messages"][1]["content"])
    assert projected.pop("available_actions") == menu
    assert_decision_projection(projected, original)
    assert payload["response_format"]["json_schema"]["schema"] == schema
    predicates = {item["condition_id"]: item for item in projected["tree"]["conditions"]}
    assert predicates["enabled"]["threshold"] == 0.0 and predicates["enabled"]["upper"] == 4.5
    assert predicates["obstruction"]["threshold"] is False
    assert predicates["enabled"]["annotation"]["value"] is None
    assert projected["evidence"] == original["evidence"]
    assert projected["observations"] == original["observations"]
    assert len(payload["messages"][1]["content"]) < len(json.dumps(original))
    assert payload["max_tokens"] == 1200 and configurations[0]["timeout"].read == 180
    prompt = payload["messages"][0]["content"]
    for phrase in ("frozen by the program", "not a fixed pipeline", "Candidates present: do not query again"):
        assert phrase in prompt
    assert client.metadata["decision_prompt_sha256"] == digest(module.DECISION_PROMPT)
    assert client.metadata["context_serialization"] == "immutable-tree-growing-evidence-prefix-v3-no-null-index"
    assert client.fingerprint == "local-" + digest(client.metadata)
    previous = {**client.metadata, "context_serialization": "immutable-tree-growing-evidence-prefix-v2"}
    assert client.fingerprint != "local-" + digest(previous)
    isolated = module._decision_context(context)
    isolated["tree"]["conditions"][0]["annotation"]["terms"].append("TEST mutate projection")
    isolated["tree"]["nodes"][0]["label"] = "TEST changed projection"
    isolated["request"]["terms"].clear()
    assert context == original and digest(context) == context_hash


@pytest.mark.parametrize("change", ["null-node", "scalar-node", "null-conditions", "null-condition", "scalar-condition"])
def test_decision_malformed_projection_context_is_known_zero_before_http(monkeypatch, change):
    context = decision_projection_context("classify")
    if change == "null-node":
        context["tree"]["nodes"][0] = None
    elif change == "scalar-node":
        context["tree"]["nodes"][0] = 42
    elif change == "null-conditions":
        context["tree"]["conditions"] = None
    elif change == "null-condition":
        context["tree"]["conditions"][0] = None
    else:
        context["tree"]["conditions"][0] = 42
    original = copy.deepcopy(context)
    fake_http(monkeypatch, lambda _: pytest.fail("Malformed projection must fail before inference"))
    with pytest.raises(ProviderFailure) as error:
        LocalProvider("http://127.0.0.1").decision(context)
    assert error.value.code == "model_context"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False
    assert context == original


def test_decision_schema_limits_ids_without_hiding_knowledge_labels(monkeypatch):
    context = public_context()
    context["intent"] = {"kind": "conveyor_diagnosis"}
    client = LocalProvider("http://127.0.0.1")
    payloads = []

    def post(_url, payload):
        payloads.append(payload)
        return envelope({"tool": "time_series_verify", "arguments": {"condition_ids": ["enabled"]},
                         "reason": "Check the visible global signal."})

    monkeypatch.setattr(client, "_post", post)
    assert client.decision(context).action.tool == "time_series_verify"
    schema = payloads[0]["response_format"]["json_schema"]["schema"]
    variant = next(item for item in schema["oneOf"]
                   if item["properties"]["tool"]["enum"] == ["time_series_verify"])
    assert variant["properties"]["arguments"]["properties"]["condition_ids"]["const"] == ["enabled"]
    projected = json.loads(payloads[0]["messages"][1]["content"])
    assert_decision_projection({key: value for key, value in projected.items() if key != "available_actions"}, context)


@pytest.mark.parametrize("action", [
    {"tool": "kg_query_candidates", "arguments": {"query": "already queried"}, "reason": "Repeat"},
    {"tool": "time_series_verify", "arguments": {"condition_ids": ["high-current"]}, "reason": "Hidden"},
    {"tool": "expand_tree_node", "arguments": {"node_id": "drive"}, "reason": "Skip root"},
])
def test_eligible_action_and_id_validation_independent_of_model_schema(monkeypatch, action):
    context = public_context()
    context["intent"] = {"kind": "conveyor_diagnosis"}
    context["candidates"] = ["normal"]
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: envelope(action))
    with pytest.raises(DomainError) as error:
        client.decision(context)
    assert error.value.code == "provider_invalid"


def complete_local_context():
    context = supported_normal_context()
    context["evidence"].extend([
        {"condition_id": cid, "evidence_id": "e-" + cid, "state": "refuted", "details": {}}
        for cid in ["high-current", "slow-belt", "difference", "clear"]
    ])
    return context


def test_submit_schema_binds_continuous_path_and_all_gate_citations(monkeypatch):
    context = complete_local_context()
    # any gate still requires citation of its refuted predicate, not just supporting evidence.
    context["evidence"][2]["state"] = "refuted"
    variants = module.decision_response_schema(context)["oneOf"]
    submit = next(item for item in variants if item["properties"]["tool"]["enum"] == ["submit_diagnosis"])
    fields = submit["properties"]["arguments"]["properties"]
    arguments = {name: value["const"] for name, value in fields.items()}
    assert arguments == {"node_id": "normal", "path": ["root", "drive", "normal"],
                         "evidence_ids": ["e-enabled", "e-normal-speed", "e-normal-current"]}
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: envelope({
        "tool": "submit_diagnosis", "arguments": arguments, "reason": "Cite the checked predicates",
    }))
    assert client.decision(context).action.arguments == arguments


@pytest.mark.parametrize("change", ["path", "missing-citation", "duplicate-citation", "extra-citation"])
def test_submission_binding_is_validated_without_trusting_grammar(monkeypatch, change):
    context = complete_local_context()
    arguments = module._submission_arguments(context, "normal")
    if change == "path":
        arguments["path"] = ["root", "normal"]
    elif change == "missing-citation":
        arguments["evidence_ids"].pop()
    elif change == "duplicate-citation":
        arguments["evidence_ids"].append(arguments["evidence_ids"][0])
    else:
        arguments["evidence_ids"].append("e-high-current")
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: envelope({
        "tool": "submit_diagnosis", "arguments": arguments, "reason": "Invalid binding",
    }))
    with pytest.raises(DomainError) as error:
        client.decision(context)
    assert error.value.code == "provider_invalid"


def test_verification_subset_is_not_silently_accepted(monkeypatch):
    context = public_context()
    context.update(intent={"kind": "diagnosis"}, expanded_nodes=["root", "drive"])
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: envelope({
        "tool": "time_series_verify", "arguments": {"condition_ids": ["enabled"]}, "reason": "Subset",
    }))
    with pytest.raises(DomainError) as error:
        client.decision(context)
    assert error.value.code == "provider_invalid"


def test_local_supported_missing_candidate_can_be_queried_once():
    context = complete_local_context()
    context["candidates"] = ["jam"]
    assert "kg_query_candidates" in module.available_actions(context)
    assert "submit_diagnosis" not in module.available_actions(context)
    context["observations"] = [{"tool": "kg_query_candidates"}]
    assert "kg_query_candidates" not in module.available_actions(context)


def test_unmanaged_and_test_sessions_are_noop_scopes():
    for client in (LocalProvider("http://127.0.0.1"), SyntheticProvider()):
        with client.session() as scoped:
            assert scoped is client


def test_narrative_separately_opt_in_and_nonmutating(monkeypatch):
    client = LocalProvider("http://127.0.0.1")
    requests = []
    draft = {"explanation": "Recorded outcome only.", "limitations": ["Unreviewed model draft."]}

    def post(_url, payload):
        requests.append(payload)
        return envelope(draft if payload["response_format"]["json_schema"]["name"] == "Narrative"
                        else None)

    monkeypatch.setattr(client, "_post", post)
    client.decision({})
    assert len(requests) == 1
    diagnosis = {"status": "END", "node_id": "jam", "evidence_ids": ["e1"]}
    before = copy.deepcopy(diagnosis)
    assert client.narrative(diagnosis) == draft
    assert diagnosis == before
    assert len(requests) == 2
    assert "OPTIONAL" in requests[-1]["messages"][0]["content"]
    assert "immutable" in requests[-1]["messages"][0]["content"]


def narrative_report(outcome="jam"):
    """Explicit report-shaped provider input; not a generated or reasoner-verified diagnosis."""
    scope = {"task_id": "task-test-report", "tree_version": "tree-test-v1",
             "tree_hash": "a" * 64, "data_hash": "b" * 64,
             "request": {"plant_id": "PLANT-A", "device_id": "conveyor-test", "event_id": "event-test",
                         "window": {"start_ms": 1000, "end_ms": 2000}, "complaint": "TEST observations"}}
    evidence = []
    for index, (state, source, details) in enumerate([
        ("supported", "signal", {"computed": 4.5, "unit": "A", "n": 10, "operator": "ge",
                                   "threshold": 4.0, "upper": 8.0, "signal": "motor_current",
                                   "other_signal": None, "statistic": "mean"}),
        ("refuted", "signal", {"computed": 0.0, "unit": "m/s", "n": 10, "operator": "gt",
                                 "threshold": 1.0, "signal": "belt_speed", "statistic": "mean"}),
        ("supported", "human", {"answer": True, "not_sensor_verified": True}),
        ("refuted", "human", {"answer": False, "not_sensor_verified": True}),
    ]):
        evidence.append({"evidence_id": f"e-test-{index}", "condition_id": f"c-test-{index}",
            "state": state, "source": source,
            **{key: value for key, value in scope.items() if key != "request"},
            **{key: value for key, value in scope["request"].items() if key != "complaint"},
            "actor_id": "operator-test", "details": {**details,
                "data_hash": scope["data_hash"], "window": scope["request"]["window"],
                "observation_note": "TEST original safe observation remains authoritative",
                "audit_metadata": "TEST repeated scope must not be sent"}})
    if outcome == "missing":
        evidence.append({**copy.deepcopy(evidence[0]), "evidence_id": "e-test-missing",
            "condition_id": "c-test-missing", "state": "missing",
            "details": {"n": 0, "reason": "TEST no samples in pinned window"}})
    refused = outcome in {"missing", "conflicting"}
    submission = None if refused else {
        **scope, "submission_id": "diagnosis-test", "content_hash": "c" * 64,
        "node_id": outcome, "label": f"TEST {outcome}", "path": ["root", "drive", outcome],
        "evidence_ids": ["e-test-0", "e-test-2"],
        "conclusion_level": "supported_under_declared_synthetic_rules",
        "limitations": ["TEST rules do not prove physical causation.", "TEST no equipment control."],
        "actor_id": "operator-test",
    }
    error = {"code": outcome + "_signals", "message": "TEST cannot submit",
             "witness": {"evidence_ids": [item["evidence_id"] for item in evidence]}} if refused else None
    return {**scope, "status": "escalated" if refused else "submitted", "error": error,
            "submission": submission, "evidence": evidence,
            "provider": {"fingerprint": "TEST provider-only fixture"}, "calls": 10, "tokens": 4000}


@pytest.mark.parametrize("outcome", ["normal", "jam", "slip", "sensor_fault", "missing", "conflicting"])
def test_narrative_projection_retains_all_outcomes_citations_and_authoritative_input(monkeypatch, outcome):
    diagnosis = narrative_report(outcome)
    before = copy.deepcopy(diagnosis)
    original_hash = digest(diagnosis)
    requests = []
    draft = {"explanation": "TEST recorded outcome. This is an unreviewed model draft.",
             "limitations": ["Not an authoritative diagnosis.", "No additional measurements inferred."]}

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=envelope(draft, tokens=73))

    configurations = fake_http(monkeypatch, handler)
    client = LocalProvider("http://127.0.0.1")
    assert client.narrative(diagnosis) == draft
    assert diagnosis == before and digest(diagnosis) == original_hash
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    projected = json.loads(payload["messages"][1]["content"])
    scope_keys = {"task_id", "status", "request", "tree_version", "tree_hash", "data_hash", "error"}
    assert {key: projected[key] for key in scope_keys} == {key: before[key] for key in scope_keys}
    assert len(projected["evidence"]) == len(before["evidence"])
    detail_keys = {"computed", "unit", "n", "reason", "answer", "operator", "threshold", "upper",
                   "signal", "other_signal", "statistic", "not_sensor_verified"}
    for projected_item, original in zip(projected["evidence"], before["evidence"], strict=True):
        assert projected_item == {
            **{key: original[key] for key in ("evidence_id", "condition_id", "state", "source")},
            "details": {key: value for key, value in original["details"].items() if key in detail_keys},
        }
    assert projected["evidence"][1]["details"]["computed"] == 0.0
    assert projected["evidence"][3]["details"]["answer"] is False
    if before["submission"] is None:
        assert projected["submission"] is None
        assert projected["error"]["witness"]["evidence_ids"] == before["error"]["witness"]["evidence_ids"]
    else:
        assert projected["submission"] == {key: before["submission"][key] for key in (
            "submission_id", "content_hash", "node_id", "label", "path", "evidence_ids",
            "conclusion_level", "limitations",
        )}
        cited = set(projected["submission"]["evidence_ids"])
        assert cited < {item["evidence_id"] for item in projected["evidence"]}
    serialized = payload["messages"][1]["content"]
    assert serialized.count('"tree_hash"') == serialized.count('"data_hash"') == 1
    assert serialized.count('"window"') == 1
    assert "audit_metadata" not in serialized and "observation_note" not in serialized
    assert "All evidence outcomes retained" in projected["projection"]
    assert payload["response_format"]["json_schema"]["name"] == "Narrative"
    assert payload["max_tokens"] == 1200 and configurations[0]["timeout"].read == 180
    prompt = payload["messages"][0]["content"]
    assert "two\nshort sentences" in prompt and "two concise limitations" in prompt
    assert "Do not infer omitted measurements" in prompt
    assert client.metadata["narrative_projection"] == "bound-scope-all-evidence-outcomes-v1"
    assert client.metadata["narrative_prompt_sha256"] == digest(module.NARRATIVE_PROMPT)
    assert client.fingerprint == "local-" + digest(client.metadata)
    without_projection = {key: value for key, value in client.metadata.items() if key != "narrative_projection"}
    assert client.fingerprint != "local-" + digest(without_projection)


@pytest.mark.parametrize("change", [
    "not-object", "null-report", "null-evidence", "noniterable-evidence", "object-evidence",
    "null-item", "missing-id", "missing-condition", "missing-state", "missing-source",
    "missing-details", "null-details", "list-details", "nonfinite-detail", "nonobject-submission",
])
def test_narrative_malformed_projection_context_fails_before_http_with_zero_usage(monkeypatch, change):
    diagnosis = narrative_report()
    if change == "not-object":
        diagnosis = []
    elif change == "null-report":
        diagnosis = None
    elif change == "null-evidence":
        diagnosis["evidence"] = None
    elif change == "noniterable-evidence":
        diagnosis["evidence"] = 42
    elif change == "object-evidence":
        diagnosis["evidence"] = {"evidence_id": "not-an-evidence-array"}
    elif change == "null-item":
        diagnosis["evidence"][0] = None
    elif change.startswith("missing-"):
        key = {"missing-id": "evidence_id", "missing-condition": "condition_id"}.get(
            change, change.removeprefix("missing-"))
        diagnosis["evidence"][0].pop(key)
    elif change in {"null-details", "list-details"}:
        diagnosis["evidence"][0]["details"] = None if change == "null-details" else []
    elif change == "nonfinite-detail":
        diagnosis["evidence"][0]["details"]["computed"] = float("inf")
    elif change == "nonobject-submission":
        diagnosis["submission"] = 42
    before = copy.deepcopy(diagnosis)
    fake_http(monkeypatch, lambda _: pytest.fail("Malformed report must not dispatch inference"))
    with pytest.raises(ProviderFailure) as error:
        LocalProvider("http://127.0.0.1").narrative(diagnosis)
    assert error.value.code == "model_context"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False
    assert diagnosis == before


def test_narrative_does_not_accept_authoritative_action(monkeypatch):
    client = LocalProvider("http://127.0.0.1")
    monkeypatch.setattr(client, "_post", lambda *_: envelope())
    with pytest.raises(DomainError) as error:
        client.narrative({"status": "END"})
    assert error.value.code == "provider_invalid"


def public_context():
    return {
        "request": {"complaint": "Conveyor not moving", "event_id": "opaque-event"},
        "tree": {"root_id": "root", "global_required": ["enabled"], "nodes": [
            {"node_id": "root", "label": "Conveyor tree", "kind": "branch", "children": ["drive"]},
            {"node_id": "drive", "label": "Drive branch", "kind": "branch",
             "children": ["jam", "slip", "normal"]},
            {"node_id": "jam", "label": "Mechanical jam", "kind": "diagnosis", "children": [],
             "gate": "all", "conditions": ["high-current", "slow-belt"], "mandatory": ["obstruction"]},
            {"node_id": "slip", "label": "Belt slip", "kind": "diagnosis", "children": [],
             "gate": "all", "conditions": ["difference", "clear"], "mandatory": []},
            {"node_id": "normal", "label": "Normal motion", "kind": "normal", "children": [],
             "gate": "any", "conditions": ["normal-speed", "normal-current"], "mandatory": []},
        ], "conditions": [
            {"condition_id": "enabled", "label": "Drive enabled", "source": "signal"},
            {"condition_id": "high-current", "label": "Motor current high", "source": "signal"},
            {"condition_id": "slow-belt", "label": "Belt motion slow", "source": "signal"},
            {"condition_id": "difference", "label": "Roller/belt discrepancy", "source": "signal"},
            {"condition_id": "clear", "label": "Path clear", "source": "signal"},
            {"condition_id": "normal-speed", "label": "Nominal motion", "source": "signal"},
            {"condition_id": "normal-current", "label": "Nominal current", "source": "signal"},
            {"condition_id": "obstruction", "label": "Physical obstruction present", "source": "human"},
        ]},
        "candidates": [], "expanded_nodes": [], "evidence": [], "observations": [], "intent": None,
        "remaining": {"calls": 30, "tools": 30, "tokens": 10000}, "pending_human": None,
    }


def drive_test_provider(order, states, candidate_ids=None):
    """Pure public-contract simulation; no repository fixture, runtime, graph or signal reads."""
    client = SyntheticProvider(order=order)
    context = public_context()
    actions = []
    for _ in range(30):
        reply = client.decision(copy.deepcopy(context))
        action = reply.action
        actions.append(action)
        context["observations"].append({"tool": action.tool, "result": {"ok": True}})
        if action.tool == "intent_classify":
            context["intent"] = {"kind": "diagnosis"}
        elif action.tool == "kg_query_candidates":
            if candidate_ids == [] and len(actions) > 2:
                context["candidates"] = ["jam"]  # A new leaf-label query obtains candidate membership.
            else:
                context["candidates"] = candidate_ids if candidate_ids is not None else [
                    {"node_id": "jam"}, {"node_id": "slip"}, {"node_id": "normal"}]
        elif action.tool == "expand_tree_node":
            nid = action.arguments["node_id"]
            assert nid == context["tree"]["root_id"] or any(
                node["node_id"] in context["expanded_nodes"] and nid in node.get("children", [])
                for node in context["tree"]["nodes"])
            assert nid not in context["expanded_nodes"]
            context["expanded_nodes"].append(nid)
        elif action.tool in {"time_series_verify", "request_human_input"}:
            ids = (action.arguments["condition_ids"] if action.tool == "time_series_verify"
                   else [action.arguments["condition_id"]])
            visible = set(context["tree"]["global_required"])
            nodes = {node["node_id"]: node for node in context["tree"]["nodes"]}
            for nid in context["expanded_nodes"]:
                for node in [nodes[nid], *(nodes[c] for c in nodes[nid].get("children", []))]:
                    visible.update(node.get("conditions", []) + node.get("mandatory", []))
            assert set(ids) <= visible
            for cid in ids:
                context["evidence"].append({"condition_id": cid, "evidence_id": "e-" + cid,
                                            "state": states.get(cid, "refuted"), "details": {}})
        elif action.tool == "submit_diagnosis":
            candidate_set = {row if isinstance(row, str) else row["node_id"]
                             for row in context["candidates"]}
            assert action.arguments["node_id"] in candidate_set
            assert set(action.arguments["path"][:-1]) <= set(context["expanded_nodes"])
            node = next(n for n in context["tree"]["nodes"]
                        if n["node_id"] == action.arguments["node_id"])
            required = set(context["tree"]["global_required"] + node["mandatory"] + node["conditions"])
            bound = {row["condition_id"] for row in context["evidence"]
                     if row["evidence_id"] in action.arguments["evidence_ids"]}
            assert required <= bound
            referenced = set(context["tree"]["global_required"])
            for tree_node in context["tree"]["nodes"]:
                referenced.update(tree_node.get("conditions", []) + tree_node.get("mandatory", []))
            signal_ids = {item["condition_id"] for item in context["tree"]["conditions"]
                          if item["source"] == "signal" and item["condition_id"] in referenced}
            observed_states = {row["condition_id"]: row["state"] for row in context["evidence"]}
            assert signal_ids <= set(observed_states)
            assert all(observed_states[cid] != "missing" for cid in signal_ids)
            return actions, context
    pytest.fail("TEST provider did not finish within bounded actions")


def test_synthetic_is_explicit_test_provider_not_inference():
    client = SyntheticProvider()
    assert client.is_test_provider is True
    assert "synthetic-test-" in client.fingerprint
    assert client.metadata["provider"] == "deterministic-TEST"
    assert client.fingerprint == SyntheticProvider().fingerprint
    assert client.fingerprint != SyntheticProvider(order="expand-first").fingerprint
    assert "TEST" in client.narrative({})["explanation"]
    with pytest.raises(ValueError):
        SyntheticProvider(order="hidden-oracle")


def test_synthetic_legal_reordering_supports_jam_with_human():
    states = dict.fromkeys(["enabled", "high-current", "slow-belt", "obstruction"], "supported")
    first, first_context = drive_test_provider("verify-first", states)
    second, second_context = drive_test_provider("expand-first", states)
    first_tools, second_tools = [a.tool for a in first], [a.tool for a in second]
    assert first_tools.index("time_series_verify") < first_tools.index("expand_tree_node")
    assert second_tools.index("expand_tree_node") < second_tools.index("time_series_verify")
    assert "request_human_input" in first_tools and "request_human_input" in second_tools
    assert first[-1].arguments == second[-1].arguments
    assert first[-1].arguments["node_id"] == "jam"
    assert first[-1].arguments["path"] == ["root", "drive", "jam"]
    for actions, context in [(first, first_context), (second, second_context)]:
        issued = {e["evidence_id"] for e in context["evidence"] if e["state"] == "supported"}
        assert set(actions[-1].arguments["evidence_ids"]) <= issued
        assert set(actions[-1].arguments["path"][:-1]) <= set(context["expanded_nodes"])
        assert all("obstruction" not in action.arguments["condition_ids"] for action in actions
                   if action.tool == "time_series_verify")


def test_synthetic_switches_refuted_candidate_based_on_evidence():
    states = dict.fromkeys(["enabled", "difference", "clear"], "supported")
    actions, _ = drive_test_provider("verify-first", states)
    assert actions[-1].arguments["node_id"] == "slip"
    assert not any(action.tool == "request_human_input" for action in actions)
    assert "e-high-current" not in actions[-1].arguments["evidence_ids"]


def test_synthetic_any_gate_normal_can_use_one_supported_predicate():
    states = {"enabled": "supported", "normal-speed": "supported"}
    actions, _ = drive_test_provider("verify-first", states, ["normal"])
    assert actions[-1].arguments["node_id"] == "normal"
    assert "e-normal-current" in actions[-1].arguments["evidence_ids"]


def test_synthetic_empty_candidates_do_not_hide_other_knowledge():
    states = dict.fromkeys(["enabled", "high-current", "slow-belt", "obstruction"], "supported")
    actions, _ = drive_test_provider("verify-first", states, [])
    assert actions[-1].arguments["node_id"] == "jam"
    assert [action.tool for action in actions].count("kg_query_candidates") == 2


def test_synthetic_does_not_use_event_name_as_oracle():
    context = public_context()
    context["intent"] = {"kind": "diagnosis"}
    context["candidates"] = ["jam", "slip"]
    other = copy.deepcopy(context)
    other["request"]["event_id"] = "EV-SLIP-OBVIOUS-ORACLE"
    assert SyntheticProvider().decision(context) == SyntheticProvider().decision(other)


def test_synthetic_pending_human_never_invents_answer_or_repeats_question():
    context = public_context()
    context["pending_human"] = {"question_id": "q1", "condition_id": "obstruction"}
    with pytest.raises(DomainError) as error:
        SyntheticProvider().decision(context)
    assert error.value.code == "provider_stuck"


def test_missing_signals_only_ask_issued_human_then_stop_without_submit():
    context = public_context()
    context["intent"] = {"kind": "diagnosis"}
    context["candidates"] = ["jam"]
    context["expanded_nodes"] = [node["node_id"] for node in context["tree"]["nodes"]]
    context["evidence"] = [
        {"condition_id": item["condition_id"], "evidence_id": "e-" + item["condition_id"],
         "state": "missing", "details": {"reason": "No numeric samples provided to model"}}
        for item in context["tree"]["conditions"] if item["source"] == "signal"
    ]
    client = SyntheticProvider()
    action = client.decision(context).action
    assert action.tool == "request_human_input"
    assert action.arguments["condition_id"] == "obstruction"
    context["evidence"].append({"condition_id": "obstruction", "evidence_id": "eh",
                                 "state": "supported", "details": {}})
    with pytest.raises(DomainError) as error:
        client.decision(context)
    assert error.value.code == "provider_stuck"


def test_synthetic_all_required_and_gate_evidence_are_needed():
    context = public_context()
    context["intent"] = {"kind": "diagnosis"}
    context["candidates"] = ["jam"]
    context["expanded_nodes"] = ["root", "drive", "jam"]
    context["evidence"] = [{"condition_id": cid, "evidence_id": "e-" + cid, "state": "supported",
                              "details": {}} for cid in ["high-current", "slow-belt", "obstruction"]]
    action = SyntheticProvider().decision(context).action
    assert action.tool == "time_series_verify"
    assert action.arguments == {"condition_ids": ["enabled"]}


def test_synthetic_never_checks_predicates_before_their_parent_is_expanded():
    context = public_context()
    context["intent"] = {"kind": "conveyor_diagnosis"}
    context["candidates"] = ["jam"]
    client = SyntheticProvider(order="verify-first")
    action = client.decision(context).action
    assert action.tool == "time_series_verify"
    assert action.arguments["condition_ids"] == ["enabled"]
    context["evidence"] = [{"condition_id": "enabled", "evidence_id": "e-enabled",
                              "state": "supported", "details": {}}]
    action = client.decision(context).action
    assert action.tool == "expand_tree_node" and action.arguments["node_id"] == "root"
    context["expanded_nodes"] = ["root"]
    action = client.decision(context).action
    assert action.tool == "expand_tree_node" and action.arguments["node_id"] == "drive"
    context["expanded_nodes"].append("drive")
    assert client.decision(context).action.tool == "time_series_verify"


def test_synthetic_any_gate_does_not_skip_unchecked_or_missing_predicates():
    context = public_context()
    context["intent"] = {"kind": "conveyor_diagnosis"}
    context["candidates"] = ["normal"]
    context["expanded_nodes"] = ["root", "drive"]
    context["evidence"] = [{"condition_id": cid, "evidence_id": "e-" + cid, "state": "supported",
                              "details": {}} for cid in ["enabled", "normal-speed"]]
    action = SyntheticProvider().decision(context).action
    assert action.tool == "time_series_verify"
    assert action.arguments["condition_ids"] == ["normal-current"]
    context["evidence"].append({"condition_id": "normal-current", "evidence_id": "e-normal-current",
                                 "state": "missing", "details": {}})
    assert SyntheticProvider().decision(context).action.tool != "submit_diagnosis"


def test_synthetic_conflicting_supported_leaves_are_not_arbitrarily_submitted():
    context = public_context()
    context["intent"] = {"kind": "conveyor_diagnosis"}
    context["candidates"] = ["jam", "slip"]
    context["expanded_nodes"] = ["root", "drive"]
    context["evidence"] = [{"condition_id": cid, "evidence_id": "e-" + cid, "state": "supported",
                              "details": {}} for cid in [
                                  "enabled", "high-current", "slow-belt", "obstruction",
                                  "difference", "clear",
                              ]]
    with pytest.raises(DomainError) as error:
        SyntheticProvider().decision(context)
    assert error.value.code == "provider_stuck"


def test_synthetic_missing_candidate_is_requeried_once_not_repeated():
    context = public_context()
    context["intent"] = {"kind": "conveyor_diagnosis"}
    context["candidates"] = ["slip"]
    context["expanded_nodes"] = ["root", "drive"]
    context["evidence"] = [{"condition_id": cid, "evidence_id": "e-" + cid, "state": "supported",
                              "details": {}} for cid in [
                                  "enabled", "high-current", "slow-belt", "obstruction",
                              ]]
    client = SyntheticProvider()
    action = client.decision(context).action
    assert action.tool == "kg_query_candidates"
    assert "Mechanical jam" in action.arguments["query"]
    context["observations"].append({"tool": "kg_query_candidates", "result": {"candidates": []}})
    with pytest.raises(DomainError) as error:
        client.decision(context)
    assert error.value.code == "provider_stuck"


def supported_normal_context():
    context = public_context()
    context["intent"] = {"kind": "conveyor_diagnosis"}
    context["candidates"] = ["normal", "jam", "slip"]
    context["expanded_nodes"] = ["root", "drive"]
    context["evidence"] = [
        {"condition_id": cid, "evidence_id": "e-" + cid, "state": "supported", "details": {}}
        for cid in ["enabled", "normal-speed", "normal-current"]
    ]
    return context


def test_supported_candidate_still_verifies_all_off_path_signals():
    context = supported_normal_context()
    action = SyntheticProvider().decision(context).action
    assert action.tool == "time_series_verify"
    assert set(action.arguments["condition_ids"]) == {
        "high-current", "slow-belt", "difference", "clear",
    }
    assert "obstruction" not in action.arguments["condition_ids"]


def test_signal_coverage_expands_off_path_parent_before_checking():
    context = supported_normal_context()
    root = context["tree"]["nodes"][0]
    root["children"].append("sensing")
    context["tree"]["nodes"].extend([
        {"node_id": "sensing", "label": "Sensing branch", "kind": "branch",
         "children": ["sensor"]},
        {"node_id": "sensor", "label": "Sensor disagreement", "kind": "diagnosis",
         "children": [], "gate": "all", "conditions": ["mismatch"], "mandatory": []},
    ])
    context["tree"]["conditions"].append({"condition_id": "mismatch", "label": "Sensor mismatch",
                                            "source": "signal"})
    context["evidence"].extend([
        {"condition_id": cid, "evidence_id": "e-" + cid, "state": "refuted", "details": {}}
        for cid in ["high-current", "slow-belt", "difference", "clear"]
    ])
    client = SyntheticProvider()
    action = client.decision(context).action
    assert action.tool == "expand_tree_node" and action.arguments["node_id"] == "sensing"
    context["expanded_nodes"].append("sensing")
    action = client.decision(context).action
    assert action.tool == "time_series_verify" and action.arguments["condition_ids"] == ["mismatch"]


def test_missing_off_path_signal_cannot_be_hidden_by_supported_normal():
    context = supported_normal_context()
    context["evidence"].extend([
        {"condition_id": cid, "evidence_id": "e-" + cid,
         "state": "missing" if cid == "difference" else "refuted", "details": {}}
        for cid in ["high-current", "slow-belt", "difference", "clear"]
    ])
    client = SyntheticProvider()
    action = client.decision(context).action
    assert action.tool == "request_human_input"
    assert action.arguments["condition_id"] == "obstruction"
    context["evidence"].append({"condition_id": "obstruction", "evidence_id": "e-human",
                                 "state": "refuted", "details": {}})
    with pytest.raises(DomainError) as error:
        client.decision(context)
    assert error.value.code == "provider_stuck"


def test_supported_normal_resolves_competing_human_predicate_before_submit():
    context = supported_normal_context()
    context["evidence"].extend([
        {"condition_id": cid, "evidence_id": "e-" + cid, "state": "refuted", "details": {}}
        for cid in ["high-current", "slow-belt", "difference", "clear"]
    ])
    context["tree"]["nodes"][1]["children"].append("human-alternative")
    context["tree"]["nodes"].append({
        "node_id": "human-alternative", "label": "Human observation alternative", "kind": "diagnosis",
        "children": [], "gate": "all", "conditions": ["normal-speed"], "mandatory": ["obstruction"],
    })
    client = SyntheticProvider()
    action = client.decision(context).action
    assert action.tool == "request_human_input" and action.arguments["condition_id"] == "obstruction"
    context["evidence"].append({"condition_id": "obstruction", "evidence_id": "e-human",
                                 "state": "refuted", "details": {}})
    assert client.decision(context).action.tool == "submit_diagnosis"


def test_unexplained_supported_fault_signal_blocks_otherwise_supported_normal():
    context = supported_normal_context()
    context["evidence"].extend([
        {"condition_id": cid, "evidence_id": "e-" + cid,
         "state": "supported" if cid == "high-current" else "refuted", "details": {}}
        for cid in ["high-current", "slow-belt", "difference", "clear"]
    ])
    with pytest.raises(DomainError) as error:
        SyntheticProvider().decision(context)
    assert error.value.code == "provider_stuck"


def test_unreferenced_signal_is_not_requested_or_required_for_submit():
    context = supported_normal_context()
    context["evidence"].extend([
        {"condition_id": cid, "evidence_id": "e-" + cid, "state": "refuted", "details": {}}
        for cid in ["high-current", "slow-belt", "difference", "clear"]
    ])
    context["tree"]["conditions"].append({"condition_id": "unused", "label": "Unbound signal",
                                            "source": "signal"})
    assert SyntheticProvider().decision(context).action.tool == "submit_diagnosis"


def test_all_signal_coverage_batches_respect_twelve_condition_limit():
    context = supported_normal_context()
    node = context["tree"]["nodes"][3]
    for index in range(14):
        cid = f"extra-{index}"
        node["conditions"].append(cid)
        context["tree"]["conditions"].append({"condition_id": cid, "label": f"Extra predicate {index}",
                                                "source": "signal"})
    action = SyntheticProvider().decision(context).action
    assert action.tool == "time_series_verify"
    assert len(action.arguments["condition_ids"]) == 12
    assert len(set(action.arguments["condition_ids"])) == 12
