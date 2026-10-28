"""Managed-runtime tests use temporary text artifacts and mocked children, never real inference."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from industrial_diagnosis import managed_provider as module
from industrial_diagnosis.managed_provider import (
    ALIASES,
    ManagedLocalProvider,
    ModelFile,
    RuntimeConfig,
)
from industrial_diagnosis.provider import LocalProvider, ProviderFailure
from industrial_diagnosis.schemas import DomainError, digest


@pytest.fixture
def runtime(tmp_path):
    # resolve() accommodates macOS's /var -> /private/var system symlink.
    root = tmp_path.resolve()
    binary = root / "llama-server"
    binary.write_bytes(b"TEST inert executable contents; never executed")
    binary.chmod(0o700)
    library = root / "libllama.dylib"
    library.write_bytes(b"TEST inert library")
    generation = root / "generator.gguf"
    generation.write_bytes(b"TEST inert generator placeholder")

    def record(path, model=False):
        data = path.read_bytes()
        result = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
        if model:
            result["bytes"] = len(data)
        return result

    config = {"binary": record(binary), "libraries": [record(library)],
              "models": {"generation": record(generation, True)},
              "lock_path": str(root / "runtime.lock"), "port": 18080}
    config_path = root / "runtime.json"
    config_path.write_text(json.dumps(config))
    return SimpleNamespace(root=root, config=config, path=config_path, binary=binary,
                           library=library, generation=generation)


def save(runtime, data):
    runtime.path.write_text(json.dumps(data))
    return ManagedLocalProvider(runtime.path)


class FakeProcess:
    def __init__(self, running=True, stubborn=False):
        self.running = running
        self.stubborn = stubborn
        self.terminated = 0
        self.killed = 0
        self.waits = []

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated += 1
        if not self.stubborn:
            self.running = False

    def kill(self):
        self.killed += 1
        self.running = False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.running:
            raise subprocess.TimeoutExpired("TEST fake process", timeout)
        return 0


class FakeSocket:
    def __init__(self, occupied=False):
        self.occupied = occupied
        self.bound = []
        self.timeout = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def settimeout(self, value):
        self.timeout = value

    def bind(self, address):
        self.bound.append(address)
        if self.occupied:
            raise OSError("TEST port occupied")


def mocked_runtime(monkeypatch, client, process=None, result=None, failure=None):
    process = process or FakeProcess()
    starts, posts, ready = [], [], []
    sock = FakeSocket()
    monkeypatch.setattr(module.socket, "socket", lambda *_: sock)

    def popen(command, **kwargs):
        starts.append((command, kwargs))
        assert os.fstat(kwargs["pass_fds"][0]).st_nlink == 1
        return process

    def post(url, payload):
        posts.append((url, payload))
        if failure:
            raise failure
        return result if result is not None else {"TEST": "one inference only"}

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(client, "_wait_ready", lambda child: ready.append(child))
    monkeypatch.setattr(LocalProvider, "_post", staticmethod(post))
    return SimpleNamespace(process=process, starts=starts, posts=posts, ready=ready, socket=sock)


def request(client):
    return client._post(client.generation_url + "/v1/chat/completions",
                        {"model": ALIASES["generation"]})


def test_only_explicit_config_read_no_inference_at_construction(runtime, monkeypatch):
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs:
                        pytest.fail("No process may be started during construction"))
    monkeypatch.setattr(module.httpx, "Client", lambda **_kwargs:
                        pytest.fail("No HTTP may be used during construction"))
    client = ManagedLocalProvider(runtime.path)
    assert set(client.config.models) == {"generation"}
    assert client._verified == {}
    assert client.generation_model == ALIASES["generation"]
    assert client.is_test_provider is False


def test_r2_three_model_shape_allowed_but_other_models_not_read_or_launched(runtime, monkeypatch):
    data = copy.deepcopy(runtime.config)
    for role in ("embedding", "rerank"):
        data["models"][role] = {"path": str(runtime.root / f"nonexistent-{role}.gguf"),
                                "sha256": "a" * 64, "bytes": 99}
    client = save(runtime, data)
    simulated = mocked_runtime(monkeypatch, client)
    request(client)
    assert len(simulated.starts) == len(simulated.posts) == 1
    assert "--embedding" not in simulated.starts[0][0]
    assert "--rerank" not in simulated.starts[0][0]
    assert set(client._verified) == {str(runtime.binary), str(runtime.library), str(runtime.generation)}


@pytest.mark.parametrize("change", [
    "extra", "roles", "partial-r2", "relative", "traversal", "duplicate", "library-dir",
    "hash", "bytes-bool", "bytes-zero", "port-bool", "port-small", "no-libraries",
])
def test_config_rejects_unrecognized_or_unsafe_fields(runtime, change):
    data = copy.deepcopy(runtime.config)
    if change == "extra":
        data["cloud_fallback"] = True
    elif change == "roles":
        data["models"] = {"cloud": data["models"]["generation"]}
    elif change == "partial-r2":
        data["models"]["embedding"] = {"path": str(runtime.root / "embed"), "sha256": "a" * 64,
                                         "bytes": 3}
    elif change == "relative":
        data["binary"]["path"] = "llama-server"
    elif change == "traversal":
        data["binary"]["path"] = str(runtime.root / ".." / "llama-server")
    elif change == "duplicate":
        data["lock_path"] = data["binary"]["path"]
    elif change == "library-dir":
        data["libraries"][0]["path"] = str(runtime.root / "other" / "lib.dylib")
    elif change == "hash":
        data["binary"]["sha256"] = "invalid"
    elif change in {"bytes-bool", "bytes-zero"}:
        data["models"]["generation"]["bytes"] = True if change == "bytes-bool" else 0
    elif change in {"port-bool", "port-small"}:
        data["port"] = True if change == "port-bool" else 80
    elif change == "no-libraries":
        data["libraries"] = []
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(data)
    with pytest.raises(DomainError) as error:
        save(runtime, data)
    assert error.value.code == "runtime_config"


@pytest.mark.parametrize("content", ["{}", "[]", "{bad", "x" * 64001,
                                    '{"binary":{},"binary":{}}'])
def test_bad_or_oversize_explicit_config_fails_cleanly(runtime, content):
    runtime.path.write_text(content)
    with pytest.raises(DomainError) as error:
        ManagedLocalProvider(runtime.path)
    assert error.value.code == "runtime_config"


def test_missing_config_never_looks_for_ambient_default(runtime):
    with pytest.raises(DomainError) as error:
        ManagedLocalProvider(runtime.root / "absent.json")
    assert error.value.code == "runtime_config"


def test_runtime_fingerprint_hashes_generation_and_engine_not_paths(runtime):
    first = ManagedLocalProvider(runtime.path)
    assert first.fingerprint == "managed-local-cpu-" + digest(first.metadata)
    assert "managed-cpu-v2-graph-scope" in first.metadata["protocol"]
    second = ManagedLocalProvider(runtime.path)
    assert first.fingerprint == second.fingerprint
    data = copy.deepcopy(runtime.config)
    data["models"]["generation"]["path"] = str(runtime.root / "relocated.gguf")
    assert save(runtime, data).fingerprint == first.fingerprint
    data["models"]["generation"]["sha256"] = "b" * 64
    assert save(runtime, data).fingerprint != first.fingerprint


def test_scoped_runtime_is_lazy_and_reuses_exactly_one_child(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session():
        assert not simulated.starts and not client._verified
        request(client)
        request(client)
        assert len(simulated.starts) == len(simulated.ready) == 1
        assert len(simulated.posts) == 2
        assert simulated.process.terminated == 0
        assert Path(simulated.starts[0][1]["cwd"]).exists()
    assert simulated.process.terminated == 1 and simulated.process.waits == [10]
    assert not Path(simulated.starts[0][1]["cwd"]).exists()
    assert not client._scope_lock.locked() and client._session.get() is None


def test_empty_scope_does_not_load_or_launch(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session():
        pass
    assert not simulated.starts and not simulated.posts and not client._verified


def test_scope_closes_on_human_pause_or_graph_error(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with pytest.raises(RuntimeError, match="graph exits"):
        with client.session():
            request(client)
            raise RuntimeError("graph exits")
    assert simulated.process.terminated == 1
    assert not client._scope_lock.locked()


def test_runtime_does_not_reuse_cache_across_graph_scopes(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session():
        request(client)
    simulated.process.running = True
    with client.session():
        request(client)
    assert len(simulated.starts) == 2 and simulated.process.terminated == 2


def test_scope_context_propagates_to_graph_executor_threads(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session(), ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(copy_context().run, request, client).result()
        executor.submit(copy_context().run, request, client).result()
    assert len(simulated.starts) == 1 and len(simulated.posts) == 2
    assert simulated.process.terminated == 1


def test_unrelated_thread_cannot_borrow_active_scope(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session(), ThreadPoolExecutor(max_workers=1) as executor:
        request(client)
        with pytest.raises(DomainError) as error:
            executor.submit(request, client).result()
        assert error.value.code == "provider_unavailable"
    assert len(simulated.posts) == len(simulated.starts) == 1


def test_nested_scope_rejected_without_releasing_outer_lease(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session():
        with pytest.raises(DomainError) as error, client.session():
            pytest.fail("Nested scope cannot enter")
        assert error.value.code == "provider_unavailable"
        request(client)
    assert len(simulated.starts) == 1


def test_failed_scope_reaps_immediately_and_never_retries(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client,
                               failure=DomainError("provider_unknown", "TEST ambiguous outcome", 503))
    with client.session():
        with pytest.raises(DomainError) as error:
            request(client)
        assert error.value.code == "provider_unknown"
        assert simulated.process.terminated == 1
        with pytest.raises(DomainError) as error:
            request(client)
        assert error.value.code == "provider_unavailable"
    assert len(simulated.starts) == len(simulated.posts) == 1


def test_dead_scoped_child_never_restarts_implicitly(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session():
        request(client)
        simulated.process.running = False
        with pytest.raises(DomainError) as error:
            request(client)
        assert error.value.code == "provider_unavailable"
    assert len(simulated.starts) == len(simulated.posts) == 1


def test_integrity_checked_between_scoped_calls(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session():
        request(client)
        runtime.library.write_bytes(b"changed while resident")
        with pytest.raises(DomainError) as error:
            request(client)
        assert error.value.code == "runtime_integrity"
    assert len(simulated.posts) == 1 and simulated.process.terminated == 1


def test_scope_retains_os_lock_between_requests(runtime, monkeypatch):
    first, second = ManagedLocalProvider(runtime.path), ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, first)
    with first.session():
        request(first)
        with pytest.raises(DomainError) as error:
            request(second)
        assert error.value.code == "provider_unavailable"
    assert len(simulated.starts) == 1


def test_stale_copied_context_cannot_restart_after_graph_exit(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with client.session():
        stale = copy_context()
        request(client)
    with pytest.raises(DomainError) as error:
        stale.run(request, client)
    assert error.value.code == "provider_unavailable"
    assert len(simulated.posts) == len(simulated.starts) == 1


def test_command_generation_cpu_only_without_shell(runtime):
    client = ManagedLocalProvider(runtime.path)
    command = client._command()
    assert isinstance(command, list) and command[0] == str(runtime.binary)
    for flag, value in {"--model": str(runtime.generation), "--alias": ALIASES["generation"],
                        "--host": "127.0.0.1", "--parallel": "1", "--n-gpu-layers": "0",
                        "--device": "none", "--threads": "4", "--ctx-size": "8192",
                        "--temp": "0", "--seed": "42"}.items():
        assert command[command.index(flag) + 1] == value
    assert "--no-op-offload" in command and "--no-webui" in command
    assert "--embedding" not in command and "--rerank" not in command


def test_verify_hashes_sizes_and_identity_cache(runtime):
    client = ManagedLocalProvider(runtime.path)
    for record in [client.config.binary, *client.config.libraries, client.config.models["generation"]]:
        client._verify(record)
    assert len(client._verified) == 3
    client._verify(client.config.models["generation"])
    data = runtime.generation.read_bytes()
    runtime.generation.write_bytes(b"Z" * len(data))
    with pytest.raises(DomainError) as error:
        client._verify(client.config.models["generation"])
    assert error.value.code == "runtime_integrity"


@pytest.mark.parametrize("change", ["size", "hash", "symlink", "directory", "missing", "executable"])
def test_file_integrity_fails_before_any_subprocess(runtime, monkeypatch, change):
    client = ManagedLocalProvider(runtime.path)
    if change == "size":
        runtime.generation.write_bytes(b"short")
    elif change == "hash":
        runtime.library.write_bytes(b"altered library")
    elif change == "symlink":
        target = runtime.root / "actual.gguf"
        runtime.generation.rename(target)
        runtime.generation.symlink_to(target)
    elif change == "directory":
        runtime.generation.unlink()
        runtime.generation.mkdir()
    elif change == "missing":
        runtime.generation.unlink()
    else:
        runtime.binary.chmod(0o600)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs:
                        pytest.fail("Invalid files cannot launch"))
    with pytest.raises(DomainError) as error:
        request(client)
    assert error.value.code == "runtime_integrity"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False
    assert not client._thread_lock.locked()


def test_symlink_parent_cannot_bypass_runtime_checks(runtime):
    actual = runtime.root / "actual"
    actual.mkdir()
    target = actual / "model.gguf"
    target.write_bytes(runtime.generation.read_bytes())
    link = runtime.root / "linked-directory"
    link.symlink_to(actual, target_is_directory=True)
    record = ModelFile(path=str(link / "model.gguf"), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                       bytes=target.stat().st_size)
    with pytest.raises(DomainError) as error:
        ManagedLocalProvider(runtime.path)._verify(record)
    assert error.value.code == "runtime_integrity"


def test_one_process_one_post_minimal_environment_and_reap(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    result = request(client)
    assert result == {"TEST": "one inference only"}
    assert len(simulated.starts) == len(simulated.posts) == len(simulated.ready) == 1
    _, options = simulated.starts[0]
    assert set(options["env"]) == {"HOME", "TMPDIR", "XDG_CACHE_HOME", "PATH", "LC_ALL"}
    assert options["env"]["PATH"] == "/usr/bin:/bin"
    assert options["env"]["HOME"] == options["env"]["TMPDIR"] == options["cwd"]
    assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.DEVNULL
    assert options["close_fds"] is True and len(options["pass_fds"]) == 1
    assert "shell" not in options
    assert not Path(options["cwd"]).exists()
    assert simulated.process.terminated == 1 and simulated.process.waits == [10]
    assert simulated.socket.bound == [("127.0.0.1", 18080)]
    assert not client._thread_lock.locked()
    # A later explicit call starts a fresh child, not a hidden long-lived service.
    simulated.process.running = True
    request(client)
    assert len(simulated.starts) == len(simulated.posts) == 2


@pytest.mark.parametrize("code", ["provider_unknown", "provider_invalid", "provider_size"])
def test_inference_error_never_retries_and_always_reaps(runtime, monkeypatch, code):
    client = ManagedLocalProvider(runtime.path)
    failure = DomainError(code, "TEST controlled result", 503)
    simulated = mocked_runtime(monkeypatch, client, failure=failure)
    with pytest.raises(DomainError) as error:
        request(client)
    assert (error.value.code, error.value.message, error.value.status) == (code, failure.message, 503)
    assert isinstance(error.value, ProviderFailure) and error.value is not failure
    assert error.value.reported_tokens is None and error.value.usage_unknown is True
    assert len(simulated.starts) == len(simulated.posts) == 1
    assert simulated.process.terminated == 1 and simulated.process.waits == [10]
    assert not client._thread_lock.locked()


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize(("failure_kind", "code", "tokens"), [
    ("http-rejected", "provider_http", 4000),
    ("http-ambiguous", "provider_unknown", 4000),
    ("http-no-usage", "provider_http", None),
    ("invalid-json", "provider_invalid", None),
    ("read-timeout", "provider_unknown", None),
    ("connect-error", "provider_unavailable", 0),
])
def test_real_http_provider_failure_metadata_survives_managed_cleanup(
        runtime, monkeypatch, scoped, failure_kind, code, tokens):
    client = ManagedLocalProvider(runtime.path)
    real_post = LocalProvider._post
    simulated = mocked_runtime(monkeypatch, client)
    requests, failures = [], []

    def handler(request):
        requests.append(request)
        if failure_kind == "read-timeout":
            raise httpx.ReadTimeout("TEST HTTP response lost", request=request)
        if failure_kind == "connect-error":
            raise httpx.ConnectError("TEST inference was not connected", request=request)
        if failure_kind == "invalid-json":
            return httpx.Response(200, content=b"{broken")
        status = 500 if failure_kind == "http-ambiguous" else 422
        body = {"error": {"message": "TEST inference failed"}}
        if tokens is not None:
            body["usage"] = {"total_tokens": tokens}
        return httpx.Response(status, json=body)

    def capture_real_failure(url, payload):
        try:
            return real_post(url, payload)
        except ProviderFailure as failure:
            failures.append(failure)
            raise

    health_http(monkeypatch, handler)
    monkeypatch.setattr(LocalProvider, "_post", staticmethod(capture_real_failure))
    with client.session() if scoped else nullcontext():
        with pytest.raises(ProviderFailure) as error:
            client.decision({})
        assert len(failures) == 1 and error.value is failures[0]
        assert error.value.code == code and error.value.reported_tokens == tokens
        assert error.value.usage_unknown is (tokens is None)
        assert len(simulated.starts) == 1 and simulated.process.terminated == 1
        assert simulated.process.waits == [10] and not client._thread_lock.locked()
        if scoped:
            with pytest.raises(ProviderFailure) as closed:
                client.decision({})
            assert closed.value.code == "provider_unavailable"
            assert closed.value.reported_tokens == 0 and closed.value.usage_unknown is False
    assert len(requests) == 1 and requests[0].method == "POST"
    assert str(requests[0].url) == client.generation_url + "/v1/chat/completions"
    assert not client._thread_lock.locked() and not client._scope_lock.locked()
    assert client._session.get() is None
    assert not Path(simulated.starts[0][1]["cwd"]).exists()


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("tokens", [0, 4000, None])
def test_typed_provider_failure_is_preserved_without_reclassification(runtime, monkeypatch, scoped, tokens):
    client = ManagedLocalProvider(runtime.path)
    # Even an ordinarily pre-dispatch code must retain authoritative typed metadata.
    failure = ProviderFailure("provider_unavailable", "TEST explicitly recorded usage", 502, tokens=tokens)
    simulated = mocked_runtime(monkeypatch, client, failure=failure)
    with client.session() if scoped else nullcontext():
        with pytest.raises(ProviderFailure) as error:
            request(client)
        assert error.value is failure
        assert error.value.reported_tokens == tokens and error.value.usage_unknown is (tokens is None)
        assert simulated.process.terminated == 1
    assert len(simulated.starts) == len(simulated.posts) == 1
    assert not client._thread_lock.locked() and not client._scope_lock.locked()


def test_stubborn_child_killed_and_reaped(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    child = FakeProcess(stubborn=True)
    mocked_runtime(monkeypatch, client, process=child)
    request(client)
    assert child.terminated == 1 and child.killed == 1
    assert child.waits == [10, 10]


def test_cleanup_timeout_is_transport_unknown_not_uncaught(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    child = FakeProcess(stubborn=True)
    monkeypatch.setattr(child, "kill", lambda: None)
    simulated = mocked_runtime(monkeypatch, client, process=child)
    with pytest.raises(DomainError) as error:
        request(client)
    assert error.value.code == "provider_unknown"
    assert error.value.reported_tokens is None and error.value.usage_unknown is True
    assert len(simulated.posts) == 1
    assert child.waits == [10, 10]


def test_startup_error_reaps_without_inference(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)

    def fail(_process):
        raise DomainError("provider_unavailable", "TEST startup failure", 503)

    monkeypatch.setattr(client, "_wait_ready", fail)
    with pytest.raises(DomainError) as error:
        request(client)
    assert error.value.code == "provider_unavailable"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False
    assert not simulated.posts and simulated.process.terminated == 1


def test_popen_failure_is_known_unsent(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)

    def fail(*_args, **_kwargs):
        raise OSError("TEST cannot execute")

    monkeypatch.setattr(module.subprocess, "Popen", fail)
    with pytest.raises(DomainError) as error:
        request(client)
    assert error.value.code == "provider_unavailable"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False
    assert not simulated.posts and not client._thread_lock.locked()


def test_occupied_port_refuses_unknown_server(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    sock = FakeSocket(occupied=True)
    monkeypatch.setattr(module.socket, "socket", lambda *_: sock)
    with pytest.raises(DomainError) as error:
        request(client)
    assert error.value.code == "provider_unavailable"
    assert not simulated.starts and not simulated.posts


def test_thread_busy_fails_without_touching_runtime(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    monkeypatch.setattr(client, "_exclusive_post", lambda *_: pytest.fail("Busy lock cannot launch"))
    client._thread_lock.acquire()
    try:
        with pytest.raises(DomainError) as error:
            request(client)
        assert error.value.code == "provider_unavailable"
    finally:
        client._thread_lock.release()


def test_os_lock_serializes_provider_instances(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with Path(runtime.config["lock_path"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DomainError) as error:
            request(client)
        assert error.value.code == "provider_unavailable"
    assert not simulated.starts and not simulated.posts
    request(client)
    assert len(simulated.posts) == 1


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "missing-parent", "fifo"])
def test_unsafe_lock_rejected(runtime, monkeypatch, kind):
    path = Path(runtime.config["lock_path"])
    if kind == "symlink":
        path.symlink_to(runtime.generation)
    elif kind == "hardlink":
        os.link(runtime.generation, path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        runtime.config["lock_path"] = str(runtime.root / "absent" / "runtime.lock")
        runtime.path.write_text(json.dumps(runtime.config))
    client = ManagedLocalProvider(runtime.path)
    simulated = mocked_runtime(monkeypatch, client)
    with pytest.raises(DomainError) as error:
        request(client)
    assert error.value.code == "runtime_config"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False
    assert not simulated.starts


@pytest.mark.parametrize(("suffix", "model"), [
    ("/v1/embeddings", ALIASES["generation"]), ("/v1/chat/completions/extra", ALIASES["generation"]),
    ("/v1/chat/completions", ALIASES["embedding"]), ("/v1/chat/completions", "cloud-model"),
])
def test_only_exact_generation_route_may_launch(runtime, monkeypatch, suffix, model):
    client = ManagedLocalProvider(runtime.path)
    monkeypatch.setattr(client, "_exclusive_post", lambda *_: pytest.fail("Wrong route cannot launch"))
    with pytest.raises(DomainError) as error:
        client._post(client.generation_url + suffix, {"model": model})
    assert error.value.code == "runtime_request"
    assert error.value.reported_tokens == 0 and error.value.usage_unknown is False


def health_http(monkeypatch, handler):
    original = httpx.Client
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(module.httpx, "Client", client)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return options


def test_health_bounded_gets_no_proxy_or_redirect(runtime, monkeypatch):
    client = ManagedLocalProvider(runtime.path)
    responses = iter([httpx.Response(503), httpx.Response(200, json={"status": "ok"})])
    requests = []

    def handler(request):
        requests.append(request)
        return next(responses)

    options = health_http(monkeypatch, handler)
    client._wait_ready(FakeProcess())
    assert len(requests) == 2 and all(r.method == "GET" for r in requests)
    assert all(str(r.url) == "http://127.0.0.1:18080/health" for r in requests)
    assert options[0]["trust_env"] is False and options[0]["follow_redirects"] is False
    assert options[0]["timeout"].read == 2 and options[0]["timeout"].connect == 1


@pytest.mark.parametrize("status", [302, 401, 500])
def test_health_nonstartup_status_fails_without_redirect(runtime, monkeypatch, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://outside.invalid"})

    health_http(monkeypatch, handler)
    with pytest.raises(DomainError) as error:
        ManagedLocalProvider(runtime.path)._wait_ready(FakeProcess())
    assert error.value.code == "provider_unavailable"
    assert len(requests) == 1


def test_health_oversize_fails_before_inference(runtime, monkeypatch):
    health_http(monkeypatch, lambda _: httpx.Response(200, content=b"x" * 4097))
    with pytest.raises(DomainError) as error:
        ManagedLocalProvider(runtime.path)._wait_ready(FakeProcess())
    assert error.value.code == "provider_unavailable"


def test_health_exited_child_fails_without_http(runtime, monkeypatch):
    health_http(monkeypatch, lambda _: pytest.fail("Exited process must not be contacted"))
    with pytest.raises(DomainError) as error:
        ManagedLocalProvider(runtime.path)._wait_ready(FakeProcess(running=False))
    assert error.value.code == "provider_unavailable"


def test_health_timeout_is_known_unsent(runtime, monkeypatch):
    health_http(monkeypatch, lambda _: pytest.fail("Timed-out startup must not be contacted"))
    clock = iter([0, module.STARTUP_TIMEOUT + 1])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(DomainError) as error:
        ManagedLocalProvider(runtime.path)._wait_ready(FakeProcess())
    assert error.value.code == "provider_unavailable"
