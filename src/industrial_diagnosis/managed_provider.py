"""Hash-verified local CPU generation; optional graph-scoped prompt-cache residency."""
from __future__ import annotations

import fcntl
import hashlib
import os
import socket
import stat
import subprocess
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import Field, ValidationError, model_validator

from .provider import PROTOCOL, LocalProvider, ProviderFailure, _json_object
from .schemas import DomainError, StrictModel, digest

ALIASES = {"generation": "qwen3-4b-instruct-2507", "embedding": "bge-m3",
           "rerank": "bge-reranker-v2-m3"}
STARTUP_TIMEOUT = 90.0
MAX_CONFIG_BYTES = 64_000
MAX_HEALTH_BYTES = 4096


class RuntimeFile(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelFile(RuntimeFile):
    bytes: int = Field(gt=0, strict=True)


class RuntimeConfig(StrictModel):
    """Trusted caller-selected config, not an API payload or ambient configuration.

    Accept generation-only or the delivered three-role R2 file shape for convenience. Embedding
    and rerank are never executed or read. No source dependency on the other project is needed.
    """

    binary: RuntimeFile
    libraries: list[RuntimeFile] = Field(min_length=1, max_length=30)
    models: dict[str, ModelFile]
    lock_path: str = Field(min_length=1)
    port: int = Field(default=18080, ge=1024, le=65535, strict=True)

    @model_validator(mode="after")
    def runtime_contract(self):
        if set(self.models) not in ({"generation"}, set(ALIASES)):
            raise ValueError("Models must contain generation only or exactly the three R2 roles")
        paths = [self.binary.path, self.lock_path,
                 *(record.path for record in self.libraries),
                 *(record.path for record in self.models.values())]
        if any(not Path(path).is_absolute() or ".." in Path(path).parts or "\x00" in path
               for path in paths):
            raise ValueError("Runtime paths must be absolute and have no traversal")
        normalized = [str(Path(path)) for path in paths]
        if len(set(normalized)) != len(paths):
            raise ValueError("Runtime paths must be distinct")
        if any(Path(record.path).parent != Path(self.binary.path).parent
               for record in self.libraries):
            raise ValueError("Libraries must be beside the configured binary")
        return self


@dataclass
class _RuntimeSession:
    resources: ExitStack = field(default_factory=ExitStack)
    process: object | None = None
    closed: bool = False
    failed: bool = False


class ManagedLocalProvider(LocalProvider):
    """No download/discovery, no inherited credentials, no GPU or model fallback.

    Explicit session() scopes reuse one CPU child's KV prefix across a single graph invocation,
    never across tasks or human pauses. Startup remains lazy and is charged to the first decision.
    Without a scope every request still starts/reaps its own child. Both modes retain a nonblocking
    thread lease, per-call serialization and an inherited OS lock until the child is reaped.
    Health polling is startup detection, never a retry of an inference request.
    """

    def __init__(self, config_path: str | Path):
        try:
            with Path(config_path).open("rb") as stream:
                content = stream.read(MAX_CONFIG_BYTES + 1)
            if len(content) > MAX_CONFIG_BYTES:
                raise ValueError("Runtime configuration byte limit")
            self.config = RuntimeConfig.model_validate(_json_object(content), strict=True)
        except (OSError, TypeError, ValueError, ValidationError, RecursionError):
            raise DomainError("runtime_config", "Explicit local runtime configuration is unreadable "
                              "or invalid", 503) from None
        super().__init__(f"http://127.0.0.1:{self.config.port}", ALIASES["generation"])
        self.metadata = {
            **self.metadata, "provider": "managed-local-cpu",
            "protocol": PROTOCOL + "-managed-cpu-v2-graph-scope",
            "binary": self.config.binary.sha256,
            "libraries": sorted(record.sha256 for record in self.config.libraries),
            "model_sha256": self.config.models["generation"].sha256,
            "model_bytes": self.config.models["generation"].bytes,
            "context": 8192, "threads": 4, "seed": 42,
        }
        self.fingerprint = "managed-local-cpu-" + digest(self.metadata)
        self._verified: dict[str, tuple] = {}
        self._thread_lock = threading.Lock()
        self._scope_lock = threading.Lock()
        self._session: ContextVar[_RuntimeSession | None] = ContextVar("industrial_runtime", default=None)

    @contextmanager
    def session(self):
        """Exclusive, lazy graph-run scope; ContextVar follows LangGraph's executor context."""
        if not self._scope_lock.acquire(blocking=False):
            raise DomainError("provider_unavailable", "Local generation graph scope is busy", 503)
        state = _RuntimeSession()
        token = self._session.set(state)
        try:
            yield self
        finally:
            state.closed = True
            try:
                state.resources.close()
            finally:
                self._session.reset(token)
                self._scope_lock.release()

    @staticmethod
    def _identity(record: os.stat_result) -> tuple:
        return (record.st_dev, record.st_ino, record.st_size, record.st_mtime_ns, record.st_ctime_ns)

    @staticmethod
    def _no_symlinks(path: Path):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Symlinks are not runtime files or directories")

    def _verify(self, record: RuntimeFile):
        path = Path(record.path)
        try:
            self._no_symlinks(path)
            before = path.stat()
            identity = self._identity(before)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Regular runtime file required")
            if isinstance(record, ModelFile) and before.st_size != record.bytes:
                raise ValueError("Model byte size mismatch")
            if record == self.config.binary and not os.access(path, os.X_OK):
                raise ValueError("Runtime binary is not executable")
            if self._verified.get(record.path) == identity:
                return
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                if identity != self._identity(os.fstat(stream.fileno())):
                    raise ValueError("Runtime identity changed before hashing")
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
            if sha.hexdigest() != record.sha256 or identity != self._identity(path.stat()):
                raise ValueError("Runtime hash mismatch or file changed during hashing")
            self._no_symlinks(path)
            self._verified[record.path] = identity
        except (OSError, ValueError):
            raise DomainError("runtime_integrity", "Local model or inference engine failed file "
                              "integrity verification", 503) from None

    def _command(self) -> list[str]:
        return [
            self.config.binary.path, "--model", self.config.models["generation"].path,
            "--alias", ALIASES["generation"], "--host", "127.0.0.1", "--port", str(self.config.port),
            "--parallel", "1", "--threads", "4", "--threads-batch", "4", "--threads-http", "2",
            "--n-gpu-layers", "0", "--device", "none", "--no-op-offload", "--flash-attn", "off",
            "--no-webui", "--no-slots", "--log-disable", "--ctx-size", "8192",
            "--batch-size", "512", "--ubatch-size", "128", "--jinja", "--temp", "0", "--seed", "42",
        ]

    def _post(self, url: str, payload: dict) -> dict:
        try:
            return self._runtime_post(url, payload)
        except DomainError as exc:
            if isinstance(exc, ProviderFailure):
                raise
            before_dispatch = exc.code in {
                "runtime_request", "runtime_integrity", "runtime_config", "provider_unavailable",
            }
            raise ProviderFailure(exc.code, exc.message, exc.status,
                                  tokens=0 if before_dispatch else None) from None

    def _runtime_post(self, url: str, payload: dict) -> dict:
        if (url != self.generation_url + "/v1/chat/completions"
                or payload.get("model") != ALIASES["generation"]):
            raise DomainError("runtime_request", "Request does not match the configured generation "
                              "model and endpoint", 503)
        scope = self._session.get()
        if scope is not None:
            if scope.closed or scope.failed:
                raise DomainError("provider_unavailable", "Local generation scope is closed or failed",
                                  503)
            if not self._thread_lock.acquire(blocking=False):
                raise DomainError("provider_unavailable", "Local generation runtime is busy", 503)
            try:
                if scope.process is None:
                    self._start_runtime(scope)
                else:
                    if scope.process.poll() is not None:
                        raise DomainError("provider_unavailable", "Scoped model exited before request",
                                          503)
                    self._verify_files()
                return super()._post(url, payload)
            except BaseException:
                scope.failed = True
                scope.resources.close()
                raise
            finally:
                self._thread_lock.release()
        if not self._scope_lock.acquire(blocking=False):
            raise DomainError("provider_unavailable", "Local generation graph scope is busy", 503)
        try:
            if not self._thread_lock.acquire(blocking=False):
                raise DomainError("provider_unavailable", "Local generation runtime is busy", 503)
            try:
                return self._exclusive_post(url, payload)
            finally:
                self._thread_lock.release()
        finally:
            self._scope_lock.release()

    def _open_lock(self):
        path = Path(self.config.lock_path)
        try:
            self._no_symlinks(path)
            if not path.parent.is_dir():
                raise ValueError("Lock parent directory required")
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("Lock must be an unlinked-to regular file")
                return os.fdopen(fd, "a+")
            except BaseException:
                os.close(fd)
                raise
        except (OSError, ValueError):
            raise DomainError("runtime_config", "Local runtime lock is invalid or unavailable", 503) \
                from None

    def _wait_ready(self, process):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        try:
            with httpx.Client(timeout=httpx.Timeout(2, connect=1), trust_env=False,
                              follow_redirects=False) as client:
                while True:
                    if process.poll() is not None:
                        raise DomainError("provider_unavailable", "Local model exited during loading; "
                                          "inference was not sent", 503)
                    if time.monotonic() >= deadline:
                        raise DomainError("provider_unavailable", "Local model loading timed out; "
                                          "inference was not sent", 503)
                    try:
                        with client.stream("GET", self.generation_url + "/health") as response:
                            if response.status_code == 200:
                                content = bytearray()
                                for chunk in response.iter_bytes(chunk_size=1024):
                                    if len(content) + len(chunk) > MAX_HEALTH_BYTES:
                                        raise DomainError("provider_unavailable", "Local model health "
                                                          "response exceeded byte limit", 503)
                                    if time.monotonic() >= deadline:
                                        raise DomainError("provider_unavailable", "Local model loading "
                                                          "timed out; inference was not sent", 503)
                                    content.extend(chunk)
                                if _json_object(content).get("status") == "ok":
                                    if process.poll() is not None:
                                        raise DomainError("provider_unavailable", "Model process "
                                                          "exited before inference", 503)
                                    return
                            elif response.status_code != 503:
                                raise DomainError("provider_unavailable", "Unexpected local model "
                                                  "health response; inference was not sent", 503)
                    except (httpx.HTTPError, ValueError, TypeError, RecursionError):
                        pass
                    time.sleep(0.1)
        except httpx.HTTPError:
            raise DomainError("provider_unavailable", "Local model startup transport failed; "
                              "inference was not sent", 503) from None

    @staticmethod
    def _stop(process):
        try:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
            else:
                process.wait()
        except (OSError, subprocess.TimeoutExpired):
            raise DomainError("provider_unknown", "Local process cleanup could not be confirmed; "
                              "operator reconciliation is required before another request", 503) \
                from None

    def _verify_files(self):
        for record in [self.config.binary, *self.config.libraries,
                       self.config.models["generation"]]:
            self._verify(record)

    def _start_runtime(self, state: _RuntimeSession):
        lock = state.resources.enter_context(self._open_lock())
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise DomainError("provider_unavailable", "Another local runtime holds the OS lock",
                              503) from None
        self._verify_files()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                # Binding, not connecting, also detects occupied non-listening endpoints.
                sock.bind(("127.0.0.1", self.config.port))
        except OSError:
            raise DomainError("provider_unavailable", "Configured port is occupied; refusing "
                              "an unknown server", 503) from None
        temporary = state.resources.enter_context(tempfile.TemporaryDirectory(prefix="industrial-model-"))
        environment = {"HOME": temporary, "TMPDIR": temporary, "XDG_CACHE_HOME": temporary,
                       "PATH": "/usr/bin:/bin", "LC_ALL": "C"}
        try:
            process = subprocess.Popen(
                self._command(), env=environment, cwd=temporary,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True, pass_fds=(lock.fileno(),),
            )
        except OSError:
            raise DomainError("provider_unavailable", "Cannot start the local inference process",
                              503) from None
        state.process = process
        state.resources.callback(self._stop, process)
        self._wait_ready(process)

    def _exclusive_post(self, url: str, payload: dict) -> dict:
        state = _RuntimeSession()
        with state.resources:
            self._start_runtime(state)
            return super()._post(url, payload)
