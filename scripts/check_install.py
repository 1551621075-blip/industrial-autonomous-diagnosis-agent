"""Run with an isolated fresh-venv Python, from a copy outside the frozen source tree.

No installation, downloads, model inference, Java, network sockets, or real data.
Only installed imports, generated launcher help/fixture, resources and in-process
HTTP routes are exercised. A temporary synthetic database is deleted on exit.
This does not assess dependency licenses, vulnerabilities, or diagnosis capability.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import importlib.resources
import json
import os
import selectors
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

PROJECT = "industrial-autonomous-diagnosis-agent"
PACKAGE = "industrial_diagnosis"
RESOURCES = ("templates/index.html", "static/app.js", "static/style.css")
MODULES = ("api", "cli", "fixtures", "knowledge", "managed_provider", "provider",
           "reports", "schemas", "service", "storage", "telemetry")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def within(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def bounded_read(path: Path, limit: int = 2 * 1024 * 1024) -> bytes:
    require(not path.is_symlink() and path.is_file(), "installed file is not regular")
    require(path.stat().st_size <= limit, "installed file exceeds byte bound")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "installed file exceeds byte bound")
    return data


def run_launcher(launcher: Path, arguments: list[str], cwd: Path) -> bytes:
    # Do not inherit model/runtime/provider/credential configuration from the host.
    env = {"PATH": str(launcher.parent), "HOME": str(cwd), "TMPDIR": str(cwd),
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"}
    output = bytearray()
    process = subprocess.Popen([str(launcher), *arguments], cwd=cwd, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 30
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                require(remaining > 0, "installed console command timed out")
                if not selector.select(timeout=min(remaining, 0.2)):
                    continue
                chunk = os.read(process.stdout.fileno(), 16_384)
                if not chunk:
                    break
                output.extend(chunk)
                require(len(output) <= 256_000, "console output exceeds byte bound")
        require(process.wait(timeout=max(0.01, deadline - time.monotonic())) == 0,
                "installed console command failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()
    return bytes(output)


def smoke(forbid_root: Path, expected_version: str) -> dict:
    require(sys.flags.isolated, "use the isolated fresh interpreter with -I")
    require(sys.prefix != sys.base_prefix, "fresh virtual environment required")
    require(not within(Path.cwd(), forbid_root), "working directory must be source-free")
    require(not within(Path(__file__), forbid_root), "copy this checker outside sources first")
    require(not within(Path(sys.prefix), forbid_root), "venv must be outside sources")
    require(not os.environ.get("PYTHONPATH"), "unset PYTHONPATH for install smoke")
    require(not any(within(Path(p), forbid_root) for p in sys.path if p),
            "source tree is on import path")
    prefix = Path(sys.prefix).resolve()
    library = Path(sysconfig.get_path("purelib")).resolve()
    require(within(library, prefix), "site-packages must be inside fresh environment")
    distribution = importlib.metadata.distribution(PROJECT)
    require(distribution.version == expected_version, "installed version mismatch")
    installed_files = distribution.files
    require(installed_files is not None and len(installed_files) <= 4096,
            "installed distribution inventory absent or oversized")
    owned = {}
    for item in installed_files:
        name = item.as_posix()
        if not name.startswith(PACKAGE + "/") or "/__pycache__/" in name:
            continue
        path = Path(distribution.locate_file(item))
        require(within(path, library) and not within(path, forbid_root),
                "installed package file escapes fresh site-packages")
        data = bounded_read(path)
        require(item.hash is not None and item.hash.mode == "sha256", "installed hash missing")
        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        require(actual == item.hash.value and len(data) == item.size,
                "installed package RECORD mismatch")
        owned[name] = data
    require(owned and sum(map(len, owned.values())) <= 32 * 1024 * 1024,
            "installed package absent or exceeds total byte bound")
    direct = distribution.read_text("direct_url.json")
    if direct:
        require(not json.loads(direct).get("dir_info", {}).get("editable"),
                "editable installs cannot satisfy source-free acceptance")
    for suffix in ("", *MODULES):
        module = importlib.import_module(PACKAGE + ("." + suffix if suffix else ""))
        location = Path(module.__file__).resolve()
        expected = library / PACKAGE / (suffix + ".py" if suffix else "__init__.py")
        require(location == expected and not within(location, forbid_root),
                "module import did not resolve to installed distribution")
    resources = importlib.resources.files(PACKAGE)
    for name in RESOURCES:
        installed = owned.get(f"{PACKAGE}/{name}")
        require(installed and resources.joinpath(*name.split("/")).read_bytes() == installed,
                "installed resource missing or inconsistent")
    entries = [entry for entry in distribution.entry_points if entry.group == "console_scripts"]
    require(len(entries) == 1 and entries[0].name == "industrial-diagnosis"
            and entries[0].value == f"{PACKAGE}.cli:main" and callable(entries[0].load()),
            "installed console entrypoint mismatch")
    launcher = Path(sysconfig.get_path("scripts")) / "industrial-diagnosis"
    require(within(launcher, prefix) and launcher.is_file() and not launcher.is_symlink(),
            "installed console launcher missing")
    from fastapi.testclient import TestClient

    from industrial_diagnosis.api import create_app
    from industrial_diagnosis.provider import SyntheticProvider

    with tempfile.TemporaryDirectory(prefix="diagnosis-install-") as temporary:
        work = Path(temporary)
        help_text = run_launcher(launcher, ["--help"], work).decode("utf-8")
        require("fixture" in help_text and "serve" in help_text, "console help incomplete")
        fixture = json.loads(run_launcher(launcher, ["fixture", "--reports-only"], work))
        require(isinstance(fixture, list) and len(fixture) == 3, "synthetic fixture export failed")
        app = create_app(work / "synthetic-data", SyntheticProvider())
        with TestClient(app) as client:
            page = client.get("/")
            require(page.status_code == 200 and page.content == owned[f"{PACKAGE}/{RESOURCES[0]}"],
                    "installed index response differs from packaged resource")
            for asset, mime in (("app.js", "javascript"), ("style.css", "text/css")):
                response = client.get(f"/static/{asset}")
                require(response.status_code == 200
                        and response.content == owned[f"{PACKAGE}/static/{asset}"]
                        and mime in response.headers["content-type"], "installed static route failed")
            require(client.get("/static/missing.js").status_code == 404,
                    "static route must reject unshipped files")
            require(client.get("/health").status_code == 200, "installed health route failed")
    return {"status": "passed", "version": distribution.version, "imported_modules": 12,
            "resources": len(RESOURCES), "console_commands": 2, "source_free": True,
            "in_process_http": True, "inference": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid-root", type=Path, required=True)
    parser.add_argument("--expected-version", default="0.1.0")
    args = parser.parse_args(argv)
    try:
        result = smoke(args.forbid_root.resolve(), args.expected_version)
    except Exception as exc:
        # Neither payloads nor machine-specific exception paths belong in smoke output.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
