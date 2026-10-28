"""Synthetic archive tests only: no package build, install, downloads or inference.

Sensitive fixtures are assembled at runtime so publication selfscan needs no bypass.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release = load_script("check_release")
install = load_script("check_install")


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "source"
    files = {
        "pyproject.toml": f'''[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"
[project]
name = "{release.PROJECT}"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["fastapi>=0.115"]
[project.scripts]
industrial-diagnosis = "{release.PACKAGE}.cli:main"
[tool.hatch.build.targets.wheel]
packages = ["src/{release.PACKAGE}"]
''',
        "README.md": "Original synthetic package.\n",
        ".gitignore": "runtime/\nmodels/\n.env\n",
        "uv.lock": "version = 1\n",
        "src/industrial_diagnosis/__init__.py": "",
        "src/industrial_diagnosis/cli.py": "def main():\n    return None\n",
        "src/industrial_diagnosis/api.py": "# Routes are tested separately.\n",
        "src/industrial_diagnosis/templates/index.html": (
            '<link rel="stylesheet" href="/static/style.css">'
            '<script src="/static/app.js"></script>'
        ),
        "src/industrial_diagnosis/static/app.js": "'use strict';\n",
        "src/industrial_diagnosis/static/style.css": "body { color: black; }\n",
        "tests/test_example.py": "def test_example():\n    assert True\n",
        "docs/usage.md": "Synthetic only.\n",
        "scripts/example.py": "print('example')\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def metadata(snapshot):
    project = snapshot.project
    lines = ["Metadata-Version: 2.4", f"Name: {project['name']}",
             f"Version: {project['version']}", f"Requires-Python: {project['requires-python']}"]
    lines.extend(f"Requires-Dist: {item}" for item in project["dependencies"])
    return ("\n".join(lines) + "\n\n").encode()


def artifacts(snapshot):
    stem = release.PROJECT.replace("-", "_") + "-" + snapshot.project["version"]
    info = stem + ".dist-info"
    wheel = {name.removeprefix("src/"): data for name, data in snapshot.files.items()
             if name.startswith("src/")}
    wheel.update({
        f"{info}/METADATA": metadata(snapshot),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{info}/entry_points.txt": b"[console_scripts]\nindustrial-diagnosis = industrial_diagnosis.cli:main\n",
    })
    add_record(wheel, info)
    sdist = {f"{stem}/{name}": data for name, data in snapshot.files.items()}
    sdist[f"{stem}/PKG-INFO"] = metadata(snapshot)
    return stem, info, wheel, sdist


def add_record(wheel, info):
    record_name = f"{info}/RECORD"
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in wheel.items():
        if name != record_name:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            writer.writerow([name, "sha256=" + digest.decode(), str(len(data))])
    writer.writerow([record_name, "", ""])
    wheel[record_name] = output.getvalue().encode()


def write_wheel(path, files):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def write_sdist(path, files, extra=None):
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if extra is not None:
            archive.addfile(extra)


def write_pair(tmp_path, stem, wheel, sdist):
    whl = tmp_path / f"{stem}-py3-none-any.whl"
    tar = tmp_path / f"{stem}.tar.gz"
    write_wheel(whl, wheel)
    write_sdist(tar, sdist)
    return whl, tar


def test_source_snapshot_and_explicit_candidates(project):
    snapshot = release.check_source(project)
    assert len(snapshot.files) == 13
    assert "src/industrial_diagnosis/static/app.js" in snapshot.files
    assert not any(name.startswith("runtime/") for name in snapshot.files)


def test_ignored_runtime_roots_are_not_enumerated_or_read(project, monkeypatch):
    excluded = {"runtime", "models", "tokens", "config", "db", ".git", ".venv", "dist"}
    for name in excluded:
        directory = project / name
        directory.mkdir()
        (directory / "private.json").write_bytes(b"\x00")
    (project / ".env").write_bytes(b"\x00")
    scan = os.scandir
    read = release.read_regular

    def guarded_scan(path):
        assert not (Path(path).is_relative_to(project)
                    and Path(path).relative_to(project).parts[0] in excluded)
        return scan(path)

    def guarded_read(path, cap):
        assert path.relative_to(project).parts[0] not in excluded | {".env"}
        return read(path, cap)

    monkeypatch.setattr(release.os, "scandir", guarded_scan)
    monkeypatch.setattr(release, "read_regular", guarded_read)
    assert len(release.check_source(project).files) == 13


@pytest.mark.parametrize("name", [
    "runtime/state.json", "models/weights.gguf", "tokens/access.txt", "config/local.json",
    ".env.example", "result.sqlite3-wal", "local.db", "runtime-config.json", "private.pem",
])
def test_sensitive_candidate_rejected_before_read(project, monkeypatch, name):
    path = project / "docs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"must not be opened")
    original = release.read_regular

    def guard(candidate, cap):
        assert candidate != path
        return original(candidate, cap)

    monkeypatch.setattr(release, "read_regular", guard)
    with pytest.raises(release.CheckError, match="sensitive publication path"):
        release.check_source(project)


def test_cache_directory_not_entered(project):
    cache = project / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"\x00")
    assert len(release.check_source(project).files) == 13


@pytest.mark.parametrize("target", ["docs/link.md", "docs/link", "README.md"])
def test_source_symlinks_rejected(project, tmp_path, target):
    path = project / target
    if path.exists():
        path.unlink()
    external = tmp_path / "external"
    external.write_text("not publication")
    path.symlink_to(external)
    with pytest.raises(release.CheckError, match="symlink"):
        release.check_source(project)


def test_unknown_and_binary_source_fail_closed(project):
    unknown = project / "docs" / "extra.csv"
    unknown.write_text("synthetic")
    with pytest.raises(release.CheckError, match="unapproved"):
        release.check_source(project)
    unknown.unlink()
    (project / "README.md").write_bytes(b"\x00")
    with pytest.raises(release.CheckError, match="binary"):
        release.check_source(project)


def test_source_count_depth_and_byte_limits(project):
    defaults = release.DEFAULT_LIMITS
    for limits in (replace(defaults, max_entries=1), replace(defaults, max_file_bytes=10),
                   replace(defaults, max_total_bytes=10), replace(defaults, max_depth=2)):
        with pytest.raises(release.CheckError, match="limit"):
            release.check_source(project, limits=limits)


def test_source_case_collision(project):
    # On case-insensitive hosts this aliases the existing file; skip that case only.
    original = project / "docs/usage.md"
    alias = project / "docs/USAGE.md"
    if alias.exists():
        pytest.skip("filesystem is case-insensitive")
    alias.write_text(original.read_text())
    with pytest.raises(release.CheckError, match="case-colliding"):
        release.check_source(project)


def secret_samples():
    return [
        "-----BEGIN " + "RSA PRIVATE KEY-----",
        "AK" + "IA" + "A" * 16,
        "gh" + "p_" + "a" * 36,
        "github_" + "pat_" + "a" * 36,
        "s" + "k-" + "a" * 32,
        "xo" + "xb-" + "a" * 32,
        "ey" + "J" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12,
        "Bearer " + "a" * 32,
        "https://" + "alice:" + "not-a-real-credential" + "@example.invalid",
        "api_key = " + repr("a" * 32),
    ]


@pytest.mark.parametrize("sample", secret_samples(), ids=lambda value: str(len(value)))
def test_recognizable_secrets_are_redacted(sample):
    with pytest.raises(release.CheckError, match="recognizable secret") as caught:
        release.scan_text("docs/example.md", sample.encode())
    assert sample not in str(caught.value)


@pytest.mark.parametrize("sample", [
    "/" + "Users/" + "example/work", "/" + "home/" + "example/work",
    "/" + "root/" + "work", "/private/" + "var/folders/" + "abc/work",
    "C:" + "\\Users\\" + "Example\\work", "D:" + "/Users/" + "Example/work",
    "%2F" + "Users%2Fexample%2Fwork",
])
def test_personal_absolute_paths_rejected(sample):
    with pytest.raises(release.CheckError, match="personal absolute"):
        release.scan_text("docs/example.md", sample.encode())


def test_source_literals_and_placeholder_paths_are_allowed():
    release.scan_text("docs/example.md", b"/absolute/path/runtime.json ${HOME} <LOCAL_TOKEN>")
    for name in ("scripts/check_release.py", "scripts/check_install.py", "tests/test_release.py"):
        release.scan_text(name, (ROOT / name).read_bytes())


def test_valid_artifact_parity_record_and_resources(project, tmp_path):
    snapshot = release.check_source(project)
    stem, _info, wheel, sdist = artifacts(snapshot)
    paths = write_pair(tmp_path, stem, wheel, sdist)
    report = release.check_artifacts(snapshot, *paths)
    assert report["record_verified"] and report["parity_verified"]
    assert report["resource_files"] == 3
    assert report["wheel_files"] == 10


@pytest.mark.parametrize("change", ["missing", "extra", "changed", "metadata", "entrypoint", "tag"])
def test_wheel_contract_mutations(project, tmp_path, change):
    snapshot = release.check_source(project)
    stem, info, wheel, sdist = artifacts(snapshot)
    if change == "missing":
        del wheel["industrial_diagnosis/static/app.js"]
    elif change == "extra":
        wheel["industrial_diagnosis/static/extra.js"] = b"extra"
    elif change == "changed":
        wheel["industrial_diagnosis/static/app.js"] = b"changed"
    elif change == "metadata":
        wheel[f"{info}/METADATA"] = wheel[f"{info}/METADATA"].replace(b"0.1.0", b"9.0.0")
    elif change == "entrypoint":
        wheel[f"{info}/entry_points.txt"] = b"[console_scripts]\nother = elsewhere:run\n"
    else:
        wheel[f"{info}/WHEEL"] = wheel[f"{info}/WHEEL"].replace(b"py3-none-any", b"py2-none-any")
    add_record(wheel, info)  # Valid hashes do not excuse parity or metadata changes.
    paths = write_pair(tmp_path, stem, wheel, sdist)
    with pytest.raises(release.CheckError):
        release.check_artifacts(snapshot, *paths)


@pytest.mark.parametrize("change", ["hash", "size", "missing", "duplicate", "unknown", "self", "algorithm"])
def test_record_rejects_incomplete_or_invalid_integrity(project, change):
    snapshot = release.check_source(project)
    _stem, info, wheel, _sdist = artifacts(snapshot)
    name = f"{info}/RECORD"
    rows = list(csv.reader(io.StringIO(wheel[name].decode())))
    if change == "hash":
        rows[0][1] = "sha256=" + "0" * 43
    elif change == "size":
        rows[0][2] = "999"
    elif change == "missing":
        rows.pop(0)
    elif change == "duplicate":
        rows[1] = rows[0]
    elif change == "unknown":
        rows[0][0] = "unknown.py"
    elif change == "self":
        rows[-1][2] = "0"
    else:
        rows[0][1] = rows[0][1].replace("sha256", "md5")
    text = io.StringIO(newline="")
    csv.writer(text).writerows(rows)
    wheel[name] = text.getvalue().encode()
    with pytest.raises(release.CheckError, match="RECORD"):
        release.check_record(wheel, name)


@pytest.mark.parametrize("change", ["missing", "extra", "changed", "root", "metadata"])
def test_sdist_contract_mutations(project, tmp_path, change):
    snapshot = release.check_source(project)
    stem, _info, wheel, sdist = artifacts(snapshot)
    name = f"{stem}/README.md"
    if change == "missing":
        del sdist[name]
    elif change == "extra":
        sdist[f"{stem}/extra.txt"] = b"extra"
    elif change == "changed":
        sdist[name] = b"modified documentation"
    elif change == "root":
        sdist["other/README.md"] = sdist.pop(name)
    else:
        sdist[f"{stem}/PKG-INFO"] = b"Name: wrong\n\n"
    paths = write_pair(tmp_path, stem, wheel, sdist)
    with pytest.raises(release.CheckError):
        release.check_artifacts(snapshot, *paths)


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py", "a\\b.py", "a/./b.py", "a//b.py"])
def test_archive_paths_are_canonical(tmp_path, name):
    path = tmp_path / "example.whl"
    write_wheel(path, {name: b"example"})
    with pytest.raises(release.CheckError):
        release.read_wheel(path, release.DEFAULT_LIMITS)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_sensitive_artifact_rejected_before_payload_read(tmp_path, monkeypatch, kind):
    def forbidden(*args, **kwargs):
        pytest.fail("sensitive archive payload opened")

    if kind == "wheel":
        path = tmp_path / "example.whl"
        write_wheel(path, {"industrial_diagnosis/models/weights.gguf": b"private"})
        monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
        reader = release.read_wheel
    else:
        path = tmp_path / "example.tar.gz"
        write_sdist(path, {"package/runtime/state.db": b"private"})
        monkeypatch.setattr(tarfile.TarFile, "extractfile", forbidden)
        reader = release.read_sdist
    with pytest.raises(release.CheckError, match="sensitive artifact path"):
        reader(path, release.DEFAULT_LIMITS)


def test_wheel_links_duplicates_and_count_lies(tmp_path):
    path = tmp_path / "example.whl"
    member = zipfile.ZipInfo("industrial_diagnosis/link.py")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "outside")
    with pytest.raises(release.CheckError, match="non-regular"):
        release.read_wheel(path, release.DEFAULT_LIMITS)
    with pytest.warns(UserWarning), zipfile.ZipFile(path, "w") as archive:
        archive.writestr("duplicate.py", "a")
        archive.writestr("duplicate.py", "b")
    with pytest.raises(release.CheckError, match="duplicate"):
        release.read_wheel(path, release.DEFAULT_LIMITS)
    write_wheel(path, {"one.py": b"one", "two.py": b"two"})
    raw = bytearray(path.read_bytes())
    end = raw.rfind(b"PK\x05\x06")
    struct.pack_into("<2H", raw, end + 8, 1, 1)
    path.write_bytes(raw)
    with pytest.raises(release.CheckError, match="count mismatch"):
        release.read_wheel(path, release.DEFAULT_LIMITS)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_sdist_links_and_special_members_rejected(tmp_path, kind):
    path = tmp_path / "example.tar.gz"
    member = tarfile.TarInfo("package/link")
    member.type = kind
    member.linkname = "outside"
    write_sdist(path, {"package/README.md": b"example"}, member)
    with pytest.raises(release.CheckError, match="special"):
        release.read_sdist(path, release.DEFAULT_LIMITS)


def test_sdist_duplicate_pax_and_trailing_payloads(tmp_path):
    path = tmp_path / "example.tar.gz"
    member = tarfile.TarInfo("package/README.md")
    write_sdist(path, {"package/README.md": b"example"}, member)
    with pytest.raises(release.CheckError, match="duplicate"):
        release.read_sdist(path, release.DEFAULT_LIMITS)
    member = tarfile.TarInfo("package/other.md")
    member.pax_headers = {"comment": "unexpected extended metadata"}
    write_sdist(path, {"package/README.md": b"example"}, member)
    with pytest.raises(release.CheckError, match="extended metadata"):
        release.read_sdist(path, release.DEFAULT_LIMITS)
    write_sdist(path, {"package/README.md": b"example"})
    raw = gzip.decompress(path.read_bytes()) + b"hidden" + b"\0" * 506
    path.write_bytes(gzip.compress(raw))
    with pytest.raises(release.CheckError, match="trailing tar"):
        release.read_sdist(path, release.DEFAULT_LIMITS)


def test_archive_allocation_and_decompression_limits(tmp_path):
    limits = release.DEFAULT_LIMITS
    path = tmp_path / "example.whl"
    write_wheel(path, {"one.py": b"x" * 4096, "two.py": b"two"})
    for bounded in (replace(limits, max_entries=1), replace(limits, max_file_bytes=20),
                    replace(limits, max_total_bytes=20), replace(limits, max_archive_bytes=20)):
        with pytest.raises(release.CheckError, match="limit"):
            release.read_wheel(path, bounded)
    path = tmp_path / "example.tar.gz"
    write_sdist(path, {"package/one.py": b"x" * 4096})
    with pytest.raises(release.CheckError, match="expanded tar"):
        release.read_sdist(path, replace(limits, max_tar_bytes=512))


def test_non_utf8_archive_and_tampered_zip_crc(tmp_path):
    path = tmp_path / "example.whl"
    write_wheel(path, {"example.py": b"\xff"})
    with pytest.raises(release.CheckError, match="UTF-8"):
        release.read_wheel(path, release.DEFAULT_LIMITS)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("example.py", b"unchanged")
    path.write_bytes(path.read_bytes().replace(b"unchanged", b"corrupted"))
    with pytest.raises(zipfile.BadZipFile):
        release.read_wheel(path, release.DEFAULT_LIMITS)


def test_assets_and_entrypoint_source_contracts(project):
    path = project / "src/industrial_diagnosis/templates/index.html"
    path.write_text('<script src="https://example.invalid/app.js"></script>')
    with pytest.raises(release.CheckError, match="shipped local"):
        release.check_source(project)


def test_release_cli_modes_and_sanitized_failures(project, tmp_path, capsys):
    assert release.main(["--root", str(project), "--source-only"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "source-only"
    snapshot = release.check_source(project)
    stem, _info, wheel, sdist = artifacts(snapshot)
    whl, tar = write_pair(tmp_path, stem, wheel, sdist)
    assert release.main(["--root", str(project), "--wheel", str(whl), "--sdist", str(tar)]) == 0
    assert json.loads(capsys.readouterr().out)["record_verified"]
    assert release.main(["--root", str(tmp_path / "absent"), "--source-only"]) == 1
    assert str(tmp_path) not in capsys.readouterr().err
    for args in ([], ["--source-only", "--wheel", str(whl)], ["--wheel", str(whl)]):
        with pytest.raises(SystemExit) as caught:
            release.main(args)
        assert caught.value.code == 2


def test_release_script_is_directly_runnable(project):
    completed = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/check_release.py"),
         "--root", str(project), "--source-only"],
        check=False, capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == "passed"


def test_install_smoke_guard_and_launcher_helper(tmp_path):
    assert install.within(tmp_path / "child", tmp_path)
    assert not install.within(tmp_path.parent, tmp_path)
    with pytest.raises(RuntimeError, match="isolated"):
        install.smoke(ROOT, "0.1.0")
    assert install.run_launcher(Path(sys.executable), ["-I", "-c", "print('ok')"], tmp_path) == b"ok\n"
    with pytest.raises(RuntimeError, match="console command failed"):
        install.run_launcher(Path(sys.executable), ["-I", "-c", "raise SystemExit(1)"], tmp_path)
    with pytest.raises(RuntimeError, match="output exceeds"):
        install.run_launcher(Path(sys.executable), ["-I", "-c", "print('x' * 256001)"], tmp_path)


def test_metadata_accepts_backend_ordering_but_not_duplicate_requirements(project):
    snapshot = release.check_source(project)
    raw = metadata(snapshot).replace(b">=3.12,<3.13", b"<3.13,>=3.12")
    release.check_metadata(raw, snapshot.project)
    raw = raw.replace(b"\n\n", b"\nRequires-Dist: fastapi>=0.115\n\n")
    with pytest.raises(release.CheckError, match="dependency"):
        release.check_metadata(raw, snapshot.project)


def test_wheel_comment_and_local_header_mutation_fail(tmp_path):
    path = tmp_path / "example.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("example.py", b"example")
        archive.comment = b"untracked metadata"
    with pytest.raises(release.CheckError, match="comments"):
        release.read_wheel(path, release.DEFAULT_LIMITS)
    write_wheel(path, {"example.py": b"example"})
    raw = bytearray(path.read_bytes())
    struct.pack_into("<L", raw, 14, 0)
    path.write_bytes(raw)
    with pytest.raises(release.CheckError, match="local/central"):
        release.read_wheel(path, release.DEFAULT_LIMITS)


def test_tar_physical_header_and_padding_limits(tmp_path):
    path = tmp_path / "example.tar.gz"
    member = tarfile.TarInfo("package/README.md")
    member.pax_headers = {"mtime": "1.5"}
    write_sdist(path, {}, member)
    with pytest.raises(release.CheckError, match="physical-header"):
        release.read_sdist(path, replace(release.DEFAULT_LIMITS, max_entries=1))
    write_sdist(path, {"package/README.md": b"example"})
    raw = bytearray(gzip.decompress(path.read_bytes()))
    raw[512 + len(b"example")] = 65
    path.write_bytes(gzip.compress(raw))
    with pytest.raises(release.CheckError, match="padding"):
        release.read_sdist(path, release.DEFAULT_LIMITS)


def test_sdist_case_collision_and_zip_case_collision(tmp_path):
    files = {"package/one.py": b"one", "package/ONE.py": b"two"}
    wheel = tmp_path / "example.whl"
    sdist = tmp_path / "example.tar.gz"
    write_wheel(wheel, files)
    write_sdist(sdist, files)
    for reader, path in ((release.read_wheel, wheel), (release.read_sdist, sdist)):
        with pytest.raises(release.CheckError, match="case-colliding"):
            reader(path, release.DEFAULT_LIMITS)


def test_recognizable_secret_in_artifact_payload(tmp_path):
    path = tmp_path / "example.whl"
    write_wheel(path, {"package/example.py": secret_samples()[1].encode()})
    with pytest.raises(release.CheckError, match="recognizable secret"):
        release.read_wheel(path, release.DEFAULT_LIMITS)


def test_non_regular_source_rejected_without_blocking(project):
    path = project / "docs" / "pipe.md"
    os.mkfifo(path)
    with pytest.raises(release.CheckError, match="regular"):
        release.check_source(project)


def test_install_script_cli_help_does_not_import_project():
    completed = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/check_install.py"), "--help"],
        check=False, capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0 and "--forbid-root" in completed.stdout
