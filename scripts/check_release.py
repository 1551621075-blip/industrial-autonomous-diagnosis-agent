"""Bounded, offline publication checks; no build, imports of project code, or Git access.

Only the named root files and src/tests/docs/scripts are source candidates. Ignored
root runtime data is never enumerated. Sensitive paths within publication trees or
archives fail before their payload is read. Frozen sources are required for artifact
parity. Recognizable-secret checks are lexical heuristics, not a semantic or complete
secret audit: constructed/obfuscated values need separate review. The CLI stops at
the first failure. Metadata checks are not a dependency vulnerability or license/legal
audit. Archive support deliberately targets ordinary pure-Python Hatchling output.
"""
from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
import gzip
import hashlib
import io
import json
import os
import re
import stat
import struct
import sys
import tarfile
import tomllib
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

PACKAGE = "industrial_diagnosis"
PROJECT = "industrial-autonomous-diagnosis-agent"
RESOURCES = ("templates/index.html", "static/app.js", "static/style.css")
ROOT_REQUIRED = frozenset({"pyproject.toml", "README.md", "uv.lock", ".gitignore"})
ROOT_OPTIONAL = frozenset({"LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.txt"})
TREES = ("src", "tests", "docs", "scripts")
CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})
SENSITIVE_PARTS = frozenset({
    ".git", ".venv", "venv", ".aws", ".ssh", ".config", ".cache", ".industrial-data", "runtime",
    "model", "models", "model_weights", "tokens", "secrets", "credentials", "config",
    "configs", "db", "database", "databases", "data", "demo-data", "logs", "outputs",
    "artifacts", "node_modules", "dist", "build",
})
SENSITIVE_SUFFIXES = frozenset({
    ".gguf", ".safetensors", ".bin", ".pkl", ".pickle", ".onnx", ".pt", ".pth",
    ".h5", ".hdf5", ".npy", ".npz", ".parquet", ".arrow", ".jar", ".dylib", ".so",
    ".dll", ".exe", ".log", ".pem", ".key", ".p12", ".pfx", ".pyc", ".pyo",
    ".zip", ".whl", ".gz", ".tar", ".owl", ".rdf",
})
SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    ("cloud-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("api-token", re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer-credential", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}={0,2}")),
    ("credential-url", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:'\"]+:[^\s/@'\"]+@")),
    ("literal-credential", re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|password|"
        r"client[_-]?secret|secret[_-]?key)\b[\"']?\s*[:=]\s*[\"']"
        r"[A-Za-z0-9_./+=-]{20,}[\"']"
    )),
)
PERSONAL_PATH = re.compile(
    r"/(?:Users|home)/[^\s/\"'<>]+|/r[o]ot/|/private/var/[f]olders/|"
    r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|"
    r"\\\\[^\s\\]+\\(?:Users|homes)\\", re.IGNORECASE
)


class CheckError(ValueError):
    """A fail-closed publication check, with no payload echoed in diagnostics."""


@dataclass(frozen=True)
class Limits:
    max_entries: int = 4096
    max_file_bytes: int = 2 * 1024 * 1024
    max_total_bytes: int = 32 * 1024 * 1024
    max_archive_bytes: int = 32 * 1024 * 1024
    max_tar_bytes: int = 64 * 1024 * 1024
    max_depth: int = 16


DEFAULT_LIMITS = Limits()


@dataclass
class Budget:
    limits: Limits
    entries: int = 0
    total_bytes: int = 0

    def entry(self):
        self.entries += 1
        require(self.entries <= self.limits.max_entries, "entry-count limit exceeded")

    def payload(self, size: int):
        require(0 <= size <= self.limits.max_file_bytes, "per-file byte limit exceeded")
        self.total_bytes += size
        require(self.total_bytes <= self.limits.max_total_bytes, "total byte limit exceeded")


@dataclass
class Snapshot:
    files: dict[str, bytes]
    project: dict = field(repr=False)


def require(condition, message: str):
    if not condition:
        raise CheckError(message)


def safe_name(name: str, limits: Limits) -> str:
    require(bool(name) and len(name) <= 512, "invalid publication path length")
    require(not any(ord(c) < 32 or ord(c) == 127 for c in name), "control in publication path")
    require("\\" not in name and ":" not in name, "non-portable publication path")
    parts = name.split("/")
    require(all(part not in {"", ".", ".."} for part in parts), "unsafe publication path")
    require(len(parts) <= limits.max_depth, "publication path depth limit exceeded")
    require(all(not part.endswith((" ", ".")) for part in parts), "ambiguous publication path")
    return name


def sensitive_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    for part in parts:
        low = part.lower()
        if (low in SENSITIVE_PARTS or low in CACHE_DIRS or low.startswith(".env")
                or low.endswith(".egg-info") or low in {".ds_store", "id_rsa", "id_ed25519"}):
            return True
        if re.search(r"\.(?:db|sqlite\d*)(?:-(?:wal|shm|journal))?$", low):
            return True
        if Path(low).suffix in SENSITIVE_SUFFIXES:
            return True
        if re.search(r"(?:^|[-_.])(?:token[s]?|secret[s]?|credentials?)(?:[-_.]|$)", low):
            return True
        if ("runtime-config" in low or low in {"runtime.json", "runtime.yaml", "runtime.yml"}
                or re.fullmatch(r"config\.(?:json|ya?ml|toml|ini)", low)):
            return True
    return False


def source_candidate(name: str) -> bool:
    if name in ROOT_REQUIRED | ROOT_OPTIONAL:
        return True
    parts = PurePosixPath(name).parts
    if len(parts) < 2:
        return False
    suffix = PurePosixPath(name).suffix
    if parts[0] == "src":
        prefix = f"src/{PACKAGE}/"
        return name.startswith(prefix) and (suffix == ".py" or name[len(prefix):] in RESOURCES)
    allowed = {"tests": {".py"}, "scripts": {".py", ".mjs", ".js"},
               "docs": {".md", ".rst", ".txt"}}
    return suffix in allowed.get(parts[0], set()) and not any(p.startswith(".") for p in parts)


def scan_text(name: str, data: bytes):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise CheckError(f"non-UTF-8 publication file: {name}") from None
    require("\x00" not in text, f"binary publication file: {name}")
    # Decode URL and JSON path escaping as well as literal personal paths.
    path_text = unquote(text).replace("\\\\", "\\")
    require(not PERSONAL_PATH.search(path_text), f"personal absolute path: {name}")
    for label, pattern in SECRET_PATTERNS:
        require(not pattern.search(text), f"recognizable secret ({label}): {name}")


def read_regular(path: Path, cap: int) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "input must be a regular, non-symlink file")
    require(before.st_size <= cap, "input byte limit exceeded")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode), "input is not a regular file")
        require((before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
                "input changed during opening")
        data = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
    require(len(data) <= cap, "input byte limit exceeded")
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            and len(data) == before.st_size, "input changed while reading; freeze sources")
    return data


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.assets: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and "src" in attrs:
            self.assets.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.assets.append(attrs.get("href", ""))


def check_package(files: dict[str, bytes]):
    for resource in RESOURCES:
        require(bool(files.get(f"{PACKAGE}/{resource}", b"").strip()),
                f"missing or empty project resource: {resource}")
    page = Page(files[f"{PACKAGE}/templates/index.html"].decode("utf-8"))
    require(sorted(page.assets) == ["/static/app.js", "/static/style.css"],
            "page must reference exactly the two shipped local static assets")
    for module in ("__init__", "cli", "api"):
        require(f"{PACKAGE}/{module}.py" in files, f"missing package module: {module}")
    try:
        tree = ast.parse(files[f"{PACKAGE}/cli.py"])
    except SyntaxError:
        raise CheckError("invalid CLI Python source") from None
    require(any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body),
            "console entrypoint main is not declared in CLI source")


def check_source(root: Path, *, limits: Limits = DEFAULT_LIMITS) -> Snapshot:
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), "source root must be a real directory")
    root = root.resolve()
    files: dict[str, bytes] = {}
    folded: set[str] = set()
    budget = Budget(limits)

    def add(path: Path, name: str):
        safe_name(name, limits)
        require(not sensitive_path(name), f"sensitive publication path: {name}")
        require(source_candidate(name), f"unapproved source candidate: {name}")
        require(name.casefold() not in folded, "case-colliding source paths")
        folded.add(name.casefold())
        size = path.lstat().st_size
        budget.payload(size)
        data = read_regular(path, limits.max_file_bytes)
        scan_text(name, data)
        files[name] = data

    def walk(directory: Path, prefix: str):
        require(not directory.is_symlink(), "symlink publication directory")
        require(directory.is_dir(), "publication tree is not a directory")
        with os.scandir(directory) as entries:
            for entry in entries:
                budget.entry()
                name = safe_name(f"{prefix}/{entry.name}", limits)
                require(not entry.is_symlink(), f"symlink publication path: {name}")
                # Only inert interpreter/tool caches are skipped inside candidate trees.
                # Runtime/config/model/token paths are rejected without opening them.
                if entry.name in CACHE_DIRS and entry.is_dir(follow_symlinks=False):
                    continue
                require(not sensitive_path(name), f"sensitive publication path: {name}")
                if entry.is_dir(follow_symlinks=False):
                    walk(Path(entry.path), name)
                else:
                    add(Path(entry.path), name)

    for name in sorted(ROOT_REQUIRED | ROOT_OPTIONAL):
        path = root / name
        if path.exists() or path.is_symlink():
            budget.entry()
            add(path, name)
        else:
            require(name not in ROOT_REQUIRED, f"missing publication input: {name}")
    for name in TREES:
        path = root / name
        if path.exists() or path.is_symlink():
            budget.entry()
            walk(path, name)
    try:
        metadata = tomllib.loads(files["pyproject.toml"].decode("utf-8"))
        project = metadata["project"]
        require(project["name"] == PROJECT, "unexpected project name")
        require(re.fullmatch(r"\d+\.\d+\.\d+", project["version"]),
                "release checker requires a static three-component version")
        require(not project.get("dynamic"), "dynamic project metadata is unsupported")
        require(project["scripts"] == {"industrial-diagnosis": f"{PACKAGE}.cli:main"},
                "unexpected declared console entrypoint")
        require(metadata["build-system"]["build-backend"] == "hatchling.build",
                "unsupported build backend")
        require(metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
                == [f"src/{PACKAGE}"], "unexpected wheel source roots")
    except (KeyError, TypeError, tomllib.TOMLDecodeError):
        raise CheckError("invalid or unsupported project metadata") from None
    check_package({name.removeprefix("src/"): data for name, data in files.items()
                   if name.startswith("src/")})
    return Snapshot(files, project)


def check_archive_names(names: list[str], limits: Limits):
    folded: set[str] = set()
    for name in names:
        safe_name(name, limits)
        require(name.casefold() not in folded, "duplicate or case-colliding archive member")
        folded.add(name.casefold())
        require(not sensitive_path(name), f"sensitive artifact path: {name}")


def read_wheel(path: Path, limits: Limits) -> dict[str, bytes]:
    raw = read_regular(path, limits.max_archive_bytes)
    # Bound central-directory object allocation before handing data to ZipFile.
    marker = raw.rfind(b"PK\x05\x06", max(0, len(raw) - 65557))
    require(marker >= 0 and marker + 22 <= len(raw), "missing ZIP end record")
    _, disk, start_disk, count_disk, count, cd_size, cd_offset, comment = struct.unpack_from(
        "<4s4H2LH", raw, marker
    )
    require(marker + 22 + comment == len(raw), "trailing or malformed ZIP data")
    require(comment == 0, "wheel archive comments are unsupported")
    require(disk == start_disk == 0 and count == count_disk, "multi-disk ZIP unsupported")
    require(0 < count <= limits.max_entries and count != 65535,
            "ZIP member count exceeds limit or uses ZIP64")
    require(0 < cd_size <= count * 2048, "ZIP metadata byte limit exceeded")
    require(cd_offset + cd_size == marker, "ZIP64, prepended, or malformed central directory")
    # Count real fixed central headers too: do not trust only EOCD's claimed count.
    offset = cd_offset
    actual = 0
    while offset < marker:
        require(offset + 46 <= marker and raw[offset:offset + 4] == b"PK\x01\x02",
                "invalid ZIP central directory")
        lengths = struct.unpack_from("<3H", raw, offset + 28)
        offset += 46 + sum(lengths)
        actual += 1
        require(actual <= limits.max_entries, "ZIP central-directory entry limit exceeded")
    require(offset == marker and actual == count, "ZIP central-directory count mismatch")
    files: dict[str, bytes] = {}
    budget = Budget(limits)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        check_archive_names([m.filename for m in members], limits)
        # Require one contiguous, ordinary local-file sequence with no hidden prefix/gaps.
        local_offset = 0
        for member in sorted(members, key=lambda item: item.header_offset):
            require(member.header_offset == local_offset
                    and raw[local_offset:local_offset + 4] == b"PK\x03\x04",
                    "non-contiguous ZIP local records")
            require(local_offset + 30 <= cd_offset, "truncated ZIP local header")
            flags, compression = struct.unpack_from("<2H", raw, local_offset + 6)
            crc, compressed_size, size = struct.unpack_from("<3L", raw, local_offset + 14)
            name_size, extra_size = struct.unpack_from("<2H", raw, local_offset + 26)
            require(not flags & 8 and extra_size == 0, "ZIP descriptor or local extras unsupported")
            require((flags, compression, crc, compressed_size, size)
                    == (member.flag_bits, member.compress_type, member.CRC,
                        member.compress_size, member.file_size), "ZIP local/central mismatch")
            name_start = local_offset + 30
            encoded_name = member.filename.encode("utf-8" if flags & 0x800 else "cp437")
            require(raw[name_start:name_start + name_size] == encoded_name,
                    "ZIP local/central filename mismatch")
            local_offset = name_start + name_size + compressed_size
            require(local_offset <= cd_offset, "ZIP local data overlaps central directory")
        require(local_offset == cd_offset, "unaccounted ZIP bytes before central directory")
        for member in members:
            budget.entry()
            require(not member.is_dir(), "wheel directory entries are unsupported")
            require(not member.flag_bits & 1, "encrypted wheel member")
            require(not member.comment and not member.extra, "wheel member extra metadata unsupported")
            require(member.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
                    "unsupported wheel compression")
            kind = stat.S_IFMT(member.external_attr >> 16)
            require(kind in {0, stat.S_IFREG}, "non-regular wheel member")
            budget.payload(member.file_size)
        for member in members:
            with archive.open(member) as stream:
                data = stream.read(limits.max_file_bytes + 1)
            require(len(data) == member.file_size, "wheel member size mismatch")
            scan_text(member.filename, data)
            files[member.filename] = data
    return files


def tar_preflight(expanded: bytes, limits: Limits):
    """Bound physical headers, including hidden PAX headers, before tarfile parses them."""
    offset = 0
    count = 0
    while offset + 512 <= len(expanded):
        header = expanded[offset:offset + 512]
        if not any(header):
            require(len(expanded) >= offset + 1024 and not any(expanded[offset:]),
                    "missing tar end markers or nonzero trailing tar data")
            return
        count += 1
        require(count <= limits.max_entries, "tar physical-header count limit exceeded")
        field = header[124:136]
        require(not field[0] & 0x80, "binary tar size encoding unsupported")
        digits = field.rstrip(b"\x00 ").lstrip(b" ")
        require(bool(re.fullmatch(rb"[0-7]+", digits)), "invalid tar size")
        size = int(digits, 8)
        kind = header[156:157]
        require(kind in {b"0", b"\x00", b"5", b"x"}, "link or special sdist member")
        if kind == b"x":
            require(size <= 4096, "tar extended-metadata byte limit exceeded")
            # Even ignored PAX values are would-be publication bytes.
            scan_text("sdist extended metadata", expanded[offset + 512:offset + 512 + size])
        elif kind == b"5":
            require(size == 0, "sdist directory has a payload")
        else:
            require(size <= limits.max_file_bytes, "per-file byte limit exceeded")
        # USTAR names must not hide arbitrary data after their terminators.
        for raw_name in (header[:100], header[345:500]):
            if b"\x00" in raw_name:
                require(not any(raw_name.split(b"\x00", 1)[1]), "nonzero tar name padding")
        end = offset + 512 + size
        next_offset = offset + 512 + ((size + 511) // 512) * 512
        require(next_offset <= len(expanded), "truncated tar payload")
        require(not any(expanded[end:next_offset]), "nonzero tar payload padding")
        offset = next_offset
    raise CheckError("missing tar end markers")


def read_sdist(path: Path, limits: Limits) -> dict[str, bytes]:
    raw = read_regular(path, limits.max_archive_bytes)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
        expanded = compressed.read(limits.max_tar_bytes + 1)
    require(len(expanded) <= limits.max_tar_bytes, "expanded tar byte limit exceeded")
    require(len(expanded) % 512 == 0, "truncated tar blocks")
    tar_preflight(expanded, limits)
    files: dict[str, bytes] = {}
    budget = Budget(limits)
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
        members = []
        for member in archive:
            budget.entry()
            members.append(member)
            require(member.isfile() or member.isdir(), "link or special sdist member")
            require(not member.sparse, "sparse sdist member")
            # PAX timestamps are normal; path/link/size overrides are not necessary here.
            require(set(member.pax_headers) <= {"mtime", "atime", "ctime"},
                    "unsupported sdist extended metadata")
            if member.isfile():
                budget.payload(member.size)
        check_archive_names([m.name for m in members], limits)
        # Tar parsers stop at the first end marker; do not silently ignore another tar.
        end = max((m.offset_data + ((m.size + 511) // 512) * 512 for m in members), default=0)
        require(len(expanded) >= end + 1024 and not any(expanded[end:]),
                "missing tar end markers or nonzero trailing tar data")
        for member in members:
            if member.isdir():
                continue
            stream = archive.extractfile(member)
            require(stream is not None, "unreadable sdist member")
            with stream:
                data = stream.read(limits.max_file_bytes + 1)
            require(len(data) == member.size, "sdist member size mismatch")
            scan_text(member.name, data)
            files[member.name] = data
    return files


def same_files(actual: dict[str, bytes], expected: dict[str, bytes], label: str):
    require(set(actual) == set(expected),
            f"{label} member-set mismatch (missing={len(set(expected) - set(actual))}, "
            f"unexpected={len(set(actual) - set(expected))})")
    for name, data in expected.items():
        require(actual[name] == data, f"{label} byte mismatch: {name}")


def check_record(files: dict[str, bytes], record_name: str):
    require(record_name in files, "missing wheel RECORD")
    try:
        rows = list(csv.reader(io.StringIO(files[record_name].decode("utf-8"), newline=""),
                               strict=True))
    except (csv.Error, UnicodeDecodeError):
        raise CheckError("invalid wheel RECORD CSV") from None
    require(len(rows) == len(files), "RECORD row count mismatch")
    seen: set[str] = set()
    for row in rows:
        require(len(row) == 3, "RECORD rows must have three columns")
        name, encoded, size = row
        require(name in files and name not in seen, "unknown or duplicate RECORD path")
        seen.add(name)
        if name == record_name:
            require(encoded == size == "", "RECORD self entry must omit hash and size")
            continue
        require(encoded.startswith("sha256="), "RECORD requires SHA-256 for every payload")
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b"=")
        require(encoded == "sha256=" + digest.decode("ascii"), f"RECORD hash mismatch: {name}")
        require(size == str(len(files[name])), f"RECORD size mismatch: {name}")
    require(seen == set(files), "RECORD coverage mismatch")


def check_metadata(data: bytes, project: dict):
    message = BytesParser(policy=policy.default).parsebytes(data)
    require(not message.defects, "malformed distribution metadata")
    for key, expected in (("Name", project["name"]), ("Version", project["version"])):
        values = message.get_all(key, [])
        require(len(values) == 1 and str(values[0]) == expected,
                f"distribution metadata mismatch: {key}")
    python_values = message.get_all("Requires-Python", [])
    require(len(python_values) == 1, "missing or duplicate Requires-Python")
    # Backends canonically reorder conjunctions of version specifiers.
    def specifiers(value):
        return Counter(part.strip() for part in str(value).split(","))

    require(specifiers(python_values[0]) == specifiers(project["requires-python"]),
            "distribution metadata mismatch: Requires-Python")

    def normalize(value):
        return re.sub(r"\s+", "", str(value)).lower()
    require(Counter(map(normalize, message.get_all("Requires-Dist", [])))
            == Counter(map(normalize, project.get("dependencies", []))),
            "distribution dependency metadata mismatch")


def check_artifacts(snapshot: Snapshot, wheel: Path, sdist: Path, *, limits: Limits = DEFAULT_LIMITS):
    project = snapshot.project
    stem = PROJECT.replace("-", "_") + "-" + project["version"]
    require(wheel.name == stem + "-py3-none-any.whl", "unexpected wheel filename or tags")
    require(sdist.name == stem + ".tar.gz", "unexpected sdist filename")
    wheel_files = read_wheel(wheel, limits)
    sdist_files = read_sdist(sdist, limits)
    info = stem + ".dist-info"
    expected_package = {name.removeprefix("src/"): data for name, data in snapshot.files.items()
                        if name.startswith("src/")}
    actual_package = {name: data for name, data in wheel_files.items()
                      if name.startswith(PACKAGE + "/")}
    same_files(actual_package, expected_package, "source/wheel package")
    check_package(actual_package)
    expected_info = {f"{info}/{name}" for name in
                     ("METADATA", "WHEEL", "entry_points.txt", "RECORD")}
    license_files = {f"{info}/licenses/{name}": data for name, data in snapshot.files.items()
                     if name in ROOT_OPTIONAL}
    require(set(wheel_files) == set(expected_package) | expected_info | set(license_files),
            "unexpected or missing wheel metadata/payload")
    for name, data in license_files.items():
        require(wheel_files[name] == data, "license/notice bytes differ from source")
    check_record(wheel_files, f"{info}/RECORD")
    check_metadata(wheel_files[f"{info}/METADATA"], project)
    wheel_metadata = BytesParser(policy=policy.default).parsebytes(wheel_files[f"{info}/WHEEL"])
    for key, expected in (("Wheel-Version", "1.0"), ("Root-Is-Purelib", "true"),
                          ("Tag", "py3-none-any")):
        require(wheel_metadata.get_all(key, []) == [expected], f"unexpected WHEEL field: {key}")
    config = configparser.ConfigParser(interpolation=None, strict=True)
    config.optionxform = str
    try:
        config.read_string(wheel_files[f"{info}/entry_points.txt"].decode("utf-8"))
    except configparser.Error:
        raise CheckError("invalid wheel entrypoint metadata") from None
    require(not config.defaults() and config.sections() == ["console_scripts"]
            and dict(config["console_scripts"]) == project["scripts"],
            "wheel console entrypoint mismatch")
    prefix = stem + "/"
    require(all(name.startswith(prefix) for name in sdist_files), "sdist must have one project root")
    flat = {name.removeprefix(prefix): data for name, data in sdist_files.items()}
    require("PKG-INFO" in flat, "missing sdist PKG-INFO")
    check_metadata(flat.pop("PKG-INFO"), project)
    same_files(flat, snapshot.files, "source/sdist publication")
    same_files({name.removeprefix("src/"): data for name, data in flat.items()
                if name.startswith("src/")}, actual_package, "sdist/wheel package")
    return {"wheel_files": len(wheel_files), "sdist_files": len(sdist_files),
            "record_verified": True, "resource_files": len(RESOURCES), "parity_verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args(argv)
    if args.source_only == (args.wheel is not None or args.sdist is not None):
        parser.error("select --source-only OR both --wheel and --sdist")
    if not args.source_only and (args.wheel is None or args.sdist is None):
        parser.error("both --wheel and --sdist are required")
    try:
        snapshot = check_source(args.root)
        result = {"status": "passed", "source_files": len(snapshot.files),
                  "source_bytes": sum(map(len, snapshot.files.values())),
                  "mode": "source-only" if args.source_only else "artifacts"}
        if not args.source_only:
            result.update(check_artifacts(snapshot, args.wheel, args.sdist))
    except (CheckError, OSError, ValueError, EOFError, zlib.error,
            zipfile.BadZipFile, tarfile.TarError) as exc:
        # OSError/parser messages can contain private host paths or archive payloads.
        reason = str(exc) if isinstance(exc, CheckError) else type(exc).__name__
        print(json.dumps({"status": "failed", "reason": reason}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
