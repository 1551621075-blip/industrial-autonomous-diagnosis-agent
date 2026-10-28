"""Local CLI contract tests: inert runtime files, no serving or model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from industrial_diagnosis import cli
from industrial_diagnosis.fixtures import original_knowledge, original_reports
from industrial_diagnosis.managed_provider import ManagedLocalProvider
from industrial_diagnosis.provider import LocalProvider, SyntheticProvider
from industrial_diagnosis.schemas import DomainError, KnowledgeBuildRequest, ReportSource
from industrial_diagnosis.storage import Store


@pytest.fixture
def runner():
    return CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"})


@pytest.fixture(autouse=True)
def prevent_server_and_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CLI test attempted to start a server or make a real inference request")

    monkeypatch.setattr(cli.uvicorn, "run", forbidden)
    monkeypatch.setattr(LocalProvider, "_post", forbidden)
    monkeypatch.setattr(ManagedLocalProvider, "_post", forbidden)


@pytest.fixture
def initialized(tmp_path, runner):
    root = tmp_path / "new-cli-root"
    result = runner.invoke(cli.app, ["init", str(root)], terminal_width=160)
    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    return SimpleNamespace(root=root, output=output, token=output["token_shown_once"])


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_init_creates_new_root_seeds_once_and_stores_only_token_hash(tmp_path, runner):
    root = tmp_path / "new-root"
    result = runner.invoke(cli.app, ["init", str(root), "--actor-id", "test-admin",
                                     "--plant", "PLANT-C"])
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    token = value["token_shown_once"]
    assert value["synthetic_only"] is True and value["actor_id"] == "test-admin"
    assert value["data_dir"] == str(root)
    assert set(value["cases"]) == {"normal", "jam", "slip", "sensor_fault", "missing", "conflicting"}
    assert {case["plant_id"] for case in value["cases"].values()} == {"PLANT-C"}
    assert len(token) >= 32 and result.stdout.count(token) == 1
    store = Store(root)
    actor = store.authenticate(token)
    assert actor.actor_id == "test-admin" and actor.role == "admin" and actor.plants == ["PLANT-C"]
    with store.connect() as db:
        row = dict(db.execute("SELECT * FROM users").fetchone())
        assert row["token_hash"] == hashlib.sha256(token.encode()).hexdigest()
        assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM telemetry_seeds").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM telemetry_events").fetchone()[0] == 8
        assert db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
    for contents in snapshot(root).values():
        assert token.encode() not in contents
    before = snapshot(root)
    repeated = runner.invoke(cli.app, ["init", str(root)])
    assert repeated.exit_code == 2
    assert "token_shown_once" not in repeated.output and token not in repeated.output
    assert snapshot(root) == before


@pytest.mark.parametrize("kind", ["empty_directory", "nonempty_directory", "file", "symlink"])
def test_init_never_overwrites_any_existing_path(tmp_path, runner, kind):
    root = tmp_path / "existing"
    target = tmp_path / "target"
    if kind == "file":
        root.write_bytes(b"ORIGINAL FILE MUST SURVIVE")
    elif kind == "symlink":
        target.mkdir()
        (target / "sentinel").write_bytes(b"ORIGINAL TARGET MUST SURVIVE")
        root.symlink_to(target, target_is_directory=True)
    else:
        root.mkdir()
        if kind == "nonempty_directory":
            (root / "sentinel").write_bytes(b"ORIGINAL DIRECTORY MUST SURVIVE")
    before = snapshot(tmp_path)
    result = runner.invoke(cli.app, ["init", str(root)])
    assert result.exit_code == 2
    assert snapshot(tmp_path) == before
    assert not (root / "industrial.sqlite3").exists()
    assert "token_shown_once" not in result.output


def test_init_rejects_dangling_symlink_instead_of_initializing_its_target(tmp_path, runner):
    root = tmp_path / "existing-dangling-link"
    target = tmp_path / "must-stay-absent"
    root.symlink_to(target, target_is_directory=True)
    result = runner.invoke(cli.app, ["init", str(root)])
    assert result.exit_code == 2, result.output
    assert root.is_symlink() and not target.exists()
    assert "token_shown_once" not in result.output


@pytest.mark.parametrize("reports_only", [False, True])
@pytest.mark.parametrize("plant", ["PLANT-A", "PLANT-C"])
def test_fixture_outputs_original_parseable_json_without_creating_state(tmp_path, monkeypatch,
                                                                       runner, reports_only, plant):
    monkeypatch.chdir(tmp_path)
    args = ["fixture", "--plant", plant] + (["--reports-only"] if reports_only else [])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    if reports_only:
        assert value == [report.model_dump(mode="json") for report in original_reports(plant)]
        assert all(ReportSource.model_validate(report) for report in value)
    else:
        assert value == original_knowledge(plant).model_dump(mode="json")
        assert KnowledgeBuildRequest.model_validate(value).draft.plant_id == plant
    assert list(tmp_path.iterdir()) == []
    assert "token_shown_once" not in result.output


@pytest.mark.parametrize(("config", "url", "test_mode"), [
    (None, None, False), (Path("explicit.json"), "http://127.0.0.1:18080", False),
    (Path("explicit.json"), None, True), (None, "http://127.0.0.1:18080", True),
    (Path("explicit.json"), "http://127.0.0.1:18080", True),
])
def test_provider_selection_is_exactly_one_explicit_option(monkeypatch, config, url, test_mode):
    constructors = [Mock(), Mock(), Mock()]
    for name, constructor in zip(("ManagedLocalProvider", "LocalProvider", "SyntheticProvider"),
                                 constructors, strict=True):
        monkeypatch.setattr(cli, name, constructor)
    with pytest.raises(typer.BadParameter, match="--runtime-config"):
        cli.provider_for(config, url, test_mode)
    assert all(not constructor.called for constructor in constructors)


def test_provider_selection_returns_only_explicit_test_or_numeric_loopback_provider():
    synthetic = cli.provider_for(None, None, True)
    assert isinstance(synthetic, SyntheticProvider) and synthetic.is_test_provider is True
    local = cli.provider_for(None, "http://127.0.0.1:18080", False)
    assert isinstance(local, LocalProvider) and local.is_test_provider is False
    assert local.generation_url == "http://127.0.0.1:18080"


def test_explicit_runtime_configuration_is_read_without_starting_or_reading_models(tmp_path):
    config = {
        "binary": {"path": str(tmp_path / "inert-server"), "sha256": "1" * 64},
        "libraries": [{"path": str(tmp_path / "inert-library"), "sha256": "2" * 64}],
        "models": {"generation": {"path": str(tmp_path / "absent-model.gguf"),
                                   "sha256": "3" * 64, "bytes": 1}},
        "lock_path": str(tmp_path / "must-not-create.lock"), "port": 18080,
    }
    path = tmp_path / "explicit-runtime.json"
    path.write_text(json.dumps(config))
    provider = cli.provider_for(path, None, False)
    assert isinstance(provider, ManagedLocalProvider) and provider.is_test_provider is False
    assert provider.generation_url == "http://127.0.0.1:18080"
    assert list(tmp_path.iterdir()) == [path]


def test_unreadable_explicit_runtime_config_never_falls_back(tmp_path, monkeypatch):
    fallback = Mock()
    monkeypatch.setattr(cli, "SyntheticProvider", fallback)
    with pytest.raises(DomainError) as error:
        cli.provider_for(tmp_path / "DO-NOT-ECHO-private-config.json", None, False)
    assert error.value.code == "runtime_config"
    assert "DO-NOT-ECHO" not in error.value.message
    fallback.assert_not_called()


@pytest.mark.parametrize("url", [
    "https://remote.invalid", "http://192.168.1.20:8000", "http://0.0.0.0:8000",
    "http://localhost:18080", "http://" + "user" + ":" + "DO-NOT-ECHO" + "@127.0.0.1:18080",
    "http://127.0.0.1:18080?token=DO-NOT-ECHO",
])
def test_cli_provider_rejects_remote_or_credentialed_urls_without_echoing_them(url):
    with pytest.raises(DomainError) as error:
        cli.provider_for(None, url, False)
    assert error.value.code == "model_endpoint"
    assert "DO-NOT-ECHO" not in error.value.message
    assert url not in error.value.message


@pytest.mark.parametrize("flags", [[], ["--test-mode", "--model-url", "http://127.0.0.1:18080"]])
def test_serve_fails_without_exactly_one_explicit_provider(initialized, runner, flags):
    result = runner.invoke(cli.app, ["serve", str(initialized.root), *flags])
    assert result.exit_code == 2, result.output
    assert "--runtime-config" in result.output and "--test-mode" in result.output
    assert initialized.token not in result.output


@pytest.mark.parametrize("port", ["0", "1023", "65536"])
def test_serve_rejects_out_of_range_port_before_starting(initialized, runner, port):
    result = runner.invoke(cli.app, ["serve", str(initialized.root), "--test-mode", "--port", port])
    assert result.exit_code == 2


def test_serve_requires_initialized_data_and_does_not_create_it(tmp_path, runner):
    root = tmp_path / "absent"
    result = runner.invoke(cli.app, ["serve", str(root), "--test-mode"])
    assert result.exit_code == 2
    assert not root.exists()


@pytest.mark.parametrize("flags", [["--test-mode"], ["--model-url", "http://127.0.0.1:18080"]])
def test_serve_is_hardwired_loopback_and_disables_access_logs(initialized, runner, monkeypatch, flags):
    serve = Mock()
    monkeypatch.setattr(cli.uvicorn, "run", serve)
    result = runner.invoke(cli.app, ["serve", str(initialized.root), *flags, "--port", "8017"])
    assert result.exit_code == 0, result.output
    serve.assert_called_once()
    assert serve.call_args.kwargs == {"host": "127.0.0.1", "port": 8017,
                                      "access_log": False, "server_header": False}
    app = serve.call_args.args[0]
    assert app.state.store.database == (initialized.root / "industrial.sqlite3").resolve()
    assert app.state.service.provider.is_test_provider is (flags == ["--test-mode"])
    assert initialized.token not in result.output


def test_serve_has_no_public_bind_override(initialized, runner):
    result = runner.invoke(cli.app, ["serve", str(initialized.root), "--test-mode",
                                     "--host", "0.0.0.0"])
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_cases_authenticates_hidden_token_and_reads_without_reseeding(initialized, runner):
    before = snapshot(initialized.root)
    result = runner.invoke(cli.app, ["cases", str(initialized.root)], input=initialized.token + "\n")
    assert result.exit_code == 0, result.output
    values = json.loads(result.stdout[result.stdout.index("["):])
    assert values and {case["plant_id"] for case in values} == {"PLANT-A"}
    assert {case["case_label"] for case in values} >= set(initialized.output["cases"])
    assert initialized.token not in result.output
    repeated = runner.invoke(cli.app, ["cases", str(initialized.root)], input=initialized.token + "\n")
    assert repeated.exit_code == 0, repeated.output
    assert repeated.stdout == result.stdout
    assert snapshot(initialized.root) == before


def diagnosis_file(initialized):
    case = initialized.output["cases"]["normal"]
    request = {"request_id": "cli-diagnose", **{key: case[key] for key in (
        "plant_id", "device_id", "event_id", "window", "complaint",
    )}}
    path = initialized.root.parent / "diagnosis.json"
    path.write_text(json.dumps(request))
    return path


@pytest.mark.parametrize("flags", [[], ["--test-mode", "--model-url", "http://127.0.0.1:18080"]])
def test_diagnose_requires_explicit_provider_before_creating_task(initialized, runner, flags):
    path = diagnosis_file(initialized)
    result = runner.invoke(cli.app, ["diagnose", str(initialized.root), str(path), *flags],
                           input=initialized.token + "\n")
    assert result.exit_code == 2, result.output
    assert "--runtime-config" in result.output
    assert initialized.token not in result.output
    with Store(initialized.root).connect() as db:
        assert db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_diagnose_request_byte_limit_precedes_service_or_model_work(initialized, runner):
    path = initialized.root.parent / "oversized-request.json"
    path.write_bytes(b" " * 8001)
    result = runner.invoke(cli.app, ["diagnose", str(initialized.root), str(path), "--test-mode"],
                           input=initialized.token + "\n")
    assert result.exit_code == 2
    assert "8 KB" in result.output
    assert initialized.token not in result.output
    with Store(initialized.root).connect() as db:
        assert db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_diagnose_test_mode_returns_waiting_human_without_auto_answer(initialized, runner, monkeypatch):
    from industrial_diagnosis import service

    path = diagnosis_file(initialized)
    instance = Mock()
    instance.create.return_value = {"task_id": "cli-test-task"}
    instance.run.return_value = {"task_id": "cli-test-task", "status": "waiting_human",
                                 "pending_human": {"question_id": "test-question"}}
    constructor = Mock(return_value=instance)
    monkeypatch.setattr(service, "DiagnosisService", constructor)
    result = runner.invoke(cli.app, ["diagnose", str(initialized.root), str(path), "--test-mode"],
                           input=initialized.token + "\n")
    assert result.exit_code == 0, result.output
    assert isinstance(constructor.call_args.args[3], SyntheticProvider)
    instance.create.assert_called_once()
    actor = instance.create.call_args.args[1]
    instance.run.assert_called_once_with("cli-test-task", actor)
    instance.answer.assert_not_called()
    value = json.loads(result.stdout[result.stdout.index("{"):])
    assert value["status"] == "waiting_human"
    assert initialized.token not in result.output


@pytest.mark.parametrize("command", [None, "init", "fixture", "build", "publish", "cases",
                                     "diagnose", "serve"])
def test_help_is_available_without_files_or_model_selection(tmp_path, monkeypatch, runner, command):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ([command] if command else []) + ["--help"], terminal_width=160)
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    if command in ("diagnose", "serve"):
        for option in ("--runtime-config", "--model-url", "--test-mode"):
            assert option in result.output
    if command is None:
        assert all(name in result.output for name in ("init", "fixture", "diagnose", "serve"))
    assert list(tmp_path.iterdir()) == []


def test_main_reports_domain_error_without_python_traceback(monkeypatch, capsys):
    def fail():
        raise DomainError("unauthorized", "Authentication required", 401)

    monkeypatch.setattr(cli, "app", fail)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "unauthorized: Authentication required\n"
    assert "Traceback" not in captured.err
