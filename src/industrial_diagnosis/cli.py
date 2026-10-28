from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from .fixtures import original_knowledge, original_reports
from .knowledge import KnowledgeStore
from .managed_provider import ManagedLocalProvider
from .provider import LocalProvider, SyntheticProvider
from .schemas import DiagnosisRequest, DomainError, KnowledgeBuildRequest, PublishRequest
from .storage import Store
from .telemetry import TelemetryStore

app = typer.Typer(no_args_is_help=True, help="原创合成输送带诊断；默认仅本地模型，无设备控制。")


def output(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def provider_for(runtime_config: Path | None, model_url: str | None, test_mode: bool):
    if sum((runtime_config is not None, model_url is not None, test_mode)) != 1:
        raise typer.BadParameter("明确选择 --runtime-config、--model-url 或仅测试的 --test-mode 之一；没有回退。")
    if runtime_config:
        return ManagedLocalProvider(runtime_config)
    return SyntheticProvider() if test_mode else LocalProvider(model_url)


def actor_for(store: Store):
    return store.authenticate(typer.prompt("本地访问令牌（不写文件）", hide_input=True))


@app.command()
def init(data_dir: Path, actor_id: str = "local-admin", plant: str = "PLANT-A"):
    """仅在新目录建立合成时序和本地管理员；令牌只输出一次。"""
    if data_dir.exists() or data_dir.is_symlink():
        raise typer.BadParameter("初始化要求尚不存在的新数据目录，不覆盖任何既有数据。")
    store = Store(data_dir)
    token = store.provision_user(actor_id, "admin", [plant])
    listing = TelemetryStore(store).seed(plant)
    output({"synthetic_only": True, "data_dir": str(data_dir), "actor_id": actor_id,
            "token_shown_once": token, "cases": listing["cases"],
            "next": "导出原创知识草稿 → build 执行真实 HermiT → 人工审核 publish；尚未自动发布。"})


@app.command("fixture")
def fixture_command(plant: str = "PLANT-A", reports_only: bool = False):
    """向标准输出导出原创报告或由报告片段提取的未发布草稿。"""
    output([r.model_dump(mode="json") for r in original_reports(plant)] if reports_only
           else original_knowledge(plant).model_dump(mode="json"))


@app.command()
def build(data_dir: Path, java_path: Path, request_file: Path | None = None):
    """真实 OWL/HermiT 检查；结果只到 checked/blocked，不自动发布。"""
    store = Store(data_dir)
    actor = actor_for(store)
    if request_file:
        with request_file.open("rb") as stream:
            raw = stream.read(256_001)
        if len(raw) > 256_000:
            raise typer.BadParameter("知识请求超过 256 KB。")
        request = KnowledgeBuildRequest.model_validate_json(raw)
    else:
        request = original_knowledge(actor.plants[0])
    output(KnowledgeStore(store, java_path=java_path).build(request, actor))


@app.command()
def publish(data_dir: Path, version_id: str, content_hash: str, action_id: str,
            review_note: str):
    """人工管理员审核后显式激活精确知识版本；不控制设备。"""
    store = Store(data_dir)
    actor = actor_for(store)
    request = PublishRequest(action_id=action_id, version_id=version_id,
                             content_hash=content_hash, review_note=review_note)
    output(KnowledgeStore(store).publish(request, actor))


@app.command()
def cases(data_dir: Path):
    """读取当前身份授权的合成事件，不重复播种。"""
    store = Store(data_dir)
    output(TelemetryStore(store).demo_cases(actor_for(store)))


@app.command()
def diagnose(data_dir: Path, request_file: Path, runtime_config: Path | None = None,
             model_url: str | None = None, test_mode: bool = False):
    """创建并运行任务，遇人工补问立即暂停；不会自动提供观察答案。"""
    from .service import DiagnosisService
    store = Store(data_dir)
    actor = actor_for(store)
    with request_file.open("rb") as stream:
        raw = stream.read(8001)
    if len(raw) > 8000:
        raise typer.BadParameter("诊断请求超过 8 KB。")
    request = DiagnosisRequest.model_validate_json(raw)
    provider = provider_for(runtime_config, model_url, test_mode)
    service = DiagnosisService(store, KnowledgeStore(store), TelemetryStore(store), provider)
    task = service.create(request, actor)
    output(service.run(task["task_id"], actor))


@app.command()
def serve(data_dir: Path, runtime_config: Path | None = None, model_url: str | None = None,
          java_path: Path | None = None, test_mode: bool = False, port: int = 8003):
    """仅监听 127.0.0.1，令牌在工作台内存保存；无自动启动守护。"""
    from .api import create_app
    if not 1024 <= port <= 65535:
        raise typer.BadParameter("端口必须位于 1024–65535。")
    if not (data_dir / "industrial.sqlite3").is_file():
        raise typer.BadParameter("请先使用 init 建立新的合成数据目录。")
    provider = provider_for(runtime_config, model_url, test_mode)
    uvicorn.run(create_app(data_dir, provider, java_path=java_path), host="127.0.0.1", port=port,
                access_log=False, server_header=False)


def main():
    try:
        app()
    except DomainError as exc:
        typer.echo(f"{exc.code}: {exc.message}", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
