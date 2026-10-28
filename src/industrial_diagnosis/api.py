import ipaddress
import json
import math
from importlib.resources import files
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import Field

from .knowledge import KnowledgeStore, request_from_reports
from .reports import ReportService
from .schemas import (
    Actor,
    DiagnosisRequest,
    DomainError,
    HumanAnswer,
    Identifier,
    KnowledgeBuildRequest,
    PublishRequest,
    ReconcileRequest,
    ReportSource,
    StrictModel,
)
from .service import DiagnosisService
from .storage import Store
from .telemetry import TelemetryStore

MAX_BODY = 256_000


class ActionRequest(StrictModel):
    action_id: Identifier


class ExtractionRequest(StrictModel):
    request_id: Identifier
    reports: list[ReportSource] = Field(min_length=1, max_length=16)


class ReviewRequest(ActionRequest):
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["approve", "reject"]
    note: str = Field(min_length=10, max_length=2000)


def create_app(data_dir, provider, *, java_path=None):
    store = Store(data_dir)
    knowledge = KnowledgeStore(store, java_path=java_path)
    telemetry = TelemetryStore(store)
    service = DiagnosisService(store, knowledge, telemetry, provider)
    reports = ReportService(store, service, provider)
    app = FastAPI(title="工业自主诊断 · 原创合成参考工程", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.store, app.state.knowledge = store, knowledge
    app.state.telemetry, app.state.service, app.state.reports = telemetry, service, reports

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError):
        return JSONResponse({"error": exc.code, "message": exc.message}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"error": "invalid_request", "message": "请求字段或类型不符合契约。"}, status_code=422)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        host = request.url.hostname
        if host != "testserver":
            try:
                if host != "localhost" and not ipaddress.ip_address(host or "").is_loopback:
                    raise ValueError("host")
            except ValueError:
                return JSONResponse({"error": "invalid_host"}, status_code=400)
        origin = request.headers.get("origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                valid = (parsed.scheme == request.url.scheme and parsed.netloc == request.url.netloc
                         and not parsed.path and not parsed.query and not parsed.fragment)
            except ValueError:
                valid = False
            if not valid:
                return JSONResponse({"error": "cross_origin_denied"}, status_code=403)
        if request.method in {"POST", "PUT", "PATCH"}:
            if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                return JSONResponse({"error": "json_required"}, status_code=415)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY:
                    return JSONResponse({"error": "body_too_large"}, status_code=413)
            request._body = bytes(body)
            try:
                def unique_object(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError("Duplicate JSON field")
                        result[key] = value
                    return result

                def invalid_constant(value):
                    raise ValueError("Non-finite JSON value")

                def finite_float(value):
                    result = float(value)
                    if not math.isfinite(result):
                        raise ValueError("Non-finite JSON number")
                    return result

                payload = json.loads(request._body, object_pairs_hook=unique_object,
                                     parse_constant=invalid_constant, parse_float=finite_float)
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
            except (ValueError, UnicodeError, RecursionError):
                return JSONResponse({"error": "invalid_request", "message": "请求必须是有效 JSON 对象。"}, status_code=422)
        response = await call_next(request)
        return response

    @app.middleware("http")
    async def response_security(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse({"error": "internal_error", "message": "请求处理失败；未返回内部异常内容。"}, status_code=500)
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        })
        return response

    def authenticated(authorization: Annotated[str | None, Header()] = None):
        if not authorization or not authorization.startswith("Bearer "):
            raise DomainError("unauthorized", "请输入本地 CLI 签发的访问令牌。", 401)
        return store.authenticate(authorization[7:])

    User = Annotated[Actor, Depends(authenticated)]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return files("industrial_diagnosis").joinpath("templates/index.html").read_text()

    @app.get("/static/{name}")
    def static(name: str):
        types = {"app.js": "application/javascript", "style.css": "text/css"}
        if name not in types:
            raise DomainError("not_found", "Static resource unavailable", 404)
        return Response(files("industrial_diagnosis").joinpath("static", name).read_text(), media_type=types[name])

    @app.get("/health")
    def health():
        return {"status": "ok", "synthetic_only": True, "device_control": False,
                "provider_mode": "test-only" if provider.is_test_provider else "local-model"}

    @app.get("/api/me")
    def identity(actor: User):
        return actor.model_dump(mode="json")

    @app.get("/api/cases")
    def cases(actor: User):
        return telemetry.demo_cases(actor)

    @app.post("/api/knowledge/extract")
    def extract(request: ExtractionRequest, actor: User):
        if actor.role not in {"analyst", "admin"}:
            raise DomainError("forbidden", "Knowledge author role required", 403)
        result = request_from_reports(request.request_id, request.reports)
        store.authorize(actor, result.draft.plant_id)
        return result.model_dump(mode="json")

    @app.post("/api/knowledge/build")
    def build(request: KnowledgeBuildRequest, actor: User):
        return knowledge.build(request, actor)

    @app.post("/api/knowledge/publish")
    def publish(request: PublishRequest, actor: User):
        return knowledge.publish(request, actor)

    @app.get("/api/knowledge")
    def versions(actor: User):
        return knowledge.list_versions(actor)

    @app.get("/api/knowledge/{version_id}")
    def version(version_id: str, actor: User):
        return knowledge.get(version_id, actor)

    @app.get("/api/tasks")
    def tasks(actor: User):
        return service.list_tasks(actor)

    @app.post("/api/tasks")
    def create(request: DiagnosisRequest, actor: User):
        return service.create(request, actor)

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str, actor: User):
        return service.get(task_id, actor)

    @app.post("/api/tasks/{task_id}/run")
    def run(task_id: str, actor: User):
        return service.run(task_id, actor)

    @app.post("/api/tasks/{task_id}/answer")
    def answer(task_id: str, request: HumanAnswer, actor: User):
        return service.answer(task_id, request, actor)

    @app.post("/api/tasks/{task_id}/reconcile")
    def reconcile(task_id: str, request: ReconcileRequest, actor: User):
        return service.reconcile(task_id, request, actor)

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel(task_id: str, request: ActionRequest, actor: User):
        return service.cancel(task_id, request.action_id, actor)

    @app.get("/api/tasks/{task_id}/audit")
    def audit(task_id: str, actor: User):
        return {"events": service.audit(task_id, actor), "model_calls": service.calls(task_id, actor)}

    @app.post("/api/tasks/{task_id}/reports")
    def generate_report(task_id: str, request: ActionRequest, actor: User):
        return reports.generate(task_id, request.action_id, actor)

    @app.get("/api/tasks/{task_id}/reports")
    def list_reports(task_id: str, actor: User):
        return reports.list_for_task(task_id, actor)

    @app.get("/api/reports/{report_id}")
    def report(report_id: str, actor: User):
        return reports.get(report_id, actor)

    @app.post("/api/reports/{report_id}/review")
    def review(report_id: str, request: ReviewRequest, actor: User):
        return reports.review(report_id=report_id, action_id=request.action_id,
                              content_hash=request.content_hash, decision=request.decision,
                              note=request.note, actor=actor)

    return app
