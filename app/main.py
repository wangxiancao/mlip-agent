"""FastAPI 装配入口（PLAN §2.5）：uvicorn app.main:app。

create_app(settings=None, model_override=None) 工厂：
- 生产：无参，Settings 走环境变量，LLM 模型 run 时解析（无 key 可启动）；
- 测试：注入 Settings(data_dir=tmp) 与 FunctionModel model_override。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent.harness import ModelOverride
from app.api import routes_chat, routes_jobs, routes_models, routes_sessions
from app.api.deps import AppServices
from app.config import Settings, get_settings
from app.state.store import SessionStore

_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

# vite dev server（PLAN §2.6）
_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app(
    settings: Settings | None = None, model_override: ModelOverride = None
) -> FastAPI:
    settings = settings or get_settings()
    services = AppServices(
        settings=settings,
        store=SessionStore(settings.data_dir),
        model_override=model_override,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services.store.sessions_dir.mkdir(parents=True, exist_ok=True)
        yield
        services.close()

    app = FastAPI(title="mlip_agent", version="0.1.0", lifespan=lifespan)
    app.state.services = services

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_sessions.router, prefix="/api")
    app.include_router(routes_chat.router, prefix="/api")
    app.include_router(routes_jobs.router, prefix="/api")
    app.include_router(routes_models.router, prefix="/api")

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "llm_configured": settings.llm_configured}

    # 生产模式：前端 build 产物存在时由 StaticFiles 托管（API 路由注册在先，优先匹配）
    if _WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")

    return app


app = create_app()
