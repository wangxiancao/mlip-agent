"""会话 CRUD（PLAN §2.5）：创建会话、会话状态快照。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agent.harness import summarize_messages
from app.api.deps import AppServices, get_services
from app.state.schemas import (
    CreateSessionRequest,
    JobInfo,
    SessionInfo,
    SessionState,
)
from app.state.store import CURRENT_STATE_VERSION, SessionRecord

router = APIRouter(tags=["sessions"])


@router.post("/sessions", status_code=201)
async def create_session(
    body: CreateSessionRequest | None = None,
    services: AppServices = Depends(get_services),
) -> SessionInfo:
    record = await services.store.create(title=(body.title if body else None))
    return SessionInfo(
        session_id=record.session_id,
        created_at=record.created_at,
        state_version=CURRENT_STATE_VERSION,
    )


def _session_state(record: SessionRecord, services: AppServices) -> SessionState:
    messages = []
    if record.harness_state_json:
        try:
            messages = summarize_messages(record.harness_state_json)
        except ValueError:
            # 磁盘上的 state 与本版 SDK 不兼容：按 stale 处理，不拖垮端点
            record.stale_state = True
    return SessionState(
        session_id=record.session_id,
        state_version=CURRENT_STATE_VERSION,
        status=record.status,  # type: ignore[arg-type]
        stale_state=record.stale_state,
        title=record.title,
        thread_id=record.thread_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        pending_approvals=record.pending_approvals,  # type: ignore[arg-type]
        jobs=[JobInfo(**j) for j in services.store.list_jobs(record.session_id)],
        messages=messages,  # type: ignore[arg-type]
        last_run=record.last_run,  # type: ignore[arg-type]
    )


@router.get("/sessions/{session_id}/state")
async def get_session_state(
    session_id: str, services: AppServices = Depends(get_services)
) -> SessionState:
    record = await services.store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"未知会话: {session_id}")
    return _session_state(record, services)
