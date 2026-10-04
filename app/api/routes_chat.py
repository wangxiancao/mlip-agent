"""对话流端点（PLAN §2.5）：POST 消息 / 审批恢复，响应即 SSE（text/event-stream）。

SSE 线格式：每个 AG-UI 事件一行 `data: {json}\n\n`；超过 HEARTBEAT_S 无事件
时发注释行 `:hb\n\n` 防代理断流。事件序列末尾恒为 MLIP_RUN_RESULT 合成事件
（见 app/agent/streaming.py），收到后即持久化会话状态。

并发：每会话一把 asyncio.Lock（store.lock），run 期间持有；同会话第二个
并发请求直接 409，不排队。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.agent.harness import make_resume_from_json, state_from_json
from app.agent.streaming import RUN_RESULT_EVENT_TYPE, stream_agui_events
from app.api.deps import AppServices, get_services
from app.state.schemas import MessageRequest, ResumeRequest
from app.state.store import SessionRecord

router = APIRouter(tags=["chat"])

HEARTBEAT_S = 15.0

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # 禁 nginx 缓冲，逐事件下发
}


def _sse_data(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _sse_run(
    services: AppServices,
    record: SessionRecord,
    *,
    prompt: str | None,
    deferred_resume: Any = None,
) -> AsyncIterator[str]:
    """消费 stream_agui_events 并编码为 SSE；终止事件落盘。

    心跳用「挂起中的 __anext__ task + asyncio.wait 超时」实现，不 cancel
    底层 harness 流（cancel 会中断 run）。
    """
    lock = await services.store.lock(record.session_id)
    async with lock:
        # 锁内重读记录：等待锁期间可能已有别的 run 改了状态
        record = await services.store.get(record.session_id) or record
        previous_state = (
            state_from_json(record.harness_state_json)
            if record.harness_state_json
            else None
        )
        executable = services.get_harness(record.session_id)
        agen = stream_agui_events(
            executable,
            prompt,
            previous_state=previous_state,
            deferred_resume=deferred_resume,
        )
        iterator = agen.__aiter__()
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(iterator.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_S)
                if not done:
                    yield ":hb\n\n"
                    continue
                task, pending = pending, None
                try:
                    event = task.result()
                except StopAsyncIteration:
                    break
                if event.get("type") == RUN_RESULT_EVENT_TYPE:
                    await services.store.save_run(record.session_id, event)
                yield _sse_data(event)
        finally:
            if pending is not None:
                pending.cancel()
                # 等被 cancel 的 __anext__ 收尾，否则下面的 aclose 会撞上
                # "asynchronous generator is already running"
                try:
                    await pending
                except (asyncio.CancelledError, Exception):
                    pass
            await agen.aclose()


async def _get_record(services: AppServices, session_id: str) -> SessionRecord:
    record = await services.store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"未知会话: {session_id}")
    return record


def _check_idle(record: SessionRecord) -> None:
    if record.status == "suspended":
        raise HTTPException(
            status_code=409,
            detail="会话处于挂起态，请先 POST /resume 处理待审批项",
        )


@router.post("/sessions/{session_id}/messages")
async def post_message(
    session_id: str,
    body: MessageRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    record = await _get_record(services, session_id)
    _check_idle(record)
    lock = await services.store.lock(session_id)
    if lock.locked():
        raise HTTPException(status_code=409, detail="该会话有 run 正在进行")
    return StreamingResponse(
        _sse_run(services, record, prompt=body.content),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/sessions/{session_id}/resume")
async def post_resume(
    session_id: str,
    body: ResumeRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    record = await _get_record(services, session_id)
    if record.status != "suspended" or not record.deferred_json:
        raise HTTPException(status_code=409, detail="会话不在挂起态，无可恢复的审批")
    lock = await services.store.lock(session_id)
    if lock.locked():
        raise HTTPException(status_code=409, detail="该会话有 run 正在进行")
    try:
        resume = make_resume_from_json(
            record.deferred_json,
            approve_all=body.approve_all,
            answers=body.answers,
            denials=body.denials,
        )
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"挂起现场损坏，无法恢复: {exc}") from exc
    return StreamingResponse(
        _sse_run(services, record, prompt=None, deferred_resume=resume),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
