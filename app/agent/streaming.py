"""Harness 流 → AG-UI dict 事件的异步适配器（供 M3 SSE 使用）。

SDK 边界：只 import a13n_stream_protocol（不 import a13n-harness）；
HarnessStreamEvent 的判定用鸭子类型（有 .result 属性即终止事件）。
deferred 的 JSON 序列化委托 app.agent.permissions（pydantic-ai 层）。

用法（M3 routes_chat.py）：
    async for event in stream_agui_events(executable, prompt, previous_state=..., deferred_resume=...):
        yield sse_encode(event)
最后一个事件恒为 {"type": "MLIP_RUN_RESULT", ...}，携带 status/output/
state_json（HarnessState.model_dump_json，供持久化与恢复）/deferred 摘要。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from a13n_stream_protocol import HarnessAguiStreamObserver

from app.agent.permissions import dump_deferred_json

# 终止事件之后的合成事件类型，M3 依赖该约定做状态持久化
RUN_RESULT_EVENT_TYPE = "MLIP_RUN_RESULT"


def _dump_event(event: Any) -> dict:
    data = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    data.setdefault("type", str(getattr(event, "type", "UNKNOWN")))
    return data


def _deferred_summary(deferred: Any) -> dict | None:
    if deferred is None:
        return None
    return {
        "approvals": [
            {
                "tool_call_id": c.tool_call_id,
                "tool_name": c.tool_name,
                "args": c.args if isinstance(c.args, str) else None,
            }
            for c in deferred.approvals
        ],
        "calls": [
            {
                "tool_call_id": c.tool_call_id,
                "tool_name": c.tool_name,
                "args": c.args if isinstance(c.args, str) else None,
            }
            for c in deferred.calls
        ],
    }


async def stream_agui_events(
    executable: Any,
    prompt: str | None = None,
    *,
    previous_state: Any = None,
    deferred_resume: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    """把一次 harness run（stream）转成 AG-UI dict 事件序列。

    末尾追加一个 RUN_RESULT_EVENT_TYPE 合成事件：
        {"type", "status", "output", "thread_id", "run_id",
         "state_json": str | None, "deferred": {...} | None,
         "deferred_json": str | None, "failure": {...} | None}
    deferred_json 是 DeferredToolRequests 的完整 JSON（harness.make_resume_from_json
    可据其在服务重启后重建恢复值）。
    """
    observer = HarnessAguiStreamObserver()
    async with executable.stream(
        prompt, previous_state=previous_state, deferred_resume=deferred_resume
    ) as stream:
        async for item in stream:
            for event in observer.observe(item):
                yield _dump_event(event)
            result = getattr(item, "result", None)
            if result is not None:  # HarnessRunResultEvent：终止事件
                state = result.state
                failure = result.failure
                yield {
                    "type": RUN_RESULT_EVENT_TYPE,
                    "thread_id": result.thread_id,
                    "run_id": result.run_id,
                    "status": result.status,
                    "output": result.output if result.status == "completed" else None,
                    "state_json": state.model_dump_json() if state is not None else None,
                    "deferred": _deferred_summary(result.deferred),
                    "deferred_json": (
                        dump_deferred_json(result.deferred)
                        if result.deferred is not None
                        else None
                    ),
                    "failure": (
                        failure.model_dump(mode="json", by_alias=True)
                        if failure is not None
                        else None
                    ),
                }
