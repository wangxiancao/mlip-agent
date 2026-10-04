"""API 层 pydantic schema（PLAN §2.5，前后端共享契约）。

SSE 事件本身（AG-UI dict）不走这里的 schema —— 其契约由
app/agent/streaming.py 定义（type 字段 + 末尾 MLIP_RUN_RESULT 合成事件）。
本模块只覆盖 REST 端点的请求/响应体。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SessionInfo(BaseModel):
    """POST /api/sessions 响应。"""

    session_id: str
    created_at: str
    state_version: int


class MessageRequest(BaseModel):
    """POST /api/sessions/{sid}/messages 请求体；响应是 SSE 流（无 JSON schema）。"""

    content: str = Field(min_length=1)


class ResumeRequest(BaseModel):
    """POST /api/sessions/{sid}/resume 请求体：对挂起 run 的审批决定。

    - approve_all: 未显式列出的审批一律通过（默认 True）；
    - answers: tool_call_id → 用户文字答复（ask_user_question 注入用）；
    - denials: tool_call_id → 拒绝理由。
    """

    approve_all: bool = True
    answers: dict[str, str] = Field(default_factory=dict)
    denials: dict[str, str] = Field(default_factory=dict)


class PendingApproval(BaseModel):
    """挂起待审批的工具调用（MLIP_RUN_RESULT.deferred.approvals 的元素形态）。"""

    tool_call_id: str
    tool_name: str
    args: str | None = None


class RunSummary(BaseModel):
    """最近一次 run 的终态摘要。"""

    thread_id: str | None = None
    run_id: str | None = None
    status: Literal["completed", "suspended", "failed", "cancelled"]
    output: str | None = None
    failure: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    """GET /state 快照里的轻量消息条目（完整内容走 SSE / artifacts）。"""

    role: Literal["user", "assistant", "tool"]
    kind: Literal["text", "tool_call", "tool_return"]
    text: str | None = None
    tool_name: str | None = None


class JobInfo(BaseModel):
    """作业目录快照（data/sessions/{sid}/jobs/{job_id}）。"""

    job_id: str
    session_id: str
    job_type: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    has_output: bool = False


class SessionState(BaseModel):
    """GET /api/sessions/{sid}/state 响应：前端刷新/断线恢复用快照。"""

    session_id: str
    state_version: int
    status: Literal["idle", "completed", "suspended", "failed", "cancelled"]
    stale_state: bool = False
    title: str | None = None
    thread_id: str | None = None
    created_at: str
    updated_at: str
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    jobs: list[JobInfo] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    last_run: RunSummary | None = None


class ModelInfo(BaseModel):
    """GET /api/models 响应元素（数据来自 tools.maple_tools.MODEL_CATALOG）。"""

    name: str
    label: str
    path: str
    backend: str
    status: str
    notes: str = ""
