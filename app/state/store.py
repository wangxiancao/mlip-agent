"""会话/作业持久化（PLAN §2.5）：JSON 文件存储，单进程 + asyncio.Lock。

布局（data_dir 缺省 ./data，不入库）：
    data/sessions/{session_id}/state.json   会话元数据 + HarnessState JSON + 挂起现场
    data/sessions/{session_id}/jobs/{job_id}/   作业工件（工具层写入，store 只扫描）

state.json 字段：
    state_version      本模块 CURRENT_STATE_VERSION；不匹配时弃旧对话开新（stale_state 标记）
    harness_state_json stream_agui_events 的 state_json（a13n HarnessState 序列化）
    deferred_json      挂起时 DeferredToolRequests 的完整 JSON（重启后 resume 用）
    pending_approvals  挂起审批摘要（前端直接渲染）
    status             idle | completed | suspended | failed | cancelled
    last_run           最近 run 摘要

store 不 import 任何 SDK：harness_state_json / deferred_json 一律按不透明字符串
处理；反序列化只在 app/agent/harness.py（SDK 隔离层，PLAN §5.1）。
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CURRENT_STATE_VERSION = 1

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SessionRecord:
    """一个会话的内存视图（state.json 的镜像）。"""

    session_id: str
    created_at: str
    updated_at: str
    title: str | None = None
    status: str = "idle"
    stale_state: bool = False
    harness_state_json: str | None = None
    deferred_json: str | None = None
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    last_run: dict[str, Any] | None = None

    @property
    def thread_id(self) -> str | None:
        if self.harness_state_json is None:
            return None
        try:
            return json.loads(self.harness_state_json)["thread_id"]
        except (json.JSONDecodeError, KeyError):
            return None


class SessionStore:
    """会话存储：内存缓存 + 磁盘 lazy 加载，asyncio.Lock 防并发写。"""

    def __init__(self, data_dir: Path | str) -> None:
        # TICKET-0012：规范化为绝对路径——缺省 ./data 相对路径会让
        # session_workspace 变成相对串，子进程 cwd=workspace 时 argv 中的
        # 相对 job 路径被二次拼接（data/sessions/<sid>/data/sessions/...）。
        self.data_dir = Path(data_dir).resolve()
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, SessionRecord] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard: asyncio.Lock | None = None

    # ---------- 锁 ----------

    async def lock(self, session_id: str) -> asyncio.Lock:
        """每会话一把锁：run 期间持有，串行化消息/恢复/状态写。"""
        if self._locks_guard is None:
            self._locks_guard = asyncio.Lock()
        async with self._locks_guard:
            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            return self._locks[session_id]

    # ---------- 路径 ----------

    @staticmethod
    def valid_session_id(session_id: str) -> bool:
        return bool(_SESSION_ID_RE.match(session_id))

    def session_dir(self, session_id: str) -> Path:
        if not self.valid_session_id(session_id):
            raise ValueError(f"非法 session_id: {session_id!r}")
        return self.sessions_dir / session_id

    def workspace(self, session_id: str) -> Path:
        """会话工作目录（build_tools 的 session_workspace 参数）。"""
        return self.session_dir(session_id)

    def _state_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "state.json"

    # ---------- 会话生命周期 ----------

    async def create(self, title: str | None = None) -> SessionRecord:
        session_id = uuid.uuid4().hex
        now = _utcnow()
        record = SessionRecord(session_id=session_id, created_at=now, updated_at=now, title=title)
        self.session_dir(session_id).mkdir(parents=True)
        self._cache[session_id] = record
        await self._persist(record)
        return record

    async def get(self, session_id: str) -> SessionRecord | None:
        """取会话；未缓存时从磁盘 lazy 加载（服务重启后的恢复路径）。"""
        if not self.valid_session_id(session_id):
            return None
        if session_id in self._cache:
            return self._cache[session_id]
        path = self._state_path(session_id)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        record = SessionRecord(
            session_id=session_id,
            created_at=raw.get("created_at", _utcnow()),
            updated_at=raw.get("updated_at", _utcnow()),
            title=raw.get("title"),
            status=raw.get("status", "idle"),
            harness_state_json=raw.get("harness_state_json"),
            deferred_json=raw.get("deferred_json"),
            pending_approvals=list(raw.get("pending_approvals") or []),
            last_run=raw.get("last_run"),
        )
        if raw.get("state_version") != CURRENT_STATE_VERSION:
            # PLAN §5.8：版本不匹配弃旧开新，保留目录与作业工件，标记 stale
            record.stale_state = True
            record.status = "idle"
            record.harness_state_json = None
            record.deferred_json = None
            record.pending_approvals = []
            record.last_run = None
            stale_backup = path.with_suffix(f".v{raw.get('state_version')}.stale.json")
            try:
                path.replace(stale_backup)
            except OSError:
                pass
            await self._persist(record)
        self._cache[session_id] = record
        return record

    async def save_run(self, session_id: str, run_event: dict[str, Any]) -> SessionRecord | None:
        """run 结束后持久化（调用方须持有会话锁）。

        Args:
            run_event: stream_agui_events 的 MLIP_RUN_RESULT 合成事件。
        """
        record = await self.get(session_id)
        if record is None:
            return None
        record.updated_at = _utcnow()
        record.status = run_event["status"]
        if run_event.get("state_json"):
            record.harness_state_json = run_event["state_json"]
        record.deferred_json = run_event.get("deferred_json")
        deferred = run_event.get("deferred") or {}
        record.pending_approvals = (
            list(deferred.get("approvals") or [])
            if run_event["status"] == "suspended"
            else []
        )
        record.last_run = {
            "thread_id": run_event.get("thread_id"),
            "run_id": run_event.get("run_id"),
            "status": run_event["status"],
            "output": run_event.get("output"),
            "failure": run_event.get("failure"),
        }
        await self._persist(record)
        return record

    async def _persist(self, record: SessionRecord) -> None:
        payload = {
            "state_version": CURRENT_STATE_VERSION,
            "session_id": record.session_id,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "status": record.status,
            "stale_state": record.stale_state,
            "harness_state_json": record.harness_state_json,
            "deferred_json": record.deferred_json,
            "pending_approvals": record.pending_approvals,
            "last_run": record.last_run,
        }
        path = self._state_path(record.session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)

    # ---------- 作业索引（扫描派生，不冗余持久化） ----------

    def list_jobs(self, session_id: str) -> list[dict[str, Any]]:
        """扫描会话 jobs 目录，产出 JobInfo dict 列表（job_id 升序）。"""
        try:
            jobs_dir = self.session_dir(session_id) / "jobs"
        except ValueError:
            return []
        if not jobs_dir.is_dir():
            return []
        return [
            self._job_info(session_id, d)
            for d in sorted(jobs_dir.iterdir())
            if d.is_dir() and _JOB_ID_RE.match(d.name)
        ]

    def find_job(self, job_id: str) -> tuple[str, Path] | None:
        """全局定位作业目录 → (session_id, job_dir)；非法名/不存在返回 None。"""
        if not job_id or not _JOB_ID_RE.match(job_id) or Path(job_id).name != job_id:
            return None
        if not self.sessions_dir.is_dir():
            return None
        for session_dir in sorted(self.sessions_dir.iterdir()):
            if not session_dir.is_dir() or not self.valid_session_id(session_dir.name):
                continue
            job_dir = session_dir / "jobs" / job_id
            if job_dir.is_dir():
                return session_dir.name, job_dir
        return None

    @staticmethod
    def _job_info(session_id: str, job_dir: Path) -> dict[str, Any]:
        artifacts = sorted(p.name for p in job_dir.iterdir() if p.is_file())
        job_type: str | None = None
        spec_path = job_dir / "job_spec.json"
        if spec_path.is_file():
            try:
                job_type = json.loads(spec_path.read_text()).get("job_type")
            except (json.JSONDecodeError, OSError):
                pass
        has_output = any(
            name.endswith(".out") or name == "result.json" for name in artifacts
        )
        return {
            "job_id": job_dir.name,
            "session_id": session_id,
            "job_type": job_type,
            "artifacts": artifacts,
            "has_output": has_output,
        }
