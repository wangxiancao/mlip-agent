"""工具层共享类型：ToolResult 约定与依赖注入上下文（PLAN §2.2）。

所有工具返回 ToolResult dict，不抛异常；应用服务（session workspace、
job 目录分配、配置）经 ToolServices 由闭包注入（registry.py 组装）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from app.config import Settings


class ToolResult(TypedDict):
    ok: bool
    data: dict[str, Any] | list[Any] | str | None
    error: str | None


def ok(data: Any = None) -> ToolResult:
    return {"ok": True, "data": data, "error": None}


def err(message: str, data: Any = None) -> ToolResult:
    return {"ok": False, "data": data, "error": message}


class JobAllocator:
    """session workspace 内的作业目录分配器（data/sessions/{sid}/jobs/{job_id}）。

    job_id 形如 job-0001，单调递增；目录即 MAPLE/NequIP 子进程的 cwd。
    """

    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = Path(jobs_dir)
        self._counter = 0

    def allocate(self) -> tuple[str, Path]:
        self._counter += 1
        while True:
            job_id = f"job-{self._counter:04d}"
            job_dir = self.jobs_dir / job_id
            if not job_dir.exists():
                job_dir.mkdir(parents=True)
                return job_id, job_dir
            self._counter += 1

    def resolve(self, job_id: str) -> Path | None:
        # 防路径穿越：只允许纯目录名
        if not job_id or Path(job_id).name != job_id:
            return None
        job_dir = self.jobs_dir / job_id
        return job_dir if job_dir.is_dir() else None


@dataclass(frozen=True)
class ToolServices:
    """注入工具闭包的应用服务集合。"""

    settings: Settings
    session_workspace: Path
    job_allocator: JobAllocator
