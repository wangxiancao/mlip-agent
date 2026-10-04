"""作业端点（PLAN §2.5）：作业状态查询与工件下载（FileResponse，防路径穿越）。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import AppServices, get_services
from app.state.schemas import JobInfo

router = APIRouter(tags=["jobs"])


def _locate(services: AppServices, job_id: str) -> tuple[str, Path]:
    found = services.store.find_job(job_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"未知作业: {job_id}")
    return found


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str, services: AppServices = Depends(get_services)
) -> JobInfo:
    session_id, job_dir = _locate(services, job_id)
    info = services.store._job_info(session_id, job_dir)  # noqa: SLF001
    return JobInfo(**info)


@router.get("/jobs/{job_id}/artifacts/{name}")
async def get_job_artifact(
    job_id: str, name: str, services: AppServices = Depends(get_services)
) -> FileResponse:
    _, job_dir = _locate(services, job_id)
    # 防路径穿越：只允许纯文件名，且解析后必须仍在作业目录内
    if not name or Path(name).name != name:
        raise HTTPException(status_code=400, detail=f"非法工件名: {name!r}")
    artifact = (job_dir / name).resolve()
    if not artifact.is_relative_to(job_dir.resolve()) or not artifact.is_file():
        raise HTTPException(status_code=404, detail=f"工件不存在: {name}")
    return FileResponse(artifact, filename=name)
