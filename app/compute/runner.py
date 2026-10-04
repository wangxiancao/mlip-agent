"""计算子进程封装（PLAN §2.3）。

- `run_compute(argv, cwd, timeout_s, gpu)`：gpu=True 时过模块级 GPU 锁（Semaphore(1)）；
- env 白名单：PATH / HOME / CUDA_VISIBLE_DEVICES=0 / HF_HUB_OFFLINE=1 /
  TRANSFORMERS_OFFLINE=1（杜绝联网下载，与 scripts/make_golden_refs.sh 一致）；
  extra_env 仅在显式传入时追加（TICKET-0002 插件后端注入用），缺省不变；
- stdout/stderr 实时落盘 job 目录（stdout.log / stderr.log），内存保留尾部 50 行；
- 超时后 kill 整个进程组（start_new_session=True + os.killpg）；
- MAPLE 成败判定：exit code 非 0 或 argv 指向的 .out 含 `ERROR:` 行 → 失败，
  error 附日志尾部。
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections import deque
from pathlib import Path
from pydantic import BaseModel, Field

# GPU 串行化：同一时刻只跑一个 GPU 计算（PLAN §0.5）
GPU_LOCK = asyncio.Semaphore(1)

_ENV_WHITELIST = ("PATH", "HOME")
_ENV_FIXED = {
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

_ERROR_MARKER = "ERROR:"
_TAIL_LINES = 50


class CompletedJob(BaseModel):
    """run_compute 返回结构。"""

    argv: list[str]
    cwd: str
    ok: bool
    exit_code: int | None
    timed_out: bool = False
    duration_s: float
    stdout_path: str
    stderr_path: str
    out_path: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


def _build_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_WHITELIST if k in os.environ}
    env.update(_ENV_FIXED)
    # 条件注入（如 TICKET-0002 插件后端）；缺省 None 时 env 一字节不变
    if extra_env:
        env.update(extra_env)
    return env


def _find_out_file(argv: list[str], cwd: Path) -> Path | None:
    for arg in argv[1:]:
        if arg.endswith(".out"):
            p = Path(arg)
            return p if p.is_absolute() else cwd / p
    return None


def _tail(lines: deque[str]) -> str:
    return "".join(lines)


async def _pump(stream: asyncio.StreamReader, fh, tail: deque[str]) -> None:
    while True:
        chunk = await stream.readline()
        if not chunk:
            break
        fh.write(chunk)
        fh.flush()
        tail.append(chunk.decode(errors="replace"))


async def _spawn_and_wait(
    argv: list[str], cwd: Path, timeout_s: int, extra_env: dict[str, str] | None = None
) -> tuple[int | None, bool, deque[str], deque[str], Path, Path]:
    stdout_path = cwd / "stdout.log"
    stderr_path = cwd / "stderr.log"
    out_tail: deque[str] = deque(maxlen=_TAIL_LINES)
    err_tail: deque[str] = deque(maxlen=_TAIL_LINES)

    with stdout_path.open("wb") as fout, stderr_path.open("wb") as ferr:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=_build_env(extra_env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pumps = [
            asyncio.create_task(_pump(proc.stdout, fout, out_tail)),  # type: ignore[arg-type]
            asyncio.create_task(_pump(proc.stderr, ferr, err_tail)),  # type: ignore[arg-type]
        ]
        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except TimeoutError:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
        await asyncio.gather(*pumps)
    return proc.returncode, timed_out, out_tail, err_tail, stdout_path, stderr_path


async def _run(
    argv: list[str], cwd: Path, timeout_s: int, extra_env: dict[str, str] | None = None
) -> CompletedJob:
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    exit_code, timed_out, out_tail, err_tail, stdout_path, stderr_path = (
        await _spawn_and_wait(argv, cwd, timeout_s, extra_env)
    )
    duration = time.monotonic() - started

    out_path = _find_out_file(argv, cwd)
    out_file_tail: deque[str] = deque(maxlen=_TAIL_LINES)
    out_has_error = False
    if out_path is not None and out_path.is_file():
        for line in out_path.read_text(errors="replace").splitlines(keepends=True):
            if _ERROR_MARKER in line:
                out_has_error = True
            out_file_tail.append(line)

    ok = not timed_out and exit_code == 0 and not out_has_error
    error: str | None = None
    if not ok:
        reasons = []
        if timed_out:
            reasons.append(f"超时（>{timeout_s}s），已 kill 进程组")
        if exit_code != 0:
            reasons.append(f"exit code = {exit_code}")
        if out_has_error and out_path is not None:
            reasons.append(f"{out_path.name} 含 `{_ERROR_MARKER}` 行")
        log_tail = _tail(err_tail) or _tail(out_tail) or _tail(out_file_tail)
        error = "; ".join(reasons) + "\n--- 日志尾部 ---\n" + log_tail

    return CompletedJob(
        argv=argv,
        cwd=str(cwd),
        ok=ok,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_s=round(duration, 3),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        out_path=str(out_path) if out_path is not None else None,
        error=error,
    )


async def run_compute(
    argv: list[str],
    cwd: str | Path,
    timeout_s: int = 1800,
    gpu: bool = True,
    extra_env: dict[str, str] | None = None,
) -> CompletedJob:
    """运行计算子进程。gpu=True 时串行化（模块级 Semaphore(1)，async with 保证释放）。

    extra_env：在白名单 env 之上条件追加的变量（如 MAPLE 插件后端的
    MAPLE_CALCULATOR_PLUGINS + PYTHONPATH，TICKET-0002）；缺省 None 时
    子进程 env 与既有行为完全一致。
    """
    cwd = Path(cwd)
    if gpu:
        async with GPU_LOCK:
            return await _run(argv, cwd, timeout_s, extra_env)
    return await _run(argv, cwd, timeout_s, extra_env)
