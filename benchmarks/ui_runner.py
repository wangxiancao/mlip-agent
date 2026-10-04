"""UI benchmark 运行器：经 Web UI 前端（Playwright headless Chromium）驱动 B 系列验收。

与 runner.py 的差别：runner.py 直连 harness（canned/live），本运行器走真实用户链路——
浏览器打开生产模式前端 → textarea 输入任务 prompt → SSE 流式渲染 → 审批卡点批准、
ask_user_question 填 tasks.yaml 的 auto_answer → 直至 run 终态。run_record.json 的
schema 与 runner.py 完全对齐（status/output/events/tool_calls/suspensions/jobs），
judge.py 直接复用判定；events/tool_calls 取自浏览器旁路录制的真实 AG-UI 事件流，
jobs 由后端 data_dir 的会话目录收集（runner.collect_jobs 复用）。

浏览器驱动在 Node 侧（benchmarks/ui_driver.mjs，playwright 为 web/ devDependency，
不污染 Python 环境）；Python 侧负责任务编排、后端生命周期、record 组装与 judge。

用法：
    python benchmarks/ui_runner.py                      # 自动起后端（动态端口），全部任务
    python benchmarks/ui_runner.py --tasks B1,B10       # 只跑部分任务
    python benchmarks/ui_runner.py --base-url http://127.0.0.1:8000 --data-dir data
    python benchmarks/ui_runner.py --headed             # 有头浏览器（调试用）

无 LLM key（/api/health llm_configured=false）时全部任务 SKIPPED，退出码 0（同 runner.py）。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from benchmarks import judge
from benchmarks.runner import RUNS_DIR, SKIP_NOTE, _tool_call_names, collect_jobs

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
DRIVER = BENCHMARKS_DIR / "ui_driver.mjs"

PROXY_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"
)

MAX_DRIVER_EXTRA_S = 120  # 子进程宽限（driver 内部已有 task timeout）


def _scrubbed_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    env.update(extra)
    return env


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_backend(data_dir: Path) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    env = _scrubbed_env(MLIP_AGENT_DATA_DIR=str(data_dir))
    proc = subprocess.Popen(
        [
            str(REPO_ROOT / ".venv" / "bin" / "uvicorn"),
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    with httpx.Client(trust_env=False, timeout=5.0) as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"后端进程提前退出（code={proc.returncode}）")
            try:
                if client.get(f"{base_url}/api/health").status_code == 200:
                    return proc, base_url
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("后端 60 s 内未就绪")


# ---------- 单任务执行 ----------


def run_task_ui(
    task: dict,
    *,
    base_url: str,
    data_dir: Path | None,
    task_dir: Path,
    headed: bool,
) -> dict:
    """驱动浏览器跑一条任务，产出与 runner.py 对齐的 record。"""
    started = datetime.now(timezone.utc)
    task_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = int(task.get("timeout_s", 1800))
    config_path = task_dir / "ui_config.json"
    result_path = task_dir / "ui_result.json"
    config = {
        "baseUrl": base_url,
        "prompt": task["prompt"],
        "autoAnswer": task.get("auto_answer"),
        "dataDir": str(data_dir) if data_dir else None,
        "resultPath": str(result_path),
        "headed": headed,
        "timeoutS": timeout_s,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    record: dict[str, Any] = {
        "task_id": task["id"],
        "mode": "ui",
        "started_at": started.isoformat(),
        "status": "error",
        "output": None,
        "error": None,
        "events": [],
        "tool_calls": [],
        "suspensions": [],
        "jobs": [],
        "workspace": None,
    }
    try:
        proc = subprocess.run(
            ["node", str(DRIVER), str(config_path)],
            cwd=REPO_ROOT,
            env=_scrubbed_env(),
            timeout=timeout_s + MAX_DRIVER_EXTRA_S,
            capture_output=True,
            text=True,
        )
        (task_dir / "driver.log").write_text(
            (proc.stdout or "") + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")
        )
        if proc.stdout:
            print(proc.stdout.rstrip(), flush=True)
        if not result_path.is_file():
            record["error"] = (
                f"driver 未产出结果（exit={proc.returncode}）: "
                f"{(proc.stderr or proc.stdout or '')[-500:]}"
            )
            return record
        ui = json.loads(result_path.read_text())
    except subprocess.TimeoutExpired:
        record["error"] = f"driver 子进程超时（{timeout_s + MAX_DRIVER_EXTRA_S} s）"
        return record

    events = ui.get("events") or []
    (task_dir / "events.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in events)
    )
    record["events"] = [e.get("type", "UNKNOWN") for e in events]
    record["tool_calls"] = _tool_call_names(events)
    record["suspensions"] = ui.get("suspensions") or []

    session_id = ui.get("session_id")
    if session_id and data_dir:
        session_dir = data_dir / "sessions" / session_id
        record["workspace"] = str(session_dir)
        record["jobs"] = collect_jobs(session_dir)

    final = ui.get("final")
    if final and final.get("type") == "MLIP_RUN_RESULT":
        record["status"] = final["status"]
        record["output"] = final.get("output")
        if final.get("failure"):
            record["error"] = json.dumps(final["failure"], ensure_ascii=False)
    # 后端 state 交叉校验/补齐（output 兜底）
    if session_id:
        try:
            with httpx.Client(trust_env=False, timeout=15.0) as client:
                state = client.get(
                    f"{base_url}/api/sessions/{session_id}/state"
                ).json()
            record["backend_status"] = state.get("status")
            last_run = state.get("last_run") or {}
            if record["output"] is None and last_run.get("output"):
                record["output"] = last_run["output"]
            if final is None and last_run.get("status"):
                record["status"] = last_run["status"]
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            record["backend_state_error"] = str(exc)
    if ui.get("error") and record["status"] == "error":
        record["error"] = ui["error"]
    return record


# ---------- 入口 ----------


def _write_skipped_records(run_dir: Path, tasks: list[dict]) -> None:
    for task in tasks:
        task_dir = run_dir / task["id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "task_id": task["id"],
            "mode": "ui",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "skipped",
            "output": None,
            "error": SKIP_NOTE,
            "events": [],
            "tool_calls": [],
            "suspensions": [],
            "jobs": [],
            "workspace": None,
        }
        (task_dir / "run_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLIP agent UI benchmark runner (Playwright)")
    parser.add_argument("--tasks", default=None, help="逗号分隔的任务 id 子集，如 B1,B10")
    parser.add_argument(
        "--base-url",
        default=None,
        help="已有后端地址；缺省时自动起 uvicorn（动态空闲端口，独立 data 目录）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="--base-url 模式下后端的 data 目录（jobs 收集用），缺省 ./data",
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    head = parser.add_mutually_exclusive_group()
    head.add_argument("--headed", action="store_true", help="有头浏览器（调试）")
    head.add_argument("--headless", action="store_true", help="无头浏览器（默认）")
    args = parser.parse_args(argv)

    tasks = judge.load_tasks()
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",")}
        tasks = [t for t in tasks if t["id"] in wanted]
        if not tasks:
            print(f"没有匹配的任务: {args.tasks}", file=sys.stderr)
            return 2

    run_dir = args.runs_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_ui"
    run_dir.mkdir(parents=True, exist_ok=True)

    backend: subprocess.Popen | None = None
    if args.base_url:
        base_url = args.base_url.rstrip("/")
        data_dir = args.data_dir or (REPO_ROOT / "data")
    else:
        data_dir = run_dir / "data"
        backend, base_url = _start_backend(data_dir)
    print(f"base_url={base_url}  data_dir={data_dir}  run_dir={run_dir}", flush=True)

    try:
        with httpx.Client(trust_env=False, timeout=15.0) as client:
            health = client.get(f"{base_url}/api/health").json()
        if not health.get("llm_configured"):
            print(SKIP_NOTE + "（设置 MLIP_AGENT_LLM_* 环境变量后重试）")
            _write_skipped_records(run_dir, tasks)
        else:
            for task in tasks:
                print(f"[{task['id']}] {task.get('description', '')} ...", flush=True)
                started = datetime.now(timezone.utc)
                record = run_task_ui(
                    task,
                    base_url=base_url,
                    data_dir=data_dir,
                    task_dir=run_dir / task["id"],
                    headed=args.headed,
                )
                record["duration_s"] = round(
                    (datetime.now(timezone.utc) - started).total_seconds(), 3
                )
                (run_dir / task["id"] / "run_record.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n"
                )
                print(
                    f"    status={record['status']} jobs={len(record['jobs'])}"
                    f" suspensions={len(record['suspensions'])}",
                    flush=True,
                )
    finally:
        if backend is not None:
            backend.terminate()
            try:
                backend.wait(timeout=15)
            except subprocess.TimeoutExpired:
                backend.kill()

    report = judge.judge_run_dir(run_dir, tasks)
    judge.print_report(report)
    summary = report["summary"]
    if summary["executed"] == 0:
        return 0
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
