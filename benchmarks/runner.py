"""M5 运行器（PLAN §3 / §4 M5）：--mode=canned（默认）| live。

canned：pydantic-ai FunctionModel 剧本回放（benchmarks/canned_scripts.py），
工具层 → GPU 子进程 → 解析 → judge 全链路真实执行，不走 LLM。
live：直连 harness（build_harness + 真实 LLM，经 stream_agui_events 消费），
需要 MLIP_AGENT_LLM_MODEL / MLIP_AGENT_LLM_BASE_URL / MLIP_AGENT_LLM_API_KEY；
无 key 时全部任务标记 SKIPPED: no LLM endpoint（退出码 0）。

两种模式共用同一条流式执行循环（run → suspended 自动批准恢复 → 终态），
每任务产出 benchmarks/runs/<timestamp>/<task_id>/：
    run_record.json   事件流摘要 + 作业列表 + 解析结果（judge 的输入）
    events.jsonl      完整 AG-UI 事件流
跑完后自动调用 judge 判定并打印逐任务 pass/fail 与汇总比率；
任一执行任务 fail → 退出码 1。

用法：
    python benchmarks/runner.py                      # canned，全部任务
    python benchmarks/runner.py --mode=live          # 需 LLM key
    python benchmarks/runner.py --tasks B1,B9        # 只跑部分任务
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_ai.models.function import FunctionModel

from app.agent.harness import build_harness, make_resume_from_json, state_from_json
from app.agent.streaming import RUN_RESULT_EVENT_TYPE, stream_agui_events
from app.compute.parsers import parse_out_file
from app.config import Settings
from app.tools.registry import build_tools
from benchmarks import canned_scripts, judge

BENCHMARKS_DIR = Path(__file__).resolve().parent
RUNS_DIR = BENCHMARKS_DIR / "runs"
REACTION_DIR = BENCHMARKS_DIR / "reaction"
SKIP_NOTE = "SKIPPED: no LLM endpoint"

# --suite 配置：tasks 路径 / runs 目录 / canned 剧本注册表（延迟导入 suite 模块）
SUITES = {
    "b": {
        "tasks": BENCHMARKS_DIR / "tasks.yaml",
        "runs": RUNS_DIR,
        "ladder": None,
    },
    "reaction": {
        "tasks": REACTION_DIR / "tasks.yaml",
        "runs": REACTION_DIR / "runs",
        "ladder": REACTION_DIR / "ladder.md",
    },
}

MAX_SUSPENSION_ROUNDS = 8


# ---------- 作业收集与挂起快照 ----------


def _snapshot_jobs(workspace: Path) -> list[dict]:
    """挂起瞬间的作业产物快照（B10 断言「挂起时无计算产物」用）。"""
    jobs_dir = workspace / "jobs"
    if not jobs_dir.is_dir():
        return []
    return [
        {
            "job_id": d.name,
            "files": sorted(p.name for p in d.iterdir() if p.is_file()),
        }
        for d in sorted(jobs_dir.iterdir())
        if d.is_dir()
    ]


def collect_jobs(workspace: Path) -> list[dict]:
    """扫描 session workspace 的作业目录，解析产物为 judge 可用的结构化结果。

    MAPLE 作业：job_spec.json + input.out → parsers 解析；
    NequIP / TACE / DPA-4 直算作业：job.json + result.json → 直接读 result
    （backend 按 result.json versions 里的标识键区分：deepmd-kit→dpa4、tace→tace、
    其余→nequip）。
    """
    jobs: list[dict] = []
    jobs_dir = workspace / "jobs"
    if not jobs_dir.is_dir():
        return jobs
    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        entry: dict[str, Any] = {
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "artifacts": sorted(p.name for p in job_dir.iterdir() if p.is_file()),
        }
        spec_path = job_dir / "job_spec.json"
        direct_job_path = job_dir / "job.json"
        if spec_path.is_file():
            entry["backend"] = "maple"
            try:
                spec = json.loads(spec_path.read_text())
                entry["job_type"] = spec.get("job_type")
            except json.JSONDecodeError:
                entry["job_type"] = None
            out_path = job_dir / "input.out"
            if out_path.is_file() and entry["job_type"]:
                try:
                    entry["parsed"] = parse_out_file(
                        out_path, entry["job_type"]
                    ).model_dump(mode="json")
                except Exception as exc:  # noqa: BLE001 - 解析失败不拖垮收集
                    entry["parse_error"] = str(exc)
        elif direct_job_path.is_file():
            try:
                entry["job_type"] = json.loads(direct_job_path.read_text()).get("task")
            except json.JSONDecodeError:
                entry["job_type"] = None
            result_path = job_dir / "result.json"
            if result_path.is_file():
                try:
                    entry["result"] = json.loads(result_path.read_text())
                except json.JSONDecodeError as exc:
                    entry["parse_error"] = str(exc)
            versions = (entry.get("result") or {}).get("versions") or {}
            if "deepmd-kit" in versions:
                entry["backend"] = "dpa4"
            elif "tace" in versions:
                entry["backend"] = "tace"
            else:
                entry["backend"] = "nequip"
        else:
            entry["backend"] = "unknown"
        jobs.append(entry)
    return jobs


# ---------- 单任务执行 ----------


def _tool_call_names(events: list[dict]) -> list[str]:
    names = []
    for e in events:
        if e.get("type") == "TOOL_CALL_START":
            name = e.get("toolCallName") or e.get("tool_call_name")
            if name:
                names.append(name)
    return names


def _failed_tool_calls(events: list[dict]) -> list[str]:
    """从 TOOL_CALL_RESULT 事件提取 ok=false 的工具调用名（恢复类任务断言用）。

    TOOL_CALL_RESULT.content 是 ToolResult 的 JSON 字符串（含 ok 字段）；
    无 ok 字段的结果（如 delegate）不算失败。
    """
    id_to_name = {}
    for e in events:
        if e.get("type") == "TOOL_CALL_START":
            name = e.get("toolCallName") or e.get("tool_call_name")
            if e.get("toolCallId") and name:
                id_to_name[e["toolCallId"]] = name
    failed = []
    for e in events:
        if e.get("type") != "TOOL_CALL_RESULT":
            continue
        name = id_to_name.get(e.get("toolCallId"))
        if not name:
            continue
        try:
            content = json.loads(e.get("content") or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(content, dict) and content.get("ok") is False:
            failed.append(name)
    return failed


def _count_failed_jobs(jobs: list[dict]) -> int:
    """失败作业数：解析报错、MAPLE 作业无解析结果、直算作业无 result 或 ok=false。"""
    n = 0
    for job in jobs:
        if job.get("parse_error"):
            n += 1
        elif job.get("backend") == "maple" and "parsed" not in job:
            n += 1
        elif job.get("backend") in ("nequip", "tace", "dpa4"):
            result = job.get("result")
            if result is None or result.get("ok") is False:
                n += 1
    return n


async def run_task(
    task: dict,
    *,
    mode: str,
    workspace: Path,
    settings: Settings,
    scripts: dict | None = None,
) -> dict:
    """执行一条任务（流式循环：run → suspended 自动批准 → 终态）。"""
    started = datetime.now(timezone.utc)
    workspace.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "task_id": task["id"],
        "mode": mode,
        "tier": task.get("tier"),
        "reaction": task.get("reaction"),
        "started_at": started.isoformat(),
        "status": "error",
        "output": None,
        "error": None,
        "events": [],
        "tool_calls": [],
        "failed_tool_calls": [],
        "suspensions": [],
        "jobs": [],
        "metrics": {},
        "workspace": str(workspace),
    }
    try:
        if mode == "canned":
            registry = scripts if scripts is not None else canned_scripts.SCRIPTS
            script = registry[task["id"]]
            models = {
                "mlip_orchestrator": FunctionModel(stream_function=script.orchestrator),
                "task_parser": FunctionModel(stream_function=script.task_parser),
                "result_validator": FunctionModel(
                    stream_function=script.result_validator
                ),
            }
            executable = build_harness(
                settings, build_tools(workspace, settings=settings), model_override=models
            )
        else:
            executable = build_harness(settings, build_tools(workspace, settings=settings))

        events: list[dict] = []
        state = None
        resume = None
        prompt: str | None = task["prompt"]
        final: dict | None = None
        for _ in range(MAX_SUSPENSION_ROUNDS):
            async for event in stream_agui_events(
                executable, prompt, previous_state=state, deferred_resume=resume
            ):
                events.append(event)
            final = events[-1] if events else None
            if not final or final.get("type") != RUN_RESULT_EVENT_TYPE:
                record["error"] = "事件流未以 MLIP_RUN_RESULT 终止"
                break
            if final["status"] != "suspended":
                break
            deferred = final.get("deferred") or {}
            record["suspensions"].append(
                {
                    "approvals": deferred.get("approvals", []),
                    "calls": deferred.get("calls", []),
                    "jobs_snapshot": _snapshot_jobs(workspace),
                }
            )
            answers = {
                ap["tool_call_id"]: task["auto_answer"]
                for ap in deferred.get("approvals", [])
                if ap["tool_name"] == "ask_user_question" and task.get("auto_answer")
            }
            state = state_from_json(final["state_json"])
            resume = make_resume_from_json(final["deferred_json"], answers=answers)
            prompt = None
        else:
            record["error"] = f"超过 {MAX_SUSPENSION_ROUNDS} 轮挂起仍未终态"

        (workspace.parent / "events.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in events)
        )
        if final and final.get("type") == RUN_RESULT_EVENT_TYPE:
            record["status"] = final["status"]
            record["output"] = final.get("output")
            if final.get("failure"):
                record["error"] = json.dumps(final["failure"], ensure_ascii=False)
        record["events"] = [e.get("type", "UNKNOWN") for e in events]
        record["tool_calls"] = _tool_call_names(events)
        record["failed_tool_calls"] = _failed_tool_calls(events)
    except Exception as exc:  # noqa: BLE001 - 单任务异常不中断整个 benchmark
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["jobs"] = collect_jobs(workspace)
        duration = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
        record["duration_s"] = duration
        record["metrics"] = {
            "n_jobs": len(record["jobs"]),
            "n_failed_jobs": _count_failed_jobs(record["jobs"]),
            "duration_s": duration,
        }
    return record


# ---------- 入口 ----------


def _write_skipped_records(run_dir: Path, tasks: list[dict], mode: str) -> None:
    for task in tasks:
        task_dir = run_dir / task["id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "task_id": task["id"],
            "mode": mode,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "skipped",
            "output": None,
            "error": SKIP_NOTE,
            "events": [],
            "tool_calls": [],
            "failed_tool_calls": [],
            "suspensions": [],
            "jobs": [],
            "metrics": {},
            "workspace": None,
        }
        (task_dir / "run_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        )


async def _run_all(
    tasks: list[dict],
    mode: str,
    run_dir: Path,
    settings: Settings,
    scripts: dict | None = None,
) -> None:
    for task in tasks:
        print(f"[{task['id']}] {task.get('description', '')} ...", flush=True)
        try:
            record = await asyncio.wait_for(
                run_task(
                    task,
                    mode=mode,
                    workspace=run_dir / task["id"] / "workspace",
                    settings=settings,
                    scripts=scripts,
                ),
                timeout=task.get("timeout_s", 1800),
            )
        except TimeoutError:
            record = {
                "task_id": task["id"],
                "mode": mode,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "output": None,
                "error": f"任务整体超时（{task.get('timeout_s', 1800)} s）",
                "events": [],
                "tool_calls": [],
                "failed_tool_calls": [],
                "suspensions": [],
                "jobs": collect_jobs(run_dir / task["id"] / "workspace"),
                "metrics": {},
                "workspace": str(run_dir / task["id"] / "workspace"),
            }
        task_dir = run_dir / task["id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "run_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        print(f"    status={record['status']} jobs={len(record['jobs'])}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLIP agent benchmark runner (M5)")
    parser.add_argument("--mode", choices=["canned", "live"], default="canned")
    parser.add_argument("--tasks", default=None, help="逗号分隔的任务 id 子集，如 B1,B9")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="b",
        help="任务套件：b（默认，B1–B19 框架回归）| reaction（§3.3 反应能垒套件）",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    suite = SUITES[args.suite]
    runs_dir = args.runs_dir or suite["runs"]
    tasks = judge.load_tasks(suite["tasks"])
    # §3.3：decision 档在 baseline prompt 后注入决策阶梯全文
    if suite["ladder"] is not None:
        ladder = suite["ladder"].read_text()
        for task in tasks:
            if task.get("tier") == "decision":
                task["prompt"] = (
                    task["prompt"]
                    + "\n\n以下是故障诊断决策阶梯，遇到问题时按优先级逐条参考：\n\n"
                    + ladder
                )
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",")}
        tasks = [t for t in tasks if t["id"] in wanted]
        if not tasks:
            print(f"没有匹配的任务: {args.tasks}", file=sys.stderr)
            return 2

    scripts = None
    if args.mode == "canned" and args.suite == "reaction":
        from benchmarks.reaction import canned_suite

        scripts = canned_suite.SCRIPTS

    run_dir = runs_dir / datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"suite={args.suite}  mode={args.mode}  run_dir={run_dir}")

    settings = Settings()
    if args.mode == "live" and not settings.llm_configured:
        print(SKIP_NOTE + "（设置 MLIP_AGENT_LLM_* 环境变量后重试）")
        _write_skipped_records(run_dir, tasks, args.mode)
    else:
        asyncio.run(_run_all(tasks, args.mode, run_dir, settings, scripts))

    report = judge.judge_run_dir(run_dir, tasks)
    judge.print_report(report)
    if args.suite == "reaction":
        from benchmarks.reaction import suite_judge

        suite_judge.print_suite_report(suite_judge.judge_suite(run_dir, tasks))
    summary = report["summary"]
    if summary["executed"] == 0:
        return 0  # 全部跳过（无 LLM key）
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
