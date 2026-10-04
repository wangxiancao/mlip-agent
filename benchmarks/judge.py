"""M5 判定器（PLAN §3.1/§3.2）：加载 tasks.yaml，对 runner 产出的 run_record.json 逐项断言。

用法：
    python benchmarks/judge.py <run_dir>     # 判定一次已完成的运行
    judge_run_dir(run_dir, tasks)            # 供 runner.py 调用

断言类型（type 字段，kind: numeric 计入数值正确率，其余计入执行成功率）：
    final_status / event_present / tool_called / tool_not_called / job_count /
    job_type / backend / artifact_present / out_contains / output_contains /
    converged / md_steps / md_steps_max / suspended_with_tool /
    no_artifact_at_suspension / tool_call_failed
    ---- 数值类 ----
    energy_hartree / energy_eV / frequencies_real / scan_energies / scan_argmin /
    geometry_h2o / energy_drift_max / n_imaginary

v2（§3.2）补充：
    - 断言级 only_mode（如 only_mode: canned）——只在 record.mode 匹配时求值，
      不匹配则跳过（live 下恢复路径由 agent 自主决策，不强制失败首试）；
    - job 索引支持负数（-1 = 最后一个作业）；
    - numeric 断言结果带结构化 abs_error（energy 取 |Δ|，freq/scan 取 max |Δ|，
      geometry 取 max 键长 |Δ|，drift 取漂移值），summarize 汇总 mean/max；
    - judge_task 透传 tier/expect 并输出 efficiency 块（min_jobs / actual_jobs /
      failed_jobs / first_try）；summarize 在既有字段之外输出
      by_tier / by_backend / by_job_type / by_model 分组比率、error_stats、
      efficiency 汇总（均 informational，不影响 all_passed 与退出码）。

汇总口径：
    执行成功率 = 全部 execution 断言通过的任务数 / 实际执行任务数
    数值正确率 = 全部 numeric 断言通过的任务数 / 含 numeric 断言的执行任务数
skipped 任务（live 无 LLM key）不计入两个比率，也不影响退出码；
任一执行任务 fail → 退出码 1。
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parent
TASKS_YAML = BENCHMARKS_DIR / "tasks.yaml"

# 与 MAPLE 汇总口径一致：ν > 5.0 cm⁻¹ 为实频
_REAL_FREQ_FLOOR_CM1 = 5.0


def load_tasks(path: str | Path = TASKS_YAML) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text())
    return data["tasks"]


# ---------- 断言取值辅助 ----------


def _fail(reason: str) -> tuple[bool, str]:
    return False, reason


def _pass(detail: str) -> tuple[bool, str]:
    return True, detail


def _job(record: dict, index: int) -> dict | None:
    jobs = record.get("jobs") or []
    if index < 0:
        index += len(jobs)
    return jobs[index] if 0 <= index < len(jobs) else None


def _parsed(record: dict, index: int) -> dict | None:
    job = _job(record, index)
    return (job or {}).get("parsed")


def _close(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _h2o_geometry(final_xyz: str) -> tuple[list[float], float]:
    """从 xyz 文本算 H2O 的两条 O−H 键长（Å）与 H−O−H 键角（度）。"""
    lines = [l.split() for l in final_xyz.splitlines()[2:] if l.strip()]
    o = next((l for l in lines if l[0] == "O"), None)
    hs = [l for l in lines if l[0] == "H"]
    if o is None or len(hs) < 2:
        raise ValueError("final_xyz 中找不到 1 个 O 与 2 个 H")
    po = [float(v) for v in o[1:4]]
    ph = [[float(v) for v in h[1:4]] for h in hs[:2]]
    dists = [math.dist(po, p) for p in ph]
    v1 = [p - q for p, q in zip(ph[0], po)]
    v2 = [p - q for p, q in zip(ph[1], po)]
    cos = sum(a * b for a, b in zip(v1, v2)) / (math.dist(po, ph[0]) * math.dist(po, ph[1]))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return dists, angle


# ---------- 结构化数值误差（v2 error_stats 用） ----------


def _abs_error(a: dict, record: dict) -> float | None:
    """numeric 断言的绝对偏差（无法求值时返回 None）。"""
    t = a["type"]
    try:
        if t == "energy_hartree":
            e = (_parsed(record, a["job"]) or {}).get("energy_hartree")
            return abs(e - a["value"]) if e is not None else None
        if t == "energy_eV":
            job = _job(record, a["job"])
            e = ((job or {}).get("result") or {}).get("energy_eV")
            return abs(e - a["value"]) if e is not None else None
        if t == "frequencies_real":
            freqs = (_parsed(record, a["job"]) or {}).get("frequencies_cm1")
            if not freqs:
                return None
            real = sorted(f for f in freqs if f > _REAL_FREQ_FLOOR_CM1)
            expected = sorted(a["values"])
            if len(real) != len(expected):
                return None
            return max(abs(r - e) for r, e in zip(real, expected))
        if t == "scan_energies":
            energies = (_parsed(record, a["job"]) or {}).get("energies_hartree")
            if not energies or len(energies) != len(a["values"]):
                return None
            return max(abs(r - e) for r, e in zip(energies, a["values"]))
        if t == "geometry_h2o":
            final_xyz = (_parsed(record, a["job"]) or {}).get("final_xyz")
            if not final_xyz:
                return None
            dists, _ = _h2o_geometry(final_xyz)
            return max(abs(d - e) for d, e in zip(sorted(dists), sorted(a["oh_angstrom"])))
        if t == "energy_drift_max":
            drift = (_parsed(record, a["job"]) or {}).get("energy_drift_ha")
            return drift
    except (KeyError, TypeError, ValueError):
        return None
    return None


# ---------- 断言求值 ----------


def evaluate_assertion(a: dict, record: dict) -> tuple[bool, str]:
    """对单条断言求值，返回 (passed, detail)。"""
    t = a["type"]
    jobs = record.get("jobs") or []

    if t == "final_status":
        ok = record.get("status") == a["value"]
        return (True, f"status={record.get('status')}") if ok else _fail(
            f"期望 status={a['value']}，实际 {record.get('status')}"
            + (f"（error: {record.get('error')}）" if record.get("error") else "")
        )
    if t == "event_present":
        ok = a["event"] in (record.get("events") or [])
        return _pass(f"事件 {a['event']} 存在") if ok else _fail(
            f"事件流中无 {a['event']}（实际: {record.get('events')}）"
        )
    if t == "tool_called":
        ok = a["name"] in (record.get("tool_calls") or [])
        return _pass(f"调用了 {a['name']}") if ok else _fail(f"未调用工具 {a['name']}")
    if t == "tool_not_called":
        ok = a["name"] not in (record.get("tool_calls") or [])
        return _pass(f"未调用 {a['name']}") if ok else _fail(f"不应调用 {a['name']}（实际调用了）")
    if t == "tool_call_failed":
        failed = record.get("failed_tool_calls") or []
        ok = a["name"] in failed
        return _pass(f"{a['name']} 出现过失败调用") if ok else _fail(
            f"{a['name']} 无失败调用记录（实际 failed: {failed}）"
        )
    if t == "job_count":
        ok = len(jobs) == a["value"]
        return _pass(f"作业数={len(jobs)}") if ok else _fail(
            f"期望作业数 {a['value']}，实际 {len(jobs)}"
        )
    if t == "job_type":
        job = _job(record, a["job"])
        if job is None:
            return _fail(f"作业 #{a['job']} 不存在（共 {len(jobs)} 个）")
        ok = job.get("job_type") == a["value"]
        return _pass(f"job[{a['job']}].job_type={job.get('job_type')}") if ok else _fail(
            f"job[{a['job']}].job_type 期望 {a['value']}，实际 {job.get('job_type')}"
        )
    if t == "backend":
        job = _job(record, a["job"])
        if job is None:
            return _fail(f"作业 #{a['job']} 不存在")
        ok = job.get("backend") == a["value"]
        return _pass(f"backend={job.get('backend')}") if ok else _fail(
            f"backend 期望 {a['value']}，实际 {job.get('backend')}"
        )
    if t == "artifact_present":
        job = _job(record, a["job"])
        if job is None:
            return _fail(f"作业 #{a['job']} 不存在")
        ok = a["name"] in (job.get("artifacts") or [])
        return _pass(f"{job['job_id']}/{a['name']} 存在") if ok else _fail(
            f"{job['job_id']} 缺少工件 {a['name']}（现有: {job.get('artifacts')}）"
        )
    if t == "out_contains":
        job = _job(record, a["job"])
        if job is None:
            return _fail(f"作业 #{a['job']} 不存在")
        out_path = Path(job["job_dir"]) / a.get("out_name", "input.out")
        if not out_path.is_file():
            return _fail(f"{out_path.name} 不存在")
        hay = _normalize_ws(out_path.read_text())
        needle = _normalize_ws(a["text"])
        ok = needle in hay
        return _pass(f"输出含 {needle!r}") if ok else _fail(f"输出不含 {needle!r}")
    if t == "output_contains":
        output = record.get("output") or ""
        ok = a["text"] in output
        return _pass(f"回答含 {a['text']!r}") if ok else _fail(
            f"回答不含 {a['text']!r}（回答: {output[:200]!r}）"
        )
    if t == "suspended_with_tool":
        names = [
            ap["tool_name"]
            for s in record.get("suspensions") or []
            for ap in s.get("approvals", [])
        ]
        ok = a["value"] in names
        return _pass(f"挂起审批含 {a['value']}") if ok else _fail(
            f"挂起审批中无 {a['value']}（实际挂起: {names or '从未挂起'}）"
        )
    if t == "no_artifact_at_suspension":
        suspensions = record.get("suspensions") or []
        k = a["suspension"]
        if k >= len(suspensions):
            return _fail(f"第 {k} 次挂起不存在（共 {len(suspensions)} 次）")
        offenders = [
            f"{s['job_id']}/{a['name']}"
            for s in suspensions[k].get("jobs_snapshot", [])
            if a["name"] in s.get("files", [])
        ]
        ok = not offenders
        return _pass(f"挂起 {k} 时无 {a['name']}") if ok else _fail(
            f"挂起 {k} 时已存在产物: {offenders}"
        )
    if t == "converged":
        parsed = _parsed(record, a["job"])
        if parsed is None:
            return _fail(f"作业 #{a['job']} 无解析结果")
        ok = parsed.get("converged") == a["value"]
        return _pass(f"converged={parsed.get('converged')}") if ok else _fail(
            f"converged 期望 {a['value']}，实际 {parsed.get('converged')}"
        )
    if t == "md_steps":
        parsed = _parsed(record, a["job"])
        if parsed is None:
            return _fail(f"作业 #{a['job']} 无解析结果")
        n = len(parsed.get("energies_hartree") or [])
        ok = n == a["value"]
        return _pass(f"MD 步数={n}") if ok else _fail(f"MD 步数期望 {a['value']}，实际 {n}")
    if t == "md_steps_max":
        parsed = _parsed(record, a["job"])
        if parsed is None:
            return _fail(f"作业 #{a['job']} 无解析结果")
        n = len(parsed.get("energies_hartree") or [])
        ok = n <= a["value"]
        return _pass(f"MD 步数={n} ≤ {a['value']}") if ok else _fail(
            f"MD 步数期望 ≤ {a['value']}，实际 {n}"
        )
    if t == "n_imaginary":
        parsed = _parsed(record, a["job"])
        if parsed is None:
            return _fail(f"作业 #{a['job']} 无解析结果")
        n = parsed.get("n_imaginary")
        ok = n == a["value"]
        return _pass(f"n_imaginary={n}") if ok else _fail(f"虚频数期望 {a['value']}，实际 {n}")

    # ---- 数值类 ----
    if t == "energy_hartree":
        parsed = _parsed(record, a["job"])
        e = (parsed or {}).get("energy_hartree")
        if e is None:
            return _fail(f"作业 #{a['job']} 无 energy_hartree")
        ok = _close(e, a["value"], a["tol"])
        return _pass(f"E={e:.10f} Ha（Δ={abs(e - a['value']):.2e} ≤ {a['tol']:g}）") if ok else _fail(
            f"E={e:.10f} Ha，pin={a['value']}，Δ={abs(e - a['value']):.2e} > {a['tol']:g}"
        )
    if t == "energy_eV":
        job = _job(record, a["job"])
        e = ((job or {}).get("result") or {}).get("energy_eV")
        if e is None:
            return _fail(f"作业 #{a['job']} 无 result.energy_eV")
        ok = _close(e, a["value"], a["tol"])
        return _pass(f"E={e:.8f} eV（Δ={abs(e - a['value']):.2e} ≤ {a['tol']:g}）") if ok else _fail(
            f"E={e:.8f} eV，pin={a['value']}，Δ={abs(e - a['value']):.2e} > {a['tol']:g}"
        )
    if t == "frequencies_real":
        parsed = _parsed(record, a["job"])
        freqs = (parsed or {}).get("frequencies_cm1")
        if not freqs:
            return _fail(f"作业 #{a['job']} 无频率解析")
        real = sorted(f for f in freqs if f > _REAL_FREQ_FLOOR_CM1)
        expected = sorted(a["values"])
        if len(real) != len(expected):
            return _fail(f"实频个数期望 {len(expected)}，实际 {len(real)}（{real}）")
        bad = [
            (r, e) for r, e in zip(real, expected) if not _close(r, e, a["tol"])
        ]
        ok = not bad
        return _pass(f"实频 {real} 全部在 ±{a['tol']:g} cm⁻¹ 内") if ok else _fail(
            f"实频偏差超限: {bad}（tol={a['tol']:g}）"
        )
    if t == "scan_energies":
        parsed = _parsed(record, a["job"])
        energies = (parsed or {}).get("energies_hartree")
        if not energies:
            return _fail(f"作业 #{a['job']} 无扫描能量")
        expected = a["values"]
        if len(energies) != len(expected):
            return _fail(f"扫描点数期望 {len(expected)}，实际 {len(energies)}")
        bad = [
            (i, r, e)
            for i, (r, e) in enumerate(zip(energies, expected))
            if not _close(r, e, a["tol"])
        ]
        ok = not bad
        return _pass(f"{len(energies)} 个扫描能量全部在 ±{a['tol']:g} Ha 内") if ok else _fail(
            f"扫描能量偏差超限（index, actual, expected）: {bad}"
        )
    if t == "scan_argmin":
        parsed = _parsed(record, a["job"])
        idx = (parsed or {}).get("argmin_index")
        if idx is None:
            return _fail(f"作业 #{a['job']} 无 argmin_index")
        ok = idx == a["value"]
        return _pass(f"argmin_index={idx}") if ok else _fail(
            f"能量最小点期望 index={a['value']}，实际 {idx}"
        )
    if t == "geometry_h2o":
        parsed = _parsed(record, a["job"])
        final_xyz = (parsed or {}).get("final_xyz")
        if not final_xyz:
            return _fail(f"作业 #{a['job']} 无 final_xyz")
        try:
            dists, angle = _h2o_geometry(final_xyz)
        except ValueError as exc:
            return _fail(f"几何计算失败: {exc}")
        oh_bad = [
            (d, e)
            for d, e in zip(sorted(dists), sorted(a["oh_angstrom"]))
            if not _close(d, e, a["oh_tol"])
        ]
        ang_ok = _close(angle, a["angle_deg"], a["angle_tol"])
        ok = not oh_bad and ang_ok
        detail = (
            f"O−H={['%.6f' % d for d in dists]} Å，∠HOH={angle:.4f}°"
            f"（pin {a['oh_angstrom']} ±{a['oh_tol']:g} / {a['angle_deg']} ±{a['angle_tol']:g}）"
        )
        return _pass(detail) if ok else _fail(detail)
    if t == "energy_drift_max":
        parsed = _parsed(record, a["job"])
        drift = (parsed or {}).get("energy_drift_ha")
        if drift is None:
            return _fail(f"作业 #{a['job']} 无 energy_drift_ha")
        ok = drift <= a["value"]
        return _pass(f"漂移 {drift:.3e} Ha ≤ {a['value']:g}") if ok else _fail(
            f"漂移 {drift:.3e} Ha > {a['value']:g}"
        )
    return _fail(f"未知断言类型: {t}")


# ---------- 任务级与汇总 ----------


def _count_failed_jobs(jobs: list[dict]) -> int:
    """失败作业数（与 runner._count_failed_jobs 同口径；旧 record 无 metrics 时兜底）。"""
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


def judge_task(task: dict, record: dict) -> dict:
    """对一条任务求值全部断言（only_mode 不匹配的断言跳过）。"""
    if record.get("status") == "skipped":
        return {
            "task_id": task["id"],
            "skipped": True,
            "note": record.get("error") or "SKIPPED",
            "passed": None,
            "assertions": [],
        }
    mode = record.get("mode")
    results = []
    for a in task.get("assertions", []):
        if a.get("only_mode") and a["only_mode"] != mode:
            continue
        passed, detail = evaluate_assertion(a, record)
        results.append(
            {
                "type": a["type"],
                "kind": a.get("kind", "execution"),
                "passed": passed,
                "detail": detail,
                "abs_error": _abs_error(a, record)
                if a.get("kind") == "numeric"
                else None,
            }
        )
    jobs = record.get("jobs") or []
    metrics = record.get("metrics") or {}
    min_jobs = task.get("min_jobs")
    actual_jobs = metrics.get("n_jobs", len(jobs))
    efficiency = {
        "min_jobs": min_jobs,
        "actual_jobs": actual_jobs,
        "failed_jobs": metrics.get("n_failed_jobs", _count_failed_jobs(jobs)),
        "first_try": (actual_jobs == min_jobs) if min_jobs is not None else None,
    }
    return {
        "task_id": task["id"],
        "skipped": False,
        "tier": task.get("tier", "guided"),
        "expect": task.get("expect") or {},
        "passed": all(r["passed"] for r in results),
        "execution_passed": all(
            r["passed"] for r in results if r["kind"] == "execution"
        ),
        "numeric_passed": all(
            r["passed"] for r in results if r["kind"] == "numeric"
        ),
        "has_numeric": any(r["kind"] == "numeric" for r in results),
        "assertions": results,
        "efficiency": efficiency,
    }


def _group_rates(reports: list[dict]) -> dict:
    """一组任务报告的覆盖率/精度（与总体口径一致）。"""
    executed = [r for r in reports if not r["skipped"]]
    numeric_tasks = [r for r in executed if r["has_numeric"]]
    exec_ok = sum(1 for r in executed if r["execution_passed"])
    num_ok = sum(1 for r in numeric_tasks if r["numeric_passed"])
    return {
        "executed": len(executed),
        "passed": sum(1 for r in executed if r["passed"]),
        "execution_success_rate": (exec_ok / len(executed)) if executed else None,
        "numeric_accuracy": (num_ok / len(numeric_tasks)) if numeric_tasks else None,
    }


def _group_by(reports: list[dict], key_fn) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in reports:
        groups.setdefault(key_fn(r), []).append(r)
    return {k: _group_rates(v) for k, v in sorted(groups.items())}


def summarize(task_reports: list[dict]) -> dict:
    executed = [r for r in task_reports if not r["skipped"]]
    skipped = [r for r in task_reports if r["skipped"]]
    numeric_tasks = [r for r in executed if r["has_numeric"]]
    exec_ok = sum(1 for r in executed if r["execution_passed"])
    num_ok = sum(1 for r in numeric_tasks if r["numeric_passed"])

    def _expect_key(r: dict, field: str, default: str) -> str:
        value = (r.get("expect") or {}).get(field)
        if value is None or value == []:
            return default
        if isinstance(value, list):
            return "+".join(str(v) for v in value)
        return str(value)

    abs_errors = [
        a["abs_error"]
        for r in executed
        for a in r["assertions"]
        if a.get("kind") == "numeric" and a.get("abs_error") is not None
    ]
    effs = [r["efficiency"] for r in executed if r.get("efficiency", {}).get("min_jobs") is not None]
    first_tries = [e for e in effs if e.get("first_try") is not None]
    return {
        "total": len(task_reports),
        "executed": len(executed),
        "skipped": len(skipped),
        "passed": sum(1 for r in executed if r["passed"]),
        "execution_success_rate": (exec_ok / len(executed)) if executed else None,
        "numeric_accuracy": (num_ok / len(numeric_tasks)) if numeric_tasks else None,
        "all_passed": bool(executed) and all(r["passed"] for r in executed),
        # ---- v2（§3.2）：分组 / 误差 / 效率三轴（informational） ----
        "by_tier": _group_by(task_reports, lambda r: r.get("tier", "guided")),
        "by_backend": _group_by(task_reports, lambda r: _expect_key(r, "backend", "behavior")),
        "by_job_type": _group_by(task_reports, lambda r: _expect_key(r, "job_types", "behavior")),
        "by_model": _group_by(task_reports, lambda r: _expect_key(r, "model", "-")),
        "error_stats": (
            {
                "n": len(abs_errors),
                "mean_abs_error": sum(abs_errors) / len(abs_errors),
                "max_abs_error": max(abs_errors),
            }
            if abs_errors
            else None
        ),
        "efficiency": (
            {
                "first_try_rate": (
                    sum(1 for e in first_tries if e["first_try"]) / len(first_tries)
                    if first_tries
                    else None
                ),
                "total_jobs": sum(e["actual_jobs"] for e in effs),
                "total_failed_jobs": sum(e["failed_jobs"] for e in effs),
            }
            if effs
            else None
        ),
    }


def judge_run_dir(run_dir: str | Path, tasks: list[dict] | None = None) -> dict:
    """判定 run_dir 下全部 run_record.json，写 judge_report.json 并返回报告。"""
    run_dir = Path(run_dir)
    tasks = tasks if tasks is not None else load_tasks()
    task_reports = []
    for task in tasks:
        record_path = run_dir / task["id"] / "run_record.json"
        if not record_path.is_file():
            task_reports.append(
                {
                    "task_id": task["id"],
                    "skipped": False,
                    "passed": False,
                    "execution_passed": False,
                    "numeric_passed": False,
                    "has_numeric": False,
                    "assertions": [],
                    "error": f"缺少 {record_path}",
                }
            )
            continue
        record = json.loads(record_path.read_text())
        task_reports.append(judge_task(task, record))
    report = {"run_dir": str(run_dir), "summary": summarize(task_reports), "tasks": task_reports}
    (run_dir / "judge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def _fmt_rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "N/A"


def _print_groups(title: str, groups: dict) -> None:
    if not groups:
        return
    print(f"  [{title}]")
    for key, g in groups.items():
        print(
            f"    {key}: 通过 {g['passed']}/{g['executed']}"
            f"，覆盖率 {_fmt_rate(g['execution_success_rate'])}"
            f"，数值正确率 {_fmt_rate(g['numeric_accuracy'])}"
        )


def print_report(report: dict) -> None:
    summary = report["summary"]
    for r in report["tasks"]:
        if r["skipped"]:
            print(f"  {r['task_id']}: SKIPPED ({r.get('note', '')})")
            continue
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['task_id']}: {mark}")
        for a in r["assertions"]:
            if not a["passed"]:
                print(f"      ✗ [{a['kind']}] {a['type']}: {a['detail']}")
        if r.get("error"):
            print(f"      ✗ {r['error']}")
    print(
        f"通过 {summary['passed']}/{summary['executed']}"
        + (f"，跳过 {summary['skipped']}" if summary["skipped"] else "")
    )
    if summary["execution_success_rate"] is not None:
        print(f"执行成功率: {summary['execution_success_rate']:.1%}")
    else:
        print("执行成功率: N/A（无执行任务）")
    if summary["numeric_accuracy"] is not None:
        print(f"数值正确率: {summary['numeric_accuracy']:.1%}")
    else:
        print("数值正确率: N/A（无数值断言任务）")
    # ---- v2（§3.2）：分层 / 误差 / 效率 ----
    _print_groups("按 tier", summary.get("by_tier") or {})
    _print_groups("按后端", summary.get("by_backend") or {})
    _print_groups("按任务类型", summary.get("by_job_type") or {})
    _print_groups("按模型", summary.get("by_model") or {})
    stats = summary.get("error_stats")
    if stats:
        print(
            f"数值偏差: mean={stats['mean_abs_error']:.3e}，max={stats['max_abs_error']:.3e}"
            f"（{stats['n']} 条 numeric 断言）"
        )
    eff = summary.get("efficiency")
    if eff:
        print(
            f"效率: 一次成功率 {_fmt_rate(eff['first_try_rate'])}"
            f"，总作业 {eff['total_jobs']}，失败作业 {eff['total_failed_jobs']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("用法: python benchmarks/judge.py <run_dir>", file=sys.stderr)
        return 2
    report = judge_run_dir(args[0])
    print_report(report)
    return 0 if report["summary"]["all_passed"] or report["summary"]["executed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
