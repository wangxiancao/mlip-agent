"""Benchmark v3 反应能垒套件判定器（PLAN §3.3）：反应感知的四维指标求值。

与 judge.py（断言级判定）分工：judge.py 只做最低执行检查；
本模块按 references/R*/ref.json 对 run_record.json 做反应感知求值——

- 物种归属：把 record.jobs 里的作业按（原子数、元素组成、charge/mult、
  与参考 xyz 的同序 RMSD）归属到 ref.json 的各物种 / TS；
- Coverage：opt 收敛（每分子物种）、freq 无虚频（每分子物种）、scan 曲线完整
  （≥5 点且能量上升）、TS freq 恰 1 虚频；
- MAE（kcal/mol，「总是计分」语义）：正向能垒 / 反向能垒 / 反应能，
  E(侧) = 该侧各物种优化（或单点）能量之和，Hartree × 627.5095；
- Attempts / Cost：actual_jobs vs min_jobs、duration_s；
- failure_analysis：每个未得分参考量给出原因与归因类别（映射 ladder Rule A–I）。

用法：
    python -m benchmarks.reaction.suite_judge <run_dir>
    suite_report = judge_suite(run_dir, tasks)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REACTION_DIR = Path(__file__).resolve().parent
HARTREE_TO_KCAL = 627.5095
_RMSD_MATCH_TOL = 1.0  # Å，同序坐标 RMSD 归属阈值

# failure_analysis 归因类别（映射 ladder.md Rule A–I）
_CATEGORY_RULES = [
    ("A", ("超时", "timeout", "exit", "退出码")),
    ("B", ("校验", "xyz_file", "JobSpec", "拒绝")),
    ("C", ("opt 未收敛", "opt 作业", "不收敛")),
    ("D", ("极小点", "虚频（极小", "未真正收敛")),
    ("E", ("扫描", "scan")),
    ("F", ("TS", "过渡态", "鞍点")),
    ("G", ("单位", "kcal", "Hartree")),
]


def _categorize(reason: str) -> str:
    for rule, keywords in _CATEGORY_RULES:
        if any(k in reason for k in keywords):
            return f"Rule {rule}"
    return "未分类"


# ---------- 物种归属 ----------


def _load_spec_structure(spec_path: Path) -> tuple[list[str], int | None, int | None]:
    """读 job_spec.json，返回（元素符号列表、charge、mult）。xyz_file 引用时读文件。"""
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, json.JSONDecodeError):
        return [], None, None
    structure = spec.get("structure") or {}
    coords = structure.get("coords")
    if coords:
        return [c["symbol"] for c in coords], spec.get("charge"), spec.get("multiplicity")
    xyz_file = structure.get("xyz_file")
    if xyz_file and Path(xyz_file).is_file():
        lines = [l.split() for l in Path(xyz_file).read_text().splitlines()[2:] if l.strip()]
        return [l[0] for l in lines], spec.get("charge"), spec.get("multiplicity")
    return [], spec.get("charge"), spec.get("multiplicity")


def _read_xyz_symbols_coords(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = [l.split() for l in path.read_text().splitlines()[2:] if l.strip()]
    return [l[0] for l in lines], [[float(v) for v in l[1:4]] for l in lines]


def _rmsd_same_order(a: list[list[float]], b: list[list[float]]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    return math.sqrt(
        sum(math.dist(pa, pb) ** 2 for pa, pb in zip(a, b)) / len(a)
    )


def _job_coord_signature(job: dict, ref_dir: Path) -> tuple[list[str], list[list[float]]]:
    """作业的（符号列表、坐标）：优先 structure.xyz 工件，其次 job_spec 内联/引用。"""
    job_dir = Path(job["job_dir"])
    structure_xyz = job_dir / "structure.xyz"
    if structure_xyz.is_file():
        return _read_xyz_symbols_coords(structure_xyz)
    spec_path = job_dir / "job_spec.json"
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text())
        except json.JSONDecodeError:
            return [], []
        structure = spec.get("structure") or {}
        coords = structure.get("coords")
        if coords:
            return (
                [c["symbol"] for c in coords],
                [[c["x"], c["y"], c["z"]] for c in coords],
            )
        xyz_file = structure.get("xyz_file")
        if xyz_file and Path(xyz_file).is_file():
            return _read_xyz_symbols_coords(Path(xyz_file))
    return [], []


def _match_species(
    job: dict, candidates: list[dict], ref_dir: Path
) -> dict | None:
    """把作业归属到 candidates（ref species 或 ts 条目）中坐标最接近的一个。"""
    symbols, coords = _job_coord_signature(job, ref_dir)
    if not symbols:
        return None
    best: tuple[float, dict] | None = None
    for cand in candidates:
        ref_symbols, ref_coords = _read_xyz_symbols_coords(ref_dir / cand["file"])
        if sorted(ref_symbols) != sorted(symbols):
            continue
        if cand.get("charge") is not None and job.get("spec_charge") is not None:
            if cand["charge"] != job["spec_charge"]:
                continue
        if ref_symbols != symbols:
            # 元素顺序不同（agent 重排过坐标）时无法用同序 RMSD，退化为组成匹配
            rmsd = 0.0
        else:
            rmsd = _rmsd_same_order(ref_coords, coords)
            if rmsd is None or rmsd > _RMSD_MATCH_TOL:
                continue
        if best is None or rmsd < best[0]:
            best = (rmsd, cand)
    return best[1] if best else None


# ---------- 单任务求值 ----------


def _energy_of(jobs: list[dict]) -> float | None:
    """优先 opt 收敛作业的能量，其次任意 opt/ts/sp 及任何带解析能量的作业。"""
    for pred in (
        lambda j: j.get("job_type") == "opt"
        and (j.get("parsed") or {}).get("converged") is True,
        lambda j: j.get("job_type") == "opt",
        lambda j: j.get("job_type") == "ts",
        lambda j: j.get("job_type") == "sp",
        lambda j: True,
    ):
        for j in jobs:
            if pred(j):
                e = (j.get("parsed") or {}).get("energy_hartree")
                if e is not None:
                    return e
    return None


def evaluate_task(task: dict, record: dict, ref_dir: Path) -> dict:
    ref = json.loads((ref_dir / "ref.json").read_text())
    jobs = record.get("jobs") or []
    for j in jobs:
        spec_path = Path(j["job_dir"]) / "job_spec.json"
        _, charge, _mult = _load_spec_structure(spec_path)
        j["spec_charge"] = charge

    ts_entry = dict(ref["ts"])
    ts_entry["role"] = "ts"
    assignments: dict[str, list[dict]] = {}  # file -> jobs
    for job in jobs:
        jt = job.get("job_type")
        if jt == "ts":
            cand = _match_species(job, [ts_entry], ref_dir)
        elif jt in ("opt", "sp", "freq"):
            cand = _match_species(job, ref["species"] + [ts_entry], ref_dir)
        else:
            cand = None
        if cand is not None:
            assignments.setdefault(cand["file"], []).append(job)

    quantities: list[dict] = []

    def add(name: str, kind: str, scored: bool, reason: str, abs_error: float | None = None, value: float | None = None):
        quantities.append(
            {
                "quantity": name,
                "kind": kind,
                "scored": scored,
                "reason": reason,
                "abs_error": abs_error,
                "value": value,
            }
        )

    molecular = [s for s in ref["species"] if s["n_atoms"] >= 2]
    for s in molecular:
        s_jobs = assignments.get(s["file"], [])
        opt_jobs = [j for j in s_jobs if j.get("job_type") == "opt"]
        converged = any(
            (j.get("parsed") or {}).get("converged") is True for j in opt_jobs
        )
        add(
            f"opt_converged[{s['file']}]",
            "coverage",
            converged,
            "opt 收敛" if converged else (
                "opt 作业未收敛" if opt_jobs else f"无 {s['file']} 的 opt 作业"
            ),
        )
        freq_jobs = [j for j in s_jobs if j.get("job_type") == "freq"]
        n_imag = [
            (j.get("parsed") or {}).get("n_imaginary") for j in freq_jobs
        ]
        n_imag = [n for n in n_imag if n is not None]
        ok_freq = any(n == 0 for n in n_imag)
        add(
            f"freq_no_imag[{s['file']}]",
            "coverage",
            ok_freq,
            "极小点 0 虚频" if ok_freq else (
                f"极小点虚频数异常（{n_imag}），几何可能未真正收敛"
                if n_imag else f"无 {s['file']} 的 freq 作业"
            ),
        )

    scan_jobs = [j for j in jobs if j.get("job_type") == "scan"]
    scan_ok = False
    scan_reason = "无 scan 作业"
    if scan_jobs:
        energies = (scan_jobs[-1].get("parsed") or {}).get("energies_hartree") or []
        if len(energies) >= 5 and energies[-1] > energies[0]:
            scan_ok = True
            scan_reason = f"扫描 {len(energies)} 点且能量上升"
        else:
            scan_reason = f"扫描曲线异常（{len(energies)} 点，末点能量未高于起点）"
    add("scan_curve", "coverage", scan_ok, scan_reason)

    ts_jobs = assignments.get("ts.xyz", [])
    ts_freq_jobs = [j for j in ts_jobs if j.get("job_type") == "freq"]
    ts_freq_n = [
        (j.get("parsed") or {}).get("n_imaginary") for j in ts_freq_jobs
    ]
    ts_freq_n = [n for n in ts_freq_n if n is not None]
    ok_ts_freq = any(n == 1 for n in ts_freq_n)
    add(
        "ts_freq_one_imag",
        "coverage",
        ok_ts_freq,
        "TS 恰 1 虚频" if ok_ts_freq else (
            f"TS 虚频数异常（{ts_freq_n}），可能收敛到高阶鞍点或极小点"
            if ts_freq_n else "无 TS 的 freq 作业"
        ),
    )

    # ---- MAE：正/反向能垒与反应能（kcal/mol，总是计分） ----
    e_ts = _energy_of([j for j in ts_jobs if j.get("job_type") == "ts"])
    reactant_e = {
        s["file"]: _energy_of(assignments.get(s["file"], []))
        for s in ref["species"]
        if s["role"] == "reactant"
    }
    product_e = {
        s["file"]: _energy_of(assignments.get(s["file"], []))
        for s in ref["species"]
        if s["role"] == "product"
    }
    barriers = ref["barriers_kcal"]

    def mae_quantity(name: str, computed: float | None, ref_value: float, missing: list[str]):
        if computed is None:
            add(name, "mae", False, f"无法计算 {name}（缺 {', '.join(missing)}）")
            return
        err = abs(computed - ref_value)
        add(
            name,
            "mae",
            True,
            f"{name}={computed:.2f} kcal/mol（参考 {ref_value}，Δ={err:.2f}）",
            abs_error=err,
            value=computed,
        )

    missing_r = [f for f, e in reactant_e.items() if e is None]
    missing_p = [f for f, e in product_e.items() if e is None]
    e_react = sum(e for e in reactant_e.values() if e is not None) if not missing_r else None
    e_prod = sum(e for e in product_e.values() if e is not None) if not missing_p else None

    mae_quantity(
        "barrier_forward",
        (e_ts - e_react) * HARTREE_TO_KCAL if e_ts is not None and e_react is not None else None,
        barriers["forward"],
        (["TS 能量"] if e_ts is None else []) + missing_r,
    )
    mae_quantity(
        "barrier_reverse",
        (e_ts - e_prod) * HARTREE_TO_KCAL if e_ts is not None and e_prod is not None else None,
        barriers["reverse"],
        (["TS 能量"] if e_ts is None else []) + missing_p,
    )
    mae_quantity(
        "reaction_energy",
        (e_prod - e_react) * HARTREE_TO_KCAL if e_prod is not None and e_react is not None else None,
        barriers["reaction_energy"],
        missing_r + missing_p,
    )

    metrics = record.get("metrics") or {}
    actual_jobs = metrics.get("n_jobs", len(jobs))
    min_jobs = task.get("min_jobs")
    return {
        "task_id": task["id"],
        "tier": task.get("tier"),
        "reaction": task.get("reaction"),
        "skipped": record.get("status") == "skipped",
        "quantities": quantities,
        "n_scored": sum(1 for q in quantities if q["scored"]),
        "n_total": len(quantities),
        "attempts": {
            "min_jobs": min_jobs,
            "actual_jobs": actual_jobs,
            "failed_jobs": metrics.get("n_failed_jobs", 0),
            "first_try": (actual_jobs == min_jobs) if min_jobs is not None else None,
        },
        "cost": {
            "duration_s": metrics.get("duration_s", record.get("duration_s")),
            "n_jobs": actual_jobs,
        },
    }


# ---------- 汇总与 failure analysis ----------


def _aggregate(reports: list[dict]) -> dict:
    scored = sum(r["n_scored"] for r in reports)
    total = sum(r["n_total"] for r in reports)
    errors = [
        q["abs_error"]
        for r in reports
        for q in r["quantities"]
        if q["kind"] == "mae" and q["scored"] and q["abs_error"] is not None
    ]
    attempts = [r["attempts"] for r in reports if r["attempts"]["min_jobs"] is not None]
    first_tries = [a for a in attempts if a["first_try"] is not None]
    return {
        "n_tasks": len(reports),
        "coverage": (scored / total) if total else None,
        "scored": scored,
        "total_quantities": total,
        "mae_kcal": (
            {
                "n": len(errors),
                "mean": sum(errors) / len(errors),
                "max": max(errors),
            }
            if errors
            else None
        ),
        "attempts": (
            {
                "mean_jobs": sum(a["actual_jobs"] for a in attempts) / len(attempts),
                "first_try_rate": (
                    sum(1 for a in first_tries if a["first_try"]) / len(first_tries)
                    if first_tries
                    else None
                ),
                "total_failed_jobs": sum(a["failed_jobs"] for a in attempts),
            }
            if attempts
            else None
        ),
        "cost": {
            "total_duration_s": sum(
                r["cost"]["duration_s"] or 0 for r in reports
            ),
            "total_jobs": sum(r["cost"]["n_jobs"] for r in reports),
        },
    }


def _failure_analysis(reports: list[dict]) -> tuple[list[dict], str]:
    items: list[dict] = []
    for r in reports:
        for q in r["quantities"]:
            if not q["scored"]:
                items.append(
                    {
                        "task_id": r["task_id"],
                        "tier": r["tier"],
                        "reaction": r["reaction"],
                        "quantity": q["quantity"],
                        "reason": q["reason"],
                        "category": _categorize(q["reason"]),
                    }
                )
    counts: dict[str, int] = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    if not items:
        discussion = "全部参考量均得分，无失败项。各档位 agent 均完成了完整六段链。"
    else:
        lines = [
            f"本次运行共 {len(items)} 个未得分参考量，归因分布："
            + "、".join(f"{k} {v} 项" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
            + "。"
        ]
        by_tier: dict[str, int] = {}
        for it in items:
            by_tier[it["tier"]] = by_tier.get(it["tier"], 0) + 1
        lines.append(
            "按档位："
            + "、".join(f"{k} {v} 项" for k, v in sorted(by_tier.items()))
            + "。"
        )
        top = sorted(counts.items(), key=lambda x: -x[1])[0]
        lines.append(
            f"主导失败模式为 {top[0]}（{top[1]} 项）。"
            "对照决策阶梯，该类失败应优先按对应规则的诊断路径修复；"
            "若 decision 档较 baseline 档此类失败减少，说明阶梯知识对该失败模式有效。"
        )
        discussion = "\n".join(lines)
    return items, discussion


def judge_suite(run_dir: str | Path, tasks: list[dict]) -> dict:
    """对 run_dir 下 reaction 套件的全部 run_record.json 求值，写 suite_report.json。"""
    run_dir = Path(run_dir)
    reports = []
    for task in tasks:
        record_path = run_dir / task["id"] / "run_record.json"
        ref_dir = REACTION_DIR / "references" / task["reaction"]
        if not record_path.is_file():
            reports.append(
                {
                    "task_id": task["id"],
                    "tier": task.get("tier"),
                    "reaction": task.get("reaction"),
                    "skipped": True,
                    "quantities": [],
                    "n_scored": 0,
                    "n_total": 0,
                    "attempts": {"min_jobs": task.get("min_jobs"), "actual_jobs": 0, "failed_jobs": 0, "first_try": None},
                    "cost": {"duration_s": None, "n_jobs": 0},
                    "error": f"缺少 {record_path}",
                }
            )
            continue
        record = json.loads(record_path.read_text())
        if record.get("status") == "skipped":
            reports.append(
                {
                    "task_id": task["id"],
                    "tier": task.get("tier"),
                    "reaction": task.get("reaction"),
                    "skipped": True,
                    "quantities": [],
                    "n_scored": 0,
                    "n_total": 0,
                    "attempts": {"min_jobs": task.get("min_jobs"), "actual_jobs": 0, "failed_jobs": 0, "first_try": None},
                    "cost": {"duration_s": None, "n_jobs": 0},
                }
            )
            continue
        reports.append(evaluate_task(task, record, ref_dir))

    executed = [r for r in reports if not r["skipped"]]
    by_tier: dict[str, list[dict]] = {}
    by_reaction: dict[str, list[dict]] = {}
    for r in executed:
        by_tier.setdefault(r["tier"] or "unknown", []).append(r)
        by_reaction.setdefault(r["reaction"] or "unknown", []).append(r)
    failures, discussion = _failure_analysis(executed)
    report = {
        "run_dir": str(run_dir),
        "summary": {
            "overall": _aggregate(executed),
            "by_tier": {k: _aggregate(v) for k, v in sorted(by_tier.items())},
            "by_reaction": {k: _aggregate(v) for k, v in sorted(by_reaction.items())},
        },
        "failure_analysis": failures,
        "discussion": discussion,
        "tasks": reports,
    }
    (run_dir / "suite_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def _fmt_agg(name: str, agg: dict) -> str:
    cov = f"{agg['coverage']:.1%}" if agg["coverage"] is not None else "N/A"
    mae = (
        f"MAE {agg['mae_kcal']['mean']:.2f}（max {agg['mae_kcal']['max']:.2f}）kcal/mol"
        if agg["mae_kcal"]
        else "MAE N/A"
    )
    att = (
        f"平均作业 {agg['attempts']['mean_jobs']:.1f}，一次成功率 "
        + (
            f"{agg['attempts']['first_try_rate']:.0%}"
            if agg["attempts"]["first_try_rate"] is not None
            else "N/A"
        )
        if agg["attempts"]
        else "attempts N/A"
    )
    cost = f"{agg['cost']['total_duration_s']:.0f}s / {agg['cost']['total_jobs']} 作业"
    return f"    {name}: 覆盖率 {cov}（{agg['scored']}/{agg['total_quantities']}），{mae}，{att}，成本 {cost}"


def print_suite_report(report: dict) -> None:
    s = report["summary"]
    print("== 反应套件四维指标（PLAN §3.3） ==")
    print(_fmt_agg("overall", s["overall"]))
    for tier, agg in s["by_tier"].items():
        print(_fmt_agg(f"tier={tier}", agg))
    for rxn, agg in s["by_reaction"].items():
        print(_fmt_agg(f"reaction={rxn}", agg))
    print("== 失败归因讨论（reasoning for failure） ==")
    print(report["discussion"])


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("用法: python -m benchmarks.reaction.suite_judge <run_dir>", file=sys.stderr)
        return 2
    from benchmarks import judge

    report = judge_suite(args[0], judge.load_tasks(REACTION_DIR / "tasks.yaml"))
    print_suite_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
