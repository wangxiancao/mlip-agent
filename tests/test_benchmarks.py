"""M5 验收：benchmarks 体系测试（PLAN §4 M5）。

- judge 单测：构造假 run_record，验证各类断言的正误两路（无 GPU/LLM）；
- canned runner 集成：B9（无 GPU 澄清流）全链路 + judge 报告；
- B1 canned 端到端真实计算（@pytest.mark.gpu，无 GPU 自动 skip）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.config import Settings
from benchmarks import canned_scripts, judge, runner

# ---------- 假 run_record 工厂 ----------


def _base_record() -> dict:
    return {
        "task_id": "X",
        "mode": "canned",
        "status": "completed",
        "output": "计算完成。单点能 = -76.3883010000 Hartree = -2078.7 eV。",
        "error": None,
        "events": ["RUN_STARTED", "TOOL_CALL_START", "TEXT_MESSAGE_START", "MLIP_RUN_RESULT"],
        "tool_calls": ["build_structure", "delegate", "write_maple_inp", "run_maple", "convert_units"],
        "failed_tool_calls": [],
        "suspensions": [
            {
                "approvals": [{"tool_call_id": "c1", "tool_name": "run_maple", "args": None}],
                "calls": [],
                "jobs_snapshot": [{"job_id": "job-0001", "files": ["input.inp", "job_spec.json"]}],
            }
        ],
        "jobs": [
            {
                "job_id": "job-0001",
                "job_dir": "/nonexistent/job-0001",
                "backend": "maple",
                "job_type": "opt",
                "artifacts": ["input.inp", "input.out", "stdout.log"],
                "parsed": {
                    "job_type": "opt",
                    "energy_hartree": -76.388301,
                    "energies_hartree": [-76.388277, -76.388301],
                    "converged": True,
                    "converged_at_iteration": 2,
                    "final_xyz": (
                        "3\ncomment\n"
                        "O  -0.0018438588 -0.0021544126  0.0000000000\n"
                        "H   0.9619411619  0.0003554475  0.0000000000\n"
                        "H  -0.2400973030  0.9317989651  0.0000000000\n"
                    ),
                    "frequencies_cm1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1966.21, 4199.58, 4607.14],
                    "n_imaginary": 0,
                    "energy_drift_ha": 2.0e-4,
                    "argmin_index": 1,
                },
            }
        ],
    }


# ---------- judge 断言：正路 ----------


@pytest.mark.parametrize(
    "assertion",
    [
        {"type": "final_status", "value": "completed"},
        {"type": "event_present", "event": "TOOL_CALL_START"},
        {"type": "tool_called", "name": "run_maple"},
        {"type": "tool_not_called", "name": "run_nequip"},
        {"type": "job_count", "value": 1},
        {"type": "job_type", "job": 0, "value": "opt"},
        {"type": "backend", "job": 0, "value": "maple"},
        {"type": "artifact_present", "job": 0, "name": "input.out"},
        {"type": "output_contains", "text": "eV"},
        {"type": "suspended_with_tool", "value": "run_maple"},
        {"type": "no_artifact_at_suspension", "suspension": 0, "name": "input.out"},
        {"type": "converged", "job": 0, "value": True},
        {"type": "n_imaginary", "job": 0, "value": 0},
        {"type": "energy_hartree", "job": 0, "value": -76.388301, "tol": 1e-6},
        {"type": "frequencies_real", "job": 0, "values": [1966.21, 4199.58, 4607.14], "tol": 20.0},
        {"type": "scan_argmin", "job": 0, "value": 1},
        {"type": "energy_drift_max", "job": 0, "value": 1e-3},
        {"type": "md_steps", "job": 0, "value": 2},
        {
            "type": "geometry_h2o",
            "job": 0,
            "oh_angstrom": [0.963788, 0.963864],
            "oh_tol": 0.005,
            "angle_deg": 104.1618,
            "angle_tol": 1.0,
        },
    ],
)
def test_assertion_pass_paths(assertion):
    passed, detail = judge.evaluate_assertion(assertion, _base_record())
    assert passed, detail


# ---------- judge 断言：误路 ----------


@pytest.mark.parametrize(
    "assertion",
    [
        {"type": "final_status", "value": "failed"},
        {"type": "event_present", "event": "STATE_SNAPSHOT"},
        {"type": "tool_called", "name": "run_nequip"},
        {"type": "tool_not_called", "name": "run_maple"},
        {"type": "job_count", "value": 2},
        {"type": "job_type", "job": 0, "value": "sp"},
        {"type": "backend", "job": 0, "value": "nequip"},
        {"type": "artifact_present", "job": 0, "name": "traj.xyz"},
        {"type": "output_contains", "text": "kcal/mol"},
        {"type": "suspended_with_tool", "value": "ask_user_question"},
        {"type": "no_artifact_at_suspension", "suspension": 0, "name": "input.inp"},
        {"type": "converged", "job": 0, "value": False},
        {"type": "n_imaginary", "job": 0, "value": 1},
        # 容差外 1e-5 > 1e-6
        {"type": "energy_hartree", "job": 0, "value": -76.388311, "tol": 1e-6},
        # 频率偏差 25 > 20
        {"type": "frequencies_real", "job": 0, "values": [1941.21, 4199.58, 4607.14], "tol": 20.0},
        # 实频个数不符（少 pin 一个）
        {"type": "frequencies_real", "job": 0, "values": [1966.21, 4199.58], "tol": 20.0},
        {"type": "scan_argmin", "job": 0, "value": 0},
        # 漂移 2e-4 > 1e-4
        {"type": "energy_drift_max", "job": 0, "value": 1e-4},
        {"type": "md_steps", "job": 0, "value": 20},
        # 键角 pin 偏离 5° > 1°
        {
            "type": "geometry_h2o",
            "job": 0,
            "oh_angstrom": [0.963788, 0.963864],
            "oh_tol": 0.005,
            "angle_deg": 109.0,
            "angle_tol": 1.0,
        },
        {"type": "job_type", "job": 5, "value": "sp"},  # 作业不存在
    ],
)
def test_assertion_fail_paths(assertion):
    passed, detail = judge.evaluate_assertion(assertion, _base_record())
    assert not passed
    assert detail  # 失败必须给出原因


def test_energy_tolerance_boundary_is_inclusive():
    record = _base_record()
    # 恰好等于容差边界的判定为通过（<=）
    a = {"type": "energy_hartree", "job": 0, "value": -76.388301 + 1e-6, "tol": 1e-6}
    passed, _ = judge.evaluate_assertion(a, record)
    assert passed


def test_out_contains_whitespace_normalized(tmp_path):
    job_dir = tmp_path / "job-0001"
    job_dir.mkdir()
    (job_dir / "input.out").write_text(
        "Frequency analysis summary:\n  Real frequencies (ν > 5.0 cm⁻¹):       21\n"
    )
    record = _base_record()
    record["jobs"][0]["job_dir"] = str(job_dir)
    ok_a = {"type": "out_contains", "job": 0, "text": "Real frequencies (ν > 5.0 cm⁻¹): 21"}
    passed, detail = judge.evaluate_assertion(ok_a, record)
    assert passed, detail
    bad_a = {"type": "out_contains", "job": 0, "text": "Real frequencies (ν > 5.0 cm⁻¹): 18"}
    passed, _ = judge.evaluate_assertion(bad_a, record)
    assert not passed


def test_energy_eV_from_nequip_result():
    record = _base_record()
    record["jobs"] = [
        {
            "job_id": "job-0001",
            "job_dir": "/nonexistent",
            "backend": "nequip",
            "job_type": "sp",
            "artifacts": ["job.json", "result.json"],
            "result": {"ok": True, "energy_eV": -13.43729751141953},
        }
    ]
    good = {"type": "energy_eV", "job": 0, "value": -13.43729751141953, "tol": 1e-6}
    assert judge.evaluate_assertion(good, record)[0]
    bad = {"type": "energy_eV", "job": 0, "value": -13.4373, "tol": 1e-6}
    assert not judge.evaluate_assertion(bad, record)[0]


# ---------- judge 汇总口径 ----------


def test_summarize_rates_and_exit_semantics():
    tasks = [
        {"id": "T1", "assertions": [{"type": "final_status", "value": "completed"}]},
        {
            "id": "T2",
            "assertions": [
                {"type": "final_status", "value": "completed"},
                {"type": "energy_hartree", "job": 0, "value": -76.0, "tol": 1e-6, "kind": "numeric"},
            ],
        },
        {"id": "T3", "assertions": [{"type": "job_count", "value": 0}]},
    ]
    record_ok = _base_record()
    record_bad_num = _base_record()  # execution 过、numeric 不过
    record_skipped = {"task_id": "T3", "status": "skipped", "error": "SKIPPED: no LLM endpoint"}

    reports = [
        judge.judge_task(tasks[0], record_ok),
        judge.judge_task(tasks[1], record_bad_num),
        judge.judge_task(tasks[2], record_skipped),
    ]
    assert reports[0]["passed"] is True
    assert reports[1]["execution_passed"] and not reports[1]["numeric_passed"]
    assert reports[1]["passed"] is False
    assert reports[2]["skipped"] and reports[2]["passed"] is None

    summary = judge.summarize(reports)
    assert summary["executed"] == 2 and summary["skipped"] == 1
    # 执行成功率：T1/T2 execution 断言都过 → 2/2
    assert summary["execution_success_rate"] == 1.0
    # 数值正确率：仅 T2 含 numeric 且未过 → 0/1
    assert summary["numeric_accuracy"] == 0.0
    assert summary["all_passed"] is False


def test_judge_run_dir_missing_record_fails(tmp_path):
    report = judge.judge_run_dir(tmp_path, [{"id": "NOPE", "assertions": []}])
    assert report["tasks"][0]["passed"] is False
    assert "缺少" in report["tasks"][0]["error"]
    assert (tmp_path / "judge_report.json").is_file()


def test_tasks_yaml_covers_all_registered_tasks():
    tasks = judge.load_tasks()
    ids = [t["id"] for t in tasks]
    # B1–B28 全连续（B20/B21/B22 新模型、B23–B28 PBC 落地后补齐编号）
    assert sorted(ids, key=lambda s: int(s[1:])) == [f"B{i}" for i in range(1, 29)]
    for t in tasks:
        assert t["prompt"] and t["assertions"]
        assert t["tier"] in ("guided", "recovery", "unguided")
        assert isinstance(t["min_jobs"], int)
        assert t["id"] in canned_scripts.SCRIPTS
    # §3.2 tier 模型：B17/B18 recovery，B19 unguided，其余全部 guided
    tiers = {t["id"]: t["tier"] for t in tasks}
    assert tiers["B17"] == tiers["B18"] == "recovery"
    assert tiers["B19"] == "unguided"
    guided_ids = [f"B{i}" for i in range(1, 17)] + ["B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28"]
    assert all(tiers[b] == "guided" for b in guided_ids)


# ---------- v2 断言与报表（§3.2） ----------


def test_tool_call_failed_assertion():
    record = _base_record()
    record["failed_tool_calls"] = ["write_maple_inp"]
    passed, _ = judge.evaluate_assertion(
        {"type": "tool_call_failed", "name": "write_maple_inp"}, record
    )
    assert passed
    passed, detail = judge.evaluate_assertion(
        {"type": "tool_call_failed", "name": "run_maple"}, record
    )
    assert not passed and detail


def test_md_steps_max_assertion():
    record = _base_record()  # parsed 能量 2 条
    assert judge.evaluate_assertion({"type": "md_steps_max", "job": 0, "value": 19}, record)[0]
    assert not judge.evaluate_assertion({"type": "md_steps_max", "job": 0, "value": 1}, record)[0]


def test_negative_job_index_selects_last():
    record = _base_record()
    second = copy.deepcopy(record["jobs"][0])
    second["job_id"] = "job-0002"
    second["job_type"] = "freq"
    record["jobs"].append(second)
    passed, _ = judge.evaluate_assertion({"type": "job_type", "job": -1, "value": "freq"}, record)
    assert passed
    passed, _ = judge.evaluate_assertion({"type": "job_type", "job": -2, "value": "opt"}, record)
    assert passed
    assert not judge.evaluate_assertion({"type": "job_type", "job": -1, "value": "sp"}, record)[0]


def test_only_mode_skips_mismatched_mode():
    task = {"id": "T", "assertions": [{"type": "job_count", "value": 5, "only_mode": "canned"}]}
    record = _base_record()
    record["mode"] = "live"  # mode 不匹配 → only_mode 断言跳过
    report = judge.judge_task(task, record)
    assert report["passed"] and report["assertions"] == []
    record["mode"] = "canned"  # mode 匹配 → 正常求值（5 != 1 → fail）
    report = judge.judge_task(task, record)
    assert not report["passed"]


def test_judge_task_v2_fields_and_abs_error():
    task = {
        "id": "T",
        "tier": "recovery",
        "min_jobs": 1,
        "expect": {"backend": "maple", "job_types": ["opt"], "model": "ani2x"},
        "assertions": [
            {"type": "final_status", "value": "completed"},
            {"type": "energy_hartree", "job": 0, "value": -76.388301, "tol": 1e-6, "kind": "numeric"},
        ],
    }
    report = judge.judge_task(task, _base_record())
    assert report["tier"] == "recovery"
    assert report["expect"]["model"] == "ani2x"
    assert report["efficiency"] == {
        "min_jobs": 1,
        "actual_jobs": 1,
        "failed_jobs": 0,
        "first_try": True,
    }
    numeric = next(a for a in report["assertions"] if a["kind"] == "numeric")
    assert numeric["abs_error"] == 0.0


def test_summarize_v2_groups_error_stats_efficiency():
    tasks = [
        {
            "id": "T1",
            "tier": "guided",
            "min_jobs": 1,
            "expect": {"backend": "maple", "job_types": ["opt"], "model": "ani2x"},
            "assertions": [
                {"type": "final_status", "value": "completed"},
                {"type": "energy_hartree", "job": 0, "value": -76.388301, "tol": 1e-6, "kind": "numeric"},
            ],
        },
        {
            "id": "T2",
            "tier": "recovery",
            "min_jobs": 2,
            "expect": {"backend": "maple", "job_types": ["opt", "freq"], "model": "ani2x"},
            "assertions": [{"type": "final_status", "value": "completed"}],
        },
    ]
    reports = [judge.judge_task(t, _base_record()) for t in tasks]
    summary = judge.summarize(reports)
    assert set(summary["by_tier"]) == {"guided", "recovery"}
    assert summary["by_tier"]["guided"]["execution_success_rate"] == 1.0
    assert summary["by_backend"]["maple"]["executed"] == 2
    assert "opt+freq" in summary["by_job_type"]
    assert summary["by_model"]["ani2x"]["passed"] == 2
    assert summary["error_stats"]["n"] == 1
    assert summary["error_stats"]["max_abs_error"] == 0.0
    # T1 first_try（1==1）；T2 非 first_try（1 != 2）
    assert summary["efficiency"]["first_try_rate"] == 0.5
    assert summary["efficiency"]["total_jobs"] == 2


# ---------- canned runner 集成（无 GPU：B9 澄清流） ----------


async def test_canned_runner_b9_end_to_end(tmp_path):
    task = next(t for t in judge.load_tasks() if t["id"] == "B9")
    record = await runner.run_task(
        task, mode="canned", workspace=tmp_path / "B9" / "workspace", settings=Settings()
    )
    assert record["status"] == "completed"
    assert record["jobs"] == []
    assert record["suspensions"][0]["approvals"][0]["tool_name"] == "ask_user_question"
    assert "ask_user_question" in record["tool_calls"]
    assert "run_maple" not in record["tool_calls"]
    assert (tmp_path / "B9" / "events.jsonl").is_file()

    report = judge.judge_task(task, record)
    assert report["passed"], [a for a in report["assertions"] if not a["passed"]]


def test_runner_skips_live_without_llm_key(tmp_path, monkeypatch):
    # setenv "" 同时覆盖环境变量与本地 .env（pydantic-settings 优先级：env > .env）
    monkeypatch.setenv("MLIP_AGENT_LLM_API_KEY", "")
    monkeypatch.setenv("MLIP_AGENT_LLM_MODEL", "")
    rc = runner.main(["--mode=live", "--tasks", "B9", "--runs-dir", str(tmp_path)])
    assert rc == 0
    run_dir = next(d for d in tmp_path.iterdir() if d.is_dir())
    record = json.loads((run_dir / "B9" / "run_record.json").read_text())
    assert record["status"] == "skipped"
    assert record["error"] == runner.SKIP_NOTE
    report = json.loads((run_dir / "judge_report.json").read_text())
    assert report["tasks"][0]["skipped"]


@pytest.mark.gpu
async def test_canned_runner_b1_end_to_end(tmp_path):
    """B1 canned 全链路真实计算：能量 pin −383.4722385376 ±1e-6 Ha。"""
    task = next(t for t in judge.load_tasks() if t["id"] == "B1")
    record = await runner.run_task(
        task, mode="canned", workspace=tmp_path / "B1" / "workspace", settings=Settings()
    )
    assert record["status"] == "completed"
    assert len(record["jobs"]) == 1
    energy = record["jobs"][0]["parsed"]["energy_hartree"]
    assert energy == pytest.approx(-383.4722385376, abs=1e-6)
    report = judge.judge_task(task, record)
    assert report["passed"], [a for a in report["assertions"] if not a["passed"]]
