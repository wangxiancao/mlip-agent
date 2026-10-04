"""benchmarks/reaction/suite_judge.py 测试（PLAN §3.3 WP4，GPU-free）。

- 假 run_record：作业目录含 job_spec.json / structure.xyz（取自真实 ref.json 的
  R1 参考几何），验证物种归属、四维指标、failure_analysis；
- runner --suite reaction 管道：live 无 key → skipped 记录 + suite_report 生成；
- decision 档 prompt 注入 ladder.md。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.config import Settings
from benchmarks import judge, runner
from benchmarks.reaction import suite_judge

REACTION_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "reaction"
H2K = 627.5095


def _make_job(tmp_path: Path, job_id: str, job_type: str, ref_xyz: Path, charge: int, mult: int, parsed: dict) -> dict:
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ref_xyz, job_dir / "structure.xyz")
    symbols, _ = suite_judge._read_xyz_symbols_coords(ref_xyz)
    (job_dir / "job_spec.json").write_text(
        json.dumps(
            {
                "job_type": job_type,
                "charge": charge,
                "multiplicity": mult,
                "structure": {"xyz_file": str(ref_xyz)},
            }
        )
    )
    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "backend": "maple",
        "job_type": job_type,
        "artifacts": ["input.out", "job_spec.json", "structure.xyz"],
        "parsed": parsed,
    }


@pytest.fixture()
def r1_record(tmp_path):
    """R1（HCN→HNC）全得分假 record：能量构造为与参考能垒完全一致。"""
    ref_dir = REACTION_DIR / "references" / "R1"
    ref = json.loads((ref_dir / "ref.json").read_text())
    e_r = -93.0
    e_p = e_r + ref["barriers_kcal"]["reaction_energy"] / H2K
    e_ts = e_r + ref["barriers_kcal"]["forward"] / H2K
    jobs = [
        _make_job(tmp_path, "j1", "opt", ref_dir / "r1.xyz", 0, 1,
                  {"converged": True, "energy_hartree": e_r}),
        _make_job(tmp_path, "j2", "freq", ref_dir / "r1.xyz", 0, 1,
                  {"n_imaginary": 0, "frequencies_cm1": [1.0, 2.0, 3.0]}),
        _make_job(tmp_path, "j3", "opt", ref_dir / "p1.xyz", 0, 1,
                  {"converged": True, "energy_hartree": e_p}),
        _make_job(tmp_path, "j4", "freq", ref_dir / "p1.xyz", 0, 1,
                  {"n_imaginary": 0, "frequencies_cm1": [1.0, 2.0, 3.0]}),
        _make_job(tmp_path, "j5", "scan", ref_dir / "r1.xyz", 0, 1,
                  {"energies_hartree": [e_r + 0.001 * i for i in range(6)]}),
        _make_job(tmp_path, "j6", "ts", ref_dir / "ts.xyz", 0, 1,
                  {"converged": True, "energy_hartree": e_ts}),
        _make_job(tmp_path, "j7", "freq", ref_dir / "ts.xyz", 0, 1,
                  {"n_imaginary": 1, "frequencies_cm1": [-100.0, 2.0, 3.0]}),
    ]
    task = next(
        t for t in judge.load_tasks(REACTION_DIR / "tasks.yaml") if t["id"] == "R1-base"
    )
    record = {
        "task_id": "R1-base",
        "mode": "canned",
        "status": "completed",
        "jobs": jobs,
        "metrics": {"n_jobs": 7, "n_failed_jobs": 0, "duration_s": 123.0},
    }
    return task, record, ref_dir


def test_evaluate_task_full_score(r1_record):
    task, record, ref_dir = r1_record
    report = suite_judge.evaluate_task(task, record, ref_dir)
    assert report["n_scored"] == report["n_total"]
    mae = {q["quantity"]: q for q in report["quantities"] if q["kind"] == "mae"}
    assert mae["barrier_forward"]["abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert mae["barrier_reverse"]["abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert mae["reaction_energy"]["abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert mae["barrier_forward"]["value"] == pytest.approx(48.1, abs=1e-3)
    assert report["attempts"]["first_try"] is True
    assert report["cost"]["duration_s"] == 123.0


def test_evaluate_task_mae_measured_in_kcal(r1_record):
    task, record, ref_dir = r1_record
    # 把 TS 能量抬高 1 kcal/mol → 正/反向能垒偏差都应是 1.0
    ts_job = next(j for j in record["jobs"] if j["job_type"] == "ts")
    ts_job["parsed"]["energy_hartree"] += 1.0 / H2K
    report = suite_judge.evaluate_task(task, record, ref_dir)
    mae = {q["quantity"]: q for q in report["quantities"] if q["kind"] == "mae"}
    assert mae["barrier_forward"]["abs_error"] == pytest.approx(1.0, abs=1e-3)
    assert mae["barrier_reverse"]["abs_error"] == pytest.approx(1.0, abs=1e-3)
    # 反应能不涉及 TS → 仍 0；MAE 量「总是计分」不因偏差大而 unscored
    assert mae["reaction_energy"]["abs_error"] == pytest.approx(0.0, abs=1e-3)
    assert all(q["scored"] for q in mae.values())


def test_evaluate_task_missing_jobs_unscored(r1_record):
    task, record, ref_dir = r1_record
    # 删掉产物 opt 与全部 freq → 对应 coverage 项 unscored，产物相关 MAE unscored
    record["jobs"] = [
        j for j in record["jobs"] if not (j["job_type"] == "opt" and "j3" in j["job_id"])
    ]
    record["jobs"] = [j for j in record["jobs"] if j["job_type"] != "freq"]
    report = suite_judge.evaluate_task(task, record, ref_dir)
    unscored = {q["quantity"] for q in report["quantities"] if not q["scored"]}
    assert "opt_converged[p1.xyz]" in unscored
    assert "freq_no_imag[r1.xyz]" in unscored
    assert "ts_freq_one_imag" in unscored
    assert "barrier_reverse" in unscored  # 缺产物能量
    assert "reaction_energy" in unscored
    # 正向能垒不依赖产物 → 仍 scored
    mae = {q["quantity"]: q for q in report["quantities"] if q["kind"] == "mae"}
    assert mae["barrier_forward"]["scored"]


def test_judge_suite_aggregate_and_failure_analysis(tmp_path, r1_record):
    task, record, _ = r1_record
    # 一个全得分任务 + 一个缺作业任务
    bad = json.loads(json.dumps(record))
    bad["jobs"] = [j for j in bad["jobs"] if j["job_type"] != "freq"]
    run_dir = tmp_path / "run"
    for tid, rec in (("R1-base", record), ("R1-dec", bad)):
        d = run_dir / tid
        d.mkdir(parents=True)
        (d / "run_record.json").write_text(json.dumps(rec))
    tasks = [
        t for t in judge.load_tasks(REACTION_DIR / "tasks.yaml") if t["id"] in ("R1-base", "R1-dec")
    ]
    report = suite_judge.judge_suite(run_dir, tasks)
    s = report["summary"]
    assert s["overall"]["coverage"] is not None
    assert s["overall"]["mae_kcal"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert set(s["by_tier"]) == {"baseline", "decision"}
    assert report["failure_analysis"]
    assert "freq" in report["failure_analysis"][0]["quantity"]
    assert (run_dir / "suite_report.json").is_file()
    # 全得分时讨论段应为「无失败项」
    for tid in ("R1-dec",):
        (run_dir / tid / "run_record.json").write_text(json.dumps(record))
    report2 = suite_judge.judge_suite(run_dir, tasks)
    assert "无失败项" in report2["discussion"]


def test_suite_runner_live_skip_and_ladder_injection(tmp_path, monkeypatch):
    """--suite reaction：live 无 key → skipped；decision 档 prompt 注入 ladder。"""
    monkeypatch.setenv("MLIP_AGENT_LLM_API_KEY", "")
    monkeypatch.setenv("MLIP_AGENT_LLM_MODEL", "")
    captured = {}
    orig_write = runner._write_skipped_records

    def fake_write_skipped(run_dir, tasks, mode):
        captured["prompts"] = {t["id"]: t["prompt"] for t in tasks}
        orig_write(run_dir, tasks, mode)

    monkeypatch.setattr(runner, "_write_skipped_records", fake_write_skipped)
    rc = runner.main(
        ["--mode=live", "--suite=reaction", "--tasks=R1-base,R1-dec", "--runs-dir", str(tmp_path)]
    )
    assert rc == 0
    assert "决策阶梯" not in captured["prompts"]["R1-base"]
    assert "Rule A" in captured["prompts"]["R1-dec"]
    run_dir = next(d for d in tmp_path.iterdir() if d.is_dir())
    assert (run_dir / "suite_report.json").is_file()
