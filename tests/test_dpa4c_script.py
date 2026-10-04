"""TICKET-0006 任务 2 验收测试：DPA4C 模型族（family="dpa4c"）子进程 sp 任务。

断言 result.json 能量与 benchmarks/references/B21.json 的 pin 值一致
（±1e-6 eV，CPU+OMP=1 逐位口径）。
注意 DPA4C-OMol25 输出总能量（H2O ≈ -2080 eV 量级），与 DPA4-beta B16 pin
（-15.71 eV，参考能约定不同）绝对值不可直接比较，只做量级合理性旁证。

在 mlip_agent/.venv（Python 3.13，无 torch）下运行：
    .venv/bin/python -m pytest tests/test_dpa4c_script.py
计算由 DPA-4 常驻 venv（/root/soft/ml_potential/dpa4_venv）子进程完成，
本环境不需要装 torch/deepmd-kit；子进程必须带 LD_LIBRARY_PATH=/usr/local/lib64。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DPA4_VENV = Path(os.environ.get("DPA4_VENV", "/root/soft/ml_potential/dpa4_venv"))
DPY = DPA4_VENV / "bin" / "python"
SCRIPT = ROOT / "app" / "compute" / "scripts" / "dpa4_ase.py"
REFDIR = ROOT / "benchmarks" / "references"

MODEL_PATH = os.environ.get(
    "DPA4C_MODEL_PATH",
    "/root/soft/ml_potential/checkpoints/dpa4/DPA4C-Air-OMol25-100M-v20260820.pt",
)


@pytest.fixture(scope="module")
def b21_ref() -> dict:
    ref = json.loads((REFDIR / "B21.json").read_text())
    if ref.get("status") == "failed" or "energy_eV" not in ref.get("values", {}):
        pytest.skip("B21 金标准生成失败，无 pin 值可比")
    return ref


def _run_job(job: dict, work_dir: Path) -> tuple[subprocess.CompletedProcess, dict]:
    job_path = work_dir / "job.json"
    job_path.write_text(json.dumps(job))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # 前置条件 1：gcc 11.3 编译产物需 GLIBCXX_3.4.29
        "LD_LIBRARY_PATH": "/usr/local/lib64",
    }
    proc = subprocess.run(
        [str(DPY), str(SCRIPT), str(job_path)],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    result_path = Path(job["output_dir"]) / "result.json"
    assert result_path.is_file(), f"未产出 result.json；stderr: {proc.stderr[-1000:]}"
    return proc, json.loads(result_path.read_text())


@pytest.fixture(scope="module")
def sp_result(tmp_path_factory) -> dict:
    if not DPY.is_file():
        pytest.skip(f"DPA4_VENV python 不存在: {DPY}")
    if not Path(MODEL_PATH).is_file():
        pytest.skip(f"DPA4C 模型文件不存在: {MODEL_PATH}")
    # device=cpu：CPU+OMP=1 逐位 pin 口径（TICKET-0006 裁定 2），无需 GPU
    out_dir = tmp_path_factory.mktemp("dpa4c_sp")
    shutil.copy(REFDIR / "B5_h2o.xyz", out_dir / "h2o.xyz")
    job = {
        "xyz_path": str(out_dir / "h2o.xyz"),
        "task": "sp",
        "family": "dpa4c",
        "model_path": MODEL_PATH,
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    proc, result = _run_job(job, out_dir)
    assert proc.returncode == 0 and result["ok"], f"任务失败: {result.get('error')}"
    return result


def test_energy_matches_pin(sp_result, b21_ref):
    pinned = b21_ref["values"]["energy_eV"]
    tol = b21_ref["tolerance"]["energy_eV"]
    assert abs(sp_result["energy_eV"] - pinned) <= tol


def test_result_schema_and_units(sp_result):
    assert sp_result["units"]["energy"] == "eV"
    assert sp_result["units"]["forces"] == "eV/Angstrom"
    assert sp_result["units"]["length"] == "Angstrom"
    assert len(sp_result["forces_eV_per_A"]) == 3  # H2O 3 原子
    assert all(len(row) == 3 for row in sp_result["forces_eV_per_A"])
    assert sp_result["task"] == "sp"
    assert sp_result["device"] == "cpu"
    assert Path(sp_result["final_structure_path"]).is_file()
    assert sp_result["versions"]["deepmd-kit"] == "3.2.0"


def test_family_dpa4c_and_head_null(sp_result):
    """family 显式 dpa4c；单域模型无 head，result.json 中 head 恒为 null。"""
    assert sp_result["family"] == "dpa4c"
    assert sp_result["head"] is None
    assert sp_result["charge_spin"] == [0.0, 1.0]


def test_cpu_mode_threads_pinned_to_one(sp_result):
    """裁定 2：CPU 模式进程内强制 OMP_NUM_THREADS=1 + torch.set_num_threads(1)。"""
    versions = sp_result["versions"]
    assert versions["device"] == "cpu"
    assert versions["omp_num_threads"] == "1"
    assert versions["torch_num_threads"] == 1


def test_versions_contain_required_keys(sp_result):
    versions = sp_result["versions"]
    for key in ("deepmd-kit", "torch", "e3nn", "vesin", "device"):
        assert key in versions and versions[key]


def test_dpa4c_rejects_head(tmp_path):
    """family=dpa4c 显式传 head 必须报错（deepmd 会静默忽略 head，显式拒绝防误用）。"""
    if not DPY.is_file():
        pytest.skip(f"DPA4_VENV python 不存在: {DPY}")
    if not Path(MODEL_PATH).is_file():
        pytest.skip(f"DPA4C 模型文件不存在: {MODEL_PATH}")
    out_dir = tmp_path / "with_head"
    shutil.copy(REFDIR / "B5_h2o.xyz", tmp_path / "h2o.xyz")
    job = {
        "xyz_path": str(tmp_path / "h2o.xyz"),
        "task": "sp",
        "family": "dpa4c",
        "head": "H2O_H2O_PD",
        "model_path": MODEL_PATH,
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    proc, result = _run_job(job, tmp_path)
    assert proc.returncode != 0
    assert result["ok"] is False
    assert "head" in result["error"]


def test_error_path_writes_result_json(tmp_path):
    """xyz 不存在时：result.json {ok:false} 且退出码非零。"""
    if not DPY.is_file():
        pytest.skip(f"DPA4_VENV python 不存在: {DPY}")
    out_dir = tmp_path / "bad"
    job = {
        "xyz_path": str(tmp_path / "nonexistent.xyz"),
        "task": "sp",
        "family": "dpa4c",
        "model_path": MODEL_PATH,
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    job_path = tmp_path / "bad_job.json"
    job_path.write_text(json.dumps(job))
    proc = subprocess.run(
        [str(DPY), str(SCRIPT), str(job_path)],
        env={"LD_LIBRARY_PATH": "/usr/local/lib64", "PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0
    result = json.loads((out_dir / "result.json").read_text())
    assert result["ok"] is False
    assert result["error"]
