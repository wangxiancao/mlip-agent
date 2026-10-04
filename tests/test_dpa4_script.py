"""TICKET-0006 验收测试：子进程跑 app/compute/scripts/dpa4_ase.py sp 任务，
断言 result.json 能量与 benchmarks/references/B16.json 的 pin 值一致（±1e-6 eV，
CPU+OMP=1 逐位口径）。

在 mlip_agent/.venv（Python 3.13，无 torch）下运行：
    .venv/bin/python -m pytest tests/test_dpa4_script.py
计算由 DPA-4 常驻 venv（/root/soft/ml_potential/dpa4_venv）子进程完成，
本环境不需要装 torch/deepmd-kit；子进程必须带 LD_LIBRARY_PATH=/usr/local/lib64
（前置条件 1，run_dpa4 工具层注入同款）。
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
    "DPA4_MODEL_PATH", "/root/soft/ml_potential/checkpoints/dpa4/DPA4-beta.pt"
)


@pytest.fixture(scope="module")
def b16_ref() -> dict:
    ref = json.loads((REFDIR / "B16.json").read_text())
    if ref.get("status") == "failed" or "energy_eV" not in ref.get("values", {}):
        pytest.skip("B16 金标准生成失败，无 pin 值可比")
    return ref


@pytest.fixture(scope="module")
def sp_result(tmp_path_factory, b16_ref) -> dict:
    if not DPY.is_file():
        pytest.skip(f"DPA4_VENV python 不存在: {DPY}")
    if not Path(MODEL_PATH).is_file():
        pytest.skip(f"DPA-4 模型文件不存在: {MODEL_PATH}")
    # device=cpu：CPU+OMP=1 逐位 pin 口径（TICKET-0006 裁定 2），无需 GPU
    out_dir = tmp_path_factory.mktemp("dpa4_sp")
    shutil.copy(REFDIR / "B5_h2o.xyz", out_dir / "h2o.xyz")
    job = {
        "xyz_path": str(out_dir / "h2o.xyz"),
        "task": "sp",
        "model_path": MODEL_PATH,
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    job_path = out_dir / "job.json"
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
        cwd=out_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    result_path = out_dir / "result.json"
    assert result_path.is_file(), f"未产出 result.json；stderr: {proc.stderr[-1000:]}"
    result = json.loads(result_path.read_text())
    assert proc.returncode == 0 and result["ok"], f"任务失败: {result.get('error')}"
    return result


def test_energy_matches_b16_pin(sp_result, b16_ref):
    pinned = b16_ref["values"]["energy_eV"]
    tol = b16_ref["tolerance"]["energy_eV"]
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


def test_head_default_and_charge_spin_fixed(sp_result):
    """TICKET-0006 前置条件 3：head 默认 H2O_H2O_PD；charge_spin 固定 [0.0,1.0]。"""
    assert sp_result["head"] == "H2O_H2O_PD"
    assert sp_result["charge_spin"] == [0.0, 1.0]


def test_cpu_mode_threads_pinned_to_one(sp_result):
    """裁定 2：CPU 模式进程内强制 OMP_NUM_THREADS=1 + torch.set_num_threads(1)。"""
    versions = sp_result["versions"]
    assert versions["device"] == "cpu"
    assert versions["omp_num_threads"] == "1"
    assert versions["torch_num_threads"] == 1


def test_versions_contain_required_keys(sp_result):
    """前置条件 3：versions 记录 deepmd-kit/torch/e3nn/vesin/device/线程数。"""
    versions = sp_result["versions"]
    for key in ("deepmd-kit", "torch", "e3nn", "vesin", "device"):
        assert key in versions and versions[key]


def test_error_path_writes_result_json(tmp_path):
    """xyz 不存在时：result.json {ok:false} 且退出码非零。"""
    if not DPY.is_file():
        pytest.skip(f"DPA4_VENV python 不存在: {DPY}")
    out_dir = tmp_path / "bad"
    job = {
        "xyz_path": str(tmp_path / "nonexistent.xyz"),
        "task": "sp",
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
