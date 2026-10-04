"""M1-B 验收测试：子进程跑 app/compute/scripts/nequip_ase.py sp 任务，
断言 result.json 能量与 benchmarks/references/B5.json 的 pin 值一致（±1e-6 eV）。

在 mlip_agent/.venv（Python 3.13，无 torch）下运行：
    .venv/bin/python -m pytest tests/test_nequip_script.py
计算由 COMPUTE_VENV 子进程完成，本环境不需要装 torch/nequip。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPUTE_VENV = Path(os.environ.get("COMPUTE_VENV", "/root/soft/ml_potential/.venv"))
CPY = COMPUTE_VENV / "bin" / "python"
SCRIPT = ROOT / "app" / "compute" / "scripts" / "nequip_ase.py"
REFDIR = ROOT / "benchmarks" / "references"

MODEL_PATH = os.environ.get(
    "NEQUIP_MODEL_PATH", "/root/soft/ml_potential/checkpoints/nequip/NequIP-OAM-XL-0.1.nequip.zip"
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def b5_ref() -> dict:
    ref = json.loads((REFDIR / "B5.json").read_text())
    if ref.get("status") == "failed" or "energy_eV" not in ref.get("values", {}):
        pytest.skip("B5 金标准生成失败，无 pin 值可比")
    return ref


@pytest.fixture(scope="module")
def sp_result(tmp_path_factory, b5_ref) -> dict:
    if not CPY.is_file():
        pytest.skip(f"COMPUTE_VENV python 不存在: {CPY}")
    if not Path(MODEL_PATH).is_file():
        pytest.skip(f"NequIP 模型文件不存在: {MODEL_PATH}")
    # GPU 可用性由脚本自身回退逻辑兜底；pin 值在 RTX 3090 cuda 下生成
    out_dir = tmp_path_factory.mktemp("nequip_sp")
    shutil.copy(REFDIR / "B5_h2o.xyz", out_dir / "h2o.xyz")
    job = {
        "xyz_path": str(out_dir / "h2o.xyz"),
        "task": "sp",
        "model_path": MODEL_PATH,
        "device": "cuda",
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
    }
    proc = subprocess.run(
        [str(CPY), str(SCRIPT), str(job_path)],
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


def test_energy_matches_b5_pin(sp_result, b5_ref):
    pinned = b5_ref["values"]["energy_eV"]
    tol = b5_ref["tolerance"]["energy_eV"]
    assert abs(sp_result["energy_eV"] - pinned) <= tol


def test_result_schema_and_units(sp_result):
    assert sp_result["units"]["energy"] == "eV"
    assert sp_result["units"]["forces"] == "eV/Angstrom"
    assert len(sp_result["forces_eV_per_A"]) == 3  # H2O 3 原子
    assert all(len(row) == 3 for row in sp_result["forces_eV_per_A"])
    assert sp_result["task"] == "sp"
    assert Path(sp_result["final_structure_path"]).is_file()
    assert sp_result["versions"]["nequip"] == "0.17.1"


def test_error_path_writes_result_json(tmp_path):
    """xyz 不存在时：result.json {ok:false} 且退出码非零。"""
    if not CPY.is_file():
        pytest.skip(f"COMPUTE_VENV python 不存在: {CPY}")
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
        [str(CPY), str(SCRIPT), str(job_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    result = json.loads((out_dir / "result.json").read_text())
    assert result["ok"] is False
    assert result["error"]
