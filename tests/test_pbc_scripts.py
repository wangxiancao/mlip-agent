"""直算脚本 PBC 接线测试（2026-10-03 PBC 落地，任务 4a）。

覆盖：
- tace_ase._select_neighborlist_backend：分子→ase、全周期→matscipy、
  混合 pbc / 带胞非周期 / 无胞周期 三类输入硬报错（TACE 静默升级坑的防线）。
- dpa4_ase / nequip_ase / tace_ase 的 pbc-无胞校验错误路径（fail-fast，
  在模型加载前报错，子进程写 result.json {ok:false}）。

在 mlip_agent/.venv（Python 3.13，无 torch/ase）下运行：
    .venv/bin/python -m pytest tests/test_pbc_scripts.py
子进程用 COMPUTE_VENV python 执行（tace_ase.py 模块级只依赖 stdlib + ase，
可直接 import；错误路径在模型加载前触发，无需 GPU/模型文件）。
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPUTE_VENV = Path(os.environ.get("COMPUTE_VENV", "/root/soft/ml_potential/.venv"))
CPY = COMPUTE_VENV / "bin" / "python"
DPA4_VENV = Path(os.environ.get("DPA4_VENV", "/root/soft/ml_potential/dpa4_venv"))
DPY = DPA4_VENV / "bin" / "python"
SCRIPTS = ROOT / "app" / "compute" / "scripts"

# 各脚本对应的解释器（dpa4_ase.py 依赖 deepmd，只能用 dpa4_venv）
SCRIPT_PYTHON = {
    "nequip_ase.py": CPY,
    "tace_ase.py": CPY,
    "dpa4_ase.py": DPY,
}

pytestmark = pytest.mark.gpu  # 与 test_tace_script.py 一致：需要 COMPUTE_VENV

_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", ""),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "LD_LIBRARY_PATH": "/usr/local/lib64",
}


def _run_py(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(CPY), "-c", textwrap.dedent(code)],
        env=_ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module", autouse=True)
def _require_compute_venv():
    if not CPY.is_file():
        pytest.skip(f"COMPUTE_VENV python 不存在: {CPY}")

def _select_backend_snippet(pbc: str, cell: str) -> str:
    return f"""
        import sys
        sys.path.insert(0, {str(SCRIPTS)!r})
        from ase import Atoms
        from tace_ase import _select_neighborlist_backend
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      pbc={pbc}, cell={cell})
        try:
            print("backend=", _select_neighborlist_backend(atoms))
        except ValueError as exc:
            print("ValueError:", exc)
    """


def test_tace_molecule_uses_ase_backend():
    out = _run_py(_select_backend_snippet("False", "None"))
    assert out.returncode == 0, out.stderr
    assert "backend= ase" in out.stdout


def test_tace_full_pbc_uses_matscipy_backend():
    out = _run_py(_select_backend_snippet("True", "[10, 10, 10]"))
    assert out.returncode == 0, out.stderr
    assert "backend= matscipy" in out.stdout


def test_tace_mixed_pbc_rejected():
    out = _run_py(_select_backend_snippet("(True, True, False)", "[10, 10, 20]"))
    assert out.returncode == 0, out.stderr
    assert "ValueError" in out.stdout and "混合 pbc" in out.stdout


def test_tace_cell_without_pbc_rejected():
    """带胞但 pbc 全 False：TACE 会静默按 3D 周期处理，必须拦截。"""
    out = _run_py(_select_backend_snippet("False", "[10, 10, 10]"))
    assert out.returncode == 0, out.stderr
    assert "ValueError" in out.stdout and "静默" in out.stdout


def test_tace_pbc_without_cell_rejected():
    out = _run_py(_select_backend_snippet("True", "None"))
    assert out.returncode == 0, out.stderr
    assert "ValueError" in out.stdout and "晶胞全零" in out.stdout


def _pbc_no_cell_job(script: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    """pbc="T T T" 但无 Lattice 的 extxyz → 脚本应在模型加载前报错。"""
    out_dir = tmp_path / script.stem
    out_dir.mkdir()
    xyz = out_dir / "bad.extxyz"
    xyz.write_text(
        '2\npbc="T T T"\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n'
    )
    job = {
        "xyz_path": str(xyz),
        "task": "sp",
        "model_path": "/nonexistent/model.pt",
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(job))
    return subprocess.run(
        [str(SCRIPT_PYTHON[script.name]), str(script), str(job_path)],
        cwd=out_dir,
        env=_ENV,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("script_name", ["nequip_ase.py", "tace_ase.py", "dpa4_ase.py"])
def test_pbc_without_cell_fails_fast(script_name: str, tmp_path: Path):
    if not SCRIPT_PYTHON[script_name].is_file():
        pytest.skip(f"解释器不存在: {SCRIPT_PYTHON[script_name]}")
    proc = _pbc_no_cell_job(SCRIPTS / script_name, tmp_path)
    out_dir = tmp_path / Path(script_name).stem
    result = json.loads((out_dir / "result.json").read_text())
    assert proc.returncode != 0
    assert result["ok"] is False
    assert "晶胞全零" in result["error"], result["error"]
