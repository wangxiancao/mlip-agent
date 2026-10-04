"""pytest 共享 fixture 与跳过规则（PLAN §1 tests/conftest.py）。

- 无 COMPUTE_VENV maple 可执行文件 → 跳过真实子进程测试；
- 无 GPU（nvidia-smi 不可用）→ 跳过 @pytest.mark.gpu；
- 无 LLM key（MLIP_AGENT_LLM_API_KEY）→ 供 M2+ 的 live 测试跳过用。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

MLIP_AGENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MLIP_AGENT_DIR.parent
REFERENCES_DIR = MLIP_AGENT_DIR / "benchmarks" / "references"
MAPLE_EXAMPLES_DIR = Path(
    os.environ.get("MAPLE_EXAMPLES", REPO_ROOT / "MAPLE" / "examples")
)
COMPUTE_VENV = Path(os.environ.get("COMPUTE_VENV", REPO_ROOT / ".venv"))
MAPLE_BIN = COMPUTE_VENV / "bin" / "maple"
COMPUTE_PYTHON = COMPUTE_VENV / "bin" / "python"


def _gpu_available() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    return subprocess.run([nvidia_smi, "-L"], capture_output=True).returncode == 0


GPU_AVAILABLE = _gpu_available()
MAPLE_AVAILABLE = MAPLE_BIN.is_file() and os.access(MAPLE_BIN, os.X_OK)
LLM_KEY_AVAILABLE = bool(os.environ.get("MLIP_AGENT_LLM_API_KEY"))

requires_maple = pytest.mark.skipif(
    not MAPLE_AVAILABLE, reason=f"COMPUTE_VENV maple 不可用: {MAPLE_BIN}"
)
requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="无可用 GPU")
requires_llm_key = pytest.mark.skipif(
    not LLM_KEY_AVAILABLE, reason="无 MLIP_AGENT_LLM_API_KEY（live 测试跳过）"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(requires_gpu)
            item.add_marker(requires_maple)


@pytest.fixture(scope="session")
def references_dir() -> Path:
    return REFERENCES_DIR


@pytest.fixture(scope="session")
def maple_examples_dir() -> Path:
    if not MAPLE_EXAMPLES_DIR.is_dir():
        pytest.skip(f"MAPLE examples 目录不存在: {MAPLE_EXAMPLES_DIR}")
    return MAPLE_EXAMPLES_DIR


@pytest.fixture(scope="session")
def maple_bin() -> Path:
    if not MAPLE_AVAILABLE:
        pytest.skip(f"COMPUTE_VENV maple 不可用: {MAPLE_BIN}")
    return MAPLE_BIN


@pytest.fixture(scope="session")
def compute_python() -> Path:
    if not COMPUTE_PYTHON.is_file():
        pytest.skip(f"COMPUTE_VENV python 不可用: {COMPUTE_PYTHON}")
    return COMPUTE_PYTHON
