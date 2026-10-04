"""MACE-MPA-0 MAPLE 插件后端（plugins/maple_mace_plugin.py）测试。

- 注册契约：经 COMPUTE_VENV 子进程 import 插件模块并校验注册/类属性
  （mlip_agent/.venv 无 maple/torch，插件只能在 COMPUTE_VENV 侧加载）；
- env 注入分派：plugin_env_for_model 对 macempa0 / nequip-oam-xl / 内建模型的
  分派（纯单测，无需子进程）；
- 数值验证（gpu）：真实 maple 子进程跑插件路径 H2O sp，能量与 B20 金标准 pin
  差 ≤1e-6 Ha（B20.json 缺省时跳过）。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from app.tools.maple_tools import (
    MACE_PLUGIN_MODULE,
    NEQUIP_PLUGIN_MODULE,
    PLUGINS_DIR,
    plugin_env_for_model,
)

ROOT = Path(__file__).resolve().parents[1]
REFDIR = ROOT / "benchmarks" / "references"
MODEL_PATH = os.environ.get(
    "MACEMPA0_MODEL_PATH", "/root/soft/ml_potential/checkpoints/mace/macempa0.pt"
)

H2O_COORDS = """\
O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000"""


# ---------- 注册契约（COMPUTE_VENV 子进程，CPU 即可） ----------


def test_plugin_registration_contract(compute_python):
    """import 插件模块即完成注册；get_registered_calculator 可取，契约属性齐全。"""
    assert (PLUGINS_DIR / f"{MACE_PLUGIN_MODULE}.py").is_file()
    code = (
        "import maple_mace_plugin; "
        "from maple.function.calculator.calculator_base import get_registered_calculator; "
        "cls = get_registered_calculator('macempa0'); "
        "assert cls.__name__ == 'MACEMPA0PluginCalculator'; "
        "assert cls.MODEL_NAMES == ('macempa0',); "
        "assert cls.MODEL_ENERGY_UNIT == 'eV'; "
        # TICKET-0008：MACE PBC 改造验收（B27）后插件翻为 True
        "assert cls.SUPPORTS_PBC is True; "
        "assert cls.SUPPORTS_CHARGE_MULT is False; "
        "assert cls.CHECKPOINT_FILENAME is None; "
        "assert cls.REQUIRES_LOCAL_MODEL_FILE is True; "
        "assert cls.MODEL_PATH_OPTION == 'model_path'; "
        "print('OK')"
    )
    proc = subprocess.run(
        [str(compute_python), "-c", code],
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(PLUGINS_DIR),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


def test_plugin_env_discovery(compute_python):
    """MAPLE_CALCULATOR_PLUGINS 机制：首个 calculator 构建时 import 注册，
    实例化按 #model= 名字匹配。"""
    code = (
        "import os; "
        "os.environ['MAPLE_CALCULATOR_PLUGINS'] = 'maple_mace_plugin'; "
        "from maple.function.calculator.calculator_base import ("
        "  load_calculator_plugins_from_env, get_registered_calculator); "
        "load_calculator_plugins_from_env(); "
        "cls = get_registered_calculator('macempa0'); "
        "assert cls.__name__ == 'MACEMPA0PluginCalculator'; "
        "print('OK')"
    )
    proc = subprocess.run(
        [str(compute_python), "-c", code],
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(PLUGINS_DIR),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


# ---------- env 注入分派（纯单测） ----------


def test_plugin_env_injected_for_macempa0():
    env = plugin_env_for_model("macempa0")
    assert env == {
        "MAPLE_CALCULATOR_PLUGINS": MACE_PLUGIN_MODULE,
        "PYTHONPATH": str(PLUGINS_DIR),
    }
    assert Path(env["PYTHONPATH"]).is_dir()


def test_plugin_env_injected_for_nequip_model():
    env = plugin_env_for_model("nequip-oam-xl")
    assert env == {
        "MAPLE_CALCULATOR_PLUGINS": NEQUIP_PLUGIN_MODULE,
        "PYTHONPATH": str(PLUGINS_DIR),
    }


@pytest.mark.parametrize("name", ["ani2x", "uma", "maceoff23s", "MACE-MPA-0", None, ""])
def test_plugin_env_absent_for_other_models(name):
    assert plugin_env_for_model(name) is None


# ---------- 数值验证（真实 GPU 子进程；对 B20 金标准 pin） ----------


@pytest.mark.gpu
async def test_plugin_path_energy_matches_b20_pin(tmp_path, maple_bin):
    """插件路径（MAPLE，Hartree）H2O sp 与 B20 金标准 pin 差 ≤1e-6 Ha。"""
    b20 = REFDIR / "B20.json"
    if not b20.is_file():
        pytest.skip("B20 金标准未生成（scripts/make_golden_refs.sh）")
    b20_ref = json.loads(b20.read_text())
    if b20_ref.get("status") == "failed" or "energy_hartree" not in b20_ref.get(
        "values", {}
    ):
        pytest.skip("B20 金标准生成失败，无 pin 值可比")
    if not Path(MODEL_PATH).is_file():
        pytest.skip(f"MACE-MPA-0 wrapper 不存在: {MODEL_PATH}")

    from app.compute.parsers import parse_out_file
    from app.compute.runner import run_compute

    inp = tmp_path / "input.inp"
    inp.write_text(
        f"#model=macempa0(model_path={MODEL_PATH})\n"
        "#sp(verbose=0)\n#device=gpu0\n\n" + H2O_COORDS + "\n"
    )
    job = await run_compute(
        [str(maple_bin), inp.name, "input.out"],
        cwd=tmp_path,
        timeout_s=1200,
        gpu=True,
        extra_env=plugin_env_for_model("macempa0"),
    )
    assert job.ok, job.error

    result = parse_out_file(tmp_path / "input.out", "sp")
    assert result.ok, result.warnings
    assert result.energy_hartree == pytest.approx(
        b20_ref["values"]["energy_hartree"], abs=1e-6
    )
