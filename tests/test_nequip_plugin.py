"""TICKET-0002 测试：MAPLE 插件后端（plugins/maple_nequip_plugin.py）。

- 注册契约：经 COMPUTE_VENV 子进程 import 插件模块并校验注册/类属性
  （mlip_agent/.venv 无 maple/torch，插件只能在 COMPUTE_VENV 侧加载）；
- env 注入条件性：纯单测，无需子进程；
- 数值验证（gpu）：真实 maple 子进程跑插件路径 H2O sp，能量与 B5 直算 pin
  （eV）换算值差 ≤1e-6 Ha——防 EV2HARTREE 双重换算回归（前置条件 3）。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from app.compute.runner import _build_env
from app.tools.maple_tools import (
    NEQUIP_PLUGIN_MODULE,
    PLUGINS_DIR,
    _maple_extra_env,
    plugin_env_for_model,
)

ROOT = Path(__file__).resolve().parents[1]
REFDIR = ROOT / "benchmarks" / "references"
MODEL_PATH = os.environ.get(
    "NEQUIP_MODEL_PATH", "/root/soft/ml_potential/checkpoints/nequip/NequIP-OAM-XL-0.1.nequip.zip"
)
EV2HARTREE = 1.0 / 27.211386245988

H2O_COORDS = """\
O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000"""


# ---------- 注册契约（COMPUTE_VENV 子进程，CPU 即可） ----------


def test_plugin_registration_contract(compute_python):
    """import 插件模块即完成注册；get_registered_calculator 可取，契约属性齐全。"""
    assert (PLUGINS_DIR / f"{NEQUIP_PLUGIN_MODULE}.py").is_file()
    code = (
        "import maple_nequip_plugin; "
        "from maple.function.calculator.calculator_base import get_registered_calculator; "
        "cls = get_registered_calculator('nequip-oam-xl'); "
        "assert cls.__name__ == 'NequIPPluginCalculator'; "
        "assert cls.MODEL_NAMES == ('nequip-oam-xl',); "
        "assert cls.MODEL_ENERGY_UNIT == 'eV'; "
        "assert cls.SUPPORTED_HESSIAN_MODES == ('numerical',); "
        "assert cls.SUPPORTS_PBC is False; "
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
    实例化按 #model= 名字匹配（措辞按 TICKET-0002 §7 审批修正）。"""
    code = (
        "import os; "
        "os.environ['MAPLE_CALCULATOR_PLUGINS'] = 'maple_nequip_plugin'; "
        "from maple.function.calculator.calculator_base import ("
        "  load_calculator_plugins_from_env, get_registered_calculator); "
        "load_calculator_plugins_from_env(); "
        "cls = get_registered_calculator('nequip-oam-xl'); "
        "assert cls.__name__ == 'NequIPPluginCalculator'; "
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


# ---------- env 注入条件性（纯单测） ----------


def test_plugin_env_injected_for_nequip_model():
    env = plugin_env_for_model("nequip-oam-xl")
    assert env == {
        "MAPLE_CALCULATOR_PLUGINS": NEQUIP_PLUGIN_MODULE,
        "PYTHONPATH": str(PLUGINS_DIR),
    }
    assert Path(env["PYTHONPATH"]).is_dir()


@pytest.mark.parametrize("name", ["ani2x", "uma", "aimnet2", "NequIP-Direct", None, ""])
def test_plugin_env_absent_for_other_models(name):
    assert plugin_env_for_model(name) is None


def test_default_subprocess_env_unchanged():
    """默认 maple 子进程 env 一字节不变：不含插件变量（前置条件 1）。"""
    env = _build_env()
    assert "MAPLE_CALCULATOR_PLUGINS" not in env
    assert "PYTHONPATH" not in env
    # 离线变量对所有路径（含插件）同样生效
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_extra_env_only_added_when_given():
    env = _build_env({"MAPLE_CALCULATOR_PLUGINS": NEQUIP_PLUGIN_MODULE})
    assert env["MAPLE_CALCULATOR_PLUGINS"] == NEQUIP_PLUGIN_MODULE


def test_maple_extra_env_from_job_spec(tmp_path):
    """以 JobSpec（job_spec.json）为注入条件的事实来源。"""
    inp = tmp_path / "input.inp"
    inp.write_text("#model=ani2x\n#sp\n")
    # ani2x 作业：不注入
    (tmp_path / "job_spec.json").write_text(
        json.dumps({"model": {"name": "ani2x"}, "job_type": "sp"})
    )
    assert _maple_extra_env(tmp_path, inp) is None
    # nequip-oam-xl 作业：注入（即使 inp 文本与 sidecar 不一致，以 JobSpec 为准）
    (tmp_path / "job_spec.json").write_text(
        json.dumps({"model": {"name": "nequip-oam-xl"}, "job_type": "sp"})
    )
    env = _maple_extra_env(tmp_path, inp)
    assert env is not None and "MAPLE_CALCULATOR_PLUGINS" in env


def test_maple_extra_env_fallback_to_inp(tmp_path):
    """无 job_spec.json（手写 inp）时回退解析 #model= 行。"""
    inp = tmp_path / "input.inp"
    inp.write_text(
        f"#model=nequip-oam-xl(model_path={MODEL_PATH})\n#sp(verbose=0)\n\n{H2O_COORDS}\n"
    )
    assert _maple_extra_env(tmp_path, inp) is not None
    inp.write_text("#model=ani2x\n#sp(verbose=0)\n")
    assert _maple_extra_env(tmp_path, inp) is None


# ---------- 数值验证（真实 GPU 子进程；前置条件 3） ----------


@pytest.mark.gpu
async def test_plugin_path_energy_matches_direct_path(tmp_path, maple_bin):
    """插件路径（MAPLE，Hartree）与直算路径（B5 pin，eV）能量差 ≤1e-6 Ha。

    防 EV2HARTREE 双重换算：若插件误乘换算系数，能量会差 27 倍，本测试必挂。
    """
    b5_ref = json.loads((REFDIR / "B5.json").read_text())
    if b5_ref.get("status") == "failed" or "energy_eV" not in b5_ref.get("values", {}):
        pytest.skip("B5 金标准生成失败，无 pin 值可比")
    if not Path(MODEL_PATH).is_file():
        pytest.skip(f"NequIP 模型文件不存在: {MODEL_PATH}")

    from app.compute.parsers import parse_out_file
    from app.compute.runner import run_compute

    inp = tmp_path / "input.inp"
    inp.write_text(
        f"#model=nequip-oam-xl(model_path={MODEL_PATH})\n"
        "#sp(verbose=0)\n#device=gpu0\n\n" + H2O_COORDS + "\n"
    )
    job = await run_compute(
        [str(maple_bin), inp.name, "input.out"],
        cwd=tmp_path,
        timeout_s=1200,
        gpu=True,
        extra_env=plugin_env_for_model("nequip-oam-xl"),
    )
    assert job.ok, job.error

    result = parse_out_file(tmp_path / "input.out", "sp")
    assert result.ok, result.warnings
    expected_ha = b5_ref["values"]["energy_eV"] * EV2HARTREE
    assert result.energy_hartree == pytest.approx(expected_ha, abs=1e-6)
