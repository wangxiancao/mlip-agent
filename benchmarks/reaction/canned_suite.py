"""reaction 套件的 canned 剧本（PLAN §3.3 WP5）：R1-repl 七段 happy path。

用途：无 LLM key 环境下验证「工具层 → MAPLE 子进程（含 ts）→ 作业收集 →
suite_judge 四维指标」全链路。剧本直接构造 JobSpec（不经 delegate），
能量/能垒在最终回答中按工作流公式汇总。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmarks.canned_scripts import (
    StreamFn,
    _as_data,
    _call,
    _n_responses,
    _returns,
    _tool_data,
    _validator_stream,
)

REFERENCES_DIR = Path(__file__).resolve().parent / "references"

HARTREE_TO_KCAL = 627.5095


def _spec(job_type: str, xyz: str, charge: int = 0, mult: int = 1, **extra) -> dict:
    spec: dict = {
        "job_type": job_type,
        "model": {"name": "aimnet2"},
        "structure": {"xyz_file": xyz},
        "charge": charge,
        "multiplicity": mult,
    }
    spec.update(extra)
    return spec


async def _orch_r1_repl_stream(messages, info):
    """R1（HCN→HNC）：opt/freq 两极小 → 刚性扫描 → prfo TS → TS freq → 能垒汇总。"""
    n = _n_responses(messages)
    ref = REFERENCES_DIR / "R1"
    if n == 0:
        yield _call("build_structure", {"source": str(ref / "r1.xyz"), "source_type": "xyz"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure", 0)["data"]["xyz_path"]
        yield _call("write_maple_inp", {"job_spec": _spec("opt", xyz)})
    elif n == 2:
        inp = _tool_data(messages, "write_maple_inp", 0)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 3:
        final_xyz = _tool_data(messages, "run_maple", 0)["data"]["parsed"]["final_xyz"]
        coords = "\n".join(final_xyz.splitlines()[2:])
        yield _call("build_structure", {"source": coords, "source_type": "inline_coords"})
    elif n == 4:
        xyz = _tool_data(messages, "build_structure", 1)["data"]["xyz_path"]
        yield _call("write_maple_inp", {"job_spec": _spec("freq", xyz)})
    elif n == 5:
        inp = _tool_data(messages, "write_maple_inp", 1)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 6:
        yield _call("build_structure", {"source": str(ref / "p1.xyz"), "source_type": "xyz"})
    elif n == 7:
        xyz = _tool_data(messages, "build_structure", 2)["data"]["xyz_path"]
        yield _call("write_maple_inp", {"job_spec": _spec("opt", xyz)})
    elif n == 8:
        inp = _tool_data(messages, "write_maple_inp", 2)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 9:
        final_xyz = _tool_data(messages, "run_maple", 2)["data"]["parsed"]["final_xyz"]
        coords = "\n".join(final_xyz.splitlines()[2:])
        yield _call("build_structure", {"source": coords, "source_type": "inline_coords"})
    elif n == 10:
        xyz = _tool_data(messages, "build_structure", 3)["data"]["xyz_path"]
        yield _call("write_maple_inp", {"job_spec": _spec("freq", xyz)})
    elif n == 11:
        inp = _tool_data(messages, "write_maple_inp", 3)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 12:
        # 刚性扫描：C−H（原子 1-3）从优化几何（≈1.07 Å）拉伸，delta=0.11，n_steps=5（6 点）
        xyz = _tool_data(messages, "build_structure", 1)["data"]["xyz_path"]
        yield _call(
            "write_maple_inp",
            {
                "job_spec": _spec(
                    "scan",
                    xyz,
                    scan={
                        "mode": "rigid",
                        "coordinates": [{"atom_i": 1, "atom_j": 3, "delta": 0.11, "n_steps": 5}],
                    },
                )
            },
        )
    elif n == 13:
        inp = _tool_data(messages, "write_maple_inp", 4)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 14:
        yield _call("build_structure", {"source": str(ref / "ts.xyz"), "source_type": "xyz"})
    elif n == 15:
        xyz = _tool_data(messages, "build_structure", 4)["data"]["xyz_path"]
        yield _call("write_maple_inp", {"job_spec": _spec("ts", xyz, ts={"method": "prfo"})})
    elif n == 16:
        inp = _tool_data(messages, "write_maple_inp", 5)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 17:
        final_xyz = _tool_data(messages, "run_maple", 5)["data"]["parsed"]["final_xyz"]
        coords = "\n".join(final_xyz.splitlines()[2:])
        yield _call("build_structure", {"source": coords, "source_type": "inline_coords"})
    elif n == 18:
        xyz = _tool_data(messages, "build_structure", 5)["data"]["xyz_path"]
        yield _call("write_maple_inp", {"job_spec": _spec("freq", xyz)})
    elif n == 19:
        inp = _tool_data(messages, "write_maple_inp", 6)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 20:
        summary = _tool_data(messages, "run_maple", 6)["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        e_r = _tool_data(messages, "run_maple", 0)["data"]["energy_hartree"]
        e_p = _tool_data(messages, "run_maple", 2)["data"]["energy_hartree"]
        e_ts = _tool_data(messages, "run_maple", 5)["data"]["parsed"]["energy_hartree"]
        n_imag_ts = _tool_data(messages, "run_maple", 6)["data"]["n_imaginary"]
        fwd = (e_ts - e_r) * HARTREE_TO_KCAL
        rev = (e_ts - e_p) * HARTREE_TO_KCAL
        rxn = (e_p - e_r) * HARTREE_TO_KCAL
        yield (
            f"HCN→HNC 全流程完成：两极小优化收敛、0 虚频，扫描 6 点，TS 搜索收敛、"
            f"TS 虚频 {n_imag_ts} 个。正向能垒 {fwd:.2f} kcal/mol，"
            f"反向能垒 {rev:.2f} kcal/mol，反应能 {rxn:.2f} kcal/mol。"
        )


async def _parser_passthrough(messages, info):
    # 套件剧本不经 delegate 构造 spec；此流仅为 harness 构建占位
    yield json.dumps({"job_type": "sp", "model": {"name": "aimnet2"}, "structure": {"coords": []}})


@dataclass(frozen=True)
class CannedScript:
    orchestrator: StreamFn
    task_parser: StreamFn
    result_validator: StreamFn = _validator_stream


SCRIPTS: dict[str, CannedScript] = {
    "R1-repl": CannedScript(_orch_r1_repl_stream, _parser_passthrough),
}
