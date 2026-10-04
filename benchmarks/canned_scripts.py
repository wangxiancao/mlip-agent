"""canned 模式的 FunctionModel 剧本（PLAN §3 / §4 M5）。

写法沿用 tests/test_agent_offline.py：pydantic-ai FunctionModel(stream_function=...)
按「消息历史中 ModelResponse 条数」推进确定性剧本；工具真实执行（run_maple /
run_nequip 走 GPU 子进程），子 agent（task_parser / result_validator）同为回放。
每个任务一个剧本（orchestrator + task_parser），经 SCRIPTS 注册表供 runner.py 取用。

复现约束（与 scripts/make_golden_refs.sh 对齐）：
- B4/B5 单点、B8 刚性扫描的能量与起始几何强相关 → 用金标准同款内联坐标；
- B2/B3/B6/B7 走 build_structure(name)（opt 收敛到同一极小，频率容差 ±20 cm⁻¹）。
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall

StreamFn = Callable[..., Awaitable[None]]  # async generator：(messages, info)

REFERENCES_DIR = Path(__file__).resolve().parent / "references"
MAPLE_SP_INP = (
    Path(__file__).resolve().parents[2] / "MAPLE" / "examples" / "sp" / "sp.inp"
)

# 金标准起始几何（make_golden_refs.sh 的 b2/b4/b5 inp 与 b8_scan.inp）
H2O_GOLDEN_COORDS = """\
O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000"""

H2O_SCAN_START_COORDS = """\
O      0.00000000      0.00000000      0.00000000
H      0.90000000      0.00000000      0.00000000
H     -0.23986000      0.92749000      0.00000000"""


# ---------- 剧本辅助（与 tests/test_agent_offline.py 同款） ----------


def _n_responses(messages) -> int:
    return sum(1 for m in messages if isinstance(m, ModelResponse))


def _returns(messages, tool_name: str) -> list:
    out = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart) and p.tool_name == tool_name:
                    out.append(p.content)
    return out


def _as_data(content):
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return {"raw": content}


def _tool_data(messages, tool_name: str, index: int = 0):
    return _as_data(_returns(messages, tool_name)[index])


def _user_text(messages) -> str:
    parts = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    parts.append(p.content)
    return "\n".join(parts)


def _call(name: str, args: dict) -> dict[int, DeltaToolCall]:
    return {0: DeltaToolCall(name=name, json_args=json.dumps(args))}


def _delegated_task(messages) -> str:
    text = _user_text(messages)
    return json.loads(re.search(r"(\{.*delegated_task.*\})", text).group(1))[
        "delegated_task"
    ]


def _xyz_of(task: str) -> str:
    return re.search(r"(\S+\.(?:ext)?xyz)", task).group(1)


# ---------- JobSpec 构造 ----------


def _b1_jobspec() -> dict:
    """从 MAPLE/examples/sp/sp.inp 解析 22 原子坐标，构造 B1/B10 JobSpec。"""
    coords = []
    for line in MAPLE_SP_INP.read_text().splitlines():
        m = re.match(
            r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
            line,
        )
        if m:
            coords.append(
                {
                    "symbol": m.group(1),
                    "x": float(m.group(2)),
                    "y": float(m.group(3)),
                    "z": float(m.group(4)),
                }
            )
    assert len(coords) == 22
    return {
        "job_type": "sp",
        "model": {"name": "ani2x"},
        "structure": {"coords": coords},
        "charge": 0,
        "multiplicity": 1,
    }


def _spec_from_delegated(task: str) -> dict:
    """按 delegated_task 前缀构造 xyz 引用型 JobSpec（opt/freq/md/scan/sp）。"""
    xyz = _xyz_of(task)
    spec: dict = {
        "job_type": "sp",
        "model": {"name": "ani2x"},
        "structure": {"xyz_file": xyz},
        "charge": 0,
        "multiplicity": 1,
    }
    if task.startswith("opt"):
        spec["job_type"] = "opt"
    elif task.startswith("freq"):
        spec["job_type"] = "freq"
    elif task.startswith("md"):
        # 20 步 NVE 0.5 fs；log_every=1 保证逐步能量行输出（MAPLE 默认 100）
        spec["job_type"] = "md"
        spec["md"] = {
            "ensemble": "nve",
            "timestep_fs": 0.5,
            "steps": 20,
            "temperature": 300.0,
            "log_every": 1,
        }
        if "不带log_every" in task:
            # B18 恢复类剧本：首跑故意不带 log_every（无逐步能量行）
            del spec["md"]["log_every"]
    elif task.startswith("scan"):
        # O-H 刚性扫描 0.9 → 1.1 Å 共 5 点（S 1 2 0.05 4，同金标准 b8_scan.inp）
        spec["job_type"] = "scan"
        spec["scan"] = {
            "mode": "rigid",
            "coordinates": [{"atom_i": 1, "atom_j": 2, "delta": 0.05, "n_steps": 4}],
        }
    elif "omat" in task:
        # B26：UMA 周期 task（PBC 落地 2026-10-03）；xyz 为带 Lattice 的 extxyz
        spec["model"] = {"name": "uma", "uma_task": "omat"}
    elif "uma" in task:
        spec["model"] = {"name": "uma"}
    elif "nequip" in task:
        # TICKET-0002：MAPLE 插件后端（与 B5 直算路径区分）
        spec["model"] = {"name": "nequip-oam-xl"}
    elif "macempa0" in task:
        # MACE-MPA-0：MAPLE 插件后端（plugins/maple_mace_plugin.py）
        spec["model"] = {"name": "macempa0"}
    elif "maceoff23s" in task:
        spec["model"] = {"name": "maceoff23s"}
    elif "aimnet2nse" in task:
        # B22（TICKET-0007）：AIMNet2-NSE 开壳层；三重态 O2（多重度 3）
        spec["model"] = {"name": "aimnet2nse"}
        if "三重态" in task or "多重度 3" in task:
            spec["multiplicity"] = 3
    elif "aimnet2" in task:
        spec["model"] = {"name": "aimnet2"}
    elif "ani1x" in task:
        spec["model"] = {"name": "ani1x"}
    return spec


# ---------- 子 agent 剧本 ----------


async def _validator_stream(messages, info):
    yield json.dumps(
        {"ok": True, "issues": [], "summary": "退出码 0，能量量级合理"}, ensure_ascii=False
    )


async def _parser_b1_stream(messages, info):
    yield json.dumps(_b1_jobspec())


async def _parser_generic_stream(messages, info):
    yield json.dumps(_spec_from_delegated(_delegated_task(messages)))


# ---------- orchestrator 剧本 ----------


async def _orch_b1_stream(messages, info):
    """B1/B10：delegate 解析 → write_maple_inp → run_maple（审批挂起）
    → result_validator → convert_units → 中文回答（Ha + eV）。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("delegate", {"subagent": "task_parser", "prompt": "用 ANI-2x 算该 22 原子分子的单点能，坐标见用户消息"})
    elif n == 1:
        spec = json.loads(_tool_data(messages, "delegate")["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 2:
        inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 3:
        summary = _tool_data(messages, "run_maple")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    elif n == 4:
        energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
        yield _call("convert_units", {"value": energy, "from_unit": "hartree", "to_unit": "eV"})
    else:
        energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
        ev = _tool_data(messages, "convert_units")["data"]["value"]
        verdict = json.loads(_tool_data(messages, "delegate", 1)["output"])
        yield (
            f"计算完成。单点能 = {energy:.10f} Hartree = {ev:.4f} eV。"
            f"校验结论：{verdict['summary']}"
        )


def _orch_single_maple_job(molecule: str, task_cn: str, text_fn) -> StreamFn:
    """build_structure(name) → delegate(opt/md 单作业) → write → run → validator → 回答。"""

    async def stream(messages, info):
        n = _n_responses(messages)
        if n == 0:
            yield _call("build_structure", {"source": molecule, "source_type": "name"})
        elif n == 1:
            xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
            yield _call("delegate", {"subagent": "task_parser", "prompt": f"{task_cn}，结构文件 {xyz}"})
        elif n == 2:
            spec = json.loads(_tool_data(messages, "delegate")["output"])
            yield _call("write_maple_inp", {"job_spec": spec})
        elif n == 3:
            inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
            yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
        elif n == 4:
            summary = _tool_data(messages, "run_maple")["data"]
            yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
        else:
            yield text_fn(_tool_data(messages, "run_maple")["data"])

    return stream


def _orch_opt_freq_chain(molecule: str, label: str) -> StreamFn:
    """build_structure → opt → final_xyz → freq 链式两作业（B3/B6 共用）。"""

    async def stream(messages, info):
        n = _n_responses(messages)
        if n == 0:
            yield _call("build_structure", {"source": molecule, "source_type": "name"})
        elif n == 1:
            xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
            yield _call("delegate", {"subagent": "task_parser", "prompt": f"opt：用 ani2x 优化结构文件 {xyz}"})
        elif n == 2:
            spec = json.loads(_tool_data(messages, "delegate", 0)["output"])
            yield _call("write_maple_inp", {"job_spec": spec})
        elif n == 3:
            inp = _tool_data(messages, "write_maple_inp", 0)["data"]["inp_path"]
            yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
        elif n == 4:
            final_xyz = _tool_data(messages, "run_maple", 0)["data"]["parsed"]["final_xyz"]
            coords = "\n".join(final_xyz.splitlines()[2:])  # 去掉 xyz 头两行
            yield _call("build_structure", {"source": coords, "source_type": "inline_coords"})
        elif n == 5:
            xyz = _tool_data(messages, "build_structure", 1)["data"]["xyz_path"]
            yield _call("delegate", {"subagent": "task_parser", "prompt": f"freq：用 ani2x 对结构文件 {xyz} 做频率分析"})
        elif n == 6:
            spec = json.loads(_tool_data(messages, "delegate", 1)["output"])
            yield _call("write_maple_inp", {"job_spec": spec})
        elif n == 7:
            inp = _tool_data(messages, "write_maple_inp", 1)["data"]["inp_path"]
            yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
        elif n == 8:
            summary = _tool_data(messages, "run_maple", 1)["data"]
            yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
        else:
            opt_e = _tool_data(messages, "run_maple", 0)["data"]["energy_hartree"]
            freq = _tool_data(messages, "run_maple", 1)["data"]["parsed"]["frequencies_cm1"]
            n_imag = _tool_data(messages, "run_maple", 1)["data"]["n_imaginary"]
            yield (
                f"{label}优化收敛，能量 {opt_e:.10f} Hartree；"
                f"频率分析共 {len(freq)} 个模式，虚频 {n_imag} 个。"
            )

    return stream


async def _orch_b4_stream(messages, info):
    """B4：金标准几何 inline → UMA sp → validator → 回答。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_GOLDEN_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"sp：用 uma(uma-s-1p1, omol) 算结构文件 {xyz} 的单点能"})
    elif n == 2:
        spec = json.loads(_tool_data(messages, "delegate")["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 3:
        inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 4:
        summary = _tool_data(messages, "run_maple")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
        yield f"UMA (uma-s-1p1, omol) 单点能 = {energy:.10f} Hartree。"


def _orch_maple_sp_golden(model_token: str, label: str, coords: str = H2O_GOLDEN_COORDS) -> StreamFn:
    """B12/B13/B14/B20（D7-4）与 B22（TICKET-0007）：金标准几何 inline → 指定模型
    sp（run_maple，审批挂起）→ validator → 回答。coords 参数支持非 H2O 体系
    （B22 的 O2 三重态）；多重度等非 0/1 时经 parser 提示词传达。"""

    async def stream(messages, info):
        n = _n_responses(messages)
        if n == 0:
            yield _call("build_structure", {"source": coords, "source_type": "inline_coords"})
        elif n == 1:
            xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
            hint = f"sp：用 {model_token} 算结构文件 {xyz} 的单点能"
            if model_token == "aimnet2nse":
                hint += "，O2 为三重态（多重度 3）"
            yield _call("delegate", {"subagent": "task_parser", "prompt": hint})
        elif n == 2:
            spec = json.loads(_tool_data(messages, "delegate")["output"])
            yield _call("write_maple_inp", {"job_spec": spec})
        elif n == 3:
            inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
            yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
        elif n == 4:
            summary = _tool_data(messages, "run_maple")["data"]
            yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
        else:
            energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
            yield f"{label} 单点能 = {energy:.10f} Hartree。"

    return stream


# O2 三重态金标准几何（O-O 1.21 Å，B22，TICKET-0007）
O2_TRIPLET_COORDS = "O 0.00000000 0.00000000 0.00000000\nO 1.21000000 0.00000000 0.00000000"


async def _orch_b5_stream(messages, info):
    """B5：金标准几何 inline → run_nequip 直算（审批挂起）→ validator → 回答（eV）。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_GOLDEN_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("run_nequip", {"job_spec": {"xyz_path": xyz, "task": "sp"}, "timeout_s": 900})
    elif n == 2:
        result = _tool_data(messages, "run_nequip")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(result, ensure_ascii=False, default=str)})
    else:
        energy = _tool_data(messages, "run_nequip")["data"]["energy_eV"]
        yield f"NequIP-OAM-XL 单点能 = {energy:.8f} eV（ASE 单位制）。"


async def _orch_b15_stream(messages, info):
    """B15（TICKET-0004）：金标准几何 inline → run_tace 直算（审批挂起）
    → validator → 回答（eV）。剧本结构与 B5 同构。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_GOLDEN_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("run_tace", {"job_spec": {"xyz_path": xyz, "task": "sp"}, "timeout_s": 900})
    elif n == 2:
        result = _tool_data(messages, "run_tace")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(result, ensure_ascii=False, default=str)})
    else:
        energy = _tool_data(messages, "run_tace")["data"]["energy_eV"]
        yield f"TACE-OAM-L 单点能 = {energy:.8f} eV（ASE 单位制）。"


async def _orch_b16_stream(messages, info):
    """B16（TICKET-0006）：金标准几何 inline → run_dpa4 直算（CPU+OMP=1 pin 口径，
    审批挂起）→ validator → 回答（eV）。剧本结构与 B5/B15 同构。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_GOLDEN_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("run_dpa4", {"job_spec": {"xyz_path": xyz, "task": "sp"}, "timeout_s": 900})
    elif n == 2:
        result = _tool_data(messages, "run_dpa4")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(result, ensure_ascii=False, default=str)})
    else:
        energy = _tool_data(messages, "run_dpa4")["data"]["energy_eV"]
        yield f"DPA-4（head=H2O_H2O_PD）单点能 = {energy:.8f} eV（CPU+OMP=1 口径）。"


async def _orch_b11_stream(messages, info):
    """B11（TICKET-0002）：金标准几何 inline → NequIP-OAM-XL 经 MAPLE 插件后端
    sp（run_maple，审批挂起）→ validator → 回答（Hartree）。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_GOLDEN_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"sp：用 MAPLE 的 nequip-oam-xl 插件后端算结构文件 {xyz} 的单点能"})
    elif n == 2:
        spec = json.loads(_tool_data(messages, "delegate")["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 3:
        inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 4:
        summary = _tool_data(messages, "run_maple")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
        yield f"NequIP-OAM-XL（MAPLE 插件后端）单点能 = {energy:.10f} Hartree。"


async def _orch_b8_stream(messages, info):
    """B8：金标准扫描起始几何 inline → rigid scan（S 1 2 0.05 4）→ 回答。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_SCAN_START_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"scan：用 ani2x 刚性扫描结构文件 {xyz} 的 O-H 键（原子 1-2），0.9 起 5 点步长 0.05 Å"})
    elif n == 2:
        spec = json.loads(_tool_data(messages, "delegate")["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 3:
        inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 4:
        summary = _tool_data(messages, "run_maple")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        data = _tool_data(messages, "run_maple")["data"]
        pts = data["parsed"]["energies_hartree"]
        yield f"扫描完成，{len(pts)} 个点，能量（Hartree）：{pts}。"


async def _orch_b9_stream(messages, info):
    """B9：信息不全 → ask_user_question 挂起；注入答复后确认，不提交计算。

    auto_answer 为拒绝/保留（"这些信息我暂时没有，先不要计算"）时，
    最终回答必须明确不提交任何计算（D7-2 行为语义澄清）。
    """
    n = _n_responses(messages)
    if n == 0:
        yield _call(
            "ask_user_question",
            {"question": "请提供：1) 使用哪个 MLIP 模型；2) 体系电荷与自旋多重度"},
        )
    else:
        rets = _returns(messages, "ask_user_question")
        answer = _as_data(rets[0]).get("data", {}).get("answer", "") if rets else ""
        yield (
            f"收到您的答复：{answer or '（无文字答复）'}。"
            "既然这些信息暂时没有，我不会提交任何计算；"
            "等您确定模型、电荷与自旋多重度后再继续。"
        )


# ---------- 周期性（PBC）剧本（2026-10-03 落地） ----------

# Si 金刚石原胞（fcc 初基矢量，a=5.431 Å，2 原子），extxyz 自带 Lattice/pbc 头
SI_EXTXYZ = str(REFERENCES_DIR / "si_diamond.extxyz")


def _orch_pbc_direct_stream(tool: str, label: str, extra_job: dict | None = None) -> StreamFn:
    """B23/B24/B25：Si 原胞（references/si_diamond.extxyz）→ build_structure(xyz)
    （自带 Lattice 头，原样保留周期信息）→ run_<backend> 直算 sp（审批挂起）
    → validator → 回答（eV）。剧本结构与 B5/B15 同构。"""

    async def stream(messages, info):
        n = _n_responses(messages)
        if n == 0:
            yield _call("build_structure", {"source": SI_EXTXYZ, "source_type": "xyz"})
        elif n == 1:
            xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
            job_spec = {"xyz_path": xyz, "task": "sp", **(extra_job or {})}
            yield _call(tool, {"job_spec": job_spec, "timeout_s": 900})
        elif n == 2:
            result = _tool_data(messages, tool)["data"]
            yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(result, ensure_ascii=False, default=str)})
        else:
            energy = _tool_data(messages, tool)["data"]["energy_eV"]
            yield f"{label} Si 金刚石原胞（PBC）单点能 = {energy:.8f} eV（ASE 单位制）。"

    return stream


async def _orch_b21_stream(messages, info):
    """B21（2026-10-03 DPA4C 接入）：金标准 H2O 几何 inline → run_dpa4 直算
    （family=dpa4c + 显式 model_path，CPU+OMP=1 pin 口径，审批挂起）→ validator → 回答（eV）。
    剧本结构与 B16 同构。DPA4C-OMol25 输出总能量，与 DPA4-beta 的 B16 pin 不互比。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_GOLDEN_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("run_dpa4", {"job_spec": {
            "xyz_path": xyz, "task": "sp", "family": "dpa4c",
            "model_path": "/root/soft/ml_potential/checkpoints/dpa4/DPA4C-Air-OMol25-100M-v20260820.pt",
        }, "timeout_s": 900})
    elif n == 2:
        result = _tool_data(messages, "run_dpa4")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(result, ensure_ascii=False, default=str)})
    else:
        energy = _tool_data(messages, "run_dpa4")["data"]["energy_eV"]
        yield f"DPA4C-Air-OMol25（family=dpa4c）单点能 = {energy:.8f} eV（总能量约定，CPU+OMP=1 口径）。"


def _orch_maple_pbc_sp(model_token: str, label: str, parser_hint: str) -> StreamFn:
    """B26/B27（PBC 落地）：Si 原胞 extxyz（自带 Lattice 头）→ build_structure(xyz)
    原样保留 → task_parser → write_maple_inp → run_maple（审批挂起）→ validator → 回答。"""

    async def stream(messages, info):
        n = _n_responses(messages)
        if n == 0:
            yield _call("build_structure", {"source": SI_EXTXYZ, "source_type": "xyz"})
        elif n == 1:
            xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
            yield _call("delegate", {"subagent": "task_parser", "prompt": f"sp：{parser_hint}，结构文件 {xyz}"})
        elif n == 2:
            spec = json.loads(_tool_data(messages, "delegate")["output"])
            yield _call("write_maple_inp", {"job_spec": spec})
        elif n == 3:
            inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
            yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
        elif n == 4:
            summary = _tool_data(messages, "run_maple")["data"]
            yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
        else:
            energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
            yield f"{label} Si 金刚石原胞（PBC）单点能 = {energy:.10f} Hartree。"

    return stream


async def _orch_b28_stream(messages, info):
    """B28（TICKET-0009）：H2O 金标准坐标 + cell=[20,20,20] → write_maple_inp
    （#pbc 渲染 + aimnet2 PBC 路径，库仑缺省自动 dsf）→ run_maple（审批挂起）
    → validator → 回答。周期任务不经 task_parser（cell 字段直给 JobSpec）。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call(
            "write_maple_inp",
            {
                "job_spec": {
                    "job_type": "sp",
                    "model": {"name": "aimnet2"},
                    "structure": {
                        "coords": _h2o_golden_coords_list(),
                        "cell": [20.0, 20.0, 20.0],
                    },
                    "charge": 0,
                    "multiplicity": 1,
                }
            },
        )
    elif n == 1:
        inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 2:
        summary = _tool_data(messages, "run_maple")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
        yield f"AIMNet2（20 Å 周期盒，库仑自动 dsf）单点能 = {energy:.10f} Hartree。"


# ---------- 注册表 ----------


def _h2o_golden_coords_list() -> list[dict]:
    """H2O_GOLDEN_COORDS 文本 → JobSpec coords 列表（B17 首试内联 freq 用）。"""
    coords = []
    for line in H2O_GOLDEN_COORDS.splitlines():
        symbol, x, y, z = line.split()
        coords.append({"symbol": symbol, "x": float(x), "y": float(y), "z": float(z)})
    return coords


async def _orch_b17_stream(messages, info):
    """B17（§3.2 recovery）：freq + 内联坐标 write_maple_inp 被 jobspec 校验拒绝（D7-3）
    → 诊断后改走 opt→freq 正确链（复用 B3 频率 pin）。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": H2O_GOLDEN_COORDS, "source_type": "inline_coords"})
    elif n == 1:
        # 首试：freq + 内联坐标（违反 D7-3 校验，工具返回 ok=false）
        yield _call(
            "write_maple_inp",
            {
                "job_spec": {
                    "job_type": "freq",
                    "model": {"name": "ani2x"},
                    "structure": {"coords": _h2o_golden_coords_list()},
                    "charge": 0,
                    "multiplicity": 1,
                }
            },
        )
    elif n == 2:
        # 诊断：freq 需要 xyz_file，且频率应在优化后的结构上计算 → 先 opt
        xyz = _tool_data(messages, "build_structure", 0)["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"opt：用 ani2x 优化结构文件 {xyz}"})
    elif n == 3:
        spec = json.loads(_tool_data(messages, "delegate", 0)["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 4:
        inp = _tool_data(messages, "write_maple_inp", 1)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 5:
        final_xyz = _tool_data(messages, "run_maple", 0)["data"]["parsed"]["final_xyz"]
        coords = "\n".join(final_xyz.splitlines()[2:])  # 去掉 xyz 头两行
        yield _call("build_structure", {"source": coords, "source_type": "inline_coords"})
    elif n == 6:
        xyz = _tool_data(messages, "build_structure", 1)["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"freq：用 ani2x 对结构文件 {xyz} 做频率分析"})
    elif n == 7:
        spec = json.loads(_tool_data(messages, "delegate", 1)["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 8:
        inp = _tool_data(messages, "write_maple_inp", 2)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 9:
        summary = _tool_data(messages, "run_maple", 1)["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        freq = _tool_data(messages, "run_maple", 1)["data"]["parsed"]["frequencies_cm1"]
        n_imag = _tool_data(messages, "run_maple", 1)["data"]["n_imaginary"]
        yield (
            "频率任务要求结构来自 xyz 文件，已先优化再分析。"
            f"频率分析共 {len(freq)} 个模式，虚频 {n_imag} 个。"
        )


async def _orch_b18_stream(messages, info):
    """B18（§3.2 recovery）：首跑 md 不带 log_every（无逐步能量行，D5-1）
    → 发现无法分析漂移 → log_every=1 重跑（复用 B7 口径）。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": "H2O", "source_type": "name"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure", 0)["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"md：用 ani2x 对水分子跑 20 步 NVE，步长 0.5 fs，不带log_every，结构文件 {xyz}"})
    elif n == 2:
        spec = json.loads(_tool_data(messages, "delegate", 0)["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 3:
        inp = _tool_data(messages, "write_maple_inp", 0)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 4:
        # 诊断：无逐步能量行无法分析漂移 → log_every=1 重跑
        xyz = _tool_data(messages, "build_structure", 0)["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"md：用 ani2x 对水分子跑 20 步 NVE，步长 0.5 fs，log_every=1，结构文件 {xyz}"})
    elif n == 5:
        spec = json.loads(_tool_data(messages, "delegate", 1)["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 6:
        inp = _tool_data(messages, "write_maple_inp", 1)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 7:
        summary = _tool_data(messages, "run_maple", 1)["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        first = _tool_data(messages, "run_maple", 0)["data"]["parsed"]
        parsed = _tool_data(messages, "run_maple", 1)["data"]["parsed"]
        yield (
            f"首次运行缺少逐步能量行（仅 {len(first.get('energies_hartree') or [])} 条），"
            f"已用 log_every=1 重跑：完成 {len(parsed['energies_hartree'])} 步，"
            f"总能量漂移 {parsed.get('energy_drift_ha')} Hartree。"
        )


async def _orch_b19_stream(messages, info):
    """B19（§3.2 unguided）：不给模型/电荷 → ask_user_question 挂起；
    注入答复（ANI-2x，0/1）→ build_structure → opt → validator → 回答。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call(
            "ask_user_question",
            {"question": "请提供：1) 使用哪个 MLIP 模型；2) 体系电荷与自旋多重度"},
        )
    elif n == 1:
        yield _call("build_structure", {"source": "C2H5OH", "source_type": "name"})
    elif n == 2:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"opt：用 ani2x 优化结构文件 {xyz}"})
    elif n == 3:
        spec = json.loads(_tool_data(messages, "delegate")["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 4:
        inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 5:
        summary = _tool_data(messages, "run_maple")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        rets = _returns(messages, "ask_user_question")
        answer = _as_data(rets[0]).get("data", {}).get("answer", "") if rets else ""
        data = _tool_data(messages, "run_maple")["data"]
        yield (
            f"按您提供的参数（{answer}），乙醇优化收敛，"
            f"最终能量 {data['energy_hartree']:.10f} Hartree。"
        )


@dataclass(frozen=True)
class CannedScript:
    orchestrator: StreamFn
    task_parser: StreamFn
    result_validator: StreamFn = _validator_stream


def _b2_text(data: dict) -> str:
    return (
        f"水分子优化收敛（迭代 {data['parsed'].get('converged_at_iteration')} 步），"
        f"最终能量 {data['energy_hartree']:.6f} Hartree。"
    )


def _b7_text(data: dict) -> str:
    parsed = data["parsed"]
    return (
        f"NVE 分子动力学完成 {len(parsed['energies_hartree'])} 步，"
        f"总能量漂移 {parsed.get('energy_drift_ha')} Hartree。"
    )


SCRIPTS: dict[str, CannedScript] = {
    "B1": CannedScript(_orch_b1_stream, _parser_b1_stream),
    "B2": CannedScript(
        _orch_single_maple_job("H2O", "opt：用 ani2x 优化水分子", _b2_text),
        _parser_generic_stream,
    ),
    "B3": CannedScript(_orch_opt_freq_chain("H2O", "水分子"), _parser_generic_stream),
    "B4": CannedScript(_orch_b4_stream, _parser_generic_stream),
    "B5": CannedScript(_orch_b5_stream, _parser_generic_stream),
    "B6": CannedScript(_orch_opt_freq_chain("C2H5OH", "乙醇"), _parser_generic_stream),
    "B7": CannedScript(
        _orch_single_maple_job(
            "H2O", "md：用 ani2x 对水分子跑 20 步 NVE，步长 0.5 fs", _b7_text
        ),
        _parser_generic_stream,
    ),
    "B8": CannedScript(_orch_b8_stream, _parser_generic_stream),
    "B9": CannedScript(_orch_b9_stream, _parser_b1_stream),
    "B10": CannedScript(_orch_b1_stream, _parser_b1_stream),
    "B11": CannedScript(_orch_b11_stream, _parser_generic_stream),
    "B12": CannedScript(_orch_maple_sp_golden("ani1x", "ANI-1x"), _parser_generic_stream),
    "B13": CannedScript(_orch_maple_sp_golden("aimnet2", "AIMNet2"), _parser_generic_stream),
    "B14": CannedScript(_orch_maple_sp_golden("maceoff23s", "MACE-OFF23(S)"), _parser_generic_stream),
    "B20": CannedScript(_orch_maple_sp_golden("macempa0", "MACE-MPA-0"), _parser_generic_stream),
    "B21": CannedScript(_orch_b21_stream, _parser_generic_stream),
    "B22": CannedScript(
        _orch_maple_sp_golden("aimnet2nse", "AIMNet2-NSE（O2 三重态）", O2_TRIPLET_COORDS),
        _parser_generic_stream,
    ),
    "B26": CannedScript(
        _orch_maple_pbc_sp("uma", "UMA (uma-s-1p1, omat)", "用 uma(uma-s-1p1, omat) 算该周期结构的单点能"),
        _parser_generic_stream,
    ),
    "B27": CannedScript(
        _orch_maple_pbc_sp("macempa0", "MACE-MPA-0（MAPLE 插件后端）", "用 MAPLE 的 macempa0 插件后端算该周期结构的单点能"),
        _parser_generic_stream,
    ),
    "B28": CannedScript(_orch_b28_stream, _parser_generic_stream),
    "B15": CannedScript(_orch_b15_stream, _parser_generic_stream),
    "B16": CannedScript(_orch_b16_stream, _parser_generic_stream),
    "B17": CannedScript(_orch_b17_stream, _parser_generic_stream),
    "B18": CannedScript(_orch_b18_stream, _parser_generic_stream),
    "B19": CannedScript(_orch_b19_stream, _parser_generic_stream),
    "B23": CannedScript(
        _orch_pbc_direct_stream("run_dpa4", "DPA-4（head=OMat24）", {"head": "OMat24"}),
        _parser_generic_stream,
    ),
    "B24": CannedScript(
        _orch_pbc_direct_stream("run_nequip", "NequIP-OAM-XL"), _parser_generic_stream
    ),
    "B25": CannedScript(
        _orch_pbc_direct_stream("run_tace", "TACE-OAM-L"), _parser_generic_stream
    ),
}
