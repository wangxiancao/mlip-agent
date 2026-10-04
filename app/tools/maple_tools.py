"""MAPLE 相关工具（PLAN §2.2）。

实现函数（*_impl）显式接收 ToolServices，便于单测；`build_maple_tools`
把服务闭包注入为 pydantic-ai 可注册的工具函数。全部返回 ToolResult，不抛异常。

模型清单数值依据 docs/decisions.md D0-3（2026-10-02 实测）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from app.compute.inp_builder import build_inp
from app.compute.jobspec import JobSpec, JobType, MapleModel
from app.compute.parsers import parse_out
from app.compute.runner import run_compute
from app.tools.common import ToolServices, err, ok

_HARTREE_TO_EV = 27.211386245988
_HARTREE_TO_KCAL_MOL = 627.5094740631
_EV_TO_CM1 = 8065.54429

_UNIT_ALIASES = {
    "hartree": "hartree",
    "ha": "hartree",
    "eh": "hartree",
    "ev": "eV",
    "kcal/mol": "kcal/mol",
    "kcal_mol": "kcal/mol",
    "kcalmol": "kcal/mol",
    "kj/mol": "kJ/mol",
    "kj_mol": "kJ/mol",
    "cm-1": "cm-1",
    "cm^-1": "cm-1",
    "cm**-1": "cm-1",
    "wavenumber": "cm-1",
}

# 以 Hartree 为枢纽的换算（cm-1 为每分子能量等效）
_TO_HARTREE = {
    "hartree": 1.0,
    "eV": 1.0 / _HARTREE_TO_EV,
    "kcal/mol": 1.0 / _HARTREE_TO_KCAL_MOL,
    "kJ/mol": 1.0 / (_HARTREE_TO_KCAL_MOL * 4.184),
    "cm-1": 1.0 / (_HARTREE_TO_EV * _EV_TO_CM1),
}

_RE_TASK_LINE = re.compile(r"^Task set to '(\w+)'", re.M)

# TICKET-0002：MAPLE 插件后端 env 注入常量。
# 机制：env 设置后插件模块在 maple 进程内首个 calculator 构建时即被 import
# 注册（load_calculator_plugins_from_env），实例化按 #model= 名字匹配；
# 因此注入严格以 JobSpec 选择插件模型名为条件，其余作业 env 一字节不变。
PLUGINS_DIR = Path(__file__).resolve().parents[2] / "plugins"
NEQUIP_PLUGIN_MODULE = "maple_nequip_plugin"
NEQUIP_PLUGIN_MODEL_NAMES = ("nequip-oam-xl",)
MACE_PLUGIN_MODULE = "maple_mace_plugin"
MACE_PLUGIN_MODEL_NAMES = ("macempa0",)

# 插件注册表：模型名 → 插件模块名（新插件模型在此登记）
_PLUGIN_REGISTRY: dict[str, str] = {
    **{name: NEQUIP_PLUGIN_MODULE for name in NEQUIP_PLUGIN_MODEL_NAMES},
    **{name: MACE_PLUGIN_MODULE for name in MACE_PLUGIN_MODEL_NAMES},
}

_RE_MODEL_NAME = re.compile(r"^#model=([A-Za-z0-9_-]+)", re.M)


def plugin_env_for_model(model_name: str | None) -> dict[str, str] | None:
    """模型名命中插件后端时返回需注入的 env，否则 None（子进程 env 完全不变）。"""
    module = _PLUGIN_REGISTRY.get((model_name or "").strip().lower())
    if module is None:
        return None
    return {
        "MAPLE_CALCULATOR_PLUGINS": module,
        "PYTHONPATH": str(PLUGINS_DIR),
    }


def _maple_extra_env(job_dir: Path, inp: Path) -> dict[str, str] | None:
    """决定一次 maple 子进程是否需要插件 env。

    首选作业目录的 job_spec.json（write_maple_inp 产物，即 JobSpec 事实来源）；
    缺 sidecar 时（手写 inp 直接 run_maple）回退解析 inp 的 #model= 行。
    """
    sidecar = job_dir / "job_spec.json"
    if sidecar.is_file():
        try:
            name = json.loads(sidecar.read_text())["model"]["name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            name = None
        else:
            return plugin_env_for_model(name)
    try:
        m = _RE_MODEL_NAME.search(inp.read_text())
    except OSError:
        return None
    return plugin_env_for_model(m.group(1) if m else None)

MODEL_CATALOG: list[dict] = [
    {
        "name": "ani2x",
        "label": "ANI-2x",
        "path": "MAPLE 内建（首次使用自动下载缓存）",
        "backend": "maple",
        "status": "available",
        "notes": "通用有机分子（C/H/N/O 等），benchmark 默认模型",
    },
    {
        "name": "ani1x",
        "label": "ANI-1x",
        "path": "MAPLE 内建（checkpoint 已缓存）",
        "backend": "maple",
        "status": "available",
        "notes": "ANI 早期版本；ani1ccx / ani1xnr 同名机制",
    },
    {
        "name": "aimnet2",
        "label": "AIMNet2",
        "path": "MAPLE 内建（checkpoint 已缓存）",
        "backend": "maple",
        "status": "available",
        "notes": (
            "支持带电体系与更多元素。支持周期体系（TICKET-0009：#pbc(...) 或带 "
            "Lattice 的 extxyz；PBC 时库仑缺省自动切 dsf，可显式 "
            "#model=aimnet2(coulomb_method=ewald)；PBC 仅承诺 sp/opt）。"
            "注意训练域为气相分子，周期计算仅分子晶体类可信。"
        ),
    },
    {
        "name": "aimnet2nse",
        "label": "AIMNet2-NSE",
        "path": "checkpoints/maple/aimnet2nse.pt（MAPLE model/ 为其符号链接，TICKET-0010；自转 TorchScript，TICKET-0007）",
        "backend": "maple",
        "status": "available",
        "notes": (
            "AIMNet2 开壳层版（ωB97M-D3/def2-TZVPP，UKS）：自由基/三重态/断键等"
            "多重度>1 体系（mult 经 JobSpec multiplicity 传入），14 元素 "
            "H B C N O F Si P S Cl As Se Br I，支持带电与周期体系（同 aimnet2 的 "
            "TICKET-0009 路径）。仅 sp 已验收（B22，O2 三重态 pin "
            "-150.4159493214 Ha），opt/freq/md 验收前勿用。许可 MIT。"
        ),
    },
    {
        "name": "maceoff23s",
        "label": "MACE-OFF23(S)",
        "path": "checkpoints/maple/maceoff23s.pt（MAPLE model/ 为其符号链接，TICKET-0010；本地 scripted wrapper，D7-4）",
        "backend": "maple",
        "status": "available",
        "notes": "有机小分子；仅 sp 已验收（forces 路径未验证，opt/freq/md 勿用）；另有 maceoff23m / maceoff23l",
    },
    {
        "name": "uma",
        "label": "UMA (uma-s-1p1, omol)",
        "path": "/root/soft/ml_potential/checkpoints/uma/uma-s-1p1.pt",
        "backend": "maple",
        "status": "available",
        "notes": "显式本地 checkpoint，禁止联网下载；uma-m-1p1 为 11GB 大模型，需用户显式确认。"
        "支持周期体系（MAPLE 内唯一 SUPPORTS_PBC 后端）：周期 task 取 omat/oc20/oc22/oc25/omc/odac"
        "（omol 与 PBC 互斥），结构用 #pbc(...) 或带 Lattice 的 extxyz；注意 MAPLE 的 UMA 无"
        " stress——opt 为定胞优化（不动晶胞），NPT 压力为理想气体近似，致密体系不可靠",
    },
    {
        "name": "nequip-oam-xl",
        "label": "NequIP-OAM-XL",
        "path": "/root/soft/ml_potential/checkpoints/nequip/NequIP-OAM-XL-0.1.nequip.zip",
        "backend": "nequip_direct",
        "status": "available",
        "notes": (
            "默认直算路径（run_nequip，能量单位 eV，支持 sp/opt）。训练域为周期性晶体"
            "（OMat24→sAlex+MPtrj），支持周期体系（extxyz 带 Lattice 即可，含混合 pbc）；"
            "opt 只动原子坐标、不优化晶胞。另有 backend=maple 插件选项"
            "（JobSpec model.name=nequip-oam-xl，TICKET-0002，能量经 MAPLE 统一为 Hartree）："
            "可参与 MAPLE 的 opt/freq/scan/md 编排（插件路径仅非周期）。注意：插件路径 freq 走"
            "数值 Hessian（每个自由度两次 forward），大分子很慢，freq 默认仍用 ANI/UMA。"
        ),
    },
    {
        "name": "macempa0",
        "label": "MACE-MPA-0 (medium)",
        "path": "/root/soft/ml_potential/checkpoints/mace/macempa0.pt",
        "backend": "maple_plugin",
        "status": "available",
        "notes": (
            "MIT 许可；mace-foundations 官方 mace-mpa-0-medium（约 9.1M 参数），"
            "训练域为 MPTrj+sAlex 周期性晶体——单分子属域外使用，数值可用但物理精度"
            "不代表分子体系。经 plugins/maple_mace_plugin.py 注册（复用 MAPLE 内建 "
            "MACECalculator 实现，能量经 MAPLE 统一为 Hartree）；支持周期体系"
            "（TICKET-0008，#pbc(...) 或带 Lattice 的 extxyz；PBC 仅承诺 sp/opt，"
            "freq/hessian 无 Γ 点投影未验证）。sp 已验收：B20（分子 pin）+ B27（周期 pin，"
            "与官方 mace ASE 路径零差对照）；forces/opt/freq/md 路径验收前勿用。"
        ),
    },
    {
        "name": "tace-oam-l",
        "label": "TACE-OAM-L",
        "path": "/root/soft/ml_potential/checkpoints/tace/TACE-OAM-L.pt",
        "backend": "tace_direct",
        "status": "available",
        "notes": (
            "直算路径（run_tace，能量单位 eV）。训练域为周期性材料（OMat24→sAlex+MPtrj），"
            "支持全三维周期体系（extxyz 带 Lattice 且 pbc=\"T T T\"；邻居表自动切 matscipy）；"
            "混合 pbc / 带胞非周期输入硬报错（TACE 会静默升级为 3D 周期），slab 用真空层 "
            ">= 10 Å。opt 只动原子坐标、不优化晶胞。"
            "元素覆盖（按 checkpoint type map 实测 89 种）：Z=1–83（H–Bi）"
            "加 Z=89–94（Ac–Pu），不含 Po/At/Rn/Fr/Ra。"
            "仅 sp/opt 已验收（B15）。"
        ),
    },
    {
        "name": "dpa-4",
        "label": "DPA-4 (DPA4-beta, head=H2O_H2O_PD)",
        "path": "/root/soft/ml_potential/checkpoints/dpa4/DPA4-beta.pt",
        "backend": "dpa4_direct",
        "status": "available",
        "notes": (
            "直算路径（run_dpa4，能量单位 eV；TICKET-0006）。支持周期体系（extxyz 带 "
            "Lattice；deepmd 一律按全三维周期处理，slab 需真空层 >= 10 Å），周期任务选"
            "周期域 head（体相 OMat24/MPTrj、表面 OC20M/OC22、合金 Alloy_APEX 等）。"
            "charge_spin 固定 [0.0,1.0]（中性闭壳层）；head 默认 H2O_H2O_PD 可覆盖"
            "（checkpoint 23 分支，以 checkpoint 分支名为准——注意 Alloy_APEX/OMat24 "
            "与目录内 input.json 的命名差异）。sp 已验收（B16，CPU+OMP=1 逐位 pin "
            "-15.710773446015082 eV，GPU 值口径不同不可混用）；opt 可交互使用但力跨进程"
            "不逐位稳定（~1e-7 eV/Å，根因未定位），不作 pin；freq/md 不支持。"
            "元素覆盖 118 种（checkpoint type map）。"
        ),
    },
    {
        "name": "dpa4c-air",
        "label": "DPA4C-Air (OMol25-100M, 0.63M 参数)",
        "path": "/root/soft/ml_potential/checkpoints/dpa4/DPA4C-Air-OMol25-100M-v20260820.pt",
        "backend": "dpa4_direct",
        "status": "available",
        "notes": (
            "DPA4C 紧凑高吞吐族（run_dpa4 需传 family=\"dpa4c\" + 显式 model_path；"
            "无 multitask head 概念，显式传 head 会报错——DeepPot 对 DPA4C 会静默忽略 "
            "head，family 字段即防线）。OMol25 分子域（电荷/自旋经 charge_spin 固定 "
            "[0.0,1.0]），元素全覆盖 H–Og。注意：输出为总能量（H2O ≈ -2080 eV），与 "
            "DPA4-beta 的 B16 参考能约定不互比。sp 已验收（B21，CPU+OMP=1 pin "
            "-2079.8665150032502 eV，GPU 口径不同不可混用）。许可 CC BY-NC 4.0（非商业）。"
        ),
    },
]


def _normalize_unit(unit: str) -> str | None:
    return _UNIT_ALIASES.get(unit.strip().lower())


def infer_job_type(out_text: str) -> JobType:
    """从 .out 文本推断任务类型（MAPLE 头部有 `Task set to '<type>'` 行）。"""
    m = _RE_TASK_LINE.search(out_text)
    if m:
        return JobType(m.group(1))
    return JobType.SP


def _job_type_for_dir(job_dir: Path, out_text: str) -> JobType:
    sidecar = job_dir / "job_spec.json"
    if sidecar.is_file():
        try:
            return JobType(json.loads(sidecar.read_text())["job_type"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return infer_job_type(out_text)


def write_maple_inp_impl(services: ToolServices, job_spec: dict) -> dict:
    try:
        spec = JobSpec.model_validate(job_spec)
    except ValidationError as exc:
        problems = [
            f"{'/'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]
        return err("JobSpec 校验失败:\n" + "\n".join(problems))

    # 插件后端缺省 model_path 走 Settings（NEQUIP_MODEL_PATH / MACEMPA0_MODEL_PATH 可覆盖）
    if (
        spec.model.name == MapleModel.NEQUIP_OAM_XL.value
        and spec.model.model_path is None
    ):
        spec.model.model_path = services.settings.nequip_model_path
    if (
        spec.model.name == MapleModel.MACEMPA0.value
        and spec.model.model_path is None
    ):
        spec.model.model_path = services.settings.macempa0_model_path

    job_id, job_dir = services.job_allocator.allocate()
    # 相对 xyz 引用解析到 session workspace，避免子进程 cwd 下找不到文件
    if spec.structure.xyz_file is not None:
        xyz = Path(spec.structure.xyz_file)
        if not xyz.is_absolute():
            xyz = (services.session_workspace / xyz).resolve()
        if not xyz.is_file():
            return err(f"xyz_file 不存在: {xyz}")
        spec.structure.xyz_file = str(xyz)

    inp_path = job_dir / "input.inp"
    inp_path.write_text(build_inp(spec))
    (job_dir / "job_spec.json").write_text(
        json.dumps(spec.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    )
    # 起始结构落盘为 xyz 工件（前端 StructureView 3D 渲染的数据源）
    structure_xyz = job_dir / "structure.xyz"
    if spec.structure.coords is not None:
        lines = [str(len(spec.structure.coords)), f"job {job_id} input structure"]
        lines += [
            f"{a.symbol} {a.x:.8f} {a.y:.8f} {a.z:.8f}" for a in spec.structure.coords
        ]
        structure_xyz.write_text("\n".join(lines) + "\n")
    else:
        # xyz_file 已在上文校验存在并解析为绝对路径
        structure_xyz.write_text(Path(spec.structure.xyz_file).read_text())
    return ok(
        {
            "job_id": job_id,
            "inp_path": str(inp_path),
            "job_dir": str(job_dir),
            "job_type": str(spec.job_type),
        }
    )


async def run_maple_impl(
    services: ToolServices, inp_path: str, timeout_s: int | None = None
) -> dict:
    inp = Path(inp_path)
    if not inp.is_file():
        return err(f"inp 文件不存在: {inp_path}")
    job_dir = inp.parent
    out_name = inp.with_suffix(".out").name
    timeout = timeout_s or services.settings.default_timeout_s

    job = await run_compute(
        [str(services.settings.maple_bin), inp.name, out_name],
        cwd=job_dir,
        timeout_s=timeout,
        gpu=True,
        # TICKET-0002 前置条件 1：仅当 JobSpec 选择插件模型名时注入
        # MAPLE_CALCULATOR_PLUGINS + PYTHONPATH；其余作业 env 不变
        extra_env=_maple_extra_env(job_dir, inp),
    )
    if not job.ok:
        return err(
            f"MAPLE 计算失败: {job.error}",
            data={"job_dir": str(job_dir), "exit_code": job.exit_code},
        )

    out_text = Path(job.out_path).read_text()
    job_type = _job_type_for_dir(job_dir, out_text)
    parsed = parse_out(out_text, job_type)
    data = {
        "job_id": job_dir.name,
        "job_dir": str(job_dir),
        "out_path": job.out_path,
        "exit_code": job.exit_code,
        "duration_s": job.duration_s,
        "parsed": parsed.model_dump(mode="json"),
        # 顶层便捷字段，便于 result_validator 与回答组织
        "job_type": job_type.value,
        "energy_hartree": parsed.energy_hartree,
        "converged": parsed.converged,
        "n_imaginary": parsed.n_imaginary,
        "parse_warnings": parsed.warnings,
    }
    if not parsed.ok:
        return err(
            "计算完成但输出解析失败: " + "; ".join(parsed.warnings), data=data
        )
    return ok(data)


def parse_maple_out_impl(out_path: str, job_type: str | None = None) -> dict:
    out = Path(out_path)
    if not out.is_file():
        return err(f".out 文件不存在: {out_path}")
    text = out.read_text()
    jt = JobType(job_type) if job_type else _job_type_for_dir(out.parent, text)
    parsed = parse_out(text, jt)
    data = parsed.model_dump(mode="json")
    if not parsed.ok:
        return err("解析失败: " + "; ".join(parsed.warnings), data=data)
    return ok(data)


def convert_units_impl(value: float, from_unit: str, to_unit: str) -> dict:
    src = _normalize_unit(from_unit)
    dst = _normalize_unit(to_unit)
    if src is None:
        return err(f"不支持的能量单位: {from_unit!r}（支持 hartree/eV/kcal/mol/kJ/mol/cm-1）")
    if dst is None:
        return err(f"不支持的能量单位: {to_unit!r}（支持 hartree/eV/kcal/mol/kJ/mol/cm-1）")
    converted = value * _TO_HARTREE[src] / _TO_HARTREE[dst]
    return ok(
        {"value": converted, "from": {"value": value, "unit": src}, "unit": dst}
    )


def get_job_artifact_impl(services: ToolServices, job_id: str, name: str) -> dict:
    job_dir = services.job_allocator.resolve(job_id)
    if job_dir is None:
        return err(f"未知作业: {job_id}")
    if not name or Path(name).name != name:
        return err(f"非法工件名（只允许文件名，不含路径）: {name!r}")
    artifact = job_dir / name
    if not artifact.is_file():
        available = sorted(p.name for p in job_dir.iterdir() if p.is_file())
        return err(f"工件不存在: {name}；该作业现有文件: {available}")
    return ok(str(artifact))


def build_maple_tools(services: ToolServices) -> list[Callable]:
    def list_available_models() -> dict:
        """列出全部可用 MLIP 模型：名称、checkpoint 路径、后端（MAPLE/NequIP 直算）、
        适用体系说明与下载状态。选择模型前先调用本工具。"""
        return ok(MODEL_CATALOG)

    def write_maple_inp(job_spec: dict) -> dict:
        """把 JobSpec 渲染为 MAPLE .inp 并写入新作业目录，返回 job_id 与 inp 路径。

        Args:
            job_spec: 完整 JobSpec dict（job_type/model/structure/charge/multiplicity
                及 opt/freq/scan/md 子段）。schema 校验失败时 error 列出问题字段。
        """
        try:
            return write_maple_inp_impl(services, job_spec)
        except Exception as exc:  # noqa: BLE001
            return err(f"write_maple_inp 内部错误: {exc}")

    async def run_maple(inp_path: str, timeout_s: int = 1800) -> dict:
        """执行 MAPLE 计算（GPU，需用户审批）。自动解析输出并返回能量/收敛/频率摘要。

        Args:
            inp_path: write_maple_inp 返回的 .inp 路径（作业目录为其父目录）。
            timeout_s: 超时秒数，超时杀进程组并返回 error。
        """
        try:
            return await run_maple_impl(services, inp_path, timeout_s)
        except Exception as exc:  # noqa: BLE001
            return err(f"run_maple 内部错误: {exc}")

    def parse_maple_out(out_path: str, job_type: str | None = None) -> dict:
        """纯解析 MAPLE .out 文件，无副作用。

        Args:
            out_path: .out 文件路径。
            job_type: sp/opt/freq/scan/md；缺省时从文件内容自动推断。
        """
        try:
            return parse_maple_out_impl(out_path, job_type)
        except Exception as exc:  # noqa: BLE001
            return err(f"parse_maple_out 内部错误: {exc}")

    def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
        """能量单位换算（Hartree / eV / kcal/mol / kJ/mol / cm-1）。

        Args:
            value: 数值。
            from_unit: 源单位（大小写不敏感，Ha、E_h、kcal_mol 等别名均可）。
            to_unit: 目标单位。
        """
        return convert_units_impl(value, from_unit, to_unit)

    def get_job_artifact(job_id: str, name: str) -> dict:
        """返回作业工件文件路径（供前端下载或后续任务引用）。

        Args:
            job_id: 作业 ID（write_maple_inp 返回，如 job-0001）。
            name: 工件文件名（input.inp / output.out / stdout.log 等）。
        """
        return get_job_artifact_impl(services, job_id, name)

    return [
        list_available_models,
        write_maple_inp,
        run_maple,
        parse_maple_out,
        convert_units,
        get_job_artifact,
    ]
