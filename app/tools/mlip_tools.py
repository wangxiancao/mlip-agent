"""NequIP / TACE / DPA-4 直算工具（M2：复用 app/compute/runner.py 统一子进程封装）。

`run_nequip` / `run_tace` 流程：写 job.json → `run_compute` 执行
`$COMPUTE_VENV/bin/python app/compute/scripts/{nequip,tace}_ase.py job.json`
（GPU asyncio 锁、env 白名单、超时杀进程组、stdout/stderr 落盘均由 runner 承担）
→ 读 result.json。

`run_dpa4`（TICKET-0006）同流程但三点不同：解释器为 DPA-4 常驻 venv
（settings.dpa4_python，mlip_agent 之外）；默认 device=cpu 且 gpu=False
（CPU+OMP=1 逐位 pin 口径，裁定 2/4）；LD_LIBRARY_PATH=/usr/local/lib64
经 extra_env 无条件注入（前置条件 1）。

job_spec 字段（与 nequip_ase.py / tace_ase.py / dpa4_ase.py 的 job.json 一一对应）：
    xyz_path   (str, 必填)  输入结构文件路径
    task       (str, 必填)  "sp" | "opt"
    model_path (str, 可选)  缺省分别为 NequIP-OAM-XL 本地 zip / TACE-OAM-L 本地 pt /
               DPA4-beta.pt 绝对路径
    device     (str, 可选)  nequip/tace 默认 "cuda"（不可用自动回退）；
               dpa4 默认 "cpu"（OMP=1 pin 口径）
    output_dir (str, 可选)  缺省时由 JobAllocator 分配 session 作业目录
    opt        (dict, 可选) {"fmax": 0.01, "steps": 200}
    head       (str, 可选, 仅 dpa4)  multitask 分支名，默认 H2O_H2O_PD

返回值统一为 ToolResult 形态 dict：{"ok": bool, "data": dict|None, "error": str|None}，
不抛异常。data 即 result.json 全文（能量单位 eV，力单位 eV/Å，见 result["units"]）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from app.compute.runner import run_compute
from app.tools.common import ToolServices, err, ok

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "compute" / "scripts"
NEQUIP_SCRIPT = _SCRIPTS_DIR / "nequip_ase.py"
TACE_SCRIPT = _SCRIPTS_DIR / "tace_ase.py"
DPA4_SCRIPT = _SCRIPTS_DIR / "dpa4_ase.py"
DEFAULT_MODEL_PATH = "/root/soft/ml_potential/checkpoints/nequip/NequIP-OAM-XL-0.1.nequip.zip"
DEFAULT_TACE_MODEL_PATH = "/root/soft/ml_potential/checkpoints/tace/TACE-OAM-L.pt"
# TICKET-0006 前置条件 1：deepmd 进程运行需 LD_LIBRARY_PATH=/usr/local/lib64
# （gcc 11.3 编译产物要 GLIBCXX_3.4.29），经 run_compute(extra_env=...) 无条件注入
DPA4_LD_LIBRARY_PATH = "/usr/local/lib64"


async def _run_direct_impl(
    services: ToolServices,
    job_spec: dict,
    script: Path,
    default_model_path: str,
    label: str,
    timeout_s: int | None = None,
    *,
    interpreter: Path | None = None,
    default_device: str = "cuda",
    gpu: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict:
    job = dict(job_spec)
    job.setdefault("model_path", default_model_path)
    job.setdefault("device", default_device)
    # 子进程 cwd=output_dir，输入路径必须在写入 job.json 前绝对化
    for key in ("xyz_path", "model_path"):
        if key in job:
            job[key] = str(Path(job[key]).resolve())

    if "output_dir" in job:
        output_dir = Path(job["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        job_id = output_dir.name
    else:
        job_id, output_dir = services.job_allocator.allocate()
        job["output_dir"] = str(output_dir)

    if not Path(job.get("xyz_path", "")).is_file():
        return err(f"xyz_path 不存在: {job.get('xyz_path')!r}")

    job_path = output_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n")

    completed = await run_compute(
        [str(interpreter or services.settings.compute_python), str(script), str(job_path)],
        cwd=output_dir,
        timeout_s=timeout_s or services.settings.default_timeout_s,
        gpu=gpu,
        extra_env=extra_env,
    )

    result_path = output_dir / "result.json"
    if not result_path.is_file():
        return err(
            f"{label} 子进程未产出 result.json：{completed.error or '未知原因'}",
            data={"job_id": job_id, "job_dir": str(output_dir)},
        )

    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError as exc:
        return err(f"result.json 解析失败: {exc}")

    if not completed.ok or not result.get("ok"):
        return err(
            result.get("error") or completed.error or f"{label} 计算失败",
            data={**result, "job_id": job_id, "job_dir": str(output_dir)},
        )
    return ok({**result, "job_id": job_id, "job_dir": str(output_dir)})


async def run_nequip_impl(
    services: ToolServices, job_spec: dict, timeout_s: int | None = None
) -> dict:
    return await _run_direct_impl(
        services,
        job_spec,
        NEQUIP_SCRIPT,
        services.settings.nequip_model_path,
        "NequIP",
        timeout_s,
    )


async def run_tace_impl(
    services: ToolServices, job_spec: dict, timeout_s: int | None = None
) -> dict:
    return await _run_direct_impl(
        services,
        job_spec,
        TACE_SCRIPT,
        services.settings.tace_model_path,
        "TACE",
        timeout_s,
    )


async def run_dpa4_impl(
    services: ToolServices, job_spec: dict, timeout_s: int | None = None
) -> dict:
    """DPA-4 直算（TICKET-0006）：独立常驻 venv 解释器 + CPU pin 口径（gpu=False
    不过 GPU 锁，裁定 4）+ LD_LIBRARY_PATH 无条件注入（前置条件 1）。"""
    return await _run_direct_impl(
        services,
        job_spec,
        DPA4_SCRIPT,
        services.settings.dpa4_model_path,
        "DPA-4",
        timeout_s,
        interpreter=services.settings.dpa4_python,
        default_device="cpu",
        gpu=False,
        extra_env={"LD_LIBRARY_PATH": DPA4_LD_LIBRARY_PATH},
    )


def build_nequip_tools(services: ToolServices) -> list[Callable]:
    async def run_nequip(job_spec: dict, timeout_s: int = 1800) -> dict:
        """执行 NequIP-OAM-XL 单点或几何优化（GPU，需用户审批；直算路径，不走 MAPLE）。
        支持周期体系（OAM-XL 训练域即周期性晶体）：结构文件用带 Lattice= 头的
        extxyz 即可（ASE 自动设 cell+pbc，混合 pbc 可用）；opt 只优化原子坐标、
        不优化晶胞。

        Args:
            job_spec: {"xyz_path": 结构文件, "task": "sp"|"opt",
                "model_path"?: 模型 zip 绝对路径, "device"?: "cuda"|"cpu",
                "opt"?: {"fmax": 0.01, "steps": 200}}。
                能量单位 eV（ASE 标准，非 Hartree）。
            timeout_s: 子进程超时秒数，超时杀进程组并返回 error。
        """
        try:
            return await run_nequip_impl(services, job_spec, timeout_s)
        except Exception as exc:  # noqa: BLE001 - 工具不抛异常
            return err(f"run_nequip 内部错误: {exc}")

    return [run_nequip]


def build_tace_tools(services: ToolServices) -> list[Callable]:
    async def run_tace(job_spec: dict, timeout_s: int = 1800) -> dict:
        """执行 TACE-OAM-L 单点或几何优化（GPU，需用户审批；直算路径，不走 MAPLE）。
        支持周期体系（OAM-L 训练域即周期性材料）：结构文件用带 Lattice= 头且
        pbc="T T T" 的 extxyz；仅支持全三维周期，混合 pbc / 带胞非周期输入会被
        硬报错（TACE 会静默升级为 3D 周期），slab 用真空层 >= 10 Å + 全周期胞。
        邻居表 backend 按周期性自动选择（分子 ase / 周期 matscipy）。opt 只优化
        原子坐标、不优化晶胞。

        Args:
            job_spec: {"xyz_path": 结构文件, "task": "sp"|"opt",
                "model_path"?: checkpoint .pt 绝对路径, "device"?: "cuda"|"cpu",
                "opt"?: {"fmax": 0.01, "steps": 200}}。
                能量单位 eV（ASE 标准，非 Hartree）。
            timeout_s: 子进程超时秒数，超时杀进程组并返回 error。
        """
        try:
            return await run_tace_impl(services, job_spec, timeout_s)
        except Exception as exc:  # noqa: BLE001 - 工具不抛异常
            return err(f"run_tace 内部错误: {exc}")

    return [run_tace]


def build_dpa4_tools(services: ToolServices) -> list[Callable]:
    async def run_dpa4(job_spec: dict, timeout_s: int = 1800) -> dict:
        """执行 DPA-4（DPA4-beta，head=H2O_H2O_PD）或 DPA4C（family="dpa4c"）
        单点或几何优化（需用户审批；直算路径，不走 MAPLE）。charge_spin 固定
        [0.0,1.0]（中性闭壳层）。
        支持周期体系：结构文件用带 Lattice= 头的 extxyz（ASE 自动设 cell+pbc），
        周期任务选周期域 head（体相 OMat24/MPTrj、表面 OC20M/OC22、合金
        Alloy_APEX 等）；deepmd 一律按全三维周期处理，slab 需真空层 >= 10 Å。
        默认 CPU+OMP=1 逐位复现口径（sp 已验收 pin，B16/B21）；
        opt 可用但力跨进程不逐位稳定（~1e-7 eV/Å），不作 pin；freq/md 不支持。

        Args:
            job_spec: {"xyz_path": 结构文件, "task": "sp"|"opt",
                "model_path"?: checkpoint .pt 绝对路径, "device"?: "cpu"|"cuda",
                "family"?: "dpa4-beta"（默认，multitask）| "dpa4c"（紧凑族，
                    无 head 概念，需显式 model_path 指向 DPA4C checkpoint；
                    输出为总能量，与 dpa4-beta 的参考能约定不互比）,
                "head"?: multitask 分支名（仅 dpa4-beta；默认 H2O_H2O_PD，
                    以 checkpoint 分支名为准）,
                "opt"?: {"fmax": 0.01, "steps": 200}}。
                能量单位 eV（deepmd 推理原生单位，非 Hartree）。
            timeout_s: 子进程超时秒数，超时杀进程组并返回 error。
        """
        try:
            return await run_dpa4_impl(services, job_spec, timeout_s)
        except Exception as exc:  # noqa: BLE001 - 工具不抛异常
            return err(f"run_dpa4 内部错误: {exc}")

    return [run_dpa4]
