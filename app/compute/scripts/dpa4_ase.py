"""DPA-4 直算脚本（TICKET-0006 阶段 2）：由 dpa4_venv 的 python 以子进程方式执行。

用法：
    LD_LIBRARY_PATH=/usr/local/lib64 $DPA4_VENV/bin/python app/compute/scripts/dpa4_ase.py job.json

输入 job.json（UTF-8 JSON）字段（与 tace_ase.py 契约对齐，另增 head/charge_spin 语义）：
    xyz_path   (str, 必填)  输入结构文件（ase.io 可读，通常 .xyz/.extxyz）。
               周期体系：extxyz 带 Lattice="9 floats" 头，ASE 自动设 cell+pbc；
               deepmd 对任何 cell 一律按全三维周期处理，slab 需真空层 >= 10 Å
               （> rcut=6 Å）+ 全周期胞（与 OC20/OMat24 训练协议一致）
    task       (str, 必填)  "sp" 单点 | "opt" 几何优化（ASE LBFGS；opt 可用但不作 pin——
                            力跨进程不逐位稳定 ~1e-7 eV/Å，根因未定位，见 D7-8）
    model_path (str, 必填)  checkpoint 绝对路径（family=dpa4-beta 时 DPA4-beta.pt；
                            family=dpa4c 时 DPA4C-*.pt）
    output_dir (str, 必填)  输出目录（result.json / traj.extxyz / final.xyz 写在这里）
    device     (str, 可选)  "cpu"（默认，CPU+OMP=1 逐位 pin 口径，TICKET-0006 裁定 2）
                            或 "cuda"；cuda 不可用时自动回退 cpu
    family     (str, 可选)  模型族："dpa4-beta"（默认，保持现状）| "dpa4c"。
                            dpa4c 为单域模型（无 multitask head 概念），DeepPot 不带
                            head 加载；job.json 显式传 head 时报错（head 参数会被
                            deepmd 静默忽略，显式拒绝避免误用）。是否走 DPA4C 路径
                            只由 family 决定，不做自动嗅探。
    head       (str, 可选)  仅 family=dpa4-beta。multitask 分支名，默认 "H2O_H2O_PD"
                            （水体系专属分支），可覆盖为 checkpoint 实际 23 分支之一
                            （以 checkpoint 分支名为准，注意 Alloy_APEX/OMat24 与目录内
                            input.json 的命名差异，见 D7-8）
    opt        (dict, 可选) {"fmax": 0.01, "steps": 200}，仅 task="opt" 使用

charge_spin 固定 [0.0, 1.0]（中性闭壳层；两族模型 has_default_chg_spin 默认即此值，
显式传与走默认逐位一致，见 dpa4_stage1_report.md §3②），不接受覆盖。

CPU 模式进程内强制：os.environ["OMP_NUM_THREADS"]="1"（import torch 之前）+
torch.set_num_threads(1)——此口径下 H2O 金标准几何能量跨进程逐位一致：
DPA4-beta -15.710773446015082 eV（B16 pin，GPU 值约 -15.7107726 eV）；
DPA4C     -2079.8665150032502 eV（B21 pin，OMol25 域输出总能量，与 DPA4-beta
的参考能约定不同，绝对值不可跨族直接比较）。CPU/GPU 两口径不可混用。

输出 output_dir/result.json 字段（与 tace_ase.py 契约对齐，另增）：
    ok, error, task, model_path, device（实际使用）,
    family            (str)   实际模型族（"dpa4-beta" / "dpa4c"）
    head              (str|null)  实际使用的分支；family=dpa4c 时恒为 null
    charge_spin       (list)  恒为 [0.0, 1.0]
    energy_eV         (float) 单点或最终能量，单位 eV（deepmd 推理原生单位）
    forces_eV_per_A   (list[[fx,fy,fz], ...]) 最终结构的力，单位 eV/Å
    converged / n_opt_steps   （仅 opt）
    final_structure_path / trajectory_path
    units             (dict)  显式单位标注
    versions          (dict)  deepmd-kit/torch/e3nn/vesin 版本 + device + 线程数
    wall_time_s       (float)

运行要求：LD_LIBRARY_PATH=/usr/local/lib64（gcc 11.3 编译产物需 GLIBCXX_3.4.29）；
run_dpa4 工具经 run_compute(extra_env=...) 无条件注入（TICKET-0006 前置条件 1）。
本脚本只依赖 dpa4_venv 已有包（stdlib + numpy + torch + ase + deepmd-kit），
不得 import mlip_agent 后端包（两端 Python 版本不同，严格隔离）。
任何异常都写 result.json {ok:false, error:...} 并以退出码 1 结束。
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("dpa4_ase")

UNITS = {"energy": "eV", "forces": "eV/Angstrom", "length": "Angstrom"}

# TICKET-0006 前置条件 3：head 默认 H2O_H2O_PD 可覆盖；charge_spin 固定 [0.0, 1.0]
DEFAULT_HEAD = "H2O_H2O_PD"
CHARGE_SPIN = [0.0, 1.0]
# 模型族：dpa4-beta（multitask，需 head）| dpa4c（单域，无 head 概念，TICKET-0006 任务 2）
FAMILIES = ("dpa4-beta", "dpa4c")
DEFAULT_FAMILY = "dpa4-beta"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _make_ase_calc(dp, type_map: list[str]):
    """DeepPot.eval 的 ASE Calculator 薄封装（sp/opt 共用同一 eval 路径）。

    自实现而非 deepmd.calculator.DP：保证 head 选择与固定 charge_spin 的语义
    与阶段 1 金标准配方（DeepPot.eval + charge_spin=[0.0,1.0]）逐字一致。
    周期体系（atoms.pbc 任一轴为真）传 3x3 晶胞矩阵；deepmd 推理对任何 cell
    一律按全三维周期处理（slab 需真空层 >= 10 Å，> rcut=6 Å），非周期传 None。
    """
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    class _Calc(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            symbols = self.atoms.get_chemical_symbols()
            atom_types = np.array([type_map.index(s) for s in symbols], dtype=np.int32)
            coords = self.atoms.get_positions().reshape(1, -1, 3)
            if self.atoms.pbc.any():
                cells = np.array(self.atoms.cell).reshape(1, 9)
            else:
                cells = None
            e, f, _v = dp.eval(coords, cells, atom_types, charge_spin=np.array(CHARGE_SPIN))
            self.results["energy"] = float(np.asarray(e).flatten()[0])
            self.results["forces"] = np.asarray(f).reshape(-1, 3)

    return _Calc()


def run_job(job: dict) -> dict:
    required = ("xyz_path", "task", "model_path", "output_dir")
    missing = [k for k in required if k not in job]
    if missing:
        raise ValueError(f"job.json 缺少必填字段: {missing}")

    task = job["task"]
    if task not in ("sp", "opt"):
        raise ValueError(f"不支持的 task: {task!r}（只能 sp/opt）")

    requested = job.get("device", "cpu")
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"不支持的 device: {requested!r}（只能 cuda/cpu）")

    family = str(job.get("family", DEFAULT_FAMILY))
    if family not in FAMILIES:
        raise ValueError(f"不支持的 family: {family!r}（只能 {FAMILIES}）")
    if family == "dpa4c":
        # DPA4C 单域无 multitask head；deepmd 会静默忽略 head 参数（实测），
        # 显式拒绝以避免"以为选了分支实际没生效"的误用
        if "head" in job:
            raise ValueError('family="dpa4c" 无 multitask head 概念，job.json 不应传 head')
        head = None
    else:
        head = str(job.get("head", DEFAULT_HEAD))

    output_dir = Path(job["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # CPU 模式（裁定 2）：import torch 之前强制 OMP_NUM_THREADS=1
    if requested == "cpu":
        os.environ["OMP_NUM_THREADS"] = "1"

    import torch

    device = requested
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("device=cuda 请求但 CUDA 不可用，回退到 cpu（OMP=1 口径）")
        device = "cpu"
        os.environ["OMP_NUM_THREADS"] = "1"
    if device == "cpu":
        torch.set_num_threads(1)
    # deepmd.pt 在 import 时读 DEVICE 环境变量选定设备，必须在 import 前设置
    os.environ["DEVICE"] = "cpu" if device == "cpu" else "cuda:0"

    import ase.io
    from deepmd.infer import DeepPot

    atoms = ase.io.read(str(job["xyz_path"]))
    logger.info(
        "结构: %s, %d 原子, pbc=%s", atoms.get_chemical_formula(), len(atoms), list(atoms.pbc)
    )
    if atoms.pbc.any() and not atoms.cell.any():
        raise ValueError("pbc 为真但晶胞全零：周期体系必须带晶胞（extxyz 的 Lattice= 头）")

    if family == "dpa4c":
        dp = DeepPot(str(job["model_path"]))
    else:
        dp = DeepPot(str(job["model_path"]), head=head)
    type_map = list(dp.get_type_map())
    logger.info(
        "模型: %s, family=%s, head=%s, type_map %d 元素",
        job["model_path"],
        family,
        head,
        len(type_map),
    )

    calc = _make_ase_calc(dp, type_map)
    atoms.calc = calc

    result = {
        "ok": True,
        "error": None,
        "task": task,
        "model_path": str(job["model_path"]),
        "device": device,
        "family": family,
        "head": head,
        "charge_spin": CHARGE_SPIN,
        "units": UNITS,
        "versions": {
            "deepmd-kit": _pkg_version("deepmd-kit"),
            "torch": _pkg_version("torch"),
            "e3nn": _pkg_version("e3nn"),
            "vesin": _pkg_version("vesin"),
            "device": device,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "torch_num_threads": torch.get_num_threads(),
        },
    }

    if task == "sp":
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        final_path = output_dir / "final.xyz"
        ase.io.write(str(final_path), atoms, format="extxyz")
        result.update(
            energy_eV=float(energy),
            forces_eV_per_A=[[float(x) for x in row] for row in forces],
            final_structure_path=str(final_path),
            trajectory_path=None,
        )
        logger.info("sp: E = %.15f eV", energy)
    else:  # opt（可用但不 pin：力跨进程不逐位稳定，见 D7-8）
        from ase.optimize import LBFGS

        opt_cfg = job.get("opt", {})
        fmax = float(opt_cfg.get("fmax", 0.01))
        max_steps = int(opt_cfg.get("steps", 200))
        traj_path = output_dir / "traj.extxyz"
        dyn = LBFGS(atoms, trajectory=str(traj_path))
        converged = bool(dyn.run(fmax=fmax, steps=max_steps))
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        final_path = output_dir / "final.xyz"
        ase.io.write(str(final_path), atoms, format="extxyz")
        result.update(
            energy_eV=float(energy),
            forces_eV_per_A=[[float(x) for x in row] for row in forces],
            converged=converged,
            n_opt_steps=int(dyn.get_number_of_steps()),
            fmax_eV_per_A=fmax,
            max_steps=max_steps,
            final_structure_path=str(final_path),
            trajectory_path=str(traj_path),
        )
        logger.info(
            "opt: converged=%s, steps=%d, E = %.15f eV",
            converged,
            dyn.get_number_of_steps(),
            energy,
        )

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    logger.info("写入 %s", result_path)
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"用法: {Path(argv[0]).name} <job.json>", file=sys.stderr)
        return 2

    job_path = Path(argv[1])
    t0 = time.monotonic()
    try:
        job = json.loads(job_path.read_text())
        result = run_job(job)
        # 追加墙钟时间后重写（保持 result.json 完整）
        result["wall_time_s"] = round(time.monotonic() - t0, 3)
        result_path = Path(job["output_dir"]).resolve() / "result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return 0
    except Exception:
        err = traceback.format_exc()
        logger.error("任务失败:\n%s", err)
        # 尽量把 result.json 写到 output_dir；output_dir 都解析不出时写到 job.json 旁边
        try:
            out_dir = Path(json.loads(job_path.read_text()).get("output_dir", "."))
        except Exception:
            out_dir = job_path.parent
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": False,
            "error": err.strip().splitlines()[-1],
            "traceback": err,
            "units": UNITS,
            "wall_time_s": round(time.monotonic() - t0, 3),
        }
        (out_dir / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
