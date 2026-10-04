"""NequIP 直算脚本（M1-B）：由 COMPUTE_VENV 的 python 以子进程方式执行。

用法：
    $COMPUTE_VENV/bin/python app/compute/scripts/nequip_ase.py job.json

输入 job.json（UTF-8 JSON）字段：
    xyz_path   (str, 必填)  输入结构文件（ase.io 可读，通常 .xyz/.extxyz）。
               周期体系：extxyz 带 Lattice="9 floats" 头，ASE 自动设 cell+pbc；
               支持混合 pbc（slab [T,T,F]，matscipy 后端）
    task       (str, 必填)  "sp" 单点 | "opt" 几何优化（ASE LBFGS）
    model_path (str, 必填)  nequip-package 模型文件（*.nequip.zip）绝对路径
    output_dir (str, 必填)  输出目录（result.json / traj.extxyz / final.xyz 写在这里）
    device     (str, 可选)  "cuda"（默认）或 "cpu"；cuda 不可用时自动回退 cpu
    opt        (dict, 可选) {"fmax": 0.01, "steps": 200}，仅 task="opt" 使用

输出 output_dir/result.json 字段：
    ok                  (bool)  成功标志；失败时 false 且进程非零退出
    error               (str|null)
    task                (str)
    model_path, device  (str)   device 为实际使用的设备（可能与请求不同，回退时）
    energy_eV           (float) 单点或最终能量，单位 eV（ASE 标准，非 Hartree）
    forces_eV_per_A     (list[[fx,fy,fz], ...]) 最终结构的力，单位 eV/Å
    converged           (bool, 仅 opt)
    n_opt_steps         (int,  仅 opt) LBFGS 实际步数
    final_structure_path (str)  最终结构 xyz（sp 时即输入结构的转写）
    trajectory_path     (str|null, 仅 opt) traj.extxyz 全轨迹
    units               (dict)  显式单位标注
    versions            (dict)  nequip/torch/ase 版本
    wall_time_s         (float)

注意：本脚本只能依赖 COMPUTE_VENV 已有包（stdlib + torch + ase + nequip），
不得 import mlip_agent 后端包（两端 Python 版本不同，严格隔离）。
任何异常都写 result.json {ok:false, error:...} 并以退出码 1 结束。
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("nequip_ase")

UNITS = {"energy": "eV", "forces": "eV/Angstrom", "length": "Angstrom"}


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("device=cuda 请求但 CUDA 不可用，回退到 cpu")
        return "cpu"
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"不支持的 device: {requested!r}（只能 cuda/cpu）")
    return requested


def run_job(job: dict) -> dict:
    import ase.io
    from nequip.integrations.ase import NequIPCalculator

    required = ("xyz_path", "task", "model_path", "output_dir")
    missing = [k for k in required if k not in job]
    if missing:
        raise ValueError(f"job.json 缺少必填字段: {missing}")

    task = job["task"]
    if task not in ("sp", "opt"):
        raise ValueError(f"不支持的 task: {task!r}（只能 sp/opt）")

    output_dir = Path(job["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(job.get("device", "cuda"))

    atoms = ase.io.read(str(job["xyz_path"]))
    logger.info(
        "结构: %s, %d 原子, pbc=%s", atoms.get_chemical_formula(), len(atoms), list(atoms.pbc)
    )
    # 周期体系：extxyz 带 Lattice= 头时 ASE 自动设 cell+pbc，NequIP 原生支持
    # （OAM-XL 训练域即周期性晶体）。混合 pbc（如 slab [T,T,F]）可用——邻居表
    # 默认 matscipy 后端支持（vesin 不支持，本环境未装）。
    if atoms.pbc.any() and not atoms.cell.any():
        raise ValueError("pbc 为真但晶胞全零：周期体系必须带晶胞（extxyz 的 Lattice= 头）")

    calc = NequIPCalculator._from_saved_model(
        str(job["model_path"]),
        device=device,
        chemical_species_to_atom_type_map=True,  # 恒等映射，跳过警告
    )
    atoms.calc = calc

    result = {
        "ok": True,
        "error": None,
        "task": task,
        "model_path": str(job["model_path"]),
        "device": device,
        "units": UNITS,
        "versions": {
            "nequip": _pkg_version("nequip"),
            "torch": _pkg_version("torch"),
            "ase": _pkg_version("ase"),
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
        logger.info("sp: E = %.10f eV", energy)
    else:  # opt
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
            "opt: converged=%s, steps=%d, E = %.10f eV",
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
