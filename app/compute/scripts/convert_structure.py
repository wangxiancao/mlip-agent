"""ASE 结构格式转换脚本：由 COMPUTE_VENV 的 python 以子进程方式执行。

用法：
    $COMPUTE_VENV/bin/python app/compute/scripts/convert_structure.py job.json

输入 job.json（UTF-8 JSON）字段：
    src_path    (str, 必填)  源结构文件（ase.io 可读的任意格式：cif/poscar/vasp/
                extxyz/xyz/pdb/gen/traj/mol/sdf 等，按扩展名自动识别）
    dst_path    (str, 必填)  目标文件（格式按扩展名自动识别）
    src_format  (str, 可选)  显式覆盖源格式（ase 格式名）
    dst_format  (str, 可选)  显式覆盖目标格式
    meta_path   (str, 可选)  conversion.json 写出路径（TICKET-0015，并行调用
                各自唯一）；缺省写 <dst_path 同目录>/conversion.json
    cell        (list[float], 可选)  补/覆盖晶胞（TICKET-0013）：3 值 (a,b,c)
                正交胞或 6 值 (a,b,c,α,β,γ)，ASE cellpar 约定；给出时对每帧
                set_cell + set_pbc(True)。用于上游（如 MAPLE opt 输出的纯
                xyz）丢失周期信息时补回晶胞。

输出 <dst_path 同目录>/conversion.json 字段：
    ok          (bool)
    error       (str|null)
    src_path, dst_path, src_format, dst_format  (str)
    n_frames    (int)   转换的帧数（多帧文件全部转换）
    n_atoms     (int)   首帧原子数
    formula     (str)
    pbc         (list[bool])
    cellpar     (list[float]|null)  a,b,c,α,β,γ（Å/度）；非周期或零胞为 null
    warnings    (list[str])  如目标格式丢失晶胞/周期信息

注意：本脚本只能依赖 COMPUTE_VENV 已有包（stdlib + ase），
不得 import mlip_agent 后端包（两端 Python 版本不同，严格隔离）。
任何异常都写 conversion.json {ok:false, error:...} 并以退出码 1 结束。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# 这些格式不承载晶胞/周期信息（写出时 ASE 会静默丢弃），用于 warnings
_CELL_LOSSY_FORMATS = {"pdb", "mol", "sdf", "gen"}


def run_job(job: dict) -> dict:
    import ase.io

    src = Path(job["src_path"])
    dst = Path(job["dst_path"])
    if not src.is_file():
        raise ValueError(f"源文件不存在: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_fmt = job.get("src_format")
    dst_fmt = job.get("dst_format")

    images = ase.io.read(str(src), index=":", format=src_fmt)
    if not images:
        raise ValueError(f"源文件不含任何结构帧: {src}")

    effective_dst_fmt = (dst_fmt or dst.suffix.lstrip(".")).lower()
    warnings: list[str] = []
    # TICKET-0013：可选 cell 补/覆盖晶胞（3 值正交 / 6 值 cellpar，ASE 执行）
    cell = job.get("cell")
    if cell is not None:
        from ase.geometry import cellpar_to_cell

        if len(cell) == 3:
            cellpar = [float(v) for v in cell] + [90.0, 90.0, 90.0]
        elif len(cell) == 6:
            cellpar = [float(v) for v in cell]
        else:
            raise ValueError(f"cell 只接受 3 值 (a,b,c) 或 6 值 (a,b,c,α,β,γ): {cell!r}")
        if any(v <= 0 for v in cellpar[:3]):
            raise ValueError(f"cell 的 a,b,c 必须为正: {cell!r}")
        if images[0].cell.any():
            warnings.append("源结构已有晶胞，被 cell 参数覆盖")
        for atoms in images:
            atoms.set_cell(cellpar_to_cell(cellpar))
            atoms.set_pbc(True)

    # POSCAR/VASP 必须带晶胞：无胞输入自动加 10 Å 真空盒（ASE vasp writer
    # 对零胞报 "Lattice vectors must be finite and non-parallel"）
    if effective_dst_fmt in ("vasp", "poscar") and not images[0].cell.any():
        import numpy as np

        for atoms in images:
            lo = atoms.positions.min(axis=0)
            hi = atoms.positions.max(axis=0)
            span = np.maximum(hi - lo, 1.0) + 10.0
            atoms.set_cell(np.diag(span))
            atoms.center()
            atoms.set_pbc(True)
        warnings.append(
            "源结构无晶胞，POSCAR 写出时自动加了 10 Å 真空盒（分子 → 周期格式的约定）"
        )

    ase.io.write(str(dst), images, format=dst_fmt)

    first = images[0]
    has_cell = bool(first.cell.any()) and first.pbc.any()
    effective_dst_fmt = (dst_fmt or dst.suffix.lstrip(".")).lower()
    if has_cell and effective_dst_fmt in _CELL_LOSSY_FORMATS:
        warnings.append(
            f"目标格式 {effective_dst_fmt} 不承载晶胞/周期信息，Lattice/pbc 已丢失"
        )
    if len(images) > 1 and effective_dst_fmt in ("poscar", "vasp", "cif", "gen"):
        warnings.append(f"多帧源（{len(images)} 帧）写入单帧格式，仅首帧被保存")

    cellpar = None
    if first.cell.any():
        cellpar = [round(float(v), 6) for v in first.cell.cellpar()]

    return {
        "ok": True,
        "error": None,
        "src_path": str(src),
        "dst_path": str(dst),
        "src_format": src_fmt or src.suffix.lstrip("."),
        "dst_format": effective_dst_fmt,
        "n_frames": len(images),
        "n_atoms": len(first),
        "formula": first.get_chemical_formula(),
        "pbc": [bool(x) for x in first.pbc],
        "cellpar": cellpar,
        "warnings": warnings,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"用法: {Path(argv[0]).name} <job.json>", file=sys.stderr)
        return 2
    job_path = Path(argv[1])
    try:
        job = json.loads(job_path.read_text())
        dst = Path(job.get("dst_path", "."))
        result = run_job(job)
        # TICKET-0015：meta_path 可选（并行调用各自唯一），缺省保持旧行为
        out_path = (
            Path(job["meta_path"]) if job.get("meta_path")
            else dst.parent / "conversion.json"
        )
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return 0
    except Exception:
        err = traceback.format_exc()
        try:
            job = json.loads(job_path.read_text())
            out_dir = (
                Path(job["meta_path"]).parent if job.get("meta_path")
                else Path(job.get("dst_path", ".")).parent
            )
        except Exception:
            out_dir = job_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"ok": False, "error": err.strip().splitlines()[-1], "traceback": err}
        (out_dir / "conversion.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
