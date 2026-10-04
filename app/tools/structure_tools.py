"""结构构建/读写工具（PLAN §2.2）。

`build_structure` 三种来源：
- name：分子名 → COMPUTE_VENV 子进程跑 `app/compute/scripts/build_structure.py`
  （ase.build.molecule / ase.collections.g2）。后端环境没有也不许装 ase，
  名称解析只能发生在计算环境侧，故选子进程而非内置坐标表（见 decisions.md D2-2）。
- xyz：本地 xyz 文件路径，解析后复制进 session workspace；源文件带 Lattice=
  头（extxyz 周期结构）时原样复制，保留周期信息。
- inline_coords：内联坐标文本（每行 `Symbol x y z`，Å）。

可选 cell（3 值正交盒 / 6 值 a,b,c,α,β,γ）：输出带 Lattice/pbc 头的 extxyz，
供周期性计算（直算三路 ase.io.read 自动解析；MAPLE xyz_reader 同样解析）。

`convert_structure`：ASE io 格式转换（cif/poscar/extxyz/xyz/pdb/gen/traj/mol/sdf
互转），COMPUTE_VENV 轻量子进程（`app/compute/scripts/convert_structure.py`），
返回目标路径 + 化学式/原子数/帧数/pbc/cellpar 元数据；POSCAR 对无胞输入自动
加 10 Å 真空盒，晶胞丢失型格式给警告。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from app.compute.runner import run_compute
from app.tools.common import ToolServices, err, ok

BUILD_STRUCTURE_SCRIPT = (
    Path(__file__).resolve().parent.parent / "compute" / "scripts" / "build_structure.py"
)

_COORD_LINE = re.compile(
    r"^\s*([A-Z][a-z]?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)


def parse_coord_lines(text: str) -> list[dict]:
    """解析内联坐标行；遇到非法行抛 ValueError。"""
    atoms: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        m = _COORD_LINE.match(line)
        if not m:
            raise ValueError(f"第 {lineno} 行不是合法坐标行: {line.strip()!r}")
        atoms.append(
            {
                "symbol": m.group(1),
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "z": float(m.group(4)),
            }
        )
    if not atoms:
        raise ValueError("坐标为空")
    return atoms


def parse_xyz_text(text: str) -> list[dict]:
    """解析标准 xyz 文本（首行原子数，次行注释）。"""
    lines = text.splitlines()
    if not lines:
        raise ValueError("xyz 内容为空")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"xyz 首行应为原子数: {lines[0].strip()!r}") from None
    body = "\n".join(lines[2 : 2 + n_atoms])
    atoms = parse_coord_lines(body)
    if len(atoms) != n_atoms:
        raise ValueError(f"xyz 声明 {n_atoms} 个原子，实际解析到 {len(atoms)} 个")
    return atoms


def formula_of(atoms: list[dict]) -> str:
    counts: dict[str, int] = {}
    for a in atoms:
        counts[a["symbol"]] = counts.get(a["symbol"], 0) + 1
    # Hill 序：C 优先，其次 H，其余字母序
    symbols = sorted(counts, key=lambda s: (s != "C", s != "H", s))
    return "".join(
        f"{s}{counts[s] if counts[s] > 1 else ''}" for s in symbols
    )


def write_xyz(atoms: list[dict], path: Path, comment: str = "") -> Path:
    lines = [str(len(atoms)), comment]
    lines.extend(f"{a['symbol']:<2} {a['x']:18.8f} {a['y']:18.8f} {a['z']:18.8f}" for a in atoms)
    path.write_text("\n".join(lines) + "\n")
    return path


def cellpar_to_lattice(cell: list[float]) -> list[list[float]]:
    """晶胞参数 → 3x3 晶格矩阵（行向量，Å）。

    3 个值 (a,b,c) 视为正交盒；6 个值 (a,b,c,α,β,γ)。公式与
    ase.geometry.Cell.fromcellpar 一致；2 个值（MAPLE 的 2D 用法约定）
    在此处不接受——extxyz 需要显式的 c（slab 用真空层）。
    """
    import math

    if len(cell) == 3:
        a, b, c = cell
        alpha = beta = gamma = 90.0
    elif len(cell) == 6:
        a, b, c, alpha, beta, gamma = cell
    else:
        raise ValueError(f"cell 需 3 或 6 个值（a,b,c[,α,β,γ]），得到 {len(cell)} 个")
    if min(a, b, c) <= 0:
        raise ValueError(f"晶胞边长必须为正: {cell}")
    if not all(0.0 < ang < 180.0 for ang in (alpha, beta, gamma)):
        raise ValueError(f"晶胞角度必须在 (0,180): {cell}")

    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    sin_g = math.sin(gr)
    # 正交盒走精确路径，避免浮点噪声
    if all(abs(ang - 90.0) < 1e-9 for ang in (alpha, beta, gamma)):
        return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
    cx = c * math.cos(br)
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0:
        raise ValueError(f"晶胞参数不构成合法三维胞: {cell}")
    return [
        [a, 0.0, 0.0],
        [b * math.cos(gr), b * sin_g, 0.0],
        [cx, cy, math.sqrt(cz_sq)],
    ]


def write_extxyz(
    atoms: list[dict], path: Path, cell: list[float], comment: str = ""
) -> Path:
    """写带 Lattice/pbc 头的 extxyz（ase.io.read 自动解析为周期结构）。"""
    lattice = cellpar_to_lattice(cell)
    flat = " ".join(f"{v:.10f}" for row in lattice for v in row)
    header = f'Lattice="{flat}" pbc="T T T" {comment}'.strip()
    return write_xyz(atoms, path, comment=header)


async def build_structure_impl(
    services: ToolServices,
    source: str,
    source_type: Literal["name", "xyz", "inline_coords"],
    cell: list[float] | None = None,
) -> dict:
    workspace = services.session_workspace
    workspace.mkdir(parents=True, exist_ok=True)

    if source_type == "name":
        if cell is not None:
            return err("source_type=name 不支持 cell（名称构建的都是分子）；周期结构请用 inline_coords/xyz + cell")
        # TICKET-0017：每次调用独立子目录，杜绝并行时 meta.json/structure.xyz
        # 互相覆盖（deliverable 一并落在子目录内，返回值指向它）
        build_dir = workspace / "build" / uuid4().hex[:8]
        build_dir.mkdir(parents=True, exist_ok=True)
        job = await run_compute(
            [
                str(services.settings.compute_python),
                str(BUILD_STRUCTURE_SCRIPT),
                source,
                str(build_dir),
            ],
            cwd=workspace,
            timeout_s=services.settings.light_timeout_s,
            gpu=False,
        )
        meta_path = build_dir / "meta.json"
        if not meta_path.is_file():
            return err(f"结构构建子进程未产出 meta.json：{job.error or '未知原因'}")
        meta = json.loads(meta_path.read_text())
        if not meta.get("ok"):
            return err(
                (meta.get("error") or "未知分子名")
                + "；请改用 inline_coords 提供坐标，或 xyz 指定结构文件"
            )
        return ok(
            {
                "xyz_path": meta["xyz_path"],
                "formula": meta["formula"],
                "n_atoms": meta["n_atoms"],
                "source_type": "name",
            }
        )

    if source_type == "xyz":
        src = Path(source)
        if not src.is_file():
            return err(f"xyz 文件不存在: {source}")
        try:
            atoms = parse_xyz_text(src.read_text())
        except ValueError as exc:
            return err(f"xyz 解析失败: {exc}")
        # 源文件自带 Lattice= 头（extxyz 周期结构）且未显式给 cell：原样复制，
        # 避免重写为 plain xyz 丢失周期信息
        src_lines = src.read_text().splitlines()
        if cell is None and len(src_lines) >= 2 and 'Lattice="' in src_lines[1]:
            dest = workspace / "structure.extxyz"
            dest.write_text(src.read_text())
            return ok(
                {
                    "xyz_path": str(dest),
                    "formula": formula_of(atoms),
                    "n_atoms": len(atoms),
                    "atoms": atoms,
                    "source_type": source_type,
                    "cell": "from_file",
                }
            )
    else:  # inline_coords
        try:
            atoms = parse_coord_lines(source)
        except ValueError as exc:
            return err(f"内联坐标解析失败: {exc}")

    if cell is not None:
        try:
            xyz_path = write_extxyz(
                atoms, workspace / "structure.extxyz", cell, comment=formula_of(atoms)
            )
        except ValueError as exc:
            return err(f"晶胞参数非法: {exc}")
    else:
        xyz_path = write_xyz(atoms, workspace / "structure.xyz", comment=formula_of(atoms))
    return ok(
        {
            "xyz_path": str(xyz_path),
            "formula": formula_of(atoms),
            "n_atoms": len(atoms),
            "atoms": atoms,
            "source_type": source_type,
            "cell": cell,
        }
    )


CONVERT_STRUCTURE_SCRIPT = (
    Path(__file__).resolve().parent.parent / "compute" / "scripts" / "convert_structure.py"
)

# 目标格式 → 扩展名（poscar/vasp 同义）
_CONVERT_FORMAT_EXT = {
    "xyz": ".xyz",
    "extxyz": ".extxyz",
    "cif": ".cif",
    "poscar": ".vasp",
    "vasp": ".vasp",
    "pdb": ".pdb",
    "gen": ".gen",
    "traj": ".traj",
    "mol": ".mol",
    "sdf": ".sdf",
}


async def convert_structure_impl(
    services: ToolServices,
    source_path: str,
    output_format: str | None = None,
    output_name: str | None = None,
    cell: list[float] | None = None,
) -> dict:
    """结构文件格式转换（ASE io，COMPUTE_VENV 子进程，轻量不需审批）。"""
    workspace = services.session_workspace
    workspace.mkdir(parents=True, exist_ok=True)

    if cell is not None:
        if len(cell) not in (3, 6):
            return err(f"cell 只接受 3 值 (a,b,c) 或 6 值 (a,b,c,α,β,γ): {cell!r}")
        try:
            cell = [float(v) for v in cell]
        except (TypeError, ValueError):
            return err(f"cell 必须是数值列表: {cell!r}")
        if any(v <= 0 for v in cell[:3]):
            return err(f"cell 的 a,b,c 必须为正: {cell!r}")

    src = Path(source_path)
    if not src.is_absolute():
        src = (workspace / src).resolve()
    if not src.is_file():
        return err(f"源文件不存在: {src}")

    fmt = (output_format or "").strip().lower() or None
    if fmt is not None and fmt not in _CONVERT_FORMAT_EXT:
        return err(
            f"不支持的目标格式: {output_format!r}"
            f"（支持 {'/'.join(sorted(_CONVERT_FORMAT_EXT))}）"
        )
    if output_name:
        if Path(output_name).name != output_name:
            return err(f"output_name 只允许文件名（不含路径）: {output_name!r}")
        dst = workspace / output_name
    else:
        ext = _CONVERT_FORMAT_EXT.get(fmt or "", None) or (
            "." + fmt if fmt else src.suffix + ".conv"
        )
        dst = workspace / f"{src.stem}{ext}"
    if dst.resolve() == src.resolve():
        return err("源与目标相同，拒绝原地覆盖")

    # TICKET-0017：每次调用独立子目录（与 jobs/job-000N 同一理念），
    # job/meta 用普通文件名，从结构上消除并行覆盖
    work_dir = workspace / "convert" / uuid4().hex[:8]
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / "job.json"
    meta_path = work_dir / "conversion.json"
    job_payload = {
        # 不把用户格式名传给 ASE（ASE format= 只认正则名，poscar/pdb 之类
        # 别名会 UnknownFileTypeError）；扩展名映射已保证 dst 后缀可识别
        "src_path": str(src),
        "dst_path": str(dst),
        "meta_path": str(meta_path),
    }
    if cell is not None:
        job_payload["cell"] = cell
    job_path.write_text(json.dumps(job_payload, ensure_ascii=False))
    job = await run_compute(
        [str(services.settings.compute_python), str(CONVERT_STRUCTURE_SCRIPT), str(job_path)],
        cwd=workspace,
        timeout_s=services.settings.light_timeout_s,
        gpu=False,
    )
    if not meta_path.is_file():
        return err(f"格式转换子进程未产出 conversion.json：{job.error or '未知原因'}")
    meta = json.loads(meta_path.read_text())
    if not meta.get("ok"):
        return err(f"格式转换失败: {meta.get('error')}")
    meta["dst_path"] = str(dst)
    return ok(meta)


def build_structure_tools(services: ToolServices) -> list[Callable]:
    async def build_structure(
        source: str,
        source_type: Literal["name", "xyz", "inline_coords"],
        cell: list[float] | None = None,
    ) -> dict:
        """构建分子/周期结构并写入 session workspace，返回结构文件路径、化学式、原子数。

        Args:
            source: source_type=name 时为分子名（如 H2O/NH3/CH4/C2H5OH）；
                xyz 时为本地 xyz/extxyz 文件路径（自带 Lattice= 头的周期结构
                会原样保留周期信息）；inline_coords 时为坐标文本
                （每行 `Symbol x y z`，单位 Å）。
            source_type: 结构来源类型，三选一。
            cell: 周期晶胞参数（可选）：3 个值 (a,b,c) 为正交盒，6 个值
                (a,b,c,α,β,γ)；给出时输出带 Lattice/pbc 头的 extxyz
                （周期体系用，如晶体、slab）。name 来源不支持 cell。
        """
        try:
            return await build_structure_impl(services, source, source_type, cell)
        except Exception as exc:  # noqa: BLE001 - 工具不抛异常
            return err(f"build_structure 内部错误: {exc}")

    async def convert_structure(
        source_path: str,
        output_format: str | None = None,
        output_name: str | None = None,
        cell: list[float] | None = None,
    ) -> dict:
        """结构文件格式转换（ASE io，支持 cif/poscar/extxyz/xyz/pdb/gen/traj/mol/sdf 等）。
        输出写入 session workspace，返回目标路径、化学式、原子数、帧数、pbc 与晶胞参数
        （cellpar: a,b,c,α,β,γ——可直接作为 JobSpec structure.cell 的值）。
        周期结构转 extxyz 会保留 Lattice/pbc 头（周期计算路径的标准输入）。

        Args:
            source_path: 源结构文件路径（绝对路径，或相对 session workspace）。
            output_format: 目标格式（如 cif/poscar/extxyz/xyz/pdb）；缺省按
                output_name 扩展名推断。
            output_name: 目标文件名（可选，不含路径）；缺省用源文件名换扩展名。
            cell: 可选晶胞（TICKET-0013），3 值 (a,b,c) 正交胞或 6 值
                (a,b,c,α,β,γ)；给出时由 ASE set_cell + set_pbc(True) 对每帧
                补/覆盖晶胞——用于源文件丢失周期信息（如 MAPLE opt 输出的
                纯 xyz）时补回晶胞。
        """
        try:
            return await convert_structure_impl(
                services, source_path, output_format, output_name, cell
            )
        except Exception as exc:  # noqa: BLE001 - 工具不抛异常
            return err(f"convert_structure 内部错误: {exc}")

    return [build_structure, convert_structure]
