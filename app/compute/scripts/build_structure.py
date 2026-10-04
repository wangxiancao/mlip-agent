"""分子名 → 坐标脚本：由 COMPUTE_VENV 的 python 以子进程方式执行（M2）。

用法：
    $COMPUTE_VENV/bin/python app/compute/scripts/build_structure.py <name> <out_dir>

在 out_dir 写入：
    structure.xyz  标准 xyz 文件（Å）
    meta.json      {"ok": bool, "name", "formula", "n_atoms", "error"}

解析顺序：`ase.build.molecule(name)`（含 H2O/NH3/CH4 等常用名）→
`ase.collections.g2[name]`（覆盖 CH3CH2OH 等 g2 集合，下划线/大小写归一化后匹配）。
两者都失败时 meta.json ok=false 且退出码 1，调用方提示改用 inline_coords。

本脚本只能依赖 COMPUTE_VENV 已有包（stdlib + ase），不得 import mlip_agent 后端包。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# 常用别名 → g2 命名（g2 用结构式命名，如乙醇是 CH3CH2OH 而非 C2H5OH）
_ALIASES = {
    "c2h5oh": "CH3CH2OH",
    "ethanol": "CH3CH2OH",
    "methanol": "CH3OH",
    "ch3oh": "CH3OH",
    "acetone": "CH3COCH3",
    "acetic_acid": "CH3COOH",
    "ch3cooh": "CH3COOH",
    "benzene": "C6H6",
    "water": "H2O",
    "ammonia": "NH3",
    "methane": "CH4",
}


def _build_atoms(name: str):
    import ase.build
    from ase.collections import g2

    try:
        return ase.build.molecule(name)
    except (KeyError, ValueError, NotImplementedError):
        pass
    normalized = name.replace("-", "").replace("_", "").lower()
    alias = _ALIASES.get(normalized)
    if alias is not None and alias in g2.names:
        return g2[alias]
    for candidate in g2.names:
        if candidate.replace("-", "").replace("_", "").lower() == normalized:
            return g2[candidate]
    raise KeyError(name)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: build_structure.py <name> <out_dir>", file=sys.stderr)
        return 2
    name, out_dir = argv[1], Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"ok": False, "name": name, "formula": None, "n_atoms": None, "error": None}
    try:
        import ase.io

        atoms = _build_atoms(name)
        xyz_path = out_dir / "structure.xyz"
        ase.io.write(xyz_path, atoms, format="xyz")
        meta.update(
            ok=True,
            formula=atoms.get_chemical_formula(),
            n_atoms=len(atoms),
            xyz_path=str(xyz_path),
        )
    except Exception as exc:  # noqa: BLE001 - 子进程边界，任何失败都落成 meta.json
        meta["error"] = f"未知分子名 {name!r}：{exc}"
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    return 0 if meta["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
