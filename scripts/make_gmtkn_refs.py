#!/usr/bin/env python3
"""从 GMTKN55 clone 提取 6 个 BH76 反应的 benchmark v3 参考数据（PLAN §3.3 WP1）。

用法：
    python scripts/make_gmtkn_refs.py /path/to/GMTKN55

产出 benchmarks/reaction/references/R{1..6}/：
    r1.xyz [r2.xyz] / p1.xyz [p2.xyz] / ts.xyz   各物种几何（TURBOMOLE struc.xyz 转标准 xyz）
    ref.json   能垒/反应能参考值（kcal/mol）、各物种 charge/multiplicity、
               scan 段建议（键长范围与原子序号）、数据集出处

数据源：GMTKN55 BH76 子集（Goerigk et al., PCCP 2017, DOI 10.1039/C7CP04913G；
GitHub grimme-lab/GMTKN55）。参考值为高级别 WFT，能量差单位 kcal/mol。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "reaction" / "references"

# 能垒来源：GMTKN55 BH76/.res（kcal/mol，正向 = 反应物侧 → TS）
# species: (文件前缀, GMTKN55 目录名, charge, multiplicity)
REACTIONS: dict[str, dict] = {
    "R1": {
        "name": "HCN→HNC 异构化",
        "type": "unimolecular_isomerization",
        "species": [("r1", "hcn", 0, 1), ("p1", "hnc", 0, 1)],
        "ts": ("hcnts", 0, 1),
        "barriers": {"forward": 48.1, "reverse": 33.0},
        "scan": {"of": "r1", "symbols": ("H", "C"), "stretch": 0.55},
    },
    "R2": {
        "name": "H + N2O → OH + N2（氢转移）",
        "type": "h_transfer",
        "species": [("r1", "h", 0, 2), ("r2", "n2o", 0, 1), ("p1", "oh", 0, 2), ("p2", "n2", 0, 1)],
        "ts": ("n2ohts", 0, 2),
        "barriers": {"forward": 17.7, "reverse": 82.6},
        "scan": {"of": "r2", "symbols": ("N", "O"), "stretch": 0.6},
    },
    "R3": {
        "name": "OH + CH4 → H2O + CH3（氢抽取）",
        "type": "h_abstraction",
        "species": [("r1", "oh", 0, 2), ("r2", "CH4", 0, 1), ("p1", "H2O", 0, 1), ("p2", "ch3", 0, 2)],
        "ts": ("RKT04", 0, 2),
        "barriers": {"forward": 6.3, "reverse": 19.5},
        "scan": {"of": "r2", "symbols": ("C", "H"), "stretch": 0.6},
    },
    "R4": {
        "name": "F− + CH3Cl → CH3F + Cl−（S_N2，阴离子）",
        "type": "sn2_anionic",
        "species": [("r1", "f-", -1, 1), ("r2", "ch3cl", 0, 1), ("p1", "ch3f", 0, 1), ("p2", "cl-", -1, 1)],
        "ts": ("fch3clts", -1, 1),
        "barriers": {"forward": -12.3, "reverse": 19.8},
        "scan": {"of": "r2", "symbols": ("C", "Cl"), "stretch": 0.8},
    },
    "R5": {
        "name": "CH3 + C2H4 → C3H7（自由基加成）",
        "type": "radical_addition",
        "species": [("r1", "ch3", 0, 2), ("r2", "c2h4", 0, 1), ("p1", "c3h7", 0, 2)],
        "ts": ("c3h7ts", 0, 2),
        "barriers": {"forward": 6.4, "reverse": 33.0},
        "scan": {"of": "r2", "symbols": ("C", "C"), "stretch": 0.3},
    },
    "R6": {
        "name": "O(3P) + CH4 → OH + CH3（三重态氧抽取氢）",
        "type": "o3p_abstraction",
        "species": [("r1", "O", 0, 3), ("r2", "CH4", 0, 1), ("p1", "oh", 0, 2), ("p2", "ch3", 0, 2)],
        "ts": ("RKT11", 0, 3),
        "barriers": {"forward": 14.4, "reverse": 8.9},
        "scan": {"of": "r2", "symbols": ("C", "H"), "stretch": 0.6},
    },
}

CITATION = {
    "dataset": "GMTKN55",
    "subset": "BH76",
    "level": "high-level WFT reference（见 GMTKN55 论文；BH76 条目源自 HTBH38/NHTBH38/DBH 系列）",
    "citation": "Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme, PCCP 2017, DOI 10.1039/C7CP04913G",
    "units": "kcal/mol（能量差，相对各自物种的分离/独立极小）",
}


def read_xyz(path: Path) -> list[list]:
    lines = [l.split() for l in path.read_text().splitlines()[2:] if l.strip()]
    return [[l[0], float(l[1]), float(l[2]), float(l[3])] for l in lines]


def write_xyz(atoms: list[list], path: Path) -> None:
    body = "\n".join(f"{s}  {x:.10f}  {y:.10f}  {z:.10f}" for s, x, y, z in atoms)
    path.write_text(f"{len(atoms)}\n\n{body}\n")


def dist(a: list, b: list) -> float:
    return math.dist(a[1:], b[1:])


def find_bond(atoms: list[list], s1: str, s2: str) -> tuple[int, int, float]:
    """找符号对 (s1,s2) 中距离最近的一对原子（1-based）及其距离。"""
    best = None
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            if i >= j or {a[0], b[0]} != {s1, s2}:
                continue
            d = dist(a, b)
            if best is None or d < best[2]:
                best = (i + 1, j + 1, d)
    if best is None:
        raise ValueError(f"找不到键 {s1}-{s2}")
    return best


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gmtkn55")
    if not (src / "BH76").is_dir():
        print(f"GMTKN55 路径无效: {src}", file=sys.stderr)
        return 2
    for rid, spec in REACTIONS.items():
        out = OUT_DIR / rid
        out.mkdir(parents=True, exist_ok=True)
        species_meta = []
        atoms_by_prefix: dict[str, list] = {}
        for prefix, dirname, charge, mult in spec["species"]:
            atoms = read_xyz(src / "BH76" / dirname / "struc.xyz")
            write_xyz(atoms, out / f"{prefix}.xyz")
            atoms_by_prefix[prefix] = atoms
            species_meta.append(
                {
                    "file": f"{prefix}.xyz",
                    "gmtkn_name": dirname,
                    "charge": charge,
                    "multiplicity": mult,
                    "role": "reactant" if prefix.startswith("r") else "product",
                    "n_atoms": len(atoms),
                }
            )
        ts_dir, ts_charge, ts_mult = spec["ts"]
        ts_atoms = read_xyz(src / "BH76" / ts_dir / "struc.xyz")
        write_xyz(ts_atoms, out / "ts.xyz")
        sc = spec["scan"]
        scan_atoms = atoms_by_prefix[sc["of"]]
        i, j, eq = find_bond(scan_atoms, *sc["symbols"])
        barriers = dict(spec["barriers"])
        barriers["reaction_energy"] = round(barriers["forward"] - barriers["reverse"], 1)
        ref = {
            "reaction": rid,
            "name": spec["name"],
            "type": spec["type"],
            "species": species_meta,
            "ts": {
                "file": "ts.xyz",
                "gmtkn_name": ts_dir,
                "charge": ts_charge,
                "multiplicity": ts_mult,
                "n_atoms": len(ts_atoms),
            },
            "barriers_kcal": barriers,
            "scan": {
                "species": sc["of"],
                "atom_i": i,
                "atom_j": j,
                "symbols": list(sc["symbols"]),
                "equilibrium_ang": round(eq, 4),
                "start_ang": round(eq, 4),
                "end_ang": round(eq + sc["stretch"], 4),
                "points": 6,
            },
            "reference": CITATION,
        }
        (out / "ref.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n")
        print(
            f"{rid} {spec['name']}: 物种 {len(species_meta)}，TS {ts_dir}，"
            f"正/反能垒 {barriers['forward']}/{barriers['reverse']}，"
            f"反应能 {barriers['reaction_energy']} kcal/mol，"
            f"scan {sc['of']} {i}-{j} {eq:.3f}→{eq + sc['stretch']:.3f} Å"
        )
    print(f"\n产出目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
