"""M2 工具层单测（不需要 GPU/LLM；build_structure name 用例走 COMPUTE_VENV 轻量子进程）。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.tools.common import JobAllocator, ToolServices
from app.tools.maple_tools import (
    build_maple_tools,
    convert_units_impl,
    parse_maple_out_impl,
    write_maple_inp_impl,
)
from app.tools.structure_tools import (
    build_structure_impl,
    convert_structure_impl,
    parse_coord_lines,
    parse_xyz_text,
)

B1_INP_COORDS = """\
C      -0.77812600     -1.06756100      0.32105900
C       1.30255300      0.05212000     -0.02829900
N       0.29116400     -0.87682100     -0.50039900
"""


@pytest.fixture
def services(tmp_path) -> ToolServices:
    workspace = tmp_path / "session"
    return ToolServices(
        settings=Settings(),
        session_workspace=workspace,
        job_allocator=JobAllocator(workspace / "jobs"),
    )


@pytest.fixture
def maple_tools_map(services) -> dict:
    return {t.__name__: t for t in build_maple_tools(services)}


# ---------- convert_units ----------


def test_convert_units_hartree_ev():
    r = convert_units_impl(-383.4722385376, "hartree", "eV")
    assert r["ok"]
    assert r["data"]["value"] == pytest.approx(-383.4722385376 * 27.211386245988)


def test_convert_units_aliases_and_roundtrip():
    r = convert_units_impl(1.0, "Ha", "kcal_mol")
    assert r["ok"]
    assert r["data"]["value"] == pytest.approx(627.5094740631)
    back = convert_units_impl(r["data"]["value"], "kcal/mol", "hartree")
    assert back["data"]["value"] == pytest.approx(1.0)


def test_convert_units_bad_unit():
    r = convert_units_impl(1.0, "hartree", "furlong")
    assert not r["ok"]
    assert "furlong" in r["error"]


# ---------- list_available_models ----------


def test_list_available_models(maple_tools_map):
    r = maple_tools_map["list_available_models"]()
    assert r["ok"]
    names = {m["name"] for m in r["data"]}
    assert {"ani2x", "uma", "nequip-oam-xl"} <= names
    uma = next(m for m in r["data"] if m["name"] == "uma")
    assert Path(uma["path"]).is_absolute()


# ---------- write_maple_inp ----------


def _b1_jobspec() -> dict:
    coords = [
        {"symbol": a["symbol"], "x": a["x"], "y": a["y"], "z": a["z"]}
        for a in parse_coord_lines(B1_INP_COORDS)
    ]
    return {
        "job_type": "sp",
        "model": {"name": "ani2x"},
        "structure": {"coords": coords},
        "charge": 0,
        "multiplicity": 1,
    }


def test_write_maple_inp_ok(services):
    r = write_maple_inp_impl(services, _b1_jobspec())
    assert r["ok"], r["error"]
    inp = Path(r["data"]["inp_path"])
    assert inp.is_file()
    text = inp.read_text()
    assert text.startswith("#model=ani2x\n#sp(verbose=0)")
    sidecar = json.loads((inp.parent / "job_spec.json").read_text())
    assert sidecar["job_type"] == "sp"
    # 作业目录在 session workspace 下
    assert str(inp).startswith(str(services.session_workspace))


def test_write_maple_inp_validation_error(services):
    bad = _b1_jobspec()
    bad["job_type"] = "scan"  # scan 缺 scan 段
    r = write_maple_inp_impl(services, bad)
    assert not r["ok"]
    assert "scan" in r["error"]


def test_write_maple_inp_missing_model(services):
    r = write_maple_inp_impl(services, {"job_type": "sp"})
    assert not r["ok"]
    assert "model" in r["error"] and "structure" in r["error"]


# ---------- parse_maple_out ----------


def test_parse_maple_out_b1_reference(references_dir):
    r = parse_maple_out_impl(str(references_dir / "B1_sp.out"))
    assert r["ok"]
    assert r["data"]["job_type"] == "sp"  # 自动推断
    assert r["data"]["energy_hartree"] == pytest.approx(-383.4722385376, abs=1e-9)


def test_parse_maple_out_freq_reference(references_dir):
    r = parse_maple_out_impl(str(references_dir / "B3_freq.out"))
    assert r["ok"]
    assert r["data"]["job_type"] == "freq"
    # MAPLE 输出全部 3N 个模式（含 6 个近零平动/转动模）
    assert len(r["data"]["frequencies_cm1"]) == 9
    assert sum(1 for f in r["data"]["frequencies_cm1"] if f > 100) == 3
    assert r["data"]["n_imaginary"] == 0


def test_parse_maple_out_missing_file(tmp_path):
    r = parse_maple_out_impl(str(tmp_path / "nope.out"))
    assert not r["ok"]


# ---------- build_structure ----------


def test_build_structure_inline(services):
    r = asyncio_run(build_structure_impl(services, B1_INP_COORDS, "inline_coords"))
    assert r["ok"], r["error"]
    assert r["data"]["n_atoms"] == 3
    assert r["data"]["formula"] == "C2N"
    xyz = Path(r["data"]["xyz_path"])
    assert xyz.is_file()
    atoms = parse_xyz_text(xyz.read_text())
    assert len(atoms) == 3


def test_build_structure_inline_bad(services):
    r = asyncio_run(build_structure_impl(services, "H 0 0", "inline_coords"))
    assert not r["ok"]


def test_build_structure_xyz(services, tmp_path):
    src = tmp_path / "in.xyz"
    src.write_text("2\ncomment\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\n")
    r = asyncio_run(build_structure_impl(services, str(src), "xyz"))
    assert r["ok"], r["error"]
    assert r["data"]["formula"] == "HO"


# ---------- build_structure 周期结构（2026-10-03 PBC 落地） ----------


def test_build_structure_inline_with_cell_writes_extxyz(services):
    r = asyncio_run(
        build_structure_impl(
            services,
            "Si 0.0 0.0 0.0\nSi 1.35775 1.35775 1.35775",
            "inline_coords",
            cell=[5.431, 5.431, 5.431],
        )
    )
    assert r["ok"], r["error"]
    assert r["data"]["cell"] == [5.431, 5.431, 5.431]
    xyz = Path(r["data"]["xyz_path"])
    assert xyz.name == "structure.extxyz"
    header = xyz.read_text().splitlines()[1]
    assert 'Lattice="5.4310000000 0.0000000000 0.0000000000' in header
    assert 'pbc="T T T"' in header


def test_build_structure_cell_six_params(services):
    r = asyncio_run(
        build_structure_impl(
            services, "Si 0.0 0.0 0.0", "inline_coords",
            cell=[5.0, 6.0, 7.0, 90.0, 90.0, 120.0],
        )
    )
    assert r["ok"], r["error"]
    header = Path(r["data"]["xyz_path"]).read_text().splitlines()[1]
    # γ=120°：b 向量 = (b·cos120, b·sin120, 0) = (-3, 5.196..., 0)
    assert "-3.0000000000 5.1961524227" in header


def test_build_structure_cell_bad_params(services):
    r = asyncio_run(build_structure_impl(services, "Si 0 0 0", "inline_coords", cell=[5.0, 5.0]))
    assert not r["ok"] and "晶胞参数非法" in r["error"]
    r = asyncio_run(
        build_structure_impl(services, "Si 0 0 0", "inline_coords", cell=[-1.0, 5.0, 5.0])
    )
    assert not r["ok"] and "晶胞参数非法" in r["error"]


def test_build_structure_name_rejects_cell(services):
    r = asyncio_run(build_structure_impl(services, "H2O", "name", cell=[10.0, 10.0, 10.0]))
    assert not r["ok"] and "不支持 cell" in r["error"]


def test_build_structure_xyz_preserves_lattice(services, tmp_path):
    """源 extxyz 自带 Lattice 头：原样复制，不丢周期信息。"""
    src = tmp_path / "si.extxyz"
    src.write_text(
        '2\nLattice="5.431 0 0 0 5.431 0 0 0 5.431" pbc="T T T"\n'
        "Si 0.0 0.0 0.0\nSi 1.35775 1.35775 1.35775\n"
    )
    r = asyncio_run(build_structure_impl(services, str(src), "xyz"))
    assert r["ok"], r["error"]
    assert r["data"]["cell"] == "from_file"
    dest = Path(r["data"]["xyz_path"])
    assert dest.name == "structure.extxyz"
    assert 'Lattice="5.431 0 0 0 5.431 0 0 0 5.431"' in dest.read_text().splitlines()[1]


# ---------- convert_structure（ASE io 格式转换，2026-10-03 新增） ----------

_SI_EXTXYZ = (
    '2\nLattice="5.431 0 0 0 5.431 0 0 0 5.431" pbc="T T T"\n'
    "Si 0.0 0.0 0.0\nSi 1.35775 1.35775 1.35775\n"
)


def test_convert_structure_extxyz_to_cif(services, compute_python, tmp_path):
    """周期结构 extxyz → cif：晶胞参数保留。"""
    src = tmp_path / "si.extxyz"
    src.write_text(_SI_EXTXYZ)
    r = asyncio_run(convert_structure_impl(services, str(src), "cif"))
    assert r["ok"], r["error"]
    data = r["data"]
    assert data["formula"] == "Si2"
    assert data["n_atoms"] == 2 and data["n_frames"] == 1
    assert data["pbc"] == [True, True, True]
    assert data["cellpar"] == pytest.approx([5.431, 5.431, 5.431, 90, 90, 90], abs=1e-4)
    dst = Path(data["dst_path"])
    assert dst.suffix == ".cif" and dst.is_file()
    assert "_cell_length_a" in dst.read_text()


def test_convert_structure_cif_to_extxyz_roundtrip(services, compute_python, tmp_path):
    """cif → extxyz：Lattice 头回来，可直接进周期计算路径。"""
    src = tmp_path / "si.extxyz"
    src.write_text(_SI_EXTXYZ)
    r1 = asyncio_run(convert_structure_impl(services, str(src), "cif"))
    assert r1["ok"], r1["error"]
    r2 = asyncio_run(
        convert_structure_impl(services, r1["data"]["dst_path"], "extxyz")
    )
    assert r2["ok"], r2["error"]
    header = Path(r2["data"]["dst_path"]).read_text().splitlines()[1]
    assert 'Lattice="' in header and 'pbc="T T T"' in header
    assert r2["data"]["cellpar"] == pytest.approx([5.431] * 3 + [90] * 3, abs=1e-4)


def test_convert_structure_xyz_to_poscar(services, compute_python, tmp_path):
    """分子 → POSCAR：无胞输入自动加 10 Å 真空盒（POSCAR 必须带胞）。"""
    src = tmp_path / "h2o.xyz"
    src.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    r = asyncio_run(convert_structure_impl(services, str(src), "poscar"))
    assert r["ok"], r["error"]
    data = r["data"]
    assert data["formula"] == "H2O"
    assert data["pbc"] == [True, True, True]
    assert data["cellpar"] is not None and all(
        v > 10.0 for v in data["cellpar"][:3]
    )
    assert any("真空盒" in w for w in data["warnings"])
    text = Path(data["dst_path"]).read_text()
    assert "O" in text and "H" in text


def test_convert_structure_cell_lossy_warns(services, compute_python, tmp_path):
    """周期结构转 pdb：给晶胞丢失警告。"""
    src = tmp_path / "si.extxyz"
    src.write_text(_SI_EXTXYZ)
    r = asyncio_run(convert_structure_impl(services, str(src), "pdb"))
    assert r["ok"], r["error"]
    assert any("Lattice" in w or "晶胞" in w for w in r["data"]["warnings"])


def test_convert_structure_errors(services, compute_python, tmp_path):
    r = asyncio_run(convert_structure_impl(services, str(tmp_path / "nope.xyz"), "cif"))
    assert not r["ok"] and "不存在" in r["error"]
    src = tmp_path / "h2o.xyz"
    src.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    r = asyncio_run(convert_structure_impl(services, str(src), "quantum-espresso-magic"))
    assert not r["ok"] and "不支持的目标格式" in r["error"]
    # 源与目标相同（源在 session workspace 内，output_name 撞名）
    ws_src = services.session_workspace / "h2o.xyz"
    services.session_workspace.mkdir(parents=True, exist_ok=True)
    ws_src.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    r = asyncio_run(
        convert_structure_impl(services, str(ws_src), "xyz", output_name="h2o.xyz")
    )
    assert not r["ok"] and "原地覆盖" in r["error"]


# ---------- convert_structure cell 参数（TICKET-0013） ----------


def test_convert_structure_cell_param_adds_cell(services, compute_python, tmp_path):
    """无胞 xyz + cell 参数 → cif 带晶胞（MAPLE opt 输出丢胞的补回路径）。"""
    src = tmp_path / "opt.xyz"
    src.write_text("3\nopt final\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    r = asyncio_run(
        convert_structure_impl(services, str(src), "cif", cell=[10.0, 12.0, 14.0])
    )
    assert r["ok"], r["error"]
    data = r["data"]
    assert data["pbc"] == [True, True, True]
    assert data["cellpar"] == pytest.approx([10.0, 12.0, 14.0, 90, 90, 90], abs=1e-4)
    text = Path(data["dst_path"]).read_text()
    assert "_cell_length_a" in text and "_cell_length_c" in text


def test_convert_structure_cell_param_six_values(services, compute_python, tmp_path):
    """6 值 cellpar（非正交胞）也支持。"""
    src = tmp_path / "opt.xyz"
    src.write_text("1\none\nSi 0 0 0\n")
    r = asyncio_run(
        convert_structure_impl(
            services, str(src), "extxyz", cell=[5.0, 6.0, 7.0, 80.0, 85.0, 70.0]
        )
    )
    assert r["ok"], r["error"]
    assert r["data"]["cellpar"] == pytest.approx(
        [5.0, 6.0, 7.0, 80.0, 85.0, 70.0], abs=1e-3
    )
    header = Path(r["data"]["dst_path"]).read_text().splitlines()[1]
    assert 'Lattice="' in header and 'pbc="T T T"' in header


def test_convert_structure_cell_param_override_warns(
    services, compute_python, tmp_path
):
    """源已有胞且给 cell：覆盖并记 warning。"""
    src = tmp_path / "si.extxyz"
    src.write_text(_SI_EXTXYZ)
    r = asyncio_run(
        convert_structure_impl(services, str(src), "cif", cell=[6.0, 6.0, 6.0])
    )
    assert r["ok"], r["error"]
    assert r["data"]["cellpar"] == pytest.approx([6.0, 6.0, 6.0, 90, 90, 90], abs=1e-4)
    assert any("覆盖" in w for w in r["data"]["warnings"])


def test_convert_structure_cell_param_bad(services, compute_python, tmp_path):
    src = tmp_path / "opt.xyz"
    src.write_text("1\none\nSi 0 0 0\n")
    r = asyncio_run(convert_structure_impl(services, str(src), "cif", cell=[1.0, 2.0]))
    assert not r["ok"] and "3 值" in r["error"]
    r = asyncio_run(
        convert_structure_impl(services, str(src), "cif", cell=[10.0, -1.0, 10.0])
    )
    assert not r["ok"] and "必须为正" in r["error"]


def test_convert_structure_parallel_no_race(services, compute_python, tmp_path):
    """TICKET-0015：并行调用各自的 job/meta 不串扰。"""
    src1 = tmp_path / "a.xyz"
    src1.write_text("1\na\nSi 0 0 0\n")
    src2 = tmp_path / "b.xyz"
    src2.write_text("2\nb\nO 0 0 0\nH 0.96 0 0\n")

    async def _both():
        return await asyncio.gather(
            convert_structure_impl(services, str(src1), "cif"),
            convert_structure_impl(services, str(src2), "cif"),
        )

    r1, r2 = asyncio_run(_both())
    assert r1["ok"], r1["error"]
    assert r2["ok"], r2["error"]
    assert r1["data"]["src_path"] == str(src1)
    assert r2["data"]["src_path"] == str(src2)
    assert r1["data"]["formula"] == "Si"
    assert r2["data"]["formula"] == "HO"


def test_build_structure_name_subprocess(services, compute_python):
    """name 来源走 COMPUTE_VENV 子进程（ase 只存在于计算环境）。"""
    r = asyncio_run(build_structure_impl(services, "H2O", "name"))
    assert r["ok"], r["error"]
    assert r["data"]["formula"] == "H2O"
    assert r["data"]["n_atoms"] == 3
    assert Path(r["data"]["xyz_path"]).is_file()


def test_build_structure_name_unknown(services, compute_python):
    r = asyncio_run(build_structure_impl(services, "NotAMolecule123", "name"))
    assert not r["ok"]
    assert "inline_coords" in r["error"]


# ---------- get_job_artifact ----------


def test_get_job_artifact(services):
    r = write_maple_inp_impl(services, _b1_jobspec())
    job_id = r["data"]["job_id"]
    tools = {t.__name__: t for t in build_maple_tools(services)}
    got = tools["get_job_artifact"](job_id, "input.inp")
    assert got["ok"]
    assert got["data"].endswith("input.inp")
    assert not tools["get_job_artifact"](job_id, "output.out")["ok"]
    assert not tools["get_job_artifact"](job_id, "../job_spec.json")["ok"]
    assert not tools["get_job_artifact"]("job-9999", "input.inp")["ok"]


def asyncio_run(coro):
    return asyncio.run(coro)
