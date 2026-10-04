"""inp_builder 单元测试：JobSpec → MAPLE .inp 文本。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.compute.inp_builder import build_inp
from app.compute.jobspec import (
    DEFAULT_MACEMPA0_MODEL_PATH,
    DEFAULT_NEQUIP_MODEL_PATH,
    DEFAULT_UMA_CHECKPOINT_PATH,
    Atom,
    JobSpec,
    MdSpec,
    ModelSpec,
    ScanCoordinate,
    ScanSpec,
    StructureSpec,
)

H2O = [
    Atom(symbol="O", x=0.0, y=0.0, z=0.0),
    Atom(symbol="H", x=0.96, y=0.0, z=0.0),
    Atom(symbol="H", x=-0.24, y=0.93, z=0.0),
]


def make_spec(**overrides) -> JobSpec:
    base = dict(
        job_type="sp",
        model=ModelSpec(name="ani2x"),
        structure=StructureSpec(coords=H2O),
    )
    base.update(overrides)
    return JobSpec(**base)


class TestModelLine:
    def test_ani2x(self):
        inp = build_inp(make_spec())
        assert inp.splitlines()[0] == "#model=ani2x"

    def test_uma_default_checkpoint(self):
        inp = build_inp(make_spec(model=ModelSpec(name="uma")))
        assert inp.splitlines()[0] == (
            "#model=uma(size=uma-s-1p1,task=omol,"
            f"checkpoint_path={DEFAULT_UMA_CHECKPOINT_PATH})"
        )

    def test_uma_custom_checkpoint(self):
        inp = build_inp(
            make_spec(
                model=ModelSpec(
                    name="uma", uma_size="uma-m-1p1", checkpoint_path="/tmp/x.pt"
                )
            )
        )
        assert (
            inp.splitlines()[0]
            == "#model=uma(size=uma-m-1p1,task=omol,checkpoint_path=/tmp/x.pt)"
        )

    def test_nequip_oam_xl_plugin_default_path(self):
        """TICKET-0002：插件后端渲染显式本地 model_path（缺省默认 zip）。"""
        inp = build_inp(make_spec(model=ModelSpec(name="nequip-oam-xl")))
        assert inp.splitlines()[0] == (
            f"#model=nequip-oam-xl(model_path={DEFAULT_NEQUIP_MODEL_PATH})"
        )

    def test_nequip_oam_xl_plugin_custom_path(self):
        inp = build_inp(
            make_spec(model=ModelSpec(name="nequip-oam-xl", model_path="/tmp/m.nequip.zip"))
        )
        assert inp.splitlines()[0] == "#model=nequip-oam-xl(model_path=/tmp/m.nequip.zip)"

    def test_macempa0_plugin_default_path(self):
        """插件后端渲染显式本地 model_path（缺省默认 macempa0.pt）。"""
        inp = build_inp(make_spec(model=ModelSpec(name="macempa0")))
        assert inp.splitlines()[0] == (
            f"#model=macempa0(model_path={DEFAULT_MACEMPA0_MODEL_PATH})"
        )

    def test_macempa0_plugin_custom_path(self):
        inp = build_inp(
            make_spec(model=ModelSpec(name="macempa0", model_path="/tmp/m.pt"))
        )
        assert inp.splitlines()[0] == "#model=macempa0(model_path=/tmp/m.pt)"


class TestTaskLines:
    def test_sp(self):
        assert "#sp(verbose=0)" in build_inp(make_spec())

    def test_opt_defaults(self):
        inp = build_inp(make_spec(job_type="opt"))
        assert "#opt(method=lbfgs,max_iter=256)" in inp
        # 默认 medium 不打印 #level
        assert "#level=" not in inp

    def test_opt_level_tight(self):
        spec = make_spec(job_type="opt")
        spec.opt.level = "tight"
        assert "#level=tight" in build_inp(spec)

    def test_freq(self):
        assert "#freq(method=mw,temperature=298.15)" in build_inp(
            make_spec(job_type="freq", structure=StructureSpec(xyz_file="a.xyz"))
        )

    def test_md_inline(self):
        inp = build_inp(
            make_spec(job_type="md", md=MdSpec(steps=20, timestep_fs=0.5))
        )
        assert "#md(ensemble=nve,timestep=0.5,steps=20,temperature=300)" in inp

    def test_md_mdp_file(self):
        inp = build_inp(make_spec(job_type="md", md=MdSpec(mdp_file="nve.mdp")))
        assert "#md(mdp=nve.mdp)" in inp

    def test_ts_default_prfo(self):
        assert "#ts(method=prfo)" in build_inp(make_spec(job_type="ts"))

    def test_ts_charged_xyz_reference(self):
        inp = build_inp(
            make_spec(
                job_type="ts",
                structure=StructureSpec(xyz_file="ts_guess.xyz"),
                charge=-1,
                multiplicity=1,
            )
        )
        assert "XYZ -1 1 ts_guess.xyz" in inp


class TestStructure:
    def test_inline_coords(self):
        inp = build_inp(make_spec())
        lines = inp.splitlines()
        assert "O" in lines[4] and "0.00000000" in lines[4]
        assert lines[5].startswith("H")
        assert lines[6].startswith("H")
        # 默认 charge=0/mult=1 不打印电荷行
        assert not lines[3].strip()

    def test_charge_mult_line(self):
        inp = build_inp(make_spec(charge=-1, multiplicity=2))
        lines = inp.splitlines()
        assert lines[4] == "-1 2"
        assert lines[5].startswith("O")

    def test_xyz_reference(self):
        inp = build_inp(make_spec(structure=StructureSpec(xyz_file="b2_opt_opt.xyz")))
        assert "XYZ b2_opt_opt.xyz" in inp.splitlines()

    def test_xyz_reference_with_charge(self):
        # charge/mult 内联进 XYZ 引用行（独立行 + XYZ 同组会被 MAPLE 判为混合坐标）
        inp = build_inp(
            make_spec(structure=StructureSpec(xyz_file="a.xyz"), charge=1)
        )
        lines = inp.splitlines()
        assert lines[4] == "XYZ 1 1 a.xyz"

    def test_md_forces_explicit_charge_mult(self):
        # MAPLE md dispatcher 强制 atoms.info 含 charge/mult
        inp_xyz = build_inp(
            make_spec(job_type="md", structure=StructureSpec(xyz_file="a.xyz"))
        )
        assert "XYZ 0 1 a.xyz" in inp_xyz.splitlines()
        inp_inline = build_inp(make_spec(job_type="md"))
        lines = inp_inline.splitlines()
        assert lines[4] == "0 1"

    def test_structure_requires_exactly_one_source(self):
        with pytest.raises(ValidationError):
            StructureSpec()
        with pytest.raises(ValidationError):
            StructureSpec(coords=H2O, xyz_file="a.xyz")
        with pytest.raises(ValidationError):
            StructureSpec(coords=[])


class TestScan:
    def test_rigid_scan_matches_b8_shape(self):
        spec = make_spec(
            job_type="scan",
            scan=ScanSpec(
                mode="rigid",
                coordinates=[ScanCoordinate(atom_i=1, atom_j=2, delta=0.05, n_steps=4)],
            ),
        )
        inp = build_inp(spec)
        lines = inp.splitlines()
        assert "#scan(mode=rigid)" in lines
        assert "S 1 2 0.05 4" in lines
        # S 行在坐标段之后，中间空行
        s_idx = lines.index("S 1 2 0.05 4")
        assert lines[s_idx - 1] == ""
        assert lines[s_idx - 2].startswith("H")

    def test_relaxed_scan(self):
        spec = make_spec(
            job_type="scan",
            scan=ScanSpec(
                mode="relaxed",
                coordinates=[
                    ScanCoordinate(atom_i=11, atom_j=52, delta=-0.1, n_steps=15),
                    ScanCoordinate(atom_i=14, atom_j=51, delta=-0.1, n_steps=15),
                ],
            ),
        )
        inp = build_inp(spec)
        assert "#scan()" in inp
        assert "S 11 52 -0.1 15" in inp
        assert "S 14 51 -0.1 15" in inp

    def test_scan_job_requires_scan_spec(self):
        with pytest.raises(ValidationError):
            make_spec(job_type="scan")


class TestDevice:
    def test_default_gpu0(self):
        assert "#device=gpu0" in build_inp(make_spec())

    def test_cpu(self):
        assert "#device=cpu" in build_inp(make_spec(device="cpu"))


class TestUmaSizeNormalization:
    """D7-3：uma_size 缩写归一化与非法值报错（live B4 根因修复）。"""

    def test_valid_enum_values_pass_through(self):
        for size in ("uma-s-1p1", "uma-s-1p2", "uma-m-1p1"):
            assert ModelSpec(name="uma", uma_size=size).uma_size == size

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("s", "uma-s-1p1"),
            ("uma-s", "uma-s-1p1"),
            ("m", "uma-m-1p1"),
            ("uma-m", "uma-m-1p1"),
            ("1p2", "uma-s-1p2"),
        ],
    )
    def test_aliases_normalized(self, alias, expected):
        assert ModelSpec(name="uma", uma_size=alias).uma_size == expected

    def test_invalid_size_error_lists_allowed_values(self):
        with pytest.raises(ValidationError) as exc_info:
            ModelSpec(name="uma", uma_size="xl")
        msg = str(exc_info.value)
        assert "uma-s-1p1" in msg and "uma-s-1p2" in msg and "uma-m-1p1" in msg

    def test_normalized_size_rendered_in_inp(self):
        inp = build_inp(make_spec(model=ModelSpec(name="uma", uma_size="s")))
        assert inp.splitlines()[0].startswith("#model=uma(size=uma-s-1p1,")


class TestFreqStructureRequirement:
    """D7-3：freq 任务禁止内联坐标（live B6 根因修复，MAPLE freq 要求 XYZ 引用）。"""

    def test_freq_with_inline_coords_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            make_spec(job_type="freq")
        assert "xyz_file" in str(exc_info.value)

    def test_freq_with_xyz_file_accepted(self):
        spec = make_spec(job_type="freq", structure=StructureSpec(xyz_file="a.xyz"))
        assert "XYZ a.xyz" in build_inp(spec)


class TestB1GoldenInp:
    """用 MAPLE/examples/sp/sp.inp 的坐标重建 B1 inp，逐行比对坐标段。"""

    def test_rebuild_b1_inp(self, maple_examples_dir):
        example = maple_examples_dir / "sp" / "sp.inp"
        lines = example.read_text().splitlines()
        coord_lines = [ln for ln in lines if ln and not ln.startswith("#")]
        coords = []
        for ln in coord_lines:
            sym, x, y, z = ln.split()
            coords.append(
                Atom(symbol=sym, x=float(x), y=float(y), z=float(z))
            )
        assert len(coords) == 22

        inp = build_inp(make_spec(structure=StructureSpec(coords=coords)))
        out_lines = inp.splitlines()
        assert out_lines[0] == "#model=ani2x"
        assert out_lines[1] == "#sp(verbose=0)"
        assert out_lines[2] == "#device=gpu0"
        built_coords = out_lines[4:]
        assert len(built_coords) == 22
        for built, src in zip(built_coords, coord_lines):
            bsym, bx, by, bz = built.split()
            ssym, sx, sy, sz = src.split()
            assert bsym == ssym
            assert float(bx) == float(sx)
            assert float(by) == float(sy)
            assert float(bz) == float(sz)


class TestPbcCell:
    """PBC 落地（2026-10-03）：structure.cell → #pbc(...) 渲染与模型门禁。"""

    def test_cell_six_values_rendered(self):
        spec = make_spec(
            model=ModelSpec(name="uma", uma_task="omat"),
            structure=StructureSpec(coords=H2O, cell=[10.0, 10.0, 10.0, 90, 90, 90]),
        )
        inp = build_inp(spec)
        assert "#pbc(10,10,10,90,90,90)" in inp.splitlines()
        # #pbc 是全局参数，位于坐标段空行之前
        assert inp.index("#pbc(") < inp.index("\n\n")

    def test_cell_two_values_2d_form(self):
        spec = make_spec(
            model=ModelSpec(name="uma", uma_task="oc20"),
            structure=StructureSpec(coords=H2O, cell=[18.19, 24.90]),
        )
        assert "#pbc(18.19,24.9)" in build_inp(spec).splitlines()

    def test_no_cell_no_pbc_line(self):
        assert "#pbc(" not in build_inp(make_spec())

    def test_cell_with_xyz_file(self, tmp_path):
        xyz = tmp_path / "si.extxyz"
        xyz.write_text("2\n\nSi 0 0 0\nSi 1.35775 1.35775 1.35775\n")
        spec = make_spec(
            model=ModelSpec(name="uma", uma_task="omat"),
            structure=StructureSpec(xyz_file=str(xyz), cell=[5.431, 5.431, 5.431]),
        )
        assert "#pbc(5.431,5.431,5.431)" in build_inp(spec).splitlines()

    def test_cell_bad_length_rejected(self):
        with pytest.raises(ValidationError, match="cell 需 2/3/6 个值"):
            StructureSpec(coords=H2O, cell=[10.0, 10.0, 10.0, 90.0])

    def test_cell_rejects_non_uma_models(self):
        with pytest.raises(ValidationError, match="周期体系"):
            make_spec(structure=StructureSpec(coords=H2O, cell=[10.0, 10.0]))

    def test_cell_rejects_nequip_plugin(self):
        with pytest.raises(ValidationError, match="周期体系"):
            make_spec(
                model=ModelSpec(name="nequip-oam-xl"),
                structure=StructureSpec(coords=H2O, cell=[10.0, 10.0]),
            )

    def test_cell_uma_omol_mutex(self):
        with pytest.raises(ValidationError, match="omol"):
            make_spec(
                model=ModelSpec(name="uma"),
                structure=StructureSpec(coords=H2O, cell=[10.0, 10.0]),
            )

    def test_cell_uma_periodic_tasks_accepted(self):
        for task in ("omat", "oc20", "oc22", "oc25", "omc", "odac"):
            make_spec(
                model=ModelSpec(name="uma", uma_task=task),
                structure=StructureSpec(coords=H2O, cell=[10.0, 10.0]),
            )
    def test_cell_allows_pbc_models(self):
        """TICKET-0008/0009 后 MACE 系（含插件 macempa0）与 AIMNet2 系支持 PBC。"""
        for model in ("maceoff23s", "maceoff23m", "maceoff23l", "macempa0", "aimnet2", "aimnet2nse"):
            make_spec(
                model=ModelSpec(name=model),
                structure=StructureSpec(coords=H2O, cell=[10.0, 10.0]),
            )

