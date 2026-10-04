"""parsers 测试：fixture 用 benchmarks/references/ 与 MAPLE/examples/** 的真实 .out。

B1/B2/B3/B8 与 pinned JSON（benchmarks/references/Bx.json）数值比对（验收要求）。
"""

from __future__ import annotations

import json
import math

import pytest

from app.compute.jobspec import JobType
from app.compute.parsers import parse_out, parse_out_file


def load_ref_json(references_dir, name: str) -> dict:
    return json.loads((references_dir / name).read_text())


class TestSp:
    def test_b1_matches_pinned(self, references_dir):
        ref = load_ref_json(references_dir, "B1.json")
        result = parse_out_file(references_dir / "B1_sp.out", "sp")
        assert result.ok
        assert result.charge == 0 and result.multiplicity == 1
        assert math.isclose(
            result.energy_hartree,
            ref["values"]["energy_hartree"],
            abs_tol=ref["tolerance"]["energy_hartree"],
        )
        # sp 能量 10 位小数精度
        assert result.energy_hartree == pytest.approx(-383.4722385376, abs=1e-10)

    def test_b4_uma_matches_pinned(self, references_dir):
        ref = load_ref_json(references_dir, "B4.json")
        result = parse_out_file(references_dir / "B4_sp.out", "sp")
        assert result.ok
        assert result.energy_hartree == pytest.approx(
            ref["values"]["energy_hartree"],
            abs=ref["tolerance"]["energy_hartree"],
        )

    def test_sp_failure_returns_partial(self):
        result = parse_out("some garbage without energy\n", "sp")
        assert not result.ok
        assert result.energy_hartree is None
        assert result.warnings


class TestOpt:
    def test_b2_matches_pinned(self, references_dir):
        ref = load_ref_json(references_dir, "B2.json")["values"]
        result = parse_out_file(references_dir / "B2_opt.out", "opt")
        assert result.ok
        assert result.converged is True
        assert result.converged_at_iteration == 2
        assert result.energy_hartree == pytest.approx(
            ref["final_energy_hartree"], abs=1e-6
        )
        assert result.energies_hartree == [-76.388277, -76.388301]

    def test_b2_final_structure(self, references_dir):
        result = parse_out_file(references_dir / "B2_opt.out", "opt")
        assert result.final_xyz is not None
        xyz_lines = result.final_xyz.splitlines()
        assert xyz_lines[0] == "3"
        assert [ln.split()[0] for ln in xyz_lines[2:]] == ["O", "H", "H"]
        # 与 pinned B2_opt.xyz 键长在容差内（.out 坐标 4 位小数，容差放宽）
        o, h1, _h2 = [[float(v) for v in ln.split()[1:]] for ln in xyz_lines[2:]]
        d_oh1 = math.dist(o, h1)
        ref = load_ref_json(references_dir, "B2.json")["values"]
        assert d_oh1 == pytest.approx(ref["oh_bond_angstrom"][0], abs=5e-4)
        assert any(p.endswith("_opt.xyz") for p in result.artifacts)

    def test_examples_lbfgs(self, maple_examples_dir):
        result = parse_out_file(
            maple_examples_dir / "opt" / "lbfgs" / "inp1.out", "opt"
        )
        assert result.ok
        assert result.converged is True
        assert result.converged_at_iteration == 54
        assert len(result.energies_hartree) == 55
        assert result.energy_hartree == pytest.approx(-383.655534)
        assert result.energies_hartree[0] == pytest.approx(-383.528090)

    def test_not_converged_flag(self):
        text = "Energy:   -76.388277 Convergence criteria  Is converged \n"
        result = parse_out(text, "opt")
        assert result.ok
        assert result.converged is False
        assert result.energy_hartree == pytest.approx(-76.388277)
        assert any("收敛行" in w for w in result.warnings)


class TestFreq:
    def test_b3_matches_pinned(self, references_dir):
        ref = load_ref_json(references_dir, "B3.json")["values"]
        result = parse_out_file(references_dir / "B3_freq.out", "freq")
        assert result.ok
        assert result.n_imaginary == ref["n_imaginary"] == 0
        real = [f for f in result.frequencies_cm1 if f > 5.0]
        assert len(result.frequencies_cm1) == 9
        assert real == pytest.approx(ref["frequencies_cm-1"], abs=1e-2)
        assert result.zpe_kcal_mol == pytest.approx(15.40)

    def test_examples_imaginary_mode(self, maple_examples_dir):
        result = parse_out_file(
            maple_examples_dir / "freq" / "mw" / "inp1.out", "freq"
        )
        assert result.ok
        assert len(result.frequencies_cm1) == 16
        assert result.n_imaginary == 1
        assert -2355.69 in result.frequencies_cm1
        assert result.zpe_kcal_mol == pytest.approx(33.85)
        assert result.gibbs_correction_kcal_mol is not None

    def test_freq_failure_returns_partial(self):
        result = parse_out("no frequencies here\n", "freq")
        assert not result.ok
        assert result.frequencies_cm1 == []
        assert result.warnings


class TestScan:
    def test_b8_matches_pinned(self, references_dir):
        ref = load_ref_json(references_dir, "B8.json")["values"]
        result = parse_out_file(references_dir / "B8_scan.out", "scan")
        assert result.ok
        assert len(result.energies_hartree) == 5
        assert result.energies_hartree == pytest.approx(
            ref["energies_hartree"], abs=1e-5
        )
        assert result.scan_values_angstrom == pytest.approx(
            ref["oh_distances_angstrom"], abs=1e-6
        )
        assert result.argmin_index == ref["argmin_index"] == 1
        assert result.energy_hartree == min(result.energies_hartree)

    def test_examples_relaxed_scan(self, maple_examples_dir):
        """relaxed scan（da.out）：每点是约束 opt 轨迹，取块内末值。"""
        result = parse_out_file(
            maple_examples_dir / "scan" / "da.out", "scan"
        )
        assert result.ok
        assert len(result.energies_hartree) == 256
        # 第一个扫描点（[3.00, 3.00]）的收敛能量
        assert result.energies_hartree[0] == pytest.approx(-386.255150)
        assert result.scan_values_angstrom[0] == pytest.approx(3.00)
        assert result.scan_values_angstrom[1] == pytest.approx(2.90)
        assert not result.warnings or all(
            "Total points" not in w for w in result.warnings
        )

    def test_examples_rigid_scan_2d(self, maple_examples_dir):
        result = parse_out_file(
            maple_examples_dir / "scan" / "da_rigid.out", "scan"
        )
        assert result.ok
        assert len(result.energies_hartree) == 256

    def test_scan_failure_returns_partial(self):
        result = parse_out("nothing scanned\n", "scan")
        assert not result.ok
        assert result.warnings


class TestMd:
    def test_examples_nve(self, maple_examples_dir):
        result = parse_out_file(
            maple_examples_dir / "md" / "nve" / "inp1.out", "md"
        )
        assert result.ok
        assert result.charge == 0 and result.multiplicity == 1
        assert len(result.energies_hartree) == 10
        assert result.md_times_fs == pytest.approx([0.1 * i for i in range(1, 11)])
        assert result.md_temperatures_K[0] == pytest.approx(299.15)
        assert result.energies_hartree[0] == pytest.approx(-40.483946)
        assert result.energy_hartree == pytest.approx(-40.483945)
        assert result.energy_drift_ha == pytest.approx(1e-6, abs=1e-6)

    def test_md_failure_returns_partial(self):
        result = parse_out("no md steps\n", "md")
        assert not result.ok
        assert result.warnings


class TestTs:
    def test_examples_prfo(self, maple_examples_dir):
        result = parse_out_file(
            maple_examples_dir / "ts" / "prfo" / "inp1.out", "ts"
        )
        assert result.ok
        assert result.converged is True
        assert len(result.energies_hartree) == 21
        assert result.energy_hartree == pytest.approx(-72.435037)
        assert result.energies_hartree[0] == pytest.approx(-72.416507)
        # 最终 TS 结构块（22 原子）
        assert result.final_xyz is not None
        assert result.final_xyz.splitlines()[0] == "22"

    def test_ts_not_converged_flag(self):
        text = (
            "Energy:   -72.4 Convergence criteria  Is converged \n"
            "Maximum Force:             0.009795     0.002850                No\n"
            "RMS Force:                 0.004193     0.001900                No\n"
            "Maximum Displacement:      0.008349     0.003150                Yes\n"
            "RMS Displacement:          0.003587     0.002100                Yes\n"
        )
        result = parse_out(text, "ts")
        assert result.ok
        assert result.converged is False
        assert result.energy_hartree == pytest.approx(-72.4)

    def test_ts_failure_returns_partial(self):
        result = parse_out("no ts iterations\n", "ts")
        assert not result.ok
        assert result.warnings


def test_parse_out_dispatch_rejects_unknown_type():
    with pytest.raises(ValueError):
        parse_out("", "neb2")
