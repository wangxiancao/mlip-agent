"""MAPLE .out 解析器：sp / opt / freq / scan / md 五类纯函数。

正则基于 benchmarks/references/ 与 MAPLE/examples/** 的真实 .out（M0 已核格式）：
- sp  : `Energy: -383.4722385376 Hartree`（10 位小数）、`Charge: 0, Multiplicity: 1`
- opt : 每步 `Energy: <6 位小数> Convergence criteria  Is converged`，
        收敛行 `<Optimizer> converged at iteration N.`，终帧写 `<inpbase>_opt.xyz`
- freq: `     6:     1966.21 cm**-1`（虚频带 `***imaginary mode***`）、
        `Zero point energy ... 15.40 kcal/mol`
- scan: 每点一块 `Scanning combination k/N: [v...]`，块内最后一个 `Energy:` 为该点能量
        （rigid 每块一个值；relaxed 块内是 opt 轨迹，取末值），
        汇总行 `Scan completed! Total points: N`、`Energy range: lo to hi Hartree`
- md  : `  Step        1        0.10 fs  T  299.15 K  E     -40.483946 Ha`，
        末尾统计 `Mean total energy:` / `Std deviation:` / `Energy drift:`

所有解析器不抛异常：缺字段时返回部分结果并在 warnings 里说明。
"""

from __future__ import annotations

import re
from pathlib import Path

from .jobspec import JobResult, JobType

RE_CHARGE_MULT = re.compile(r"^Charge:\s*(-?\d+),\s*Multiplicity:\s*(\d+)", re.M)
RE_SP_ENERGY = re.compile(r"^Energy:\s+(-?\d+\.\d+)\s+Hartree", re.M)
RE_OPT_STEP_ENERGY = re.compile(
    r"^Energy:\s+(-?\d+\.\d+)\s+Convergence criteria", re.M
)
RE_ANY_ENERGY = re.compile(r"^Energy:\s+(-?\d+\.\d+)", re.M)
RE_CONVERGED = re.compile(r"([A-Za-z]+) converged at iteration (\d+)")
RE_FINAL_FRAME = re.compile(r"Final frame written to (\S+)")
RE_COORD_ROW = re.compile(
    r"^\s*\d+\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
RE_FREQ_LINE = re.compile(
    r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1(\s+\*\*\*imaginary mode\*\*\*)?",
    re.M,
)
RE_ZPE = re.compile(r"^Zero point energy\s+\.\.\.\s+(-?\d+\.\d+)\s+kcal/mol", re.M)
RE_GIBBS = re.compile(
    r"^Final Gibbs free energy corr\.\s+\.\.\.\s+(-?\d+\.\d+)\s+kcal/mol", re.M
)
RE_SCAN_HEADER = re.compile(r"Scanning combination (\d+)/(\d+): \[([^\]]*)\]")
RE_SCAN_TOTAL = re.compile(r"Scan completed! Total points: (\d+)")
RE_SCAN_RANGE = re.compile(
    r"^Energy range:\s+(-?\d+\.\d+) to (-?\d+\.\d+) Hartree", re.M
)
RE_SCAN_RESULTS = re.compile(r"Results saved to: (\S+)")
RE_MD_STEP = re.compile(
    r"^\s*Step\s+(\d+)\s+(-?\d+(?:\.\d+)?)\s*fs\s+T\s+(-?\d+(?:\.\d+)?)\s*K\s+"
    r"E\s+(-?\d+(?:\.\d+)?)\s*Ha",
    re.M,
)
RE_MD_MEAN_E = re.compile(r"Mean total energy:\s+(-?\d+(?:\.\d+)?)\s+Ha")
RE_MD_STD_E = re.compile(r"Std deviation:\s+(-?\d+(?:\.\d+)?)\s+Ha")
RE_MD_DRIFT_KCAL = re.compile(r"Energy drift:\s+([+-]?\d+(?:\.\d+)?)\s+kcal/mol")
RE_TS_CRITERION = re.compile(
    r"^(?:Maximum Force|RMS Force|Maximum Displacement|RMS Displacement):"
    r".*\b(Yes|No)\s*$",
    re.M,
)


def _parse_charge_mult(text: str, warnings: list[str]) -> tuple[int | None, int | None]:
    m = RE_CHARGE_MULT.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    warnings.append("未找到 Charge/Multiplicity 行")
    return None, None


def _last_coord_block_xyz(text: str, comment: str) -> str | None:
    """取 .out 中最后一个坐标块，渲染为 xyz 文本。"""
    blocks: list[list[tuple[str, str, str, str]]] = []
    current: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        m = RE_COORD_ROW.match(line)
        if m:
            current.append((m.group(1), m.group(2), m.group(3), m.group(4)))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    if not blocks:
        return None
    atoms = blocks[-1]
    lines = [str(len(atoms)), comment]
    lines.extend(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
    return "\n".join(lines) + "\n"


def parse_sp(text: str) -> JobResult:
    warnings: list[str] = []
    charge, mult = _parse_charge_mult(text, warnings)
    energies = [float(x) for x in RE_SP_ENERGY.findall(text)]
    result = JobResult(
        job_type=JobType.SP,
        charge=charge,
        multiplicity=mult,
        energies_hartree=energies,
        warnings=warnings,
    )
    if energies:
        result.energy_hartree = energies[-1]
        if len(energies) > 1:
            result.warnings.append(f"sp 任务出现 {len(energies)} 个能量行，取最后一个")
    else:
        result.ok = False
        result.warnings.append("未找到 `Energy: <float> Hartree` 行")
    return result


def parse_opt(text: str) -> JobResult:
    warnings: list[str] = []
    charge, mult = _parse_charge_mult(text, warnings)
    energies = [float(x) for x in RE_OPT_STEP_ENERGY.findall(text)]
    result = JobResult(
        job_type=JobType.OPT,
        charge=charge,
        multiplicity=mult,
        energies_hartree=energies,
        warnings=warnings,
    )
    if energies:
        result.energy_hartree = energies[-1]
    else:
        result.ok = False
        result.warnings.append("未找到任何迭代能量行（`Energy: ... Convergence criteria`）")

    m = RE_CONVERGED.search(text)
    result.converged = m is not None
    if m:
        result.converged_at_iteration = int(m.group(2))
    else:
        result.warnings.append("未找到收敛行（`<optimizer> converged at iteration N`）")

    for path in RE_FINAL_FRAME.findall(text):
        if path not in result.artifacts:
            result.artifacts.append(path)

    if energies:
        xyz = _last_coord_block_xyz(text, comment=f"Energy = {energies[-1]:.6f}")
        if xyz:
            result.final_xyz = xyz
        else:
            result.warnings.append("未能从 .out 提取最终坐标块")
    return result


def parse_freq(text: str) -> JobResult:
    warnings: list[str] = []
    charge, mult = _parse_charge_mult(text, warnings)
    freqs: list[float] = []
    n_imag_marked = 0
    for _idx, value, imag_flag in RE_FREQ_LINE.findall(text):
        freqs.append(float(value))
        if imag_flag:
            n_imag_marked += 1
    result = JobResult(
        job_type=JobType.FREQ,
        charge=charge,
        multiplicity=mult,
        frequencies_cm1=freqs,
        warnings=warnings,
    )
    if not freqs:
        result.ok = False
        result.warnings.append("未找到频率行（`<idx>: <float> cm**-1`）")
    else:
        # 与 MAPLE 汇总口径一致：ν < -5.0 cm⁻¹ 计为虚频
        n_imag = sum(1 for f in freqs if f < -5.0)
        if n_imag_marked and n_imag_marked != n_imag:
            result.warnings.append(
                f"imaginary mode 标记数（{n_imag_marked}）与 ν<-5 计数（{n_imag}）不一致，采用后者"
            )
        result.n_imaginary = n_imag

    m = RE_ZPE.search(text)
    if m:
        result.zpe_kcal_mol = float(m.group(1))
    else:
        result.warnings.append("未找到 Zero point energy 行")
    m = RE_GIBBS.search(text)
    if m:
        result.gibbs_correction_kcal_mol = float(m.group(1))
    return result


def parse_scan(text: str) -> JobResult:
    warnings: list[str] = []
    charge, mult = _parse_charge_mult(text, warnings)
    headers = list(RE_SCAN_HEADER.finditer(text))
    result = JobResult(
        job_type=JobType.SCAN, charge=charge, multiplicity=mult, warnings=warnings
    )

    energies: list[float] = []
    first_coord_values: list[float] = []
    for k, header in enumerate(headers):
        start = header.end()
        end = headers[k + 1].start() if k + 1 < len(headers) else len(text)
        block = text[start:end]
        block_energies = RE_ANY_ENERGY.findall(block)
        if block_energies:
            # rigid 每点一个值；relaxed 块内是 opt 轨迹，末值即该点收敛能量
            energies.append(float(block_energies[-1]))
        else:
            warnings.append(f"扫描点 {header.group(1)} 块内未找到 Energy 行")
        values = [float(v) for v in header.group(3).split(",")]
        first_coord_values.append(values[0])
    result.energies_hartree = energies
    result.scan_values_angstrom = first_coord_values
    if energies:
        result.energy_hartree = min(energies)
        result.argmin_index = energies.index(result.energy_hartree)

    m = RE_SCAN_TOTAL.search(text)
    if m:
        n_total = int(m.group(1))
        if energies and n_total != len(energies):
            result.warnings.append(
                f"Total points={n_total} 与解析到的点数 {len(energies)} 不一致"
            )
    elif not headers:
        result.warnings.append("未找到 `Scan completed! Total points:` 汇总行")

    m = RE_SCAN_RESULTS.search(text)
    if m:
        result.artifacts.append(m.group(1))

    if not energies:
        result.ok = False
        result.warnings.append("未解析到任何扫描点能量")
    return result


def parse_md(text: str) -> JobResult:
    warnings: list[str] = []
    charge, mult = _parse_charge_mult(text, warnings)
    steps = RE_MD_STEP.findall(text)
    result = JobResult(
        job_type=JobType.MD, charge=charge, multiplicity=mult, warnings=warnings
    )
    if not steps:
        result.ok = False
        result.warnings.append("未找到 MD 步进行（`Step ... fs T ... K E ... Ha`）")
        return result

    result.md_times_fs = [float(s[1]) for s in steps]
    result.md_temperatures_K = [float(s[2]) for s in steps]
    result.energies_hartree = [float(s[3]) for s in steps]
    result.energy_hartree = result.energies_hartree[-1]
    result.energy_drift_ha = abs(
        result.energies_hartree[-1] - result.energies_hartree[0]
    )

    m = RE_MD_MEAN_E.search(text)
    if not m:
        result.warnings.append("未找到 `Mean total energy:` 统计行")
    if not RE_MD_STD_E.search(text):
        result.warnings.append("未找到 `Std deviation:` 统计行")
    m = RE_MD_DRIFT_KCAL.search(text)
    if not m:
        result.warnings.append("未找到 `Energy drift:` 行（非 NVE 或无统计段）")
    return result


def parse_ts(text: str) -> JobResult:
    """TS 搜索（prfo 等）：迭代能量行同 opt；收敛判定取最后一个准则块
    （Maximum/RMS Force、Maximum/RMS Displacement 四行）是否全部 Yes。"""
    warnings: list[str] = []
    charge, mult = _parse_charge_mult(text, warnings)
    energies = [float(x) for x in RE_ANY_ENERGY.findall(text)]
    result = JobResult(
        job_type=JobType.TS,
        charge=charge,
        multiplicity=mult,
        energies_hartree=energies,
        warnings=warnings,
    )
    if energies:
        result.energy_hartree = energies[-1]
    else:
        result.ok = False
        result.warnings.append("未找到任何迭代能量行（`Energy: <float>`）")

    flags = RE_TS_CRITERION.findall(text)
    if len(flags) >= 4:
        result.converged = all(f == "Yes" for f in flags[-4:])
    else:
        result.converged = False
        result.warnings.append("未找到完整的收敛准则块（4 行 Yes/No）")

    if energies:
        xyz = _last_coord_block_xyz(text, comment=f"TS Energy = {energies[-1]:.6f}")
        if xyz:
            result.final_xyz = xyz
        else:
            result.warnings.append("未能从 .out 提取 TS 坐标块")
    return result


_PARSERS = {
    JobType.SP: parse_sp,
    JobType.OPT: parse_opt,
    JobType.FREQ: parse_freq,
    JobType.SCAN: parse_scan,
    JobType.MD: parse_md,
    JobType.TS: parse_ts,
}


def parse_out(text: str, job_type: JobType | str) -> JobResult:
    """按任务类型分派解析。"""
    return _PARSERS[JobType(job_type)](text)


def parse_out_file(path: str | Path, job_type: JobType | str) -> JobResult:
    return parse_out(Path(path).read_text(), job_type)
