"""JobSpec → MAPLE .inp 文本。

语法依据 MAPLE 源码（maple/function/read/command_control.py、input_reader.py）
与 benchmarks/references 金标准 inp（scripts/make_golden_refs.sh）：
- 全局参数 `#key=value`，任务行 `#<task>(k=v,...)`，空行后是坐标段；
- 周期晶胞用全局关键字 `#pbc(a,b[,c][,α,β,γ])`（command_control.py:111）；
- 坐标段首行可选 `<charge> <multiplicity>`（缺省 0 1），之后是内联坐标
  （`Symbol x y z`，Å）；XYZ 引用外部坐标文件时 charge/mult 内联进引用行
  （`XYZ <charge> <mult> <file>`），md 任务一律显式输出（dispatcher 强制）；
- scan 坐标行在坐标段之后：`S <i> <j> <delta Å> <n_steps>`（1-based）。
"""

from __future__ import annotations

from .jobspec import JobSpec, JobType, MapleModel, StructureSpec


def _model_line(spec: JobSpec) -> str:
    m = spec.model
    if m.name == MapleModel.UMA.value:
        return (
            f"#model=uma(size={m.uma_size},task={m.uma_task},"
            f"checkpoint_path={m.resolved_checkpoint()})"
        )
    if m.name == MapleModel.NEQUIP_OAM_XL.value:
        # TICKET-0002：插件后端（plugins/maple_nequip_plugin.py），显式本地 zip
        return f"#model=nequip-oam-xl(model_path={m.resolved_model_path()})"
    if m.name == MapleModel.MACEMPA0.value:
        # 插件后端（plugins/maple_mace_plugin.py），显式本地 scripted wrapper
        return f"#model=macempa0(model_path={m.resolved_model_path()})"
    return f"#model={MapleModel(m.name).value}"


def _task_line(spec: JobSpec) -> str:
    jt = spec.job_type
    if jt == JobType.SP.value:
        return "#sp(verbose=0)"
    if jt == JobType.OPT.value:
        return f"#opt(method={spec.opt.method},max_iter={spec.opt.max_iter})"
    if jt == JobType.FREQ.value:
        return (
            f"#freq(method={spec.freq.method},"
            f"temperature={spec.freq.temperature:g})"
        )
    if jt == JobType.SCAN.value:
        assert spec.scan is not None
        if spec.scan.mode == "rigid":
            return "#scan(mode=rigid)"
        return "#scan()"
    if jt == JobType.MD.value:
        md = spec.md
        if md.mdp_file:
            return f"#md(mdp={md.mdp_file})"
        parts = (
            f"ensemble={md.ensemble}",
            f"timestep={md.timestep_fs:g}",
            f"steps={md.steps}",
            f"temperature={md.temperature:g}",
        )
        if md.log_every is not None:
            parts += (f"log_every={md.log_every}",)
        if md.traj_every is not None:
            parts += (f"traj_every={md.traj_every}",)
        return f"#md({','.join(parts)})"
    if jt == JobType.TS.value:
        return f"#ts(method={spec.ts.method})"
    raise ValueError(f"未知任务类型: {jt}")


def _structure_lines(structure: StructureSpec) -> list[str]:
    assert structure.coords is not None
    return [
        f"{a.symbol:<2} {a.x:18.8f} {a.y:18.8f} {a.z:18.8f}" for a in structure.coords
    ]


def build_inp(spec: JobSpec) -> str:
    """渲染 JobSpec 为 MAPLE .inp 文本。"""
    lines: list[str] = [_model_line(spec), _task_line(spec)]

    if spec.job_type in (JobType.OPT.value, JobType.SCAN.value) and spec.opt.level != "medium":
        lines.append(f"#level={spec.opt.level}")
    lines.append(f"#device={spec.device}")
    # 周期晶胞：MAPLE 全局关键字 #pbc（command_control.py:111 注册）；
    # 2/3/6 个值分别对应 2D 用法（自动补 c=1000 Å）/正交盒/完整晶胞参数
    if spec.structure.cell is not None:
        lines.append("#pbc(" + ",".join(f"{v:g}" for v in spec.structure.cell) + ")")
    lines.append("")

    # MAPLE md 强制显式 charge/multiplicity（dispatcher 校验 atoms.info）；
    # XYZ 引用时须内联进引用行（`XYZ <charge> <mult> <path>`，xyz_reader 支持），
    # 独立的 `0 1` 行与 XYZ 行同组会被判为混合坐标报错
    explicit_cm = (
        spec.charge != 0
        or spec.multiplicity != 1
        or spec.job_type == JobType.MD.value
    )
    if spec.structure.xyz_file is not None:
        if explicit_cm:
            lines.append(
                f"XYZ {spec.charge} {spec.multiplicity} {spec.structure.xyz_file}"
            )
        else:
            lines.append(f"XYZ {spec.structure.xyz_file}")
    else:
        if explicit_cm:
            lines.append(f"{spec.charge} {spec.multiplicity}")
        lines.extend(_structure_lines(spec.structure))

    if spec.job_type == JobType.SCAN.value:
        assert spec.scan is not None
        lines.append("")
        for sc in spec.scan.coordinates:
            lines.append(f"S {sc.atom_i} {sc.atom_j} {sc.delta:g} {sc.n_steps}")

    return "\n".join(lines) + "\n"
