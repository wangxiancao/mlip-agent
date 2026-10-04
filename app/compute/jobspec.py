"""JobSpec / JobResult schemas for the MAPLE compute layer.

JobSpec 描述一次计算任务（M2 工具层 `write_maple_inp` / `run_maple` 的输入契约）；
JobResult 是解析器（app/compute/parsers.py）对 MAPLE .out 的统一解析产物。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# PLAN §0.3：UMA 一律用显式本地 checkpoint，禁止联网下载
# （全部 checkpoint 集中存放于 /root/soft/ml_potential/checkpoints/，TICKET-0010）
DEFAULT_UMA_CHECKPOINT_PATH = (
    "/root/soft/ml_potential/checkpoints/uma/uma-s-1p1.pt"
)
# TICKET-0002：nequip-oam-xl 走 MAPLE 插件后端，显式本地 zip，禁止联网下载
DEFAULT_NEQUIP_MODEL_PATH = (
    "/root/soft/ml_potential/checkpoints/nequip/NequIP-OAM-XL-0.1.nequip.zip"
)
# macempa0 走 MAPLE 插件后端（plugins/maple_mace_plugin.py），显式本地
# scripted wrapper（scripts/make_macempa0_wrapper.py 产物），禁止联网下载
DEFAULT_MACEMPA0_MODEL_PATH = "/root/soft/ml_potential/checkpoints/mace/macempa0.pt"


class JobType(str, Enum):
    SP = "sp"
    OPT = "opt"
    FREQ = "freq"
    SCAN = "scan"
    MD = "md"
    TS = "ts"


class MapleModel(str, Enum):
    """MAPLE 计算器路由名（小写紧凑名）。

    内建名见 set_calculator._BUILTIN_NAME_TO_MODULE；nequip-oam-xl 为
    mlip_agent/plugins/maple_nequip_plugin.py 注册的插件名（TICKET-0002，
    经 MAPLE_CALCULATOR_PLUGINS 注入，非 MAPLE 内建）；macempa0 同理，
    由 mlip_agent/plugins/maple_mace_plugin.py 注册。
    """

    ANI1X = "ani1x"
    ANI2X = "ani2x"
    ANI1CCX = "ani1ccx"
    ANI1XNR = "ani1xnr"
    AIMNET2 = "aimnet2"
    AIMNET2NSE = "aimnet2nse"
    MACEOFF23S = "maceoff23s"
    MACEOFF23M = "maceoff23m"
    MACEOFF23L = "maceoff23l"
    UMA = "uma"
    NEQUIP_OAM_XL = "nequip-oam-xl"
    MACEMPA0 = "macempa0"


# MAPLE 支持的 UMA size（MAPLE/maple/function/read/command_control.py:
# SUPPORTED_UMA_SIZES）；错误信息会经 write_maple_inp 的 ToolResult 回给 LLM
SUPPORTED_UMA_SIZES = ("uma-s-1p1", "uma-s-1p2", "uma-m-1p1")
# UMA 的周期性 task（command_control.py:918-920：omol 与 PBC 互斥；
# _uma_calculator.py:43-44：周期体系必须显式指定下列 task 之一）
SUPPORTED_UMA_PERIODIC_TASKS = ("omat", "oc20", "oc22", "oc25", "omc", "odac")
# D7-3：live B4 实证 task_parser 会输出 "s" 等缩写，归一化到完整枚举值
_UMA_SIZE_ALIASES = {
    "s": "uma-s-1p1",
    "uma-s": "uma-s-1p1",
    "m": "uma-m-1p1",
    "uma-m": "uma-m-1p1",
    "1p2": "uma-s-1p2",
}


class Atom(BaseModel):
    symbol: str
    x: float
    y: float
    z: float


class StructureSpec(BaseModel):
    """结构来源：内联坐标或外部 xyz 文件（`XYZ <file>` 引用），二者恰居其一。

    cell（可选）：周期晶胞参数，渲染为 MAPLE 全局关键字 `#pbc(...)`。
    取值个数对齐 MAPLE `_parse_pbc`（command_control.py:320-341）：
    2 个（a,b；自动补 c=1000 Å，2D 材料/slab 用法）、3 个（正交盒）或
    6 个（a,b,c,α,β,γ）。xyz_file 路径下亦可使用（MAPLE 对 XYZ 引用同样
    应用 #pbc，input_reader.py:649-654）；若 xyz_file 是带 Lattice= 头的
    extxyz，MAPLE 自解析周期信息，无需 cell 字段。
    """

    coords: list[Atom] | None = None
    xyz_file: str | None = None
    cell: list[float] | None = None

    @field_validator("cell")
    @classmethod
    def _check_cell(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) not in (2, 3, 6):
            raise ValueError(f"cell 需 2/3/6 个值（a,b[,c][,α,β,γ]），得到 {len(v)} 个")
        return v

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "StructureSpec":
        if (self.coords is None) == (self.xyz_file is None):
            raise ValueError("structure 需要且只能提供 coords 或 xyz_file 之一")
        if self.coords is not None and not self.coords:
            raise ValueError("coords 不能为空")
        return self


class ModelSpec(BaseModel):
    """模型选择。UMA 需 size/task/checkpoint_path；插件后端（nequip-oam-xl /
    macempa0）需 model_path（缺省分别用 DEFAULT_NEQUIP_MODEL_PATH /
    DEFAULT_MACEMPA0_MODEL_PATH）；其余模型忽略这些字段。"""

    name: MapleModel
    uma_size: str = "uma-s-1p1"
    uma_task: str = "omol"
    checkpoint_path: str | None = None
    model_path: str | None = None

    @field_validator("uma_size", mode="before")
    @classmethod
    def _normalize_uma_size(cls, v: object) -> object:
        """常见缩写归一化到完整枚举值；无法识别时报含允许值清单的错误。"""
        if not isinstance(v, str):
            return v
        key = v.strip().lower()
        if key in SUPPORTED_UMA_SIZES:
            return key
        if key in _UMA_SIZE_ALIASES:
            return _UMA_SIZE_ALIASES[key]
        raise ValueError(
            f"Unsupported UMA size: {v!r}。uma_size 允许值: "
            f"{', '.join(SUPPORTED_UMA_SIZES)}（缩写 s/m/1p2 会自动归一化）"
        )

    def resolved_checkpoint(self) -> str:
        return self.checkpoint_path or DEFAULT_UMA_CHECKPOINT_PATH

    def resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        if self.name == MapleModel.MACEMPA0:
            return DEFAULT_MACEMPA0_MODEL_PATH
        return DEFAULT_NEQUIP_MODEL_PATH


class OptSpec(BaseModel):
    method: Literal["lbfgs", "rfo", "sd", "cg", "sdcg"] = "lbfgs"
    max_iter: int = Field(default=256, gt=0)
    # MAPLE 全局参数 #level=（Gaussian 风格收敛阈值，缺省 medium）
    level: Literal[
        "extratight", "tight", "medium", "loose", "extraloose", "superloose"
    ] = "medium"


class FreqSpec(BaseModel):
    method: Literal["mw", "nonmw", "both"] = "mw"
    temperature: float = Field(default=298.15, gt=0)


class ScanCoordinate(BaseModel):
    """一条扫描坐标：`S <atom_i> <atom_j> <delta Å> <n_steps>`，1-based 原子序号。

    产出 n_steps + 1 个扫描点（含起始结构）。
    """

    atom_i: int = Field(gt=0)
    atom_j: int = Field(gt=0)
    delta: float
    n_steps: int = Field(gt=0)


class ScanSpec(BaseModel):
    # rigid = 逐点单点；relaxed = 逐点约束优化（MAPLE 缺省）
    mode: Literal["rigid", "relaxed"] = "rigid"
    coordinates: list[ScanCoordinate] = Field(min_length=1)


class MdSpec(BaseModel):
    """mdp_file 优先；缺省时用下方字段渲染 `#md(...)` 内联参数。"""

    mdp_file: str | None = None
    ensemble: Literal["nve", "nvt", "npt"] = "nve"
    timestep_fs: float = Field(default=0.5, gt=0)
    steps: int = Field(default=20, gt=0)
    temperature: float = Field(default=300.0, gt=0)
    # 缺省 None = 用 MAPLE 内建默认（log_every=100 / traj_every=100）；
    # 短步数 MD（如 B7 的 20 步）需显式 log_every=1 才会输出逐步能量行
    log_every: int | None = Field(default=None, gt=0)
    traj_every: int | None = Field(default=None, gt=0)


class TsSpec(BaseModel):
    """过渡态搜索。方法名与 MAPLE examples/ts/ 对齐：prfo（默认，RS-PRFO
    本征值跟随，需合理初猜）、neb/string/autoneb/dimer/dmf。"""

    method: Literal["prfo", "neb", "string", "autoneb", "dimer", "dmf"] = "prfo"


class JobSpec(BaseModel):
    """一次 MAPLE 计算的完整描述（inp_builder 的输入）。"""

    model_config = ConfigDict(use_enum_values=True)

    job_type: JobType
    model: ModelSpec
    structure: StructureSpec
    charge: int = 0
    multiplicity: int = Field(default=1, ge=1)
    device: Literal["gpu0", "cpu"] = "gpu0"
    opt: OptSpec = Field(default_factory=OptSpec)
    freq: FreqSpec = Field(default_factory=FreqSpec)
    scan: ScanSpec | None = None
    md: MdSpec = Field(default_factory=MdSpec)
    ts: TsSpec = Field(default_factory=TsSpec)

    @model_validator(mode="after")
    def _scan_required_for_scan_job(self) -> "JobSpec":
        if self.job_type == JobType.SCAN and self.scan is None:
            raise ValueError("job_type=scan 时必须提供 scan 段")
        return self

    @model_validator(mode="after")
    def _freq_requires_xyz_file(self) -> "JobSpec":
        # D7-3：live B6 实证 MAPLE freq 要求 XYZ 文件引用，内联坐标必失败；
        # 提前报错，错误信息经 write_maple_inp 的 ToolResult 回给 LLM
        if self.job_type == JobType.FREQ and self.structure.coords is not None:
            raise ValueError(
                "freq 任务的 structure 必须用 xyz_file（MAPLE freq 要求 XYZ 文件引用，"
                "不支持内联坐标）；opt→freq 链式任务请用 opt 作业的优化输出结构文件"
            )
        return self

    @model_validator(mode="after")
    def _pbc_model_gate(self) -> "JobSpec":
        """PBC 门禁（2026-10-03 PBC 落地）：cell 存在即周期体系。MAPLE 内当前
        SUPPORTS_PBC=True 的后端：UMA（_uma_calculator.py:73，需显式周期 task，
        omol 与 PBC 互斥）、MACE 系（TICKET-0008，含插件 macempa0）与 AIMNet2 系
        （TICKET-0009，PBC 时库仑缺省自动切 dsf，可显式 ewald）。其余模型遇 PBC
        在 MAPLE 工厂层报错（set_calculator.py:222-226），此处前置为更明确的
        校验错误。
        """
        if self.structure.cell is None:
            return self
        pbc_models = {
            MapleModel.MACEOFF23S.value,
            MapleModel.MACEOFF23M.value,
            MapleModel.MACEOFF23L.value,
            MapleModel.MACEMPA0.value,
            MapleModel.AIMNET2.value,
            MapleModel.AIMNET2NSE.value,
        }
        if self.model.name in pbc_models:
            return self
        if self.model.name != MapleModel.UMA.value:
            raise ValueError(
                f"周期体系（structure.cell）当前 MAPLE 路径支持 uma / MACE 系 / AIMNet2 系；"
                f"得到 {self.model.name!r}（NequIP/TACE/DPA-4 周期任务请走各自直算工具，"
                "extxyz 带 Lattice 头）"
            )
        task = (self.model.uma_task or "omol").strip().lower()
        if task not in SUPPORTED_UMA_PERIODIC_TASKS:
            raise ValueError(
                f"周期体系的 uma_task 必须是 "
                f"{'/'.join(SUPPORTED_UMA_PERIODIC_TASKS)} 之一（omol 与 PBC 互斥），"
                f"得到 {self.model.uma_task!r}"
            )
        return self


class JobResult(BaseModel):
    """parsers 对 MAPLE .out 的统一解析产物（M2 工具层 parse_maple_out 的返回）。

    解析失败不抛异常：ok=False / 字段置 None，并在 warnings 里说明。
    """

    job_type: JobType
    ok: bool = True
    charge: int | None = None
    multiplicity: int | None = None

    # sp / opt 终态能量（Hartree）
    energy_hartree: float | None = None
    # opt 每步能量 / scan 每点能量 / md 每步总能量
    energies_hartree: list[float] = Field(default_factory=list)

    # opt
    converged: bool | None = None
    converged_at_iteration: int | None = None
    final_xyz: str | None = None

    # freq
    frequencies_cm1: list[float] = Field(default_factory=list)
    n_imaginary: int | None = None
    zpe_kcal_mol: float | None = None
    gibbs_correction_kcal_mol: float | None = None

    # scan
    scan_values_angstrom: list[float] = Field(default_factory=list)
    argmin_index: int | None = None

    # md
    md_times_fs: list[float] = Field(default_factory=list)
    md_temperatures_K: list[float] = Field(default_factory=list)
    energy_drift_ha: float | None = None

    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
