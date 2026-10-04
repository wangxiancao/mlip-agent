"""MAPLE calculator plugin：把 NequIP-OAM-XL 注册为 MAPLE 后端（TICKET-0002）。

本文件只依赖 COMPUTE_VENV 已有包（maple / ase / nequip / torch），不改 MAPLE 源码。
接入方式（mlip_agent/app/tools/maple_tools.py 负责，仅当 JobSpec 选择插件模型名
nequip-oam-xl 时注入）：

    MAPLE_CALCULATOR_PLUGINS=maple_nequip_plugin
    PYTHONPATH=<mlip_agent/plugins>

机制（措辞按 TICKET-0002 §7 审批修正）：env 设置后，本模块在 MAPLE 进程内
**首个 calculator 构建时**即被 import（`set_calculator._discover_calculator_class`
经 `load_calculator_plugins_from_env` 触发），import 时 @register_calculator 完成
注册；实例化则按 `#model=` 的名字匹配——只有 `#model=nequip-oam-xl(...)` 才构造
本计算器，其余模型行为不变。

单位契约（AUTHORING.md）：calculate() 产出 eV（MODEL_ENERGY_UNIT='eV'），
eV→Hartree 换算由 CalcABC._finalize_results 统一负责，**本插件禁止自行乘
EV2HARTREE**；数值 Hessian 由 CalcABC 默认路径（numerical_hessian_from_atoms）
回调 calculate() 提供，天然继承 Hartree 单位。

模型加载与 B5 直算路径（app/compute/scripts/nequip_ase.py，D1-1）同款：
`NequIPCalculator._from_saved_model`（nequip 0.17.1 半私有 API，nequip 升级时
与直算脚本同步复查，见 docs/decisions.md D7-1）。
"""

from __future__ import annotations

from ase.calculators.calculator import all_changes

from maple.function.calculator.calculator_base import CalcABC, register_calculator


@register_calculator
class NequIPPluginCalculator(CalcABC):
    """NequIP（nequip-package zip）后端，委托给 ASE NequIPCalculator。"""

    MODEL_NAMES = ("nequip-oam-xl",)
    MODEL_ENERGY_UNIT = "eV"  # NequIP ASE 计算器原生单位；换算归 _finalize_results
    SUPPORTED_HESSIAN_MODES = ("numerical",)  # NequIP 不暴露可微 autograd forward
    SUPPORTS_CHARGE_MULT = False  # NequIP-OAM-XL 为闭壳层中性分子训练
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None  # 无 HuggingFace 自动下载
    REQUIRES_LOCAL_MODEL_FILE = True  # 必须显式 model_path（本地 zip 绝对路径）
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = "model_path"

    implemented_properties = ["energy", "free_energy", "forces", "hessian"]

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs["model_path"] = resolved_model_path
        return kwargs

    def __init__(
        self,
        device,
        model="nequip-oam-xl",
        *,
        model_path=None,
        implicit="none",
        solvent="none",
        **_ignored,
    ):
        super().__init__()
        if model_path is None:
            raise ValueError(
                "nequip-oam-xl 需要显式 model_path=/path/to/model.nequip.zip"
                "（在 #model=nequip-oam-xl(model_path=...) 中给出）；"
                "本插件不支持联网下载。"
            )
        self.device = device
        self.hessian = "numerical"

        from nequip.integrations.ase import NequIPCalculator

        # D1-1：zip 是 nequip-package 产物，from_compiled_model 不适用
        self.nequip_calc = NequIPCalculator._from_saved_model(
            str(model_path),
            device=device,
            chemical_species_to_atom_type_map=True,
        )

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)  # PBC/solvent guards

        # 委托给被包装的 NequIPCalculator；其结果为标准 ASE 单位（eV / eV·Å⁻¹）
        self.nequip_calc.calculate(
            atoms, properties=["energy", "forces"], system_changes=all_changes
        )
        energy_ev = float(self.nequip_calc.results["energy"])
        forces_np = (
            self.nequip_calc.results["forces"] if "forces" in properties else None
        )

        hessian = None
        if "hessian" in properties:
            hessian = self.get_hessian(atoms)

        # _finalize_results 负责 eV→Hartree 与 results 写入；勿自行换算
        self._finalize_results(atoms, energy=energy_ev, forces=forces_np, hessian=hessian)
