"""MAPLE calculator plugin：把 MACE-MPA-0 (medium) 注册为 MAPLE 后端（macempa0）。

本文件只依赖 COMPUTE_VENV 已有包（maple / torch），不改 MAPLE 源码。
接入方式（mlip_agent/app/tools/maple_tools.py 负责，仅当 JobSpec 选择插件模型名
macempa0 时注入）：

    MAPLE_CALCULATOR_PLUGINS=maple_mace_plugin
    PYTHONPATH=<mlip_agent/plugins>

机制（与 plugins/maple_nequip_plugin.py 同款）：env 设置后，本模块在 MAPLE 进程内
**首个 calculator 构建时**即被 import（`set_calculator._discover_calculator_class`
经 `load_calculator_plugins_from_env` 触发），import 时 @register_calculator 完成
注册；实例化则按 `#model=` 的名字匹配——只有 `#model=macempa0(...)` 才构造
本计算器，其余模型行为不变。

实现策略：直接子类化 MAPLE 内建 MACECalculator
（maple/function/calculator/mace/_mace_calculator.py），只覆盖注册名与
checkpoint 解析类属性；data 构造（build_data_from_atoms）、dtype 推断
（model_float_dtype）、autograd forces/analytic Hessian、_finalize_results 的
eV→Hartree 换算全部继承，**本插件禁止自行乘 EV2HARTREE**。

模型文件：/root/soft/ml_potential/checkpoints/mace/macempa0.pt，由
scripts/make_macempa0_wrapper.py 从 mace-foundations 官方
mace-mpa-0-medium.model（MIT 许可，MPTrj+sAlex 晶体训练域）经 e3nn
jit.compile 生成；与 maceoff23s.pt 同一 wrapper 契约（r_max/atomic_numbers
buffer + forward(data, local_or_ghost) → per-graph eV）。
"""

from __future__ import annotations

from maple.function.calculator.calculator_base import register_calculator
from maple.function.calculator.mace._mace_calculator import MACECalculator


@register_calculator
class MACEMPA0PluginCalculator(MACECalculator):
    """MACE-MPA-0 medium（本地 scripted wrapper），实现完全继承 MACECalculator。"""

    MODEL_NAMES = ("macempa0",)
    CHECKPOINT_FILENAME = None  # 无 HuggingFace 自动下载
    REQUIRES_LOCAL_MODEL_FILE = True  # 必须显式 model_path（本地 .pt 绝对路径）
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = "model_path"
    # TICKET-0008：基类 MACECalculator 已支持 PBC（matscipy 周期邻居表，
    # 上游 MACE 约定）；本插件 data 构造全继承，Si 金刚石原胞对照官方
    # mace ASE calculator 能量差 0.0 eV（容差 1e-6），显式翻开能力位。
    SUPPORTS_PBC = True
    # MODEL_ENERGY_UNIT='eV' / SUPPORTED_HESSIAN_MODES=('analytic','numerical')
    # / SUPPORTS_CHARGE_MULT=False 均继承 MACECalculator
