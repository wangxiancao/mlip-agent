## TICKET-0002：NequIP-OAM-XL 注册为 MAPLE 后端（calculator plugin，不动 MAPLE 源码）

- 状态：待审批 ｜ 提出：2026-10-02 ｜ 审批人：____

### 1. 改动对象

**不在 MAPLE/ 内改动任何文件**。全部新增内容在 mlip_agent/ 内：

- 新增 `mlip_agent/plugins/maple_nequip_plugin.py`：一个按
  `MAPLE/maple/function/calculator/AUTHORING.md` 契约编写的插件模块，
  内含 `@register_calculator` 装饰的 `NequIPPluginCalculator(CalcABC)`；
- `mlip_agent/app/tools/maple_tools.py` / `app/compute/runner.py`：为 maple
  子进程 env 增加 `MAPLE_CALCULATOR_PLUGINS=maple_nequip_plugin` 与
  `PYTHONPATH=<mlip_agent/plugins>`（仅当 JobSpec 选择 nequip 插件名时注入，
  其余作业 env 不变）；
- 模型清单 `MODEL_CATALOG` 的 `nequip-oam-xl` 条目增加 `backend: maple` 选项说明。

冻结资产触动情况：MAPLE/ 源码零改动；COMPUTE_VENV 零装包（nequip 0.17.1 已在
COMPUTE_VENV 中，插件只 import 既有包）；checkpoint 文件不改动。

### 2. 动机与问题描述

现状（PLAN §0.3、docs/decisions.md D1-1）：NequIP-OAM-XL 走「直接 ASE 计算器」
旁路——后端生成 job.json，`$COMPUTE_VENV/bin/python app/compute/scripts/nequip_ase.py`
执行，能量单位 eV，与 MAPLE 路径（Hartree）语义分离。这带来两个长期成本：

1. **能力割裂**：MAPLE 侧的任务编排（opt/freq/scan/md dispatcher、溶剂、Dimer
   TS 搜索等）对 NequIP 不可用；直算脚本只支持 sp/opt。
2. **双轨维护**：parsers/inp_builder/前端结果面板都要为两套输出格式各写一份
   （`result.json` eV vs `.out` Hartree，见 web/src/api/agui.ts 的两个分支）。

MAPLE 自带插件机制（`maple/function/calculator/calculator_base.py:38-47`
`import_calculator_plugin` / `load_calculator_plugins_from_env`，
`set_calculator.py:172-181` 的三层发现：`module=` 选项 → 内建表 →
`MAPLE_CALCULATOR_PLUGINS` 环境变量），官方示例
`MAPLE/examples/calculator_plugin/` 验证了「不改 MAPLE 源码即可接入新后端」。

### 3. 提案（影响面分析）

按模板属 **B. 框架内新增**（不改 MAPLE 数值行为），但**间接影响金标准语义**，
单列如下。

**插件设计要点**（依据 AUTHORING.md 契约与 examples/calculator_plugin 模板）：

- `MODEL_NAMES = ('nequip-oam-xl',)`；`MODEL_ENERGY_UNIT = 'eV'`（NequIP ASE
  计算器原生单位），**不自行乘 EV2HARTREE**，交给 `_finalize_results` 换算；
- `CHECKPOINT_FILENAME = None`、`REQUIRES_LOCAL_MODEL_FILE = True`、
  `MODEL_PATH_OPTION = 'model_path'`：模型 zip 用显式本地绝对路径
  （`/root/soft/ml_potential/nequip/NequIP-OAM-XL-0.1.nequip.zip`），杜绝联网下载；
- 加载走 `NequIPCalculator._from_saved_model(path, device=...,
  chemical_species_to_atom_type_map=True)`（D1-1：zip 是 nequip-package 产物，
  `from_compiled_model` 不适用）；`calculate()` 委托给被包装的
  NequIPCalculator 取 energy/forces；
- `SUPPORTED_HESSIAN_MODES = ('numerical',)`：NequIP 计算器不暴露对位置可微的
  autograd forward，解析 Hessian 不做（数值 Hessian 由 CalcABC 默认路径提供）；
- `SUPPORTS_PBC = False`、`SUPPORTS_CHARGE_MULT = False`（NequIP-OAM-XL 为
  闭壳层中性分子训练，与直算路径现状一致）。

**受影响面**：

| 项 | 影响 |
|---|---|
| MAPLE 数值行为 | 无。插件只在显式 `#model=nequip-oam-xl(...)` 时被加载 |
| 既有金标准 B1–B4、B8 | 无影响（不经插件路径） |
| B5（NequIP sp） | 直算路径保留不动；插件路径若启用，MAPLE 输出 Hartree = −13.43729751141953 eV / 27.211386245988 ≈ **−0.4938115754 Ha**（同机容差 1e-6 Ha），需新增独立 pin（不复用 B5 的 eV pin） |
| 新能力 | NequIP 可参与 MAPLE 的 opt/freq（数值 Hessian，慢）/scan/md |
| 性能 | 数值 Hessian 对大分子 freq 很慢（每个自由度两次 forward）；建议 freq 默认仍走 ANI/UMA |

### 4. 风险

- **半私有 API**：`_from_saved_model` 带下划线（D1-1 已记录），nequip 升级时
  插件与直算脚本同时失效，需同步复查；
- **双注册冲突**：若用户同时在 `#model=` 写 `module=` 且 env 也有
  `MAPLE_CALCULATOR_PLUGINS`，`register_calculator` 对同名重复注册会 raise
  （calculator_base.py:24-29）——同一模块重复 import 不会重复注册
  （import 幂等），风险低；
- **单位混淆**：eV→Hartree 换算由 `_finalize_results` 负责，插件若误乘
  EV2HARTREE 会导致能量错 27 倍——验证计划第 2 条专门防此回归；
- **回滚成本**：极低，插件是纯新增文件 + env 注入开关。

### 5. 回滚方案

`git revert` 本工单对应提交即可；MAPLE/ 与 COMPUTE_VENV 全程零改动，
无环境回滚需求。B5 直算路径与金标准不受影响。

### 6. 验证计划

1. 单元：`$COMPUTE_VENV/bin/python -c "import maple_nequip_plugin"`（PYTHONPATH
   指向 plugins/）注册成功，`get_registered_calculator('nequip-oam-xl')` 可取；
2. 数值：B5 金标准同款 H2O 几何，经 MAPLE CLI（插件路径）跑 sp，
   断言 `Energy: -0.4938115754 Hartree` ±1e-6（防单位换算回归）；
3. 集成：benchmark 新增 B11（plugin 路径 sp）或改造 B5 参数化两条路径，
   `python benchmarks/runner.py --mode=canned` 全绿；
4. 回归：`tests/` 全量 + B1–B10 canned 不受影响（插件未加载时行为不变）。

### 7. 审批结论

**批准** — 审批 agent，2026-10-02

- 定性：框架内新增功能（B 类），定性准确。MAPLE/ 零改动、COMPUTE_VENV 零装包；插件契约逐条核实属实（register_calculator calculator_base.py:16-47、三层发现 set_calculator.py:175-198、MODEL_ENERGY_UNIT 统一换算 calculator_base.py:126-140/322-353）。
- 影响面：B1–B4/B8 金标准无影响；B5 直算路径与 eV pin 保持原样；插件路径新建独立 Hartree pin（−0.4938115754 Ha 换算已验算）。
- 风险：半私有 API 与 D1-1 同款不新增风险类别；单位错乘 27 倍由前置条件 3 对症；回滚 = git revert，完全可逆。
- 措辞修正：env 设置后插件模块在进程首个 calculator 构建时即被 import（注册），实例化才按名字匹配；实施与文档按机制实情描述。

**批准附带前置条件**：
1. env 注入（MAPLE_CALCULATOR_PLUGINS + PYTHONPATH）严格以 JobSpec 选择插件名为条件，默认 maple 子进程 env 一字节不变；HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 在插件路径同样生效。
2. B5 直算路径与其 eV pin 保持原样，插件路径新建独立 Hartree pin（不复用、不改造 B5 断言）。
3. 数值验证先于 benchmark 接入：同 H2O 几何插件路径与直算路径能量差 ≤1e-6 Ha（防 EV2HARTREE 双重换算）。
4. decisions.md 新增条目：插件路径与 D1-1 共享 nequip 升级时 _from_saved_model 复查义务。
5. MODEL_CATALOG 的 backend: maple 选项说明标注：freq 走数值 Hessian、大分子慢，freq 默认仍引导至 ANI/UMA。
6. 完成后跑 pytest 全量 + canned benchmark（B1–B10 全绿 + 新增 B11 通过），结果回填本节。

**实施结果回填**：（实施完成，2026-10-02）

- **改动清单**（全部在 mlip_agent/ 内，MAPLE/ 与 COMPUTE_VENV 零改动）：
  - 新增 `plugins/maple_nequip_plugin.py`：`@register_calculator` 的
    `NequIPPluginCalculator(CalcABC)`，`MODEL_NAMES=('nequip-oam-xl',)`、
    `MODEL_ENERGY_UNIT='eV'`（不自行乘 EV2HARTREE）、
    `SUPPORTED_HESSIAN_MODES=('numerical',)`、`SUPPORTS_PBC=False`、
    `SUPPORTS_CHARGE_MULT=False`、`CHECKPOINT_FILENAME=None`、
    `REQUIRES_LOCAL_MODEL_FILE=True`、`MODEL_PATH_OPTION='model_path'`；
    加载走 `NequIPCalculator._from_saved_model`（import 路径
    `nequip.integrations.ase`，与 D1-1 同款）。
  - `app/compute/jobspec.py`：`MapleModel.NEQUIP_OAM_XL` +
    `ModelSpec.model_path` + `DEFAULT_NEQUIP_MODEL_PATH`；
    `app/compute/inp_builder.py`：渲染 `#model=nequip-oam-xl(model_path=...)`。
  - `app/compute/runner.py`：`run_compute` 新增 `extra_env`（缺省 None 时 env
    一字节不变）；`app/tools/maple_tools.py`：`_maple_extra_env` 严格以
    job_spec.json（JobSpec）为注入条件（手写 inp 回退解析 `#model=` 行），
    `write_maple_inp_impl` 缺省 model_path 走 Settings（NEQUIP_MODEL_PATH 可覆盖）；
    MODEL_CATALOG nequip 条目 notes 标注 backend=maple 插件选项与
    「freq 走数值 Hessian、大分子慢，freq 默认仍用 ANI/UMA」（前置条件 5）。
  - `app/agent/instructions.py`：task_parser/orchestrator prompt 增补
    nequip-oam-xl（插件路径仅用户明确要求或需 MAPLE 编排时使用，默认仍直算）。
  - `benchmarks/`：tasks.yaml 新增 B11（prompt「用 MAPLE 的 NequIP 后端算水分子
    单点能」，断言插件路径能量 ±1e-6 Ha + backend=maple + tool_not_called
    run_nequip）；canned_scripts.py 新增 B11 剧本与 parser 分支；
    references/ 新增 B11.json + B11_sp.inp + B11_sp.out（独立 Hartree pin，
    B5 eV pin 原样未动）；scripts/make_golden_refs.sh 补 B11 段。
  - `docs/decisions.md` 新增 D7-1：插件路径与 D1-1 共享 nequip 升级时
    `_from_saved_model` 复查义务（前置条件 4）。
  - `tests/test_nequip_plugin.py`（注册契约/env 注入条件性/默认 env 不变/
    真实子进程数值验证）+ test_inp_builder.py 渲染用例；
    test_benchmarks.py 覆盖断言更新为 B1–B11；test_runner.py fake_run 签名适配。
- **数值验证**（前置条件 3，先于 benchmark 接入完成）：同 B5 H2O 几何，插件路径
  （MAPLE CLI + env 注入）实测 `Energy: -0.4938115754 Hartree`；B5 直算 pin
  −13.43729751141953 eV / 27.211386245988 = −0.4938115754172834 Ha；
  **差 1.7e-11 Ha ≤ 1e-6 Ha**（无双重换算）。
- **B11 pin**：−0.4938115754 Ha（.out 打印精度 10 位小数），容差 1e-6 Ha。
- **测试结果**：
  - `.venv/bin/python -m pytest tests/ -q` → **145 passed**（含
    test_nequip_plugin.py 9 项：契约 2 + env 条件性 6 + 真实 GPU 数值验证 1）；
  - `.venv/bin/python benchmarks/runner.py --mode=canned` → **11/11 PASS**，
    执行成功率 100.0%，数值正确率 100.0%（B1–B10 回归全绿 + B11 通过）。
- **commit**：见 git log（mlip-agent 提交，仅 mlip_agent/ 内文件）。
