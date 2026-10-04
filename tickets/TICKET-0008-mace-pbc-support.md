## TICKET-0008：MAPLE MACE calculator 周期性（PBC）支持

- 状态：已批准（待实施回填） ｜ 提出：2026-10-03 ｜ 审批人：用户

### 1. 改动对象

- MAPLE 源码：`MAPLE/maple/function/calculator/mace/_mace_calculator.py`
  - `build_data_from_atoms`（:20-87）：当前 `radius_graph_no_pbc` + `cell=torch.zeros(3,3)` + `unit_shifts` 全零。改为：`atoms.pbc.any()` 时用周期邻居表（COMPUTE_VENV 已有 matscipy 1.2.0，`matscipy.neighbours.neighbour_list`，与上游 MACE 官方 ASE calculator 同一实现）产出 `edge_index` + `unit_shifts`，`shifts = unit_shifts @ cell`，传真实 `cell`（(1,3,3)）。
  - `MACECalculator.SUPPORTS_PBC`：`False` → `True`。
- 不改 checkpoint：`maceoff23s.pt` 等 scripted wrapper 内已含完整 MACE 周期前向（unit_shifts/cell/shifts/virials 路径），当前只是被喂全零。
- mlip_agent 侧配套（非冻结资产，仅列示）：`MODEL_CATALOG` 能力标注更新；新增周期性 benchmark。

### 2. 动机与问题描述

mlip_agent 落地周期性体系计算（2026-10-03 用户决策：除 ANI 外各框架均打通 PBC，暂不做 cell relaxation）。MACE 框架原生支持周期体系（MACE-MP 系列即晶体模型），在用 checkpoint 的周期机制全部在场，唯一缺口是 MAPLE wrapper 的邻居表/cell 传递。非周期路径保持逐字节不变（pbc 全 False 时走原代码路径）。

### 3. 影响面分析

**A. 影响库逻辑**（受限）：
- 受影响面：仅"周期体系输入"这一当前被 `_reject_unsupported_pbc` 拦截报错的路径——此前该路径**必然失败**（`set_calculator.py:223-227` 工厂预检 + `calculator_base.py:284-286` 双重拦截），不存在任何既有金标准依赖。
- 非周期路径行为不变：改动以 `atoms.pbc.any()` 为唯一分支条件，pbc 全 False 时与现状逐字节一致；B14（maceoff23s H2O sp，−76.4797837970 Ha）等既有金标准不受影响，实施后重跑 B14 复核。
- 新增金标准：周期性 benchmark（Si bulk sp）新 pin 值，energy-only pin（力不逐位 pin，沿用 D7-8 口径）。

### 4. 风险

- 邻居表实现差异（matscipy vs 上游 MACE 的 ASE 默认）可能引入 cutoff 边界邻居计数差异 → 以官方 mace ASE calculator（scratch venv）在 Si bulk 算例上能量对照（容差 1e-6 eV）为验收。
- MACE-OFF23 训练域为气相分子，周期计算仅对分子晶体/液体类体系物理可信；离子固体/表面属域外 → 能力边界写入 MODEL_CATALOG 与 agent 指令，不做静默放开。
- 数值 Hessian（freq）在 PBC 下无周期性专属处理（无 Γ 点投影），freq+PBC 本轮标注为未验证，不进 benchmark。

### 5. 回滚方案

`git revert` MAPLE 侧改动提交（MAPLE 为 git 仓库，单文件改动可整提交回退）；mlip_agent 侧金标准/benchmark 一并回退。

### 6. 验证计划

- 对照验证：Si 金刚石原胞 sp，MAPLE（改造后）vs 官方 mace ASE calculator（scratch venv，torch 2.8.0 + mace-torch），能量容差 1e-6 eV。
- 回归：B14 重跑逐位复现；`pytest tests/` 全绿。
- 新周期性 benchmark 三次复现逐位一致后 pin。

### 7. 审批结论

**批准**（用户，2026-10-03）：同意修改 `_mace_calculator.py` 支持 PBC（非周期路径逐字节不变为前提）。实施完成后回填验证结果（Si bulk 对照、B14/B20 回归、B27 pin）。

#### 实施回填（2026-10-03）

- 改动：`_mace_calculator.py` 新增 `_periodic_radius_graph`（matscipy `neighbour_list('ijS')`，约定与上游 mace-torch 0.3.10 `get_neighborhood` 逐行镜像）；`build_data_from_atoms` 以 `atoms.pbc.any()` 为唯一分支条件，非周期路径逐字节不变；`SUPPORTS_PBC=True`。插件 `macempa0` 同步翻 True。
- 对照验证：Si 金刚石原胞，MAPLE 改造路径 vs 官方 mace_mp ASE 路径能量差 **0.000e+00 eV**（scratch venv 对照脚本 `/tmp/ticket0008_compare.py`）。
- 回归：B14（-76.4797837970 Ha）/B20（-0.5064442892 Ha）重跑逐位一致。
- 新 pin：B27（Si PBC sp 经插件）= **-0.3978120290 Ha**，3 次逐位一致；与 DPA-4/NequIP/TACE/UMA 同结构互证差 <0.02 eV。
- 能力边界：PBC 仅承诺 sp/opt；freq/hessian（无 Γ 点投影）未验证。
- 验证：mlip_agent pytest 全绿；canned B27 PASS。
