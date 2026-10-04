## TICKET-0009：MAPLE AIMNet2 calculator 周期性（PBC）支持 + ewald 解禁

- 状态：已批准并实施回填（2026-10-03，B28 pin −76.4342780468 Ha） ｜ 提出：2026-10-03 ｜ 审批人：用户

### 1. 改动对象

- MAPLE 源码：`MAPLE/maple/function/calculator/aimnet/_aimnet2_calculator.py`
  - 邻居表（当前 `nblist_dense_padded` 暴力无 PBC，:26-47）：`atoms.pbc.any()` 时改用周期邻居表，产出 `nbmat+shifts`、`nbmat_lr+shifts_lr`，data 增加 `cell`。
  - `_set_lrcoulomb_method`（:139-143）：放开 `ewald` 门禁（当前显式 `NotImplementedError`，注释自述"requires validated cell/PBC/MIC support"——本工单即完成该验证）；PBC 时默认切 `dsf`（对齐上游 aimnetcentral 检测到 cell 自动 simple→dsf 的行为）。
  - `AIMNet2Calculator.SUPPORTS_PBC`：`False` → `True`。
- 不改 checkpoint：`aimnet2.pt`（aimnet2_wb97m_d3_0 导出）内 lrcoulomb 三方法（simple/dsf/ewald）与 `calc_distances` 的 shifts×cell MIC 分支已全部脚本化，2026-10-03 调研实测：给 data 加 cell 键即可跑（H2O 20 Å 盒，simple −2081.0462 eV / ewald −2081.0469 eV / dsf −2079.8803 eV）。
- mlip_agent 侧配套（非冻结资产，仅列示）：`MODEL_CATALOG` 能力标注更新；新增周期性 benchmark。

### 2. 动机与问题描述

同 TICKET-0008（mlip_agent PBC 落地决策）。AIMNet2 上游框架完整支持 PBC（DSF+Ewald 长程库仑 + 周期邻居表 + stress，`aimnet/modules/lr.py:54-182`、`aimnet/ops.py:139-221`，官方教程 `docs/tutorials/periodic_systems.md`），在用 checkpoint 的周期机制已脚本化进 JIT 图，唯一缺口是 MAPLE wrapper 从不提供 cell/shifts 且禁用 ewald。非周期路径保持逐字节不变。

### 3. 影响面分析

**A. 影响库逻辑**（受限）：
- 受影响面一：周期体系输入路径（当前必然被 PBC 拦截报错，无既有金标准依赖）。
- 受影响面二：`ewald` 从报错变为可用——仅当用户显式传 `coulomb_method=ewald` 时行为变化；缺省 `simple` 不变。既有 benchmark（均走缺省）不受影响，实施后重跑 aimnet2 相关金标准复核。
- 新增金标准：周期性 benchmark 新 pin 值，energy-only pin。

### 4. 风险

- Ewald 精度参数硬编码（该 checkpoint 导出时 accuracy=1e-8 写死），长程收敛行为以对照实验为准。
- AIMNet2 训练域为气相分子，官方明示"未对体相/表面验证"；周期计算仅分子晶体类体系可信 → 能力边界写入 MODEL_CATALOG 与 agent 指令。
- 周期邻居表的 MIC 约定必须与 checkpoint 内嵌 `calc_distances` 的 shifts×cell 分支逐字对齐 → 以上游 aimnetcentral ASE calculator 对照为验收。

### 5. 回滚方案

`git revert` MAPLE 侧改动提交；mlip_agent 侧金标准/benchmark 一并回退。

### 6. 验证计划

- 对照验证：H2O 分子 20 Å 周期盒，MAPLE（改造后）vs aimnetcentral 上游 `AIMNet2ASECalculator`，simple/dsf/ewald 三法能量对照（容差 1e-6 eV）；分子晶体算例（如苯晶体小胞）sp 合理性检查。
- 回归：aimnet2 既有金标准重跑逐位复现；`pytest tests/` 全绿。
- 新周期性 benchmark 三次复现逐位一致后 pin。

### 7. 审批结论

**批准**（用户，2026-10-03）：同意修改 `_aimnet2_calculator.py` 支持 PBC 并放开 ewald（非周期路径逐字节不变为前提）。实施顺序注意：待 TICKET-0007 的 aimnet2nse.pt 置入与验证完成后再动本文件（避免并行改动同文件）。实施完成后回填验证结果（三法对照、B13 回归、B28 pin）。

#### 实施回填（2026-10-03）

- 改动（仅 `_aimnet2_calculator.py`，MAPLE 未提交 commit）：新增 `nblist_periodic_padded`（matscipy `neighbour_list('ijS')` MIC 表 → padded `(N+1, M)` int32 nbmat + `(N+1, M, 3)` float32 单位晶格 shifts，约定同 checkpoint 脚本化 `calc_distances`：`r_ij = coord[j] − coord[i] + shifts @ cell`）；`_build_data` 以 `atoms_has_pbc(atoms)` 为唯一分支条件，周期时追加 `shifts`/`shifts_lr`/`cell` 三键，非周期路径逐字节不变；`SUPPORTS_PBC=True`；`SUPPORTED_COULOMB_METHODS += 'ewald'`；`_set_lrcoulomb_method` 放开 ewald 并修复模块查找（`name.endswith('.lrcoulomb')`——checkpoint 实际挂载点为 `outputs.lrcoulomb`，旧精确匹配是静默 no-op）；ewald 仅限周期体系（非周期 calculate() 抛 NotImplementedError）；`coulomb_method` 缺省 None：非周期 = simple（逐字节不变），周期首个 calculate() 自动切 dsf（对齐上游 `prepare_input` 行为），显式 simple 仍可强制。
- 对照验证（CPU，H2O 20 Å 立方盒；MAPLE matscipy 路径 vs 上游 nvalchemiops 路径跑**同一 checkpoint**、方法直接设在 JIT 模块上）：simple −2081.0461085663 / dsf −2079.8826618866 / ewald −2081.0467415527 eV，**三条全部逐位一致（Δ = 0.0 eV）**；上游官方 ASE 全流程（`AIMNet2ASE`，检测到 cell 自动 simple→dsf，但该 legacy checkpoint 内嵌 Coulomb 不可运行时切换，实际跑 simple）−2081.0461092965 eV，与 MAPLE simple 差 7.3e-7 eV < 容差 1e-6。对照脚本 `/tmp/nvalchemi_xval.py`、`/tmp/upstream_xval.py`。
- 回归（COMPUTE_VENV 实跑 `maple`，HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0）：B13 = **−76.4770339061 Ha**、B22 = **−150.4159493214 Ha**，物理内容（剔时间戳/计时行）逐字节一致。
- 新 pin：B28（H2O 20 Å 立方盒，`#pbc(20,20,20)` 缺省即 dsf）= **−76.4342780468 Ha**，GPU 连续 3 次逐位一致；金标准暂存 `/tmp/golden_staging/B28/`。ewald 对照 −76.4770571854 Ha（记录不 pin）；显式 simple 强制 −76.4770339061 Ha（与气相 B13 逐位一致，20 Å 盒无跨边界对，符合预期）。
- 冒烟：4×H2O 10 Å 周期胞 sp = −305.7398672163 Ha（−76.4350/水，量级合理）；aimnet2nse PBC dsf 冒烟 −76.4671299771 Ha 正常；PBC forces 路径（autograd）CPU 冒烟正常。
- 能力边界：PBC 仅承诺 sp/opt（freq/hessian 无 Γ 点投影未验证）；AIMNet2 训练域为气相分子，官方明示未对体相/表面验证——周期计算仅分子晶体类体系可信；混合维度 PBC（部分轴周期）显式拒绝；ewald 精度参数 1e-8 为 checkpoint 导出时写死。
- 副作用修复说明：旧代码 `coulomb_method='dsf'` 因模块查找精确匹配失效为静默 no-op（模型仍跑 simple 但 nbmat_lr 按 15 Å 构建）；本次修复后显式 dsf 真正生效——属 bug 修复，非周期缺省（simple）路径不受影响（B13/B22 逐位回归为证）。
