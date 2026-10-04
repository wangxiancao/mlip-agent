# TICKET-0016：MAPLE opt 输出改用 ASE 写出（轨迹带 Lattice，最终结构落 cif）

- 状态：待审批 ｜ 提出：2026-10-04 ｜ 审批人：____

### 1. 改动对象

**冻结资产（MAPLE 源码）**，故必须先有本工单：
- `/root/soft/ml_potential/MAPLE/maple/function/dispatcher/optimization/algorithm/_common.py`
  `write_xyz`：xyz 路径从手写文本拼接改为 `ase.io.write(format="extxyz")`；
  新增按 `.cif` 扩展名分派的 ASE cif 写出。
- 同目录 `LBFGS.py` / `RFO.py` / `SDCG.py` 的 `_finalize_run`：最终结构在
  `_opt.xyz` 之外额外写 `<base>_opt.cif`。

### 2. 动机与问题描述

web UI 实战（会话 a0b58838，job-0001/0002）：MACE/UMA 的 **PBC** opt 计算
本身周期完整（`input_reader.py:569-573` 给 atoms 设了 cell+pbc，能量/力
全程周期边界），但优化器输出 `write_xyz` 是手写 XYZ 拼接
（`_common.py:12-47`），只写原子数+注释+坐标，从不读 `atoms.cell/pbc`
——**丢 Lattice 发生在写文件环节，不是计算环节**。导致优化结果转 cif
丢失周期信息（mlip_agent 侧已用 TICKET-0013 的 cell 参数兜底）。

用户明确要求：结构读写一律走 ASE；opt 过程持续写 xyz（现状已逐迭代
append+fsync，保持）；最终结构保存为 cif。

### 3. 影响面分析

A. 影响库逻辑（但不改数值）：
- 优化器产生的坐标/能量/收敛行为**完全不变**，仅序列化格式变化：
  `_opt_traj.xyz` / `_opt.xyz` 由纯 xyz 变为 extxyz（多 Lattice/pbc 头，
  注释行保持 `Image N  Energy = <E>` 兼容既有文本消费者）；
  新增 `_opt.cif` 产物。
- 受影响任务类型：opt（LBFGS/RFO/SDCG 三算法共用 `_common.write_xyz`）。
- 金标准：pin 值全部来自 `.out` 解析（能量/频率/收敛行），不解析 xyz
  产物内容 → 无需重 pin；`parsers.py` 只读 `.out`。`.out` 日志新增一行
  cif 路径（finalize 的 log_final_paths 段），不影响任何 pin 解析。
- 数值复查义务：跑 canned benchmark 全量复验（B 系列含 opt 任务）。

### 4. 风险

- 外部若有脚本按纯 xyz 解析 `_opt*.xyz`：extxyz 是 xyz 超集，ASE/常见
  解析器均兼容；手写 regex 解析注释行的消费者所见注释文本不变。
- 分子（无胞）opt：extxyz 在无胞时行为由 ASE 决定（pbc="F F F"），
  坐标数值不变。
- 回滚成本：git 级 revert（MAPLE 工作区）。

### 5. 回滚方案

`git -C /root/soft/ml_potential/MAPLE revert`（或 checkout）对应改动；
mlip_agent 侧无需联动改动。

### 6. 验证计划

- 手动：小体系 opt（分子 H2O + 周期 Si extxyz 各一）确认 `_opt_traj.xyz`
  逐迭代追加、周期体系带 Lattice、`_opt.cif` 生成且 cellpar 正确。
- `.venv/bin/python -m pytest tests/ -q` 全绿。
- `benchmarks/runner.py --mode=canned` 28/28 PASS（数值口径复验）。

### 7. 审批结论

批准（用户指令 2026-10-04）。
实施回填（MAPLE 仓库 dev 分支 commit `e7fac2d`）：
- `_common.write_xyz` 改经 ASE 写出：`.cif` 走 ASE cif（取末帧）；xyz 走
  extxyz（columns=symbols/positions，write_info/write_results=False，
  旧 `Image N  Energy = <E>` 文本前置到注释行）。
  注意：ASE `comment` kwarg 会覆盖整条头部（含 Lattice/Properties/pbc），
  首版实现踩中此坑（出现 7 列无 Properties 头的损坏帧），改为 ASE 生成
  帧头后前置文本，已修正。
- LBFGS/RFO/SDCG `_finalize_run` 额外写 `<base>_opt.cif`。
- 验证：分子（H2O/ANI-2x）与周期（Si/macempa0+#pbc）opt 各一，轨迹逐迭代
  追加、周期帧带 Lattice/pbc、`_opt.cif` cellpar 正确、ASE 回读往返无损；
  mlip_agent pytest 243 绿；canned benchmark 28/28、数值正确率 100%。
