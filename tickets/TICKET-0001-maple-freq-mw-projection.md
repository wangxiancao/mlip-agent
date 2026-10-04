## TICKET-0001：MAPLE freq(mw) 平动/转动投影与质量加权顺序疑似 bug

- 状态：待审批 ｜ 提出：2026-10-02（M0 金标准生成中发现）｜ 审批人：____

### 1. 改动对象

`MAPLE/maple/function/dispatcher/frequency/frequency.py`，`MassWeightedFrequency.compute_frequencies()`（约 1009–1123 行）。属冻结资产；**M0 未做任何改动，本工单仅为记录与审批依据**。

### 2. 动机与问题描述

现行实现先对 Cartesian Hessian 做平动/转动投影（`P H P`），再做质量加权（`M^{-1/2} (P H P) M^{-1/2}`）。标准做法是先质量加权再在质量加权坐标下投影（`P (M^{-1/2} H M^{-1/2}) P`）。两者仅在等质量体系等价，对 H2O 这类混合质量体系产生系统性偏差。

数值证据（ANI-2x，收敛 H2O，max|F| = 3.3e-7，复现脚本见下）：

| 模式 | MAPLE mw 输出 | 标准顺序 | 气相实验参考 |
|---|---|---|---|
| 弯曲 | 1967.09 cm⁻¹ | 1721.44 cm⁻¹ | ~1595 |
| 对称伸缩 | 4200.32 cm⁻¹ | 3803.08 cm⁻¹ | ~3657 |
| 反对称伸缩 | 4605.88 cm⁻¹ | 3921.21 cm⁻¹ | ~3757 |

复现要点：`ANICalculator.get_hessian()` 取 Ha/Å² Hessian → 按 `frequency.py` 的 `_build_translation_rotation_basis` + QR 构造 P → 两种顺序分别对角化，转换因子 2721.1383 cm⁻¹（该因子本身正确）。MAPLE 顺序精确复现 CLI 输出（1967.09/4200.32/4605.88）。

### 3. 影响面分析

**A. 影响库逻辑**。若修复，所有 `freq`（method=mw，默认）任务的频率与热化学量（ZPE、熵、G）都会变。受影响金标准：B3（频率）、B6（freq 链式）、以及未来任何含 freq 的 benchmark。B3.json 需重新 pin。

### 4. 风险

- 修复改变 MAPLE 公开数值行为，依赖现有输出的下游结果全部失效。
- 不修复的风险：频率系统性偏高（H2O 弯曲模 +23%），thermochemistry 连带偏差。
- 注意 MAPLE examples/freq/mw 的参考输出也是在现行（疑似错误）逻辑下生成的，修复后 example 输出不复现。

### 5. 回滚方案

git revert 修复提交；重跑 `scripts/make_golden_refs.sh` 恢复 pin 值。

### 6. 验证计划

修复后：H2O 三频率应约为 1721/3803/3921 cm⁻¹（ANI-2x）；`MAPLE/examples/freq/mw/inp1.inp` 重跑，虚频计数不变（1 个）；重 pin B3。

### 7. 审批结论

**暂缓（保持现状、转上游报告）** — 审批 agent，2026-10-02

- 定性：影响库项目逻辑（A 类）。疑似 bug 认定成立（frequency.py:1028-1038 先 Cartesian 投影再质量加权，与标准顺序不等价；B3 pin 数值证据逐位一致）。
- 理由：修复责任在上游（MAPLE 为第三方项目）；本项目 benchmark 语义（D0-5）刻意 pin MAPLE 实际行为，本地修复将使 B3 金标准失效、动摇验收体系，并造成与上游的永久分叉。收益不抵成本。
- 行动：工单状态转为「已确认-不修复-转上游」，作为已知问题永久留档；向 MAPLE 上游提交 issue（附本工单复现脚本与数值对照）；用户文档标注 freq(mw) 频率系统性偏高（H2O 弯曲模约 +23%）。
- B3 保持现有 pin 值与 ±20 cm⁻¹ 容差不变。若未来吸收上游修复：先重跑 make_golden_refs.sh 重 pin B3、实测 B6 计数断言、更新 D0-5 与 PLAN §3。
