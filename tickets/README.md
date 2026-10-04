# tickets/ — 审批工单

## 何时必须开工单

以下改动**先开工单、获批后实施**（PLAN §0.6、§6）：

1. 修改 `MAPLE/` 源码（MAPLE/MLIP 库源码冻结）。
2. 向 COMPUTE_VENV（`/root/soft/ml_potential/.venv`）安装/升级/卸载任何包。
3. 修改或替换本地模型 checkpoint 文件。
4. 升级 `a13n-harness` / `a13n-stream-protocol` 钉死版本（先评审 changelog，PLAN §5.1）。
5. 任何会改变既有金标准数值的改动（需同时给出重新 pin 计划）。

不需要工单：仅在 `mlip_agent/` 内新增框架代码、不改变既有数值行为。

## 流程

1. 复制 `TEMPLATE.md` → `TICKET-<序号>-<短标题>.md`，序号递增。
2. 填写 1–6 节，状态置「待审批」。
3. 人工审批填写第 7 节结论。
4. 批准后实施，回填验证结果；若影响金标准，同步重跑 `scripts/make_golden_refs.sh` 并更新 `docs/decisions.md`。

## 现有工单

| 工单 | 主题 | 状态 |
|---|---|---|
| TICKET-0001 | MAPLE freq(mw) 投影顺序疑似 bug | 待审批（仅记录，M0 不改动 MAPLE） |
| TICKET-0002 | NequIP-OAM-XL 经 MAPLE_CALCULATOR_PLUGINS 注册为 MAPLE 后端 | 待审批（仅提案与影响面分析，获批后实施） |
| TICKET-0003 | maceoff23s.pt 自制 wrapper 置入 MAPLE 模型目录（先执行后补票） | 已追认（附 5 条件，已全部落实） |
| TICKET-0007 | aimnet2nse.pt 转换并置入 MAPLE 模型目录 | 已批准并实施回填（2026-10-03，B22 pin −150.4159493214 Ha） |
| TICKET-0008 | MAPLE MACE calculator 周期性（PBC）支持 | 已批准并实施回填（2026-10-03，B27 pin） |
| TICKET-0009 | MAPLE AIMNet2 calculator 周期性（PBC）支持 + ewald 解禁 | 已批准并实施回填（2026-10-03，B28 pin −76.4342780468 Ha） |
| TICKET-0010 | 全部 checkpoint 集中迁移至 /root/soft/ml_potential/checkpoints/（MAPLE model/ 改符号链接） | 已批准并实施回填（2026-10-03，pytest 231 绿 + canned 28/28） |
