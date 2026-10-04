# PROJECT.md — 项目目标、范围与完成标准

## 目标

构建 **MLIP Agent**：一个自然语言驱动的机器学习势（MLIP）计算框架。用户用中文描述计算需求（单点能、几何优化、频率、扫描、分子动力学），由 LLM agent 解析意图、补全缺失参数、经审批后调用本地计算栈执行，并通过 Web 界面流式展示过程与结果。

## 范围

### 计算后端

- **MAPLE CLI 子进程**（主路径）：ANI-1x/2x、AIMNet2、MACE-OFF23(s)、UMA(uma-s-1p1, omol)；sp/opt/freq/scan/md 全任务类型。
- **直算 ASE 子进程**：NequIP-OAM-XL、TACE-OAM-L、DPA-4（各自独立 venv/入口，job.json → result.json 契约）。
- **MAPLE 插件**：NequIP-OAM-XL、MACE-MPA-0 经 `MAPLE_CALCULATOR_PLUGINS` 注册进 MAPLE 编排（plugins/）。

### 系统组成

- 后端：FastAPI + a13n-harness（Python 3.13，与计算环境严格隔离），SSE 流式对话、审批挂起/恢复、会话持久化。
- 前端：React 18 + Vite + TS，对话面板、工具调用卡、审批框、结果面板（能量/频率表、收敛曲线、3Dmol.js 结构）。
- 验收：`benchmarks/` B1–B27 任务集（v2：guided/recovery/unguided 三档 tier，PLAN §3.2；v1.3 扩展含 PBC 与新模型任务，PLAN §3.4），canned（FunctionModel 回放，默认）与 live（真实 LLM）双模式 + judge 数值判定；`benchmarks/reaction/` 反应能垒套件（v3：GMTKN55 BH76 × baseline/decision/replication 三档知识梯度 + 覆盖率/MAE/尝试次数/成本四维指标，PLAN §3.3）用于 live 评测 agent 能力。

### 边界（不做）

- 不修改 MAPLE/MLIP 库源码（确需改动走 `tickets/` 审批工单）。
- 不联网下载模型（子进程强制 `HF_HUB_OFFLINE=1`，全部本地绝对路径）。
- v1 不做多 worker、不做 sqlite 作业索引。
- 周期性体系（2026-10-03 起支持，PLAN §3.4）：直算三路 + UMA/MACE 经 MAPLE；**不做 cell relaxation**（无 stress，opt 只动原子坐标）；slab 用真空层 ≥10 Å + 全周期胞；ANI 全系保持分子域。

## 完成标准

1. `pytest tests/` 全绿（当前 167 项）。
2. `python benchmarks/runner.py --mode=canned` B1–B27 全部 PASS；无 LLM key 时 live 模式正确 skip。
3. 有 LLM key 时 live 端到端演示可跑通（里程碑 M6 验收项）。
4. 数值金标准：各 benchmark 能量 pin 值在同机同配置下逐位/容差内复现（详见 resource/decisions.md）。

## 权威文档

- 实施计划（唯一权威）：`PLAN.md`
- 架构决策记录（ADR 全量）：`resource/decisions.md`
- 环境与启动：`README.md`
