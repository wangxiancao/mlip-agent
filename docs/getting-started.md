# Agent 入门教程（十分钟上手）

本教程面向刚接入 MLIP Agent 项目的 agent，目标是：读懂项目结构、跑通验证命令、知道动手的边界。人类用户的快速上手见根目录 `README.md`。

## 1. 这个项目是什么

用 LLM agent 编排机器学习势（MLIP）分子计算的框架：用户自然语言提需求 → agent 解析任务、补全参数 → 审批后调用本地计算栈 → 解析结果并中文回答。支持 sp/opt/freq/scan/md 五类任务，9 个模型。目标与完成标准的权威定义见 `PROJECT.md`。

## 2. 两个 Python 环境（最重要的背景知识）

| 环境 | 路径 | Python | 角色 |
|---|---|---|---|
| 后端环境 | `mlip_agent/.venv` | 3.13 | FastAPI + a13n-harness，**绝不 import 计算栈** |
| COMPUTE_VENV | `/root/soft/ml_potential/.venv` | 3.12 | MAPLE/torch/nequip/tace，只经子进程调用 |
| DPA4_VENV | `/root/soft/ml_potential/dpa4_venv` | 3.12 | DPA-4 专用（deepmd-kit 源码重编译），独立子进程 |

后端与计算栈的交互入口只有子进程：`$COMPUTE_VENV/bin/maple`、`$COMPUTE_VENV/bin/python`、`$DPA4_VENV/bin/python`。所有计算子进程走 `app/compute/runner.py`（GPU 全局锁、超时杀进程组、env 白名单、`HF_HUB_OFFLINE=1`）。

## 3. 验证命令（改动后必跑）

```bash
# 单测（167 项，无 GPU 自动 skip）
.venv/bin/python -m pytest tests/ -q

# benchmark：canned 模式（FunctionModel 剧本回放，无需 LLM key，默认验收口径）
.venv/bin/python benchmarks/runner.py --mode=canned

# e2e 冒烟（真实 uvicorn + httpx 全链路）
bash scripts/e2e_smoke.sh

# 前端
cd web && npm run build && npm test
```

live 模式（`--mode=live`、`benchmarks/ui_runner.py`）需要 `MLIP_AGENT_LLM_*` 环境变量；无 key 时会正确 skip，不算失败。

## 4. 动手前的规则

1. **工单先行**：修改 MAPLE 源码、动 COMPUTE_VENV 的包、改模型 checkpoint、升级钉死的 a13n 包、任何改变金标准数值的改动 —— 先复制 `tickets/TEMPLATE.md` 开工单获批。仅在 mlip_agent/ 内新增框架代码且不改数值行为不需要工单。
2. **金标准不可随意改**：`benchmarks/references/` 的 pin 值是同机同配置逐位复现的，改动数值口径必须同步更新 `resource/decisions.md` 记录原因。
3. **SDK 隔离**：a13n-harness 的 import 只允许出现在 `app/agent/harness.py` 与 `app/agent/streaming.py`。
4. **禁止联网下载模型**：模型一律本地绝对路径；工具层 lint 掉 URL 形态的模型参数。
5. 决策记录追加到 `resource/decisions.md`（ADR 格式：背景、决定、理由、后果），新决定同步更新根目录 `DECISIONS.md` 的索引。

## 5. 目录一分钟地图

```
app/agent/     harness 组装、instructions、子 agent、权限规则、SSE 流适配
app/tools/     agent 工具（闭包注入 session workspace + job 服务）
app/compute/   子进程封装、JobSpec、inp 生成、.out 解析、直算脚本（scripts/）
app/api/       FastAPI 路由（sessions/chat/jobs/models）
app/state/     会话持久化（纯 JSON 文件）
web/           React 前端（ChatPanel/ResultPanel/StructureView）
benchmarks/    B1–B19 验收任务集（v2 tier：guided/recovery/unguided）、runner、judge、canned 剧本
  └── reaction/  v3 反应能垒套件（GMTKN55 BH76 × 三档知识梯度，live 评测，PLAN §3.3）
  ├── references/  金标准 pin 值（入库）
  ├── runs/        运行记录（不入库）
  └── archive/     已验证的历史运行记录（AGENTS.md 说的 archive 就是这个）
plugins/       MAPLE calculator 插件（nequip 注册进 MAPLE 编排）
resource/      资料：ADR 全量（decisions.md）、专题报告
scripts/       bootstrap / 金标准生成 / venv 重建 / e2e
tickets/       审批工单
```

## 6. 常见任务的第一站

| 要做什么 | 先读 |
|---|---|
| 改 agent 行为/指令 | `app/agent/instructions.py` + PLAN §2.1 + D7-2/D7-3（live 修复史） |
| 加新模型后端 | `app/compute/scripts/tace_ase.py`（直算模板）+ D7-6/D7-8（接入流程）+ 工单 |
| 改 benchmark | `benchmarks/tasks.yaml` + `judge.py` + D5-2（汇总口径） |
| 排查计算失败 | 作业目录 `data/sessions/{sid}/jobs/{jid}/` 的 stdout/stderr/.out |
| 升级依赖 | 根目录 `DECISIONS.md` 的复查义务清单 |
