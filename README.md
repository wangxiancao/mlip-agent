# MLIP Agent 计算框架

用 LLM agent 编排机器学习势（MLIP）分子计算：自然语言提需求 → agent 解析任务、
生成 MAPLE 输入、（经用户审批后）在 GPU 上执行、解析结果并用中文回答
（能量同时给 Hartree 和 eV）。支持 sp / opt / freq / scan / md 五类任务，
模型覆盖 ANI / AIMNet2 / MACE-OFF / UMA（MAPLE 后端）与 NequIP-OAM-XL（直算后端）。

## 架构（文字版）

```
用户 ──► web/ (React+Vite, ChatPanel/ResultPanel/StructureView)
           │  REST + SSE（AG-UI 事件流）
           ▼
        app/main.py (FastAPI, uvicorn, 单进程)
           │  POST /api/sessions/{sid}/messages|resume → SSE
           │  GET  /api/sessions/{sid}/state · /api/jobs/* · /api/models
           ▼
        app/agent/ (a13n-harness，SDK 隔离在 harness.py/streaming.py)
           │  mlip_orchestrator + 子 agent：
           │    task_parser（NL→JobSpec JSON）
           │    result_validator（收敛/虚频/漂移校验）
           │    approval_advisor（高危操作风险说明）
           ▼
        app/tools/ (build_structure/write_maple_inp/run_maple/run_nequip/...)
           │  run_maple / run_nequip / ask_user_question 需审批（挂起-恢复）
           ▼
        app/compute/runner.py（子进程封装：GPU 全局锁 Semaphore(1)、
           超时杀进程组、env 白名单 HF_HUB_OFFLINE=1）
           │  双环境隔离：后端 .venv (Py3.13) 绝不 import 计算栈
           ▼
        COMPUTE_VENV (Py3.12)
           ├── bin/maple  → MAPLE CLI → ANI/AIMNet2/MACE-OFF/UMA
           └── bin/python → app/compute/scripts/nequip_ase.py → NequIP-OAM-XL
```

持久化：`data/sessions/{sid}/state.json`（HarnessState + 挂起现场）+
`jobs/{job_id}/`（input.inp / input.out / structure.xyz / 日志），纯 JSON 文件，
服务重启后 lazy 恢复。

## 环境搭建

```bash
cd mlip_agent
bash scripts/bootstrap.sh   # uv 建 .venv（Py3.13）+ uv sync --locked + COMPUTE_VENV/前端工具链校验
```

COMPUTE_VENV（默认 `/root/soft/ml_potential/.venv`）是既有计算环境，bootstrap
只做存在性校验，绝不修改。金标准（benchmarks/references/）已由
`scripts/make_golden_refs.sh` 生成并入库，一般无需重跑。

注意：全部模型 checkpoint 集中存放于 `/root/soft/ml_potential/checkpoints/`，
`MAPLE/maple/function/calculator/model` 是指向其中 `maple/` 子目录的符号链接。
其中 `checkpoints/maple/maceoff23s.pt` 是本地自制 scripted wrapper（权重许可为
ASL，仅限学术研究用途），**不在** MAPLE 官方发行/全新 checkout 中；换机或重新
clone MAPLE 后需用 `scripts/make_maceoff23s_wrapper.py` 重新生成（脚本内含 pin
住的权重下载地址与 SHA256 校验），否则 B14 与 maceoff23s 相关功能不可用。

## 启动

```bash
# 后端（生产模式会托管 web/dist，若已 build）
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# 前端开发（vite proxy /api → 8000）
cd web && npm install && npm run dev

# 前端生产构建（产物由后端 StaticFiles 托管）
cd web && npm run build
```

## LLM 配置

三个环境变量（OpenAI 兼容端点，见 `.env.example`）：

| 变量 | 说明 |
|---|---|
| `MLIP_AGENT_LLM_MODEL` | 模型名 |
| `MLIP_AGENT_LLM_BASE_URL` | 端点 URL |
| `MLIP_AGENT_LLM_API_KEY` | API key |

无 key 时服务可正常启动，canned benchmark / FunctionModel 单测 / e2e 冒烟全部
离线可跑；live 对话与 live benchmark 需要 key。

## 测试与验收

```bash
# 后端单测（GPU 测试在无 GPU 时自动 skip）
.venv/bin/python -m pytest tests/ -q

# benchmark：canned（默认，FunctionModel 剧本回放，无 key 可跑）
.venv/bin/python benchmarks/runner.py --mode=canned
# benchmark：live（需 MLIP_AGENT_LLM_*；无 key 时整体 SKIPPED，退出码 0）
.venv/bin/python benchmarks/runner.py --mode=live

# benchmark：UI 驱动（Playwright + 系统 Chrome 走真实前端页面；自动起后端，需 LLM key）
.venv/bin/python benchmarks/ui_runner.py                  # 全部任务
.venv/bin/python benchmarks/ui_runner.py --tasks B1,B10   # 子集
# 复用已有后端：--base-url http://127.0.0.1:8000 --data-dir data；调试用 --headed
# 依赖：web/ 的 devDependency playwright（npm install 即可，浏览器用系统 Chrome）

# e2e 冒烟：真实 uvicorn + httpx 全 API 链路（FunctionModel 剧本，真实 GPU 计算）
bash scripts/e2e_smoke.sh    # 退出码 0 = 通过

# 前端
cd web && npm run build && npm test
```

## 目录速查

- `app/` — 后端（agent / tools / compute / api / state）
- `web/` — 前端（React 18 + Vite + TS；3Dmol.js 按需加载渲染 3D 结构）
- `benchmarks/` — B1–B19 验收任务集（guided/recovery/unguided 三档 tier）+ judge + canned 剧本；`benchmarks/reaction/` — 反应能垒套件（GMTKN55 BH76 × 三档知识梯度，live 评测 agent 能力）
- `scripts/` — bootstrap.sh / make_golden_refs.sh / make_dpa4_venv.sh / e2e_smoke.sh(.py)
- `plugins/` — MAPLE 插件后端（NequIP / MACE）
