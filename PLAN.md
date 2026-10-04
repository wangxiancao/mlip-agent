# MLIP Agent 计算框架 — 实施计划

版本：v1.3（2026-10-03）。本文档是唯一权威计划；变更需更新本文档并在 `resource/decisions.md` 记录原因。
v1.1 变更：§3 拆分为 3.1（v1 任务集 B1–B16，现行有效）与 3.2（benchmark v2 重构，已落地）。
v1.2 变更：新增 §3.3（benchmark v3：GMTKN55 反应能垒复合任务 × 三档知识梯度，agent 能力评测）。
v1.3 变更：新增 §3.4（2026-10-03 扩展：周期性体系 PBC 落地 + MACE-MPA-0/DPA4C/AIMNet2-NSE 接入 + ASE 格式转换工具）。

## 0. 核心架构决策（已定，不再讨论）

1. **双 Python 环境，严格隔离**
   - Agent/后端环境：`mlip_agent/.venv`，uv 创建，**Python 3.13**（a13n-harness 硬性要求），装 fastapi/uvicorn/a13n-harness/a13n-stream-protocol/pydantic-ai。
   - 计算环境：现有 `/root/soft/ml_potential/.venv`（Python 3.12.12，下称 **COMPUTE_VENV**），**只通过子进程调用**，后端绝不 import 其中的包。入口只有两个：`$COMPUTE_VENV/bin/maple` 和 `$COMPUTE_VENV/bin/python`。
2. **单 pyproject，不用 uv workspace**。后端是一个可安装包 `mlip_agent`（`app/` 为源码目录），前端独立 npm 项目。
3. **计算路径取舍**：
   - ANI / AIMNet2 / MACE-OFF / UMA → 走 **MAPLE CLI 子进程**（UMA 用 `#model=uma(size=uma-s-1p1,task=omol,checkpoint_path=/root/soft/ml_potential/checkpoints/uma/uma-s-1p1.pt)`，显式本地路径，禁止联网下载；全部 checkpoint 集中存放于 `/root/soft/ml_potential/checkpoints/`，TICKET-0010）。
   - NequIP（NequIP-OAM-XL）→ **直接 ASE 计算器路径**：后端生成 job JSON，`$COMPUTE_VENV/bin/python app/compute/scripts/nequip_ase.py job.json` 子进程执行，输出 result.json + traj.extxyz。MAPLE 插件机制（`MAPLE_CALCULATOR_PLUGINS` + `@register_calculator`）留作 M6 可选项。
4. **LLM 端点全部走环境变量**（OpenAI 兼容）：`MLIP_AGENT_LLM_MODEL` / `MLIP_AGENT_LLM_BASE_URL` / `MLIP_AGENT_LLM_API_KEY`。当前环境无 key，所有自动化测试用 pydantic-ai `FunctionModel` 离线跑。
5. **GPU 串行化**：后端内全局 `asyncio.Semaphore(1)`（GPU job 锁），同一时刻只跑一个 GPU 计算。
6. **MAPLE/MLIP 库源码冻结**。确需改动走 `tickets/` 审批工单。

## 1. 目录结构

```
mlip_agent/
├── PLAN.md                      # 本文档
├── README.md                    # 快速上手：环境搭建、启动命令
├── pyproject.toml               # 后端包，requires-python = ">=3.13"
├── uv.lock                      # 提交锁定
├── .env.example                 # LLM 端点、路径配置模板（.env 不入库）
├── .gitignore                   # .venv/ data/ node_modules/ dist/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 装配入口（uvicorn app.main:app）
│   ├── config.py                # pydantic-settings：全部路径/端点配置
│   ├── agent/
│   │   ├── harness.py           # build_harness()：HarnessBuilder 组装，SDK 隔离层
│   │   ├── instructions.py      # 主 agent + 子 agent system prompt 常量
│   │   ├── subagents.py         # SubagentDefinition 声明
│   │   ├── permissions.py       # ToolPermissions 规则、审批恢复逻辑
│   │   └── streaming.py         # HarnessAguiStreamObserver → SSE 适配
│   ├── tools/
│   │   ├── registry.py          # 工具闭包工厂：注入 session workspace + job queue
│   │   ├── maple_tools.py       # MAPLE 相关工具（见 §2.2）
│   │   ├── structure_tools.py   # 结构构建/读写工具
│   │   └── mlip_tools.py        # NequIP 直算工具
│   ├── compute/
│   │   ├── runner.py            # 子进程封装（超时、日志、退出码、GPU 锁）
│   │   ├── jobspec.py           # JobSpec/JobResult pydantic schema
│   │   ├── parsers.py           # .out 解析：能量/频率/收敛/轨迹
│   │   ├── inp_builder.py       # JobSpec → MAPLE .inp 文本
│   │   └── scripts/             # 由 COMPUTE_VENV python 执行的脚本
│   │       ├── nequip_ase.py    # job.json → result.json + traj.extxyz
│   │       └── pinspect.py      # 环境自检（打印 torch/ase/nequip 版本）
│   ├── api/
│   │   ├── deps.py              # 依赖注入（session store、harness 池）
│   │   ├── routes_sessions.py   # 会话 CRUD
│   │   ├── routes_chat.py       # POST 消息 + SSE 流、审批恢复
│   │   ├── routes_jobs.py       # 作业状态/工件下载
│   │   └── routes_models.py     # 可用 MLIP 模型清单
│   └── state/
│       ├── store.py             # 会话/作业持久化（JSON 文件，作业索引扫描派生；见 D3-2）
│       └── schemas.py           # 前后端共享 API schema
├── web/
│   ├── package.json / vite.config.ts / tsconfig.json / index.html
│   └── src/
│       ├── main.tsx / App.tsx
│       ├── api/client.ts        # REST + SSE fetch 封装
│       ├── api/agui.ts          # AG-UI 事件类型 + 解析 reducer
│       ├── components/
│       │   ├── ChatPanel.tsx        # 对话主面板
│       │   ├── MessageList.tsx      # 消息流（文本/工具调用/审批卡）
│       │   ├── ToolCallCard.tsx     # 单次工具调用展开卡（inp/out 摘要）
│       │   ├── ApprovalDialog.tsx   # 审批确认对话框
│       │   ├── ResultPanel.tsx      # 结果区：能量表/频率表/下载
│       │   ├── ConvergenceChart.tsx # opt/scan 曲线（纯 SVG，不引图表库）
│       │   └── StructureView.tsx    # M6：3Dmol.js 3D 结构（可选）
│       └── styles.css
├── benchmarks/
│   ├── tasks.yaml               # 验收任务集（见 §3）
│   ├── runner.py                # 运行器：--mode=live|canned
│   ├── judge.py                 # 数值判定：解析 artifacts，按容差比对
│   └── references/              # 金标准 .out / pinned JSON（M0 生成）
├── tests/
│   ├── conftest.py              # 跳过规则：无 GPU/无 LLM key 时 skip
│   ├── test_inp_builder.py
│   ├── test_parsers.py          # 用 MAPLE/examples 的 .out 做解析 fixture
│   ├── test_runner.py           # 真实子进程跑 examples/sp/sp.inp
│   ├── test_agent_offline.py    # FunctionModel 全链路
│   └── test_api.py              # httpx ASGI 测试
├── tickets/
│   ├── TEMPLATE.md              # 审批工单模板
│   └── README.md                # 何时开工单（改 MAPLE/装包到 COMPUTE_VENV 等）
├── docs/
│   └── decisions.md             # 架构决策记录
├── data/                        # 运行时生成，不入库
│   ├── sessions/{session_id}/state.json
│   └── sessions/{session_id}/jobs/{job_id}/{input.inp,output.out,job.json,result.json,stdout.log,stderr.log}
└── scripts/
    ├── bootstrap.sh             # M0 一键环境搭建
    └── make_golden_refs.sh      # 生成 benchmark 金标准
```

## 2. 组件设计

### 2.1 Agent 层（`app/agent/`）

主协调 agent + 3 个 inline 子 agent，全部通过 `HarnessBuilder` 一处组装（`harness.py` 是唯一 import a13n-harness 的模块，SDK 升级只改这一个文件）。

- **主 agent `mlip_orchestrator`**：`AgentSpec(model=<来自 config>, instructions=ORCHESTRATOR_PROMPT)`，`output_type=str`。持有全部计算工具（闭包注入 session workspace 和 GPU 锁）。职责：理解用户意图 → 必要时 `ask_user_question` 补全（电荷/自旋多重度/模型选择）→ 委派 task_parser 出 JobSpec → 调工具执行 → 委派 result_validator 校验 → 组织中文回答（能量同时给 Hartree 和 eV）。
- **子 agent `task_parser`**（SubagentCapability）：纯文本转换，无工具。输入自然语言，输出严格 JSON JobSpec。校验失败（缺字段/单位不明）时在输出里列 `missing_fields`，主 agent 据此向用户追问。
- **子 agent `result_validator`**：输入计算摘要（退出码、能量、收敛标志、频率列表），输出 `{"ok": bool, "issues": [...], "summary": ...}`。检查项：opt 是否收敛、freq 有无虚频（opt 后不应有，ts 应恰有 1 个）、MD 能量漂移、能量数量级 sanity。
- **子 agent `approval_advisor`**（M6 才启用）：对高危操作（TS 搜索、长时间 MD、uma-m 11GB 模型）生成给用户的风险说明文本。
- **审批规则**（`permissions.py`）：`ToolPermissions(rules={"run_maple": "ask", "run_nequip": "ask"})`，其余工具 auto。挂起后前端弹 ApprovalDialog，用户确认走 resume 端点（`DeferredToolResume`）。

### 2.2 工具层（`app/tools/maple_tools.py` 等）

所有工具是有类型注解 + docstring 的普通函数，统一返回 `ToolResult(ok: bool, data: dict | None, error: str | None)`，**不抛异常**。清单与签名：

| 工具 | 签名 | 说明 / 错误处理 |
|---|---|---|
| `list_available_models` | `() -> list[dict]` | 返回 ANI-1x/ANI-2x/AIMNet2/MACE-OFF/UMA-s/NequIP-OAM-XL 及 checkpoint 路径、适用体系说明 |
| `build_structure` | `(source: str, source_type: Literal["name","xyz","inline_coords"]) -> dict` | name 走 `ase.build.molecule`（H2O/NH3/CH4 等 g2 库）；返回原子数、公式、写入 session 的 xyz 路径。未知名 → error 提示改用 inline |
| `write_maple_inp` | `(job_spec: dict) -> str` | 调 `inp_builder` 生成 .inp，返回路径；schema 校验失败 → error 列出问题字段 |
| `run_maple` | `(inp_path: str, timeout_s: int = 1800) -> dict` | **需审批**。拿 GPU 锁 → 子进程 `$COMPUTE_VENV/bin/maple input.inp output.out`（cwd=job 目录，stdout/stderr 落盘）→ 判 exit code + 扫 `ERROR:` 行 → 自动调 parse，返回能量/收敛/频率摘要。超时杀进程并 error |
| `parse_maple_out` | `(out_path: str) -> dict` | 纯解析，无副作用。见 §2.4 |
| `run_nequip` | `(job_spec: dict, timeout_s: int = 1800) -> dict` | **需审批**。写 job.json → `$COMPUTE_VENV/bin/python app/compute/scripts/nequip_ase.py job.json` → 读 result.json |
| `convert_units` | `(value: float, from_unit: str, to_unit: str) -> dict` | Hartree/eV/kcal-mol、cm⁻¹ 等 |
| `get_job_artifact` | `(job_id: str, name: str) -> str` | 返回工件文件路径供前端下载 |

### 2.3 计算子进程层（`app/compute/runner.py`）

```python
async def run_compute(argv: list[str], cwd: Path, timeout_s: int, gpu: bool) -> CompletedJob
```

- `gpu=True` 时 `async with gpu_lock`（模块级 `asyncio.Semaphore(1)`）。
- env 白名单：`PATH`、`HOME`、`CUDA_VISIBLE_DEVICES=0`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`（杜绝联网下载）。
- stdout/stderr 实时写入 job 目录日志；`asyncio.wait_for` 超时后 `kill()` 进程组（`start_new_session=True` + `os.killpg`）。
- MAPLE 成败判定：exit code 非 0 或 output.out 含 `ERROR:` → 失败，附最后 50 行日志进 error。

### 2.4 解析器（`app/compute/parsers.py`）

按任务类型从 .out 提取（正则基于 MAPLE examples 的真实 .out，fixture 已确认格式，如 `Energy: -383.4722385600 Hartree`）：
- sp：`Energy: <float> Hartree`、`Charge/Multiplicity`。
- opt：每步能量序列（收敛曲线数据）、最终能量、收敛标志、最终结构（extxyz 段）。
- freq：频率列表 cm⁻¹、虚频计数、ZPE。
- scan：坐标-能量点列。
- md：时间-能量序列、漂移。
解析器做成纯函数 + 对 `MAPLE/examples/**` 参考 .out 的 fixture 测试（tests/test_parsers.py），解析失败返回部分结果 + `warnings`，不让工具整体失败。

### 2.5 FastAPI 层（`app/main.py` + `app/api/`）

- `POST /api/sessions` → 创建会话（建目录、初始化 HarnessState）。
- `POST /api/sessions/{sid}/messages` → 提交用户消息，**响应即 SSE 流**（`text/event-stream`）：内部 `async with executable.stream(prompt)`，经 `HarnessAguiStreamObserver` 转 AG-UI 事件逐个 yield。心跳注释行每 15s 防代理断流。
- `POST /api/sessions/{sid}/resume` → body 为审批决定，调 `DeferredToolResume` 恢复挂起的 run，同样 SSE 返回。
- `GET /api/sessions/{sid}/state` → 会话快照（消息历史 + 作业列表），前端刷新恢复用。
- `GET /api/models`、`GET /api/jobs/{jid}`、`GET /api/jobs/{jid}/artifacts/{name}`（FileResponse）。
- 状态持久化：每轮 run 结束把 HarnessState 序列化到 `data/sessions/{sid}/state.json`（加 `state_version` 字段），服务重启后按需 lazy 重建 harness 并恢复。v1 不做多 worker，单进程 uvicorn。
- 启动：`uvicorn app.main:app --host 0.0.0.0 --port 8000`（用 mlip_agent/.venv）。

### 2.6 前端（`web/`）

React 18 + Vite + TypeScript，无重型 UI 库（手写样式 + 纯 SVG 图），3D 查看器 M6 再引 3Dmol.js。
- `api/client.ts`：`fetch` + ReadableStream 解析 SSE（`Accept: text/event-stream`），事件回调进 `agui.ts` 的 reducer。
- AG-UI 事件渲染映射：`TEXT_MESSAGE_*` → 流式气泡；`TOOL_CALL_START/ARGS/END` → ToolCallCard（进行中 spinner，完成后展开摘要）；`CUSTOM`（审批类）/ run suspended 状态 → ApprovalDialog；STATE 快照 → 刷新恢复。
- ResultPanel：能量表（Hartree/eV/kcal·mol⁻¹ 三列）、频率表、ConvergenceChart（opt 步-能量折线、scan 点线）、artifact 下载按钮（xyz/out/traj）。
- 开发：`npm run dev`（vite proxy `/api` → 8000）；生产：`npm run build` 后由 FastAPI StaticFiles 托管 `web/dist`。

## 3. benchmarks/ 验收任务集

### 3.1 v1 任务集（B1–B16，现行有效）

`tasks.yaml` 每条：`id / prompt（自然语言）/ expect: {job_type, model, assertions: [...]} / tol / timeout`。断言由 `judge.py` 对 job artifacts（result.json / parsed out）求值。

| ID | 自然语言输入 | 任务 | 预期与容差来源 |
|---|---|---|---|
| B1 | "用 ANI-2x 计算这个分子的单点能：C -0.778126 …（22 原子内联坐标）" | sp | E = −383.4722385600 ± 1e-6 Ha（金标准：MAPLE/examples/sp/sp.out，同机同库可复现） |
| B2 | "优化水分子的几何结构（用 ANI-2x）" | opt | 收敛；O−H 键长 0.95–0.97 Å、键角 103–106°（M0 跑一遍 pin 精确值 ±0.005 Å/±1°） |
| B3 | "优化水分子并计算振动频率" | opt+freq | 3 条频率，无幂零虚频外的负频；pinned 值 ±20 cm⁻¹（M0 pin；与气相水 ~1595/3657/3757 cm⁻¹ 同量级） |
| B4 | "用 UMA (uma-s-1p1, omol) 算水分子的单点能" | sp | pinned 能量 ±1e-6 Ha（M0 金标准，checkpoint 用本地绝对路径） |
| B5 | "用 NequIP-OAM-XL 模型算水分子的单点能" | sp（直算路径） | pinned ±1e-6 eV |
| B6 | "先优化乙醇的结构，再对优化后的结构做频率分析" | 链式 opt→freq | 两个作业产物齐全；freq 无虚频；乙醇 18 条频率 |
| B7 | "对水分子跑 20 步 NVE 分子动力学，步长 0.5 fs" | md | 总能量漂移 < 1e-3 Ha（相对首步） |
| B8 | "扫描水分子一个 O−H 键从 0.9 到 1.1 Å，5 个点" | scan | 5 个能量点，最小值出现在中间点附近（pin 形状） |
| B9 | "帮我算一下这个体系的能量"（不给模型/电荷） | 澄清行为 | agent 必须触发 `ask_user_question`，不得擅自提交计算 |
| B10 | 同 B1 但断言审批 | 审批流 | `run_maple` 调用前会话进入 suspended；runner 自动批准后才出现作业产物 |

判定脚本思路：`judge.py` 加载 tasks.yaml → 对每条任务读取 runner 产出的 `run_record.json`（消息事件流 + 作业列表 + parsed 结果）→ 逐项断言（数值容差、事件类型存在性、作业数、状态）→ 输出 `{task_id: pass/fail + 原因}` 和汇总成功率/数值正确率，任一 fail 退出码 1。

`runner.py` 两种模式：
- `--mode=live`：起 uvicorn（或直连 harness），需要 `MLIP_AGENT_LLM_*` 环境变量；无 key 时整体 skip 并在报告标注 "SKIPPED: no LLM endpoint"。
- `--mode=canned`：用 pydantic-ai `FunctionModel` 按每个任务预录的「工具调用剧本」回放（不走真 LLM），验证工具层→子进程→解析→judge 全链路数值正确。这是 CI/日常验收的默认模式。

### 3.2 Benchmark v2 重构（2026-10-03 立项并落地）

定位：B1–B19 为**框架级回归集**（canned 数值/行为回归），tier（guided/recovery/unguided）为框架回归语义；agent 能力的三档知识梯度评测由 §3.3 v3 承担。

借鉴 "Can Autonomous LLM Agents Execute Multireference Quantum Chemistry Calculations?"
（arXiv:2609.13357）的评测设计，将 benchmark 从单档执行保真度测试重构为
**分层信息梯度 + 覆盖率/精度/效率三轴报表**。

**硬约束**：B1–B16 的 prompt/断言/references 金标准逐字节不动（动金标准需工单）；
`judge.load_tasks/judge_run_dir/print_report` 签名兼容（ui_runner 复用）；
不触碰 MAPLE / 计算环境 / agent 指令。

**tier 模型**：每任务加 `tier` 与 `min_jobs` 字段。

- guided：B1–B16 全部（给全信息 + pin 值，对应论文 Replication 档）
- recovery：B17/B18（首试必失败，测诊断-修复能力，对应论文决策阶梯价值点）
- unguided：B19（不给模型/电荷，测信息不全行为，对应论文 Baseline 档）

**新增任务**：

- B17（recovery）：内联金标准坐标直接做 freq → write_maple_inp 被 jobspec
  校验拒绝（D7-3）→ 改走 opt→freq 正确链。复用 B3 频率 pin（±20 cm⁻¹，
  D5-3 已论证对起始点不敏感）+ n_imaginary=0 + tool_call_failed 断言，min_jobs=2。
- B18（recovery）：20 步 NVE 不提 log_every → 首跑无逐步能量行（D5-1）→
  诊断后 log_every=1 重跑。断言 job_count=2、job0 md_steps_max=19、
  job1 md_steps=20 + drift ≤1e-3（复用 B7 口径），min_jobs=2。
- B19（unguided）："优化乙醇的结构"（不给模型）→ 必须 ask_user_question；
  auto_answer 注入 "用 ANI-2x，电荷 0，多重度 1" → opt 收敛、工件齐全
  （复用 B6 opt 断言，无新 pin），min_jobs=1。

**runner.py**：run_record 新增 failed_tool_calls（从事件流提取 ok=false 的
工具调用；若 TOOL_CALL_END 无结果载荷，兜底改断言 tool_call_count≥2）与
metrics 块（n_jobs / n_failed_jobs / duration_s）；tool_calls 保持 list[str]。

**judge.py**：新断言 tool_call_failed、md_steps_max；numeric 断言记录结构化
abs_error；judge_task 透传 tier 并输出 efficiency 块；summarize 多维
（by_tier/by_backend/by_job_type/by_model 分组 coverage+accuracy、
error_stats mean/max abs_error、efficiency first_try_rate/total_jobs），
既有字段名不变；print_report 末尾加分层表格 + 误差统计 + 效率行。

**canned_scripts.py**：新增 B17/B18/B19 剧本（既有剧本不动）；B18 的
_spec_from_delegated md 分支支持无 log_every 变体。

**tests/test_benchmarks.py**：任务清单断言更新为 B1–B19 + tier 字段；
新断言正误两路单测；summarize 新字段单测；既有测试全保持。

**验收**：pytest 全绿；canned B1–B19 全 PASS 且 judge_report 含分组/误差/
效率块；`git diff benchmarks/references/` 为空；B1–B16 任务块 diff 仅
tier/min_jobs 两行；live 无 key 正确 skip。

### 3.3 Benchmark v3：GMTKN55 反应能垒复合任务 × 三档知识梯度（2026-10-03 立项，待实施）

对标 arXiv:2609.13357 的完整评测形态：外部高精度金标准 + 复合链式任务 +
同一任务三档知识梯度 + 覆盖率/MAE/尝试次数/成本四维指标 + 失败归因讨论。

**数据集（替代被否决的 T1x）**：GMTKN55（Goerigk et al., PCCP 2017）——AIMNet2
官方论文（Chem. Sci. 2025, d4sc08572h）的原生 benchmark，参考值为高级别 WFT
（CCSD(T)/CBS 至 W2-F12 级），可与已发表数据直接对照。取 **BH76 子集**（76 个
能垒，含反应物/产物/TS 三套几何 + 高精度正/反向能垒；BH76RC 给反应能），选
6 个反应（类型多样：H 转移/重原子转移/S_N2/单分子；体系 ≤6 原子）。含自由基
（mult=2）与 Cl——**套件模型定为 aimnet2**（支持 charge/mult、元素覆盖匹配、
原生 benchmark）。参考值是能量差（kcal/mol），适合跨理论级别比较。
获取：uni-bonn Grimme 组官网 tar（镜像 grimme-lab/GMTKN55、refdata）。

**复合任务链（每反应六段）**：opt 反应物 → opt 产物 → freq 两极小（各 0 虚频）
→ scan（沿断键刚性扫描）→ TS 搜索（初猜用数据集 ts_guess 或扫描最高点）→
TS freq（恰 1 虚频）。派生量：正向能垒 = E_TS − E_reactant、反向能垒 =
E_TS − E_product、反应能 = E_product − E_reactant。
判定参考量（每反应 3 MAE + 5 coverage = 8 项）：MAE——正/反向能垒、反应能
（kcal/mol）；coverage——两 opt 收敛、两 freq 无虚频、scan 曲线完整
（≥5 点且含上升段）、TS freq 恰 1 虚频。min_jobs=7。

**三档 guidance（prompt 层变体，不动 instructions.py）**：
- baseline：只给基本信息（三个 xyz 路径 + 一句话目标 + aimnet2 + 各物种
  charge/multiplicity）；
- decision：baseline + `benchmarks/reaction/ladder.md` 决策阶梯全文注入；
- replication：完整工作流逐步写明（每段输入来源/参数/产物接力/换算公式）。
tasks 存 `benchmarks/reaction/tasks.yaml`（tier=baseline/decision/replication，
reaction=R{i} 分组），6 反应 × 3 档 = 18 条。

**四维指标（judge 扩展）**：Coverage（scored/48 每档，unscored 计入覆盖损失
不污染 MAE）；MAE（能垒/反应能 vs GMTKN55，kcal/mol，mean/max 按档位与反应
类型分解）；Attempts（actual_jobs vs min_jobs=7，分布+mean）；Cost
（Σduration_s + total_jobs）。新断言 barrier_mae / reaction_energy_mae /
scan_curve（「总是计分」语义，B 系列断言语义不动）。报表 by_tier × by_reaction
对比矩阵。

**failure reasoning**：judge_report 新增 failure_analysis——每个 unscored/failed
参考量收集证据（record.error、failed_tool_calls、.out 末尾 ERROR 行、agent
输出相关句）→ 映射归因类别（Rule A–I 关键词规则 + 兜底未分类）→ 中文讨论段。

**决策阶梯 ladder.md（Rule A–I）**：
- A 执行层失败（exit≠0/超时/缺 .out）→ 读 stderr.log 与 .out 末尾 ERROR 行；
  区分基础设施（超时→调大 timeout_s）与输入错误。
- B 输入校验拒绝 → 按 write_maple_inp 错误提示修 JobSpec：freq 必须 xyz_file、
  scan 原子对 1-based、模型-体系匹配（元素/电荷支持）。
- C opt 不收敛 → 检查起始几何键长量级与 charge/multiplicity（自由基 mult=2
  常见漏配）；增大 opt.max_iter。
- D 极小点 freq 有虚频 → 几何未真正收敛：提高收敛等级重优化后重算 freq。
- E scan 曲线异常 → 检查起始几何已调至扫描起点、n_steps=N−1、断键原子对
  与反应坐标一致。
- F TS 搜索失败或虚频≠1 → 初猜取 ts_guess 或扫描最高点附近；prfo 不收敛换
  更近初猜；虚频>1 为高价鞍点，沿最大虚频模式位移后重搜；禁止盲目重交同一输入。
- G 能量/能垒数量级异常 → 查单位制（1 Ha=627.5095 kcal/mol）、charge/multiplicity、
  能垒公式方向（正/反勿颠倒）。
- H 与参考系统偏差但物理合理 → 方法学差异（MLIP 训练级别 vs WFT 参考），
  如实计入 MAE，不做「修复」。
- I 成功确认：六段全部收敛 + 虚频数符合驻点类型（极小 0 / TS 恰 1）+
  能垒方向合理 + 与参考同量级。

**配套框架改动**：JobSpec 增 ts 任务类型（TsSpec method=prfo 等，inp_builder
渲染 `#ts(method=)`，parsers 解析 ts .out，fixture 用 MAPLE/examples/ts/prfo）。

**验收**：pytest 全绿；canned B1–B19 不回归；R1 canned 冒烟链 PASS 且产出四维
指标段；B 系列金标准零改动。live 三档评测需 MLIP_AGENT_LLM_*（成本估算
6 反应 × 3 档 ≈ 1.5–4 GPU 小时 + LLM 时间），运行手册见 benchmarks/reaction/README.md。

### 3.4 2026-10-03 扩展：PBC 落地 + 新模型 + ASE io（已落地）

**前提裁定（用户）**：暂不做 cell relaxation（不接 stress 输出，opt 只动原子坐标）；
ANI 不改 PBC；其余框架全部打通周期性。工单 TICKET-0007/0008/0009 已批准实施。

**PBC 落地**（D9-1/D9-4）：
- 直算三路（NequIP/TACE/DPA-4）接 PBC：周期结构统一为带 `Lattice=` 头的 extxyz；
  TACE 邻居表按周期性分派 + 混合 pbc/带胞非周期硬报错（TACE 静默升级坑）。
- UMA 经 MAPLE：JobSpec `structure.cell` → `#pbc(...)`；周期 task 门禁
  （omat/oc20/oc22/oc25/omc/odac，omol 互斥）。
- MAPLE MACE calculator PBC（TICKET-0008）：matscipy 周期邻居表，与官方
  mace ASE 零差对照；AIMNet2 PBC + ewald 解禁（TICKET-0009，实施中）。
- 金标准 B23–B28（Si 金刚石原胞 sp 多路互证 + H2O 周期盒）。

**新模型**：
- MACE-MPA-0 medium（macempa0，MAPLE 插件路线，MIT；B20/B27）——D9-2。
- DPA4C-Air-OMol25（family="dpa4c"，run_dpa4 直算，CC BY-NC；B21）——D9-3；
  注意总能量约定与 DPA4-beta 不互比、DeepPot 对 DPA4C 静默忽略 head。
- AIMNet2-NSE（aimnet2nse，开壳层/三重态，MIT；B22）——D9-5；
  转换脚本 scripts/make_aimnet2nse_ts.py。

**ASE io 工具**：`convert_structure`（cif/poscar/extxyz/xyz/pdb 等互转，
COMPUTE_VENV 子进程跑 ASE，轻量免审批；周期信息保留/丢失警告、POSCAR 自动
真空盒、cellpar 返回可直接填 JobSpec structure.cell）。

**能力边界**：所有路径 opt 不优化晶胞；MAPLE/UMA 无 stress（NPT 不可靠）；
MACE/AIMNet2 PBC 仅承诺 sp/opt；ANI 全系保持分子域。

## 4. 里程碑与并行任务包

依赖图：M0 → M1 → M2 → M3 → {M4, M5} → M6。M4 与 M5 并行；M1 内部 A/B 包并行。

**M0 环境引导与验证（1 个 agent，先行）**
- 产出：`scripts/bootstrap.sh`、`scripts/make_golden_refs.sh`、`benchmarks/references/*`、`resource/decisions.md` 初稿。
- 步骤：`uv python install 3.13 && uv venv --python 3.13`（在 mlip_agent/）；`uv add fastapi uvicorn pydantic pydantic-settings a13n-harness==0.5.2 a13n-stream-protocol`（装完即 `uv lock` 并提交）；验证 `node --version && npm --version`（若无 node → 记录在 risks，先装 node ≥20 再继续 M4）；`$COMPUTE_VENV/bin/python app/compute/scripts/pinspect.py` 自检；`$COMPUTE_VENV/bin/maple --test 1` 冒烟；跑 golden refs（B1–B5、B8 的真实计算，pin 数值到 references/）。
- 验收：bootstrap.sh 在干净 shell 跑通；references/ 至少 5 个金标准文件。

**M1-A MAPLE 计算层（agent α）**
- 产出：`app/compute/{runner,jobspec,inp_builder,parsers}.py`、`tests/test_{inp_builder,parsers,runner}.py`。
- 验收：pytest 全绿；test_runner 真实跑 `MAPLE/examples/sp/sp.inp` 复制件并解析出 −383.4722385600；parsers 覆盖 sp/opt/freq/scan/md 各至少 1 个 examples fixture。

**M1-B NequIP 直算脚本（agent β，与 M1-A 并行）**
- 产出：`app/compute/scripts/nequip_ase.py`（读 job.json：xyz 路径/任务 sp|opt/输出路径；`nequip.ase.NequIPCalculator.from_compiled_model` 加载 zip；opt 用 ASE LBFGS；写 result.json + traj.extxyz）、`app/tools/mlip_tools.py` 初版、B5 金标准。
- 验收：手动 `$COMPUTE_VENV/bin/python nequip_ase.py sample_job.json` 产出正确能量（与 B5 pin 一致）。

**M2 agent 层（1 个 agent；依赖 M1-A/B 接口）**
- 产出：`app/agent/*` 全部、`app/tools/*` 全部、`app/config.py`、`tests/test_agent_offline.py`。
- 验收：FunctionModel 离线测试覆盖 B1/B6/B9/B10 剧本（含 suspended→resume）；工具单测全绿；`python -c "from app.agent.harness import build_harness"` 可无 key 构建（模型延迟到 run 时解析）。

**M3 API 层（1 个 agent；依赖 M2）**
- 产出：`app/main.py`、`app/api/*`、`app/state/*`、`tests/test_api.py`。
- 验收：httpx ASGI 测试：建会话→发 canned 消息→SSE 事件序列含 TOOL_CALL 与 TEXT_MESSAGE→审批恢复→artifacts 可下载；服务重启后会话 state 恢复。

**M4 前端（1 个 agent；与 M5 并行，依赖 M3 的 API 契约 + 一份录制 SSE 样例 mock）**
- 产出：`web/` 全部。
- 验收：`npm run build` 通过；mock SSE 下消息流/工具卡/审批框/结果面板渲染正确；`npm run dev` 手动联调通过 B1、B6。

**M5 benchmarks（1 个 agent；与 M4 并行，依赖 M2/M3）**
- 产出：`benchmarks/{tasks.yaml,runner.py,judge.py}` + 10 条任务。
- 验收：`python benchmarks/runner.py --mode=canned` 10/10 pass；live 模式无 key 时正确 skip。

**M6 收尾（1 个 agent）**
- 产出：approval_advisor 子 agent、StructureView（3Dmol.js）、可选 MAPLE calculator plugin（需先开 ticket 评审）、README、AGENTS.md 增补（mlip_agent 启动命令）。
- 验收：端到端 live 演示 B1–B16（有 LLM key 时）。

## 5. 风险与对策

1. **a13n-harness 0.x API 漂移**：`pyproject.toml` 精确钉 `a13n-harness==0.5.2` 与配套 `a13n-stream-protocol==0.5.2`（包名已于 2026-10-02 在 PyPI 验证存在，见 resource/decisions.md D0-2），`uv.lock` 入库；所有 SDK 引用集中在 `app/agent/harness.py` + `streaming.py` 两个文件，升级时先开 ticket 评审 changelog。
2. **无 LLM API key**：live benchmark 自动 skip；canned 模式 + FunctionModel 单测保证无 key 下 CI 全绿；`.env.example` 写清三个变量，拿到任何 OpenAI 兼容端点即可切 live。
3. **GPU 争用/显存溢出**：全局 semaphore 串行 GPU 作业；默认模型 uma-s-1p1（1.1GB），uma-m-1p1（11GB）在 `list_available_models` 标 "大模型，需确认" 且工具层要求显式确认参数；子进程崩溃/超时必须释放锁（`async with` 保证）。
4. **Node 不可用**：M0 第一步验证；若系统无 node 且不可装，降级方案为服务端渲染极简页面（不作为默认路径，先开 ticket）。
5. **联网下载模型**：子进程 env 强制 `HF_HUB_OFFLINE=1`；所有 `#model=` 与 checkpoint 用绝对本地路径；工具层 lint 掉任何 URL 形态的模型参数。
6. **SSE 长任务断流**：心跳 + 前端断线后用 `GET /sessions/{id}/state` 恢复；超过 timeout_s 的作业后台继续，完成后通过 state 拉取。
7. **MAPLE 输出格式变化**：parsers 只依赖 examples 验证过的格式并有 fixture 测试；MAPLE 被升级时 fixture 测试会第一时间暴露。
8. **HarnessState 版本不兼容**：state.json 带 `state_version`，不匹配时弃旧开新会话并提示用户，不硬迁移。

## 6. 立即可执行的下一步

按顺序派单：M0 → (M1-A ∥ M1-B) → M2 → M3 → (M4 ∥ M5) → M6。每个编码 agent 开工前读本文件对应章节 + `resource/decisions.md`；任何超出本文件范围的改动（尤其触碰 MAPLE/、.venv/ 计算环境装包）先写 `tickets/` 工单。

下一步：实施 §3.3 benchmark v3（GMTKN55 反应能垒复合任务 × 三档知识梯度），实施顺序与验收以 §3.3 为准。
