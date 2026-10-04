# 架构决策记录（ADR）

M0 初稿，2026-10-02。每条决策含：背景、决定、理由、后果。

## D0-1 双 Python 环境严格隔离

- **决定**：agent/后端环境 `mlip_agent/.venv`（uv 管理，Python 3.13.11）与计算环境 `COMPUTE_VENV=/root/soft/ml_potential/.venv`（Python 3.12.12）完全隔离。后端只通过 `$COMPUTE_VENV/bin/maple` 和 `$COMPUTE_VENV/bin/python` 两个入口以子进程调用计算栈。
- **理由**：a13n-harness 0.5.2 硬性要求 Python≥3.13；计算栈（torch 2.8.0+cu128 / maple 0.1.4 / nequip 0.17.1 / fairchem-core 2.13.0）已在 3.12 环境验证可用，升级风险高。
- **验证**：`mlip_agent/.venv` 中 `import torch` 失败（符合预期）；`scripts/bootstrap.sh` 在 `env -i` 干净 shell 中幂等跑通。

## D0-2 a13n 包钉版

- **决定**：`a13n-harness==0.5.2` 精确钉死；`a13n-stream-protocol` 已验证为 PyPI 真实包名（当前版本 0.5.2，requires-python≥3.13），同样钉 `==0.5.2`。`uv.lock` 入库。
- **事实**：两包均于 2026-10-02 在 PyPI 查证存在。`a13n-harness` 传递依赖带入 `pydantic-ai 2.52.0`（FunctionModel 离线测试可用，无需另装）与 `ag-ui-protocol 1.0.0`。
- **后果**：SDK 引用只允许出现在 `app/agent/harness.py` 与 `app/agent/streaming.py`（PLAN §5.1）。

## D0-3 模型可用性清单（2026-10-02 实测）

| 模型 | 状态 | 说明 |
|---|---|---|
| ANI-2x | ✅ 可用 | `MAPLE/maple/function/calculator/model/ani2x.pt`（53MB，首次运行自动从 HuggingFace 下载，网络通畅） |
| ANI-1x / ANI-1ccx / ANI-1xnr | ✅ ani1x 可用 | ani1x.pt 已自动下载缓存（B12 实测通过）；ani1ccx/ani1xnr 同机制，首次使用联网下载 |
| UMA uma-s-1p1 (omol) | ✅ 可用 | 本地 checkpoint `uma/uma_folder/checkpoints/uma-s-1p1.pt`，B4 实测通过 |
| UMA uma-m-1p1 / uma-s-1 | 📦 本地存在，未实测 | 11GB 大模型，按 PLAN §5.3 需显式确认 |
| NequIP-OAM-XL | ✅ 可用 | `nequip/NequIP-OAM-XL-0.1.nequip.zip`；直算 B5（D1-1）与 MAPLE 插件 B11（D7-1）均已验收 |
| AIMNet2 | ✅ 可用 | aimnet2.pt 已自动下载缓存（B13 实测通过） |
| MACE-OFF23(S) | ✅ 可用（本地自制 wrapper） | 无官方自动下载；`scripts/make_maceoff23s_wrapper.py` 生成 maceoff23s.pt（B14 实测通过，见 D7-4）；maceoff23m 可自动下载，maceoff23l 未备 |

- 后续新增模型（本表之外）：TACE-OAM-L（直算，D7-6，B15）、DPA-4（独立 venv 直算，D7-8，B16）。
- 所有框架内子进程 env 强制 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`（金标准生成脚本已如此），模型文件一律绝对本地路径。

## D0-4 B1 能量 pin 值采用本机复现值

- **事实**：MAPLE 仓库自带 `examples/sp/sp.out` 参考能量 −383.4722385600 Ha；本机（RTX 3090）复现为 **−383.4722385376 Ha**，差 2.2e-8 Ha，在 B1 容差 1e-6 内。
- **决定**：B1.json pin 本机复现值（同机同库回归用），JSON `note` 字段记录与仓库参考值的差异。

## D0-5 B3 频率 pin 值为 MAPLE 实际输出，疑似 MAPLE freq(mw) 投影顺序 bug 已开工单

- **事实**：MAPLE `freq`（默认 method=mw）对收敛 H2O（max|F|=3e-7）输出 1966/4200/4607 cm⁻¹；同一 Hessian 按标准顺序（先质量加权再投影平动/转动）得 1721/3803/3921 cm⁻¹（与 ANI-2x 物理预期一致）。已数值复现：差异 100% 来自 MAPLE 先投影（Cartesian）再质量加权的顺序——两者仅在等质量体系等价。
- **决定**：B3.json pin MAPLE CLI 实际输出（benchmark 测的是框架复现 MAPLE 的能力，不是 MAPLE 的物理正确性）；疑似 bug 记录于 `tickets/TICKET-0001-maple-freq-mw-projection.md`，是否修复 MAPLE 由人工审批决定。PLAN §3 B3 的"与气相水同量级"预期需知晓此偏差（弯曲模偏高约 23%）。

## D0-6 前端工具链

- node v22.23.1 / npm 10.9.8（登录 shell），干净 PATH 下为 node v20.19.5 / npm 10.8.2，均满足 ≥20。M4 无阻塞。

## D0-7 扫描任务用刚性扫描 pin 金标准

- B8 用 `#scan(mode=rigid)`，`S 1 2 0.05 4`（O-H 从 0.9 Å 起 5 点）。MAPLE scan 输出能量仅 6 位小数，B8 容差放宽为 1e-5 Ha（其余能量 pin 1e-6 Ha）。

## D1-1 NequIP zip 模型走 `_from_saved_model`，不是 `from_compiled_model`

- **事实**：`NequIP-OAM-XL-0.1.nequip.zip` 是 `nequip-package` 产物（torch.package 格式）；nequip 0.17.1 的 `NequIPCalculator.from_compiled_model` 只接受 `nequip-compile` 产物（AOTI `.nequip.pth`/`.pt2`），对 zip 会在 `load_compiled_model` 处失败。zip 的正确入口是 `NequIPCalculator._from_saved_model(path, device=..., chemical_species_to_atom_type_map=True)`（`nequip/integrations/mixins.py:77`，内部按扩展名分发到 `ModelFromPackage`）。
- **决定**：`app/compute/scripts/nequip_ase.py` 使用 `_from_saved_model`（带下划线的半私有 API，nequip 官方测试 `tests/integration/test_state_restoration.py` 同款用法）；`chemical_species_to_atom_type_map=True` 显式恒等映射，消除警告。nequip 升级时需复查此 API。
- **后果**：B5 pin 值 −13.43729751141953 eV（RTX 3090 / cuda / eager float32），同机两次运行位级一致，容差 1e-6 eV 安全。单位是标准 ASE eV/Å，与 MAPLE 的 Hartree 语义严格区分（result.json 显式 `units` 字段）。

## D2-1 权限规则的 selector 必须写全限定 tool_id（侦察结论修正）

- **事实**：PLAN §2.1 写 `ToolPermissions(rules={"run_maple": "ask", ...})`，但 a13n-harness 0.5.2 的权限匹配（`_tool_selectors.match_selector`）作用于**全限定 tool_id**，函数工具的 tool_id 为 `tool/<capability_id>/<tool_name>`（`tools/identity.py: source_tool_id`，Capability id 为 None 时前缀是 `native`）。裸工具名永远匹配不到。
- **决定**：`app/agent/permissions.py` 的 PERMISSION_RULES 用全限定 id：`tool/mlip.maple/run_maple`、`tool/mlip.nequip/run_nequip`、`tool/mlip.dialog/ask_user_question`；capability id 常量与 `app/tools/registry.py` 共享。
- **验证**：FunctionModel 原型与 tests/test_agent_offline.py 实测，规则生效（run_maple 挂起、其余工具直通）。

## D2-2 build_structure 的分子名解析走 COMPUTE_VENV 子进程，不维护内置坐标表

- **背景**：mlip_agent/.venv 无 ase（双环境隔离，PLAN §0.1），任务书允许「子进程调 COMPUTE_VENV ase」与「内置常见分子坐标表」二选一。
- **决定**：子进程方案——新增 `app/compute/scripts/build_structure.py`（`ase.build.molecule` → `ase.collections.g2` 两级回退 + 小别名表），由 `run_compute(gpu=False)` 执行，产物 structure.xyz + meta.json 落 session workspace。
- **理由**：g2 集合覆盖数百个分子且坐标来自标准库，内置表只能覆盖个位数分子且要手抄坐标（易错）；子进程单次约 0.5–1 s，不走 GPU 锁，成本可忽略。
- **后果**：name 来源依赖 COMPUTE_VENV 可用（conftest 已有 compute_python fixture 跳过规则）；未知名返回 error 并提示改用 inline_coords。

## D2-3 ask_user_question 用「ask 审批」实现挂起，答复经 override_args 注入

- **背景**：B9 要求信息不全时挂起追问；a13n-harness 没有内置 ask_user 工具。PLAN §2.1 只把 run_maple/run_nequip 列为 ask，未规定 ask_user_question 机制。
- **决定**：ask_user_question 注册为普通函数工具（`tool/mlip.dialog/ask_user_question`），权限规则置 "ask" → 调用即挂起（deferred.approvals）。恢复时用户答复经 `ToolApproved(override_args={..., "answer": <用户文本>})` 注入（`permissions.build_approval_results(answers={tool_call_id: text})`）；拒绝走 `ToolDenied`。这比对 deferred.calls（external 工具）路径简单，且批准语义即「用户已看到问题」。
- **验证**：test_b9_ask_user_question_suspends / test_b9_denial_path 覆盖挂起、答复注入、拒绝三条路径。

## D2-4 mlip_tools.py 重写为 async + run_compute（接口变化说明）

- **决定**：M1-B 的模块级同步 `run_nequip(job_spec, timeout_s)` 改为 `run_nequip_impl(services, job_spec, timeout_s)`（async，经 `run_compute` 拿 GPU 锁/超时杀进程组/env 白名单），工具闭包由 `build_nequip_tools(services)` 产出。模块级 `COMPUTE_VENV` 常量删除，路径统一走 Settings。
- **理由**：PLAN §2.2 要求应用服务闭包注入；M1-B 的 TODO 注释本就指明要切到 runner。对外工具契约（job_spec 字段、ToolResult 形态、需审批）不变；`output_dir` 缺省时改为 JobAllocator 分配 session 作业目录（原来是系统临时目录），使 get_job_artifact 可统一定位工件。test_nequip_script.py 不 import 本模块，不受影响。

## D2-5 FunctionModel 离线测试必须提供 stream_function

- **事实**：harness 的 run/stream 内部统一走模型的 `request_stream`（流式请求），FunctionModel 只给 `function=` 会在 `function.py:204` 断言失败。侦察结论中「stream_function 可 yield DeltaToolCalls」准确，且**只能**用 stream_function 形式。
- **决定**：离线剧本统一 `FunctionModel(stream_function=...)`；yield `{0: DeltaToolCall(name=..., json_args=...)}` 发工具调用、yield str 发最终文本。剧本按「消息历史中 ModelResponse 条数」推进（挂起/恢复不改变计数语义，resume 后历史完整）。

## D2-6 流式适配器终止事件契约（M3 依赖）

- **决定**：`app/agent/streaming.py: stream_agui_events()` 转发 `HarnessAguiStreamObserver` 的 AG-UI 事件（dict，`type` 字段为 EventType 值），末尾追加一个合成事件 `{"type": "MLIP_RUN_RESULT", "status", "output", "state_json", "deferred", "failure"}`。`state_json` 是 `HarnessState.model_dump_json()`，实测可 `model_validate_json` 往返；suspended 时 observer 产出带 `RunFinishedInterruptOutcome`（含 interrupts 列表）的 RUN_FINISHED，M3 据此弹审批卡。
- **注意**：observer 是状态机，每次 run 须新建实例（适配器内部已如此）。

## D3-1 挂起现场持久化：deferred 走 TypeAdapter 全量 JSON，不靠摘要重建

- **事实**：`HarnessState` 只含 thread_id/消息历史/环境状态，**不含** DeferredToolRequests；streaming 的 `deferred` 摘要字段只保留 `args` 的字符串形态，不足以重建 `ToolCallPart`（`build_approval_results` 的 answers 注入需要 `args_as_dict()`）。`DeferredToolRequests` 是 pydantic dataclass，`TypeAdapter(DeferredToolRequests)` 可无损 JSON 往返（实测含 tool_kind/part_kind 等默认字段）。
- **决定**：`app/agent/permissions.py` 增加 `dump_deferred_json` / `load_deferred_json`；streaming 的 MLIP_RUN_RESULT 事件新增 `deferred_json` 字段；`harness.make_resume_from_json(deferred_json, ...)` 直接构造 `DeferredToolResume`（绕开「重启后没有 HarnessRunResult」的问题——make_resume 需要一个内存态 result，重启后只有磁盘 JSON）。
- **后果**：SDK/反序列化仍只出现在 harness.py / streaming.py / permissions.py；store.py 把 state_json / deferred_json 当不透明字符串。

## D3-2 会话存储：纯 JSON 文件 + 扫描派生作业索引，不上 sqlite

- **决定**：`data/sessions/{sid}/state.json`（`state_version=1` + harness_state_json + deferred_json + pending_approvals + last_run），原子写（tmp + replace）；每会话一把 `asyncio.Lock` 串行化 run 与状态写。作业索引不冗余持久化：`GET /api/jobs/{jid}` 实时扫描 `sessions/*/jobs/{jid}`（job_id 白名单正则防穿越），`GET /sessions/{sid}/state` 的 jobs 列表同理扫描派生。PLAN §1 注释提到的 sqlite 索引在 v1 单进程规模下属于过度设计，作业量到千级再评估。
- **版本不匹配**：弃旧对话（state.json 归档为 state.v{N}.stale.json），`stale_state=true` 标记，作业工件保留（PLAN §5.8）。

## D3-3 SSE 心跳用「挂起的 __anext__ task + asyncio.wait」，不 cancel 底层流

- **事实**：`asyncio.wait_for(agen.__anext__(), 15)` 超时即 cancel 内部 anext，取消会传播进 harness stream 上下文，有中断 run 的风险。
- **决定**：把 `__anext__()` 包成 Task，`asyncio.wait({task}, timeout=15)` 超时只发 `:hb` 注释行、Task 继续挂起，下轮继续等同一个 Task。心跳间隔常量 `routes_chat.HEARTBEAT_S = 15.0`。
- **并发语义**：同会话第二个并发 message/resume 请求直接 409（不排队）；会话挂起态（suspended）下 POST /messages 也 409，引导前端走 /resume。

## D3-4 create_app 工厂 + app.state.services 依赖注入

- **决定**：`app/main.py: create_app(settings=None, model_override=None)`；测试注入 `Settings(data_dir=tmp_path)` 与 FunctionModel 字典，生产用模块级 `app = create_app()`。AppServices（settings/store/harness 缓存/model_override）挂 `app.state.services`，路由经 `deps.get_services(request)` 取；harness 每会话构建一次并缓存（工具闭包绑定该会话 workspace），重启后 lazy 重建。
- **已知限制**：httpx ASGITransport 不跑 lifespan，测试路径依赖 `SessionStore.__init__` 即建目录（lifespan 只做幂等 mkdir + 关闭清理），两端行为一致。

## D5-1 MdSpec 增加 log_every/traj_every；md 任务强制显式 charge/multiplicity

- **背景**：M5 B7（20 步 NVE）暴露两个 inp_builder 缺口：① MAPLE md dispatcher 强制 `atoms.info` 含 charge/mult（`dispatcher.py:112`），缺省 0 1 不渲染会直接报错；② charge/mult 独立行与 `XYZ <file>` 引用同组会被 input_reader 判为「混合坐标」报错，正确语法是 `XYZ <charge> <mult> <path>`（`xyz_reader._parse_charge_mult` 支持）。
- **决定**：`build_inp` 对 md 任务一律显式输出 charge/mult；XYZ 引用时内联进引用行，内联坐标时保持独立首行。`MdSpec` 新增可选 `log_every`/`traj_every`（缺省 None = MAPLE 内建默认 100；B7 的 20 步短 MD 需 `log_every=1` 才有逐步能量行供漂移解析）。
- **后果**：`test_xyz_reference_with_charge` 的期望从独立行改为 `XYZ 1 1 a.xyz`（旧期望本身在 MAPLE 侧不可运行）；其余 inp 渲染不变（既有测试全绿）。

## D5-2 benchmark 汇总口径与断言分类

- **决定**：tasks.yaml 每条断言带 `kind: execution|numeric`（缺省 execution）。执行成功率 = execution 断言全过的任务数 / 实际执行任务数；数值正确率 = numeric 断言全过的任务数 / 含 numeric 断言的执行任务数。数值比较一律 `|Δ| ≤ tol`（闭区间，容差边界算过）。
- **细节**：B3/B6 的频率断言只比实频（ν > 5.0 cm⁻¹，与 MAPLE 汇总行口径一致），解析出的 6 个近零投影模不参与比对；B6 实频总数（21）以 MAPLE 汇总行为权威（`out_contains` 空白归一化后匹配），见 TICKET-0001 与 D0-5。
- **skip 语义**：live 模式无 `MLIP_AGENT_LLM_*` 时整体写 skipped run_record（"SKIPPED: no LLM endpoint"），judge 不计入两个比率、不影响退出码。

## D5-3 canned 剧本的几何来源：sp/scan 用金标准内联坐标，opt/freq/md 用 build_structure(name)

- **决定**：B4/B5（单点）与 B8（刚性扫描）的能量与起始几何强相关，剧本用 `build_structure(inline_coords)` 喂金标准同款坐标（make_golden_refs.sh 的 b4/b5/b8 几何）；B2/B3/B6/B7 走 `build_structure(name)`（D2-2 子进程路径）——opt 收敛到同一极小（±1e-6 内），频率容差 ±20 cm⁻¹ 对起始点不敏感。
- **runner 实现**：canned/live 共用同一条 `stream_agui_events` 流式循环（挂起 → `make_resume_from_json` 自动批准 → 终态），同时覆盖 D3-1 的 deferred JSON 恢复路径；挂起瞬间对作业目录做产物快照，支撑 B10「挂起时无产物」断言。

## D6-1 approval_advisor 子 agent 接入（M6）

- **决定**：`approval_advisor` 作为第三个 inline 子 agent 加入 `SUBAGENT_SPECS`（纯文本转换，无工具）；主 agent prompt 增加规则：高危操作（TS 搜索、长 MD、uma-m-1p1 11GB 大模型）在提交 run_maple/run_nequip 审批前先委派 approval_advisor 生成风险说明，原样呈现给用户。
- **实现成本**：harness.py 零改动（`_resolve_models` 与 SubagentDefinition 构造均遍历 SUBAGENT_SPECS）；既有 FunctionModel 剧本的 model_override 字典不含 advisor 时 model=None（未委派即不解析），M2/M5 剧本不受影响。
- **验证**：test_approval_advisor_high_risk_consultation（GPU-free 剧本）覆盖委派、SUBAGENT_STARTED/FINISHED 事件、风险文本进入审批追问与最终回答。

## D6-2 write_maple_inp 落盘 structure.xyz 工件（M6 StructureView 数据源）

- **背景**：StructureView 需要「从 job artifact 的 xyz 获取内容」，但此前 MAPLE 作业目录无 xyz 工件（sp 作业尤其如此）。
- **决定**：`write_maple_inp_impl` 额外写 `structure.xyz`（coords 内联渲染 / xyz_file 引用则复制内容），保证每个 MAPLE 作业都有可 3D 渲染的结构工件；NequIP 作业用既有 traj.extxyz。
- **后果**：tests/test_api.py 的工件集合断言同步更新；不影响数值行为（不改 inp 内容）。

## D6-3 e2e 冒烟走「进程内 uvicorn + httpx 真 HTTP」

- **决定**：`scripts/e2e_smoke.py` 用 `uvicorn.Server` 在进程内起真实服务（`create_app(settings, model_override=B1 剧本 FunctionModel)`），httpx 走 TCP 全流程（sessions → messages SSE 挂起 → resume 批准 → state → artifact 下载断言 pin 能量）。httpx 客户端 `trust_env=False`（本机 socks 代理环境变量会污染 127.0.0.1 连接）。
- **理由**：与 tests/test_api.py 的 ASGITransport 不同，真实 uvicorn 覆盖 SSE 分帧、心跳、并发锁等线上路径；这是「无 LLM key 下全链路（除真实 LLM 外）」的端到端证明。

## D6-4 前端 3dmol 按需加载

- **决定**：StructureView 用动态 `import("3dmol")`，主包 164KB / 3Dmol 独立 chunk 588KB 按需加载；npm 依赖仅新增 `3dmol`（2.5.5，自带类型，@types/3dmol 不存在于 npm）。

## D7-1 NequIP-OAM-XL 注册为 MAPLE 插件后端（TICKET-0002）

- **决定**：新增 `plugins/maple_nequip_plugin.py`（`@register_calculator` 的 `NequIPPluginCalculator(CalcABC)`，`MODEL_NAMES=('nequip-oam-xl',)`、`MODEL_ENERGY_UNIT='eV'`、仅 numerical Hessian、无 PBC/电荷支持）；maple 子进程 env 注入（`MAPLE_CALCULATOR_PLUGINS=maple_nequip_plugin` + `PYTHONPATH=mlip_agent/plugins`）经 `runner.run_compute(extra_env=...)` 实现，**严格以 JobSpec 选择插件模型名为条件**（`maple_tools._maple_extra_env`，job_spec.json 为事实来源，手写 inp 回退解析 `#model=` 行）；其余作业 env 一字节不变，`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` 对插件路径同样生效。
- **机制**（按审批修正措辞）：env 设置后插件模块在 maple 进程内**首个 calculator 构建时**即被 import 注册（`load_calculator_plugins_from_env`）；实例化按 `#model=` 名字匹配，非插件模型行为完全不变。
- **单位**：插件 `calculate()` 产出 eV，eV→Hartree 由 `_finalize_results` 统一换算（禁止自行乘 EV2HARTREE）；B11 实测 −0.4938115754 Ha 与 B5 pin 换算值 −0.4938115754172834 Ha 差 1.7e-11 Ha。B5 直算路径与其 eV pin 保持原样。
- **复查义务**：插件与 D1-1 **共享** nequip 升级时 `_from_saved_model` 半私有 API 复查义务——两者加载同一 zip 的同一入口（`nequip.integrations.ase.NequIPCalculator._from_saved_model`），nequip 升级时插件与 `app/compute/scripts/nequip_ase.py` 会同时失效，必须同步复查。
- **后果**：NequIP 可参与 MAPLE 的 opt/freq/scan/md 编排；freq 走数值 Hessian（每自由度两次 forward），大分子很慢，MODEL_CATALOG 已标注 freq 默认仍引导至 ANI/UMA。回滚 = git revert。

## D7-2 live 首轮（deepseek-v4.1-flash，4/11）暴露的 5 类任务集/指令缺陷与修复

- **性质**：这是任务集迭代与 agent 指令补强，**不是放松判定**——所有数值 pin、容差、金标准几何来源（D5-3）一律不变。
- **缺陷与修复**：
  1. **B4/B5/B11 坐标缺失**（数值错误/多作业）：prompt 只说「水分子」，live 每任务全新会话，LLM 走 `build_structure(name)` 得 ase 默认几何，能量与 pin 对不上（B5 差 3.9e-2 eV、B11 差 1.4e-3 Ha）。修复：把金标准同款 H2O 内联坐标（`benchmarks/references/B5_h2o.xyz`，O−H 0.96 Å）直接嵌入三条 prompt（参照 B1 已内嵌 22 原子坐标的做法）。
  2. **B10 跨任务引用失效**（0 作业）：prompt 写「22 原子内联坐标同 B1」，但 live 无 B1 上下文，LLM 正确地回答「缺坐标」而未计算。修复：把 B1 的 22 原子坐标完整嵌入 B10 prompt，任务自包含。
  3. **B9 auto_answer 与断言自相矛盾**（追问后仍计算）：旧 auto_answer 提供了完整信息（模型/电荷/多重度/体系），agent 问完拿到答案自然继续算，与「不得擅自提交计算」的断言冲突。**B9 行为语义澄清：追问后用户拒绝提供信息（auto_answer 改为「这些信息我暂时没有，先不要计算」），则必须停止、不得计算**。断言维持「问了 + 未计算」（suspended_with_tool=ask_user_question + 三个 tool_not_called + job_count=0）；删除 `output_contains: ANI-2x`——该断言耦合旧 auto_answer 文本，拒绝式答复下不再成立，且最终措辞本就不应 pin 死。canned B9 剧本最终回答同步改为「确认收到 + 明确不提交计算」（仍回显用户答复文本）。
  4. **B7 短 MD 缺 log_every**（md_steps=0）：MAPLE 默认 log_every=100，20 步 MD 的 .out 无逐步能量行。修复：orchestrator/task_parser 指令增补「md.steps < 200 必须 log_every=1」（D5-1 已有字段，补的是 LLM 侧知识）。
  5. **B8 扫描点数 off-by-one**（6 点而非 5 点）：MAPLE `S i j delta n_steps` 产出 n_steps+1 个点。修复：指令增补「用户要 N 个点 → n_steps = N-1」。
- **指令层附加规则**（instructions.py）：显式坐标必须用 inline_coords 原样使用、禁止改用分子名重建几何；UMA 直接用本地 checkpoint（checkpoint_path 留空即解析到本机 uma-s-1p1.pt），不先试不存在的默认路径；每个任务只提交必要作业，失败先分析修正再交，避免重复失败作业。

## D7-3 live 二轮（8/11，SR 72.7% / CAR 70%）剩 B4/B6/B8 的输入校验与指令修复

- **性质**：同 D7-2——修的是 agent 知识与输入校验，judge 与金标准（数值 pin、容差、几何来源）一律未动。根因均用 live run 记录（`benchmarks/runs/20261002T110337/judge_report.json`）实证。
- **B4（job_count=2）**：task_parser 输出 `uma_size="s"`（缩写），MAPLE 报 `Unsupported UMA size: 's'`，首作业失败、重试才成功。双管齐下：
  1. `jobspec.py` ModelSpec 加 `uma_size` field_validator：完整枚举值（uma-s-1p1/uma-s-1p2/uma-m-1p1，与 MAPLE `SUPPORTED_UMA_SIZES` 对齐）原样通过；常见缩写归一化（s/uma-s→uma-s-1p1，m/uma-m→uma-m-1p1，1p2→uma-s-1p2）；无法识别时抛含完整允许值清单的错误——错误经 write_maple_inp 的 ToolResult 回给 LLM。
  2. instructions 的 task_parser 段写明 uma_size 必须是完整枚举值、不要输出缩写。
- **B6（opt→freq 链式，7 个作业）**：实证三种踩坑——(i) 正确写法的 inp 写好后没有立即 run，转去试别的变体；(ii) 用内联坐标做 freq（MAPLE freq 要求 XYZ 文件引用）；(iii) 试 `method=both` 触发 MAPLE NotImplementedError。修复：
  1. `jobspec.py` JobSpec 加校验：freq + 内联 coords 提前报错「freq 任务的 structure 必须用 xyz_file」，附 opt→freq 链的取路径指引。
  2. instructions（orchestrator + task_parser）加「链式工作流」规则：freq 的 xyz_file 路径取自 opt 作业 run_maple 返回摘要里 parsed.final_xyz（经 build_structure(inline_coords) 落盘）或 opt 作业目录的 input_opt.xyz；freq 只用 method=mw（mw 已含 ZPE 与热化学量）；写好 inp 后立即 run_maple；失败先读 .out 的 ERROR 行定位再修，不要不诊断就换写法重写。
- **B8（扫描起点错）**：MAPLE rigid scan `S i j delta n_steps` 从初始几何出发逐步改变坐标；用户要「O−H 从 0.9 扫到 1.1」时初始结构的 O−H 必须先设为 0.9 Å，否则 5 个点从 0.96 起步全错。修复 instructions（orchestrator + task_parser）：扫描任务必须先把被扫描坐标调整到起始值（直接修改相关原子坐标，其余原子不动），delta=（终值−始值）/(N−1)，n_steps=N−1。
- **测试**：`test_inp_builder.py` 新增 TestUmaSizeNormalization（合法枚举原样、5 个缩写归一、非法值报错含清单、归一化值正确渲染进 inp）与 TestFreqStructureRequirement（freq+内联报错含 xyz_file、freq+xyz_file 通过）；既有 `test_freq` 改用 xyz_file 构造。

## D7-4 benchmark 模型覆盖扩展：B12/B13/B14 补齐 ani1x / aimnet2 / maceoff23s

- **动机**：MODEL_CATALOG 6 个模型此前只有 ani2x / uma / nequip-oam-xl 有验收任务；为其余三个 MAPLE 内建模型各补一条 H2O sp 任务（与 B4 同构：金标准内联坐标 `B5_h2o.xyz` 同款、charge 0 / mult 1、能量 pin ±1e-6 Ha；三个 .out 均打印 10 位小数，无需放宽到 B8 的 1e-5）。
- **pin 值（2026-10-02 同机复现，均逐位可复现）**：
  - B12 ani1x：**−76.3884429293 Ha**
  - B13 aimnet2：**−76.4770339061 Ha**
  - B14 maceoff23s：**−76.4797837970 Ha**
- **checkpoint 获取（三模型路径各不相同）**：
  - ani1x / aimnet2：首次运行由 MAPLE 自动从 `Wayne7815/MAPLE_models`（pinned revision d5d8eb9）下载到 `MAPLE/maple/function/calculator/model/`，属 MAPLE 运行时行为；落盘后 `HF_HUB_OFFLINE=1` 下同机复现逐位一致。注意 runner 子进程 env 白名单不带代理变量，**checkpoint 必须先缓存好**，否则 canned/live 子进程下载失败。
  - maceoff23s：**无自动下载**——`MACECalculator.REQUIRES_LOCAL_MODEL_FILE=True` 且 HF 仓库只有 maceoff23m/egret1s。已用官方权重（ACEsuit/mace-off 的 `MACE-OFF23_small.model`）在 scratch 环境（torch 2.8.0 CPU + mace-torch 0.3.10，与 COMPUTE_VENV 的 torch 2.8.0 同版本）经 `e3nn.util.jit.compile` 封装为 scripted wrapper（`scripts/make_maceoff23s_wrapper.py`，forward 契约同 mace `LAMMPS_MACE` 能量路径：per-graph local energy，MAPLE 侧 `.sum()`；`MODEL_ENERGY_UNIT='eV'` 由 `_finalize_results` 统一换算，与 D7-1 同机制），放置到 `MAPLE/maple/function/calculator/model/maceoff23s.pt`（模型工件，非源码改动；同机三次复现逐位一致，H2O 能量与 wrapper 自检 eV 值 / 27.211386245988 吻合）。
- **模型特有注意事项**：aimnet2 支持 charge/mult（charge 0 / mult 1 合法，`SUPPORTS_CHARGE_MULT`）；maceoff23s 为有机中性分子力场（H/C/N/O/F/P/S/Cl/Br/I 十元素），不支持电荷/多重度（`SUPPORTS_CHARGE_MULT=False`），H2O 合法无需换体系；三模型 H2O 均在元素支持范围内，无体系替换。
- **任务/剧本**：tasks.yaml B12–B14 断言对齐 B4（final_status / tool_called run_maple / job_count=1 / job_type=sp / energy pin）；canned 剧本复用 B4 结构（`_orch_maple_sp_golden` 工厂），`_spec_from_delegated` 增加 ani1x/aimnet2/maceoff23s 模型名分支。test_benchmarks.py 任务数断言同步 B1–B14；README/PLAN/根 AGENTS.md 的「B1–B11」口径同步 B1–B14。

## D7-5 TICKET-0003 追认附带条件：maceoff23s wrapper 的复查义务与能力边界

- **背景**：maceoff23s.pt 自制 wrapper 置入 MAPLE 模型目录先执行后补票，审批以追认了结（B 类，见 `tickets/TICKET-0003-maceoff23s-model-file.md`），附带条件落实如下。
- **复查义务（条件 2）**：MAPLE mace calculator 契约（`MACECalculator` 对 wrapper 的属性/forward 签名要求）变动，或 COMPUTE_VENV torch 升级时，必须用 `scripts/make_maceoff23s_wrapper.py` 重新生成 wrapper 并复验 B14（canned 复跑，能量 pin −76.4797837970 Ha）。scripted 产物依赖 torch 版本，torch 升级后旧 .pt 可能无法加载或数值漂移。
- **能力边界（条件 3）**：maceoff23s 仅 sp 路径经 B14 验收；wrapper forward 固定 `compute_force=False`，**forces 路径未做对照验证**。在完成 forces 对照验证（如与 mace-torch ASE calculator 直接比对）之前，不得新增 maceoff23s 的 opt/freq/md benchmark 任务；MODEL_CATALOG notes 已同步标注「仅 sp 已验收（forces 路径未验证，opt/freq/md 勿用）」。

## D7-6 TACE-OAM-L 直算后端接入与 B15 pin 值（TICKET-0004）

- **决定**：复刻 nequip_direct 模式接入 TACE——`app/compute/scripts/tace_ase.py`（job.json/result.json 契约与 nequip_ase.py 一致，ASE eV/Å 单位制）+ `run_tace` 工具（`tool/mlip.tace/run_tace`，"ask"，D2-1 全限定 tool_id）+ MODEL_CATALOG `tace-oam-l` 条目 + benchmark B15（H2O sp，断言对齐 B5）。MAPLE 零改动、COMPUTE_VENV 零装包。
- **pin 值**：B15 pin **−13.926433563232422 eV**（±1e-6）。**注意与 TICKET-0004 提案的预估值 −13.933330535888672 eV 不同**（Δ≈6.9e-3 eV）：预估值无法在本机复现——cuda/float32（模型默认）、cuda/float64、cpu、matscipy+center(vacuum=10)、ase+center(vacuum=10) 各变体均落在 ~−13.9264 eV。按前置条件 2（pin 前同机连续 ≥2 次逐位一致），以交付脚本实际产出且逐位可复现的值为准。
- **逐位一致证据（2026-10-02，RTX 3090 / cuda / float32）**：同机连续 2 次运行 `tace_ase.py`（/tmp/mlip_golden_b15.2jFHcH/b15_out{1,2}），能量 `repr` 均为 `-13.926433563232422`，forces 列表逐元素相等（脚本 assert 通过）。金标准工件：`benchmarks/references/B15_job.json` / `B15_result.json` / `B15.json`（几何复用 B5_h2o.xyz）。
- **backend 固定理由（前置条件 1）**：`TACEAseCalc` 默认 `neighborlist_backend='matscipy'`，对零胞（非周期）分子 matscipy 求最近邻会 Singular matrix；脚本显式固定 `neighborlist_backend='ase'`，并写入 result.json 顶层 `neighborlist_backend` 字段（B15.json 同步记录）。`versions` 字段含 tace 0.2.0 / torch 2.8.0+cu128 / ase 3.27.0 / device / dtype（前置条件 2）。
- **元素覆盖（前置条件 5，按 checkpoint type map 实测）**：`readout_fn.representation.atomic_numbers` 共 89 种——Z=1–83（H–Bi）加 Z=89–94（Ac–Pu），缺 Po/At/Rn/Fr/Ra；分子体系（非周期），仅 sp/opt 经 B15 验收，freq/scan/md 勿用。
- **复查义务**：tace 升级时需复查 `TACEAseCalc(model=..., device=..., neighborlist_backend=...)` 签名（当前 tace 0.2.0，`tace/interface/ase/calculator.py`）并复跑 B15；runner 作业收集按 result.json versions 有无 `tace` 键区分 tace/nequip 直算作业。回滚 = git revert。

## D7-7 DPA-4 阶段 1 零足迹验证结论：torch 2.8 可行（须源码重编译 deepmd-kit），H2O_H2O_PD 分支（TICKET-0005）

- **结论**：DPA4-beta.pt（SeZM 23 分支 multitask checkpoint）在 torch 2.8.0+cu128 / Python 3.12 scratch venv（/tmp/dpa4_venv，已删）上**可加载、可推理、数值正常**，阶段 2 建议「可行」。完整记录见 `resource/dpa4_stage1_report.md`（含 pip freeze、构建配方、四问答案）。本阶段未改 mlip_agent 代码、未加 MODEL_CATALOG、未 pin benchmark（前置条件 4）。
- **关键机制发现**：PyPI 的 deepmd-kit 3.2.0 二进制 wheel 针对 torch 2.11 编译（`run_config.ini` 内 `PT_VERSIOn = 2.11.0`），import deepmd.pt 即因 ABI 不符 raise——必须按官方提示 `--no-binary deepmd-kit` 源码重编译（DP_VARIANT=cuda + /usr/local/cuda-12.8，约 4.5 min）。另需单独补装 mpich（import 期 libmpi）与 vesin+vesin-torch（SeZM 邻居表强制依赖），均不触发 [torch] extra；运行期需 `LD_LIBRARY_PATH=/usr/local/lib64`（gcc 11.3 编译产物要 GLIBCXX_3.4.29）。
- **四问摘要**：① DeepEval/DeepPot 直接加载训练 checkpoint（`head=` 选分支），无需 restart/导出；② head 固定 `H2O_H2O_PD`（H2O 领域专属分支，input.json data_dict 实证；checkpoint 分支名与目录内 input.json 有两处差异，以 checkpoint 为准），fparam 无维度（dim_fparam=0），chg_spin 固定 `[0.0,1.0]`（显式传与走默认逐位一致）；③ H2O（B5_h2o.xyz）能量：GPU −15.710772596299648 eV（跨进程漂 ~2e-7）、CPU+OMP=1 **−15.710773446015082 eV（跨进程逐位一致，sha256 b82700b7…）**；④ SeZM+e3nn 0.6.0 在 torch 2.8 上零 warning 零报错。
- **pin 策略建议（阶段 2 已裁定，见 D7-8）**：能量在固定设备+线程配置下逐位可复现，但**力在任何配置下均不能跨进程逐位复现**（~1e-7 eV/Å，疑似邻居排序/scatter 累加顺序非确定，根因未定位）——benchmark 只 pin 能量（CPU+OMP=1 金标准或 GPU ±1e-5 容差），不 pin 力。
- **清理**：/tmp/dpa4_venv（6.9 GB）已删除，零足迹；阶段 2 按报告 §1 配方 ~15 min 可重建常驻 venv。

## D7-8 DPA-4 正式接入（阶段 2）：常驻 venv + run_dpa4 + B16 pin（TICKET-0006）

- **决定**：按 TICKET-0005 预设形态正式接入 DPA-4——独立常驻 venv + `app/compute/scripts/dpa4_ase.py` 子进程脚本（job.json/result.json 契约对齐 tace_ase.py）+ `run_dpa4` 工具（`tool/mlip.dpa4/run_dpa4`，"ask"，D2-1 全限定 tool_id，capability 常量 `CAP_DPA4` 在 registry.py/permissions.py 共享）+ MODEL_CATALOG `dpa-4` 条目 + benchmark B16（H2O sp，断言对齐 B15）。MAPLE 零改动、COMPUTE_VENV 零改动、mlip_agent/.venv 零改动。
- **常驻 venv 位置（裁定 1）**：`/root/soft/ml_potential/dpa4_venv`（mlip_agent 之外、与 COMPUTE_VENV 平级；根 .gitignore 白名单天然排除）。实测磁盘占用 **6.8 GB**（torch+CUDA 库占大头；构建期依赖 cmake/ninja/scikit-build-core 等只进 /tmp scratch 构建 venv，产出 deepmd-kit wheel 后即删除，未装入常驻 venv）。重建：`scripts/make_dpa4_venv.sh`（幂等：显式校验目标路径非 COMPUTE_VENV/mlip_agent/.venv 后 rm -rf 重建；网络命令一律 `env -u *_PROXY` 剥离本机 socks 代理；实测重建约 5 min）。Settings 新增 `dpa4_venv`（env `DPA4_VENV`）+ `dpa4_model_path`（env `DPA4_MODEL_PATH`），`dpa4_python` 为 `dpa4_venv/bin/python` 派生属性。
- **pin 口径（裁定 2）**：CPU + OMP_NUM_THREADS=1 逐位金标准 **−15.710773446015082 eV ±1e-6**，只 pin 能量、不 pin 力。`dpa4_ase.py` CPU 模式（默认 device=cpu）进程内强制 `os.environ["OMP_NUM_THREADS"]="1"`（import torch 之前）+ `torch.set_num_threads(1)`；`run_dpa4` 以 `gpu=False` 提交（不过 GPU semaphore，裁定 4）。**GPU 值约 −15.7107726 eV，与 CPU pin 口径不可混用**（TF32/AMP 路径差异，见 D7-7）。
- **逐位一致证据（2026-10-02）**：① 脚本级同机连续 2 次 sp（/tmp/dpa4_test{1,2}），能量 repr 均为 `-15.710773446015082`；② canned B16 同机连续 2 次（benchmarks/runs/20261002T221516、20261002T221526），judge 全 PASS 且 run_record 能量 repr 均为 `-15.710773446015082`、backend=dpa4。金标准工件：`benchmarks/references/B16_job.json` / `B16_result.json` / `B16.json`（几何复用 B5_h2o.xyz）。
- **head 默认与 charge_spin（前置条件 3）**：job.json `head` 默认 `H2O_H2O_PD` 允许覆盖（checkpoint 实际 23 分支之一）；`charge_spin` 固定 `[0.0,1.0]` 不接受覆盖（中性闭壳层；模型默认即此值，显式传与走默认逐位一致，D7-7）。result.json versions 记录 deepmd-kit 3.2.0 / torch 2.8.0+cu128 / e3nn 0.6.0 / vesin 0.6.2 / device / omp_num_threads / torch_num_threads。
- **力的已知非确定性（裁定 3，根因未定位）**：力在任何配置下均不能跨进程逐位复现（~1e-7 eV/Å，疑似邻居排序/scatter 累加顺序非确定，D7-7 已记录，本阶段未定位根因）——MODEL_CATALOG 标注「sp 已验收（CPU pin）；opt 可交互使用、力不逐位稳定不作 pin；freq/md 不支持」。opt 路径经 LBFGS 冒烟（H2O 4 步收敛）可用但无 benchmark pin。
- **分支命名差异（前置条件 6，以 checkpoint 为准）**：checkpoint 实际分支名与 `DPA-4/input.json` 有两处差异——`Alloy_tongqi`→`Alloy_APEX`、`Omat24`→`OMat24`（目录内 input.json 非最终训练版）；head 覆盖时分支名一律以 checkpoint 内嵌名单（23 分支）为准，已写入 MODEL_CATALOG notes 与 dpa4_ase.py docstring。
- **LD_LIBRARY_PATH（前置条件 1）**：`LD_LIBRARY_PATH=/usr/local/lib64` 经 `run_dpa4_impl` → `run_compute(extra_env=...)` 无条件注入（gcc 11.3 编译产物需 GLIBCXX_3.4.29）；`-static-libstdc++` 重链加固**未做**（可选项，注入通道已足够，后续如需可补并回填本条）。
- **runner 作业收集**：`benchmarks/runner.py collect_jobs` 的 backend 判定扩展为 versions 标识键 `deepmd-kit`→dpa4、`tace`→tace、其余→nequip。
- **复查义务**：deepmd-kit / torch 升级时须按 `scripts/make_dpa4_venv.sh` 重建常驻 venv 并复跑 B16（连续两次逐位 hit pin）；deepmd-kit 源码重编译配方与 ABI 注意事项见 D7-7 / dpa4_stage1_report.md §1。回滚 = git revert + 删 `/root/soft/ml_potential/dpa4_venv`。

### 常驻 venv pip freeze（uv pip freeze，2026-10-02；deepmd-kit 行为源码重编译 wheel，sha256=72636f624a8b2ee00be4e51008de10ac5514bdd5f362a922e9fc5b794321e037）

```
annotated-types==0.8.0
array-api-compat==1.15.0
ase==3.29.0
bracex==3.0.1
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
dargs==0.5.1
deepmd-kit==3.2.0
deprecated==1.3.1
e3nn==0.6.0
einops==0.8.2
filelock==3.32.3
flexcache==0.3
flexparser==0.4
fonttools==4.66.1
fsspec==2026.7.0
h5py==3.16.0
jinja2==3.1.6
kiwisolver==1.5.1
lmdb==2.3.0
markupsafe==3.0.3
matplotlib==3.11.2
mendeleev==1.3.0
ml-dtypes==0.6.0
mpich==5.0.2
mpmath==1.3.0
msgpack==1.2.3
networkx==3.6.1
numpy==2.5.3
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
opt-einsum==3.4.0
opt-einsum-fx==0.1.4
packaging==26.3
pandas==3.0.6
pillow==12.3.0
pint==0.24.4
platformdirs==4.12.2
pydantic==2.13.5
pydantic-core==2.46.5
pyfiglet==0.8.post1
pygments==2.21.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
pyyaml==6.0.3
scipy==1.18.1
setuptools==78.1.0
six==1.17.0
sqlalchemy==2.1.2
sympy==1.14.0
torch==2.8.0+cu128
triton==3.4.0
typeguard==4.6.0
typing-extensions==4.16.0
typing-inspection==0.4.4
vesin==0.6.2
vesin-torch==0.6.2
wcmatch==11.0.1
wrapt==2.5.0
```

（cmake / ninja / scikit-build-core / setuptools_scm / hatch_fancy_pypi_readme / dependency_groups / hatchling 等纯构建期依赖已按前置条件 2 剔除，只存在于 /tmp scratch 构建 venv，用后即删；matplotlib/pandas/sqlalchemy 等随 mendeleev/ase 运行期依赖链带入。）

## D8-1 Benchmark v2 重构：tier 信息梯度 + 覆盖率/精度/效率三轴报表（PLAN §3.2）

- **背景**：v1 任务集（B1–B16）全部是「给全信息 + pin 值」的单档执行保真度测试，缺信息梯度、恢复能力、效率三个评测维度。借鉴 "Can Autonomous LLM Agents Execute Multireference Quantum Chemistry Calculations?"（arXiv:2609.13357）的三档信息梯度（Baseline/Decision/Replication）、覆盖率与精度分离、尝试次数/成本指标设计。
- **决定**：
  1. **tier 模型**：每任务加 `tier`（guided/recovery/unguided）与 `min_jobs`（效率基准）。B1–B16 全部 guided；新增 B17/B18（recovery，首试必失败测诊断修复）与 B19（unguided，不给模型须先追问）。
  2. **恢复类任务复用既有金标准，零新增 pin**：B17 复用 B3 频率 pin（±20 cm⁻¹ 对起始点不敏感，D5-3）——首试 write_maple_inp(freq+内联坐标) 被 D7-3 校验拒绝后改走 opt→freq 正确链；B18 复用 B7 口径——首跑 md 不带 log_every（无逐步能量行，D5-1）后 log_every=1 重跑；B19 复用 B6 opt 断言（追问注入答复后完成乙醇优化）。
  3. **only_mode 断言作用域**：恢复路径断言（tool_call_failed、job_count=2、md_steps_max）标 `only_mode: canned`——canned 剧本回放失败首试以验证框架支持恢复；live 下 agent 已有指令知识（D7-2/D7-3），一次做对是最优行为，不强制失败首试，其表现由 efficiency.first_try 反映。job 索引支持负数（-1=最后作业）以兼容两种模式的作业序列。
  4. **judge 三轴报表**：summarize 在既有字段（execution_success_rate/numeric_accuracy/all_passed，ui_runner 与测试依赖，不变）之外新增 by_tier/by_backend/by_job_type/by_model 分组比率、error_stats（numeric 断言结构化 abs_error 的 mean/max）、efficiency（first_try_rate/total_jobs/total_failed_jobs），均 informational 不影响退出码。
  5. **runner**：run_record 新增 failed_tool_calls（从 TOOL_CALL_RESULT 事件 content 的 ok 字段提取）与 metrics 块（n_jobs/n_failed_jobs/duration_s）；tool_calls 保持 list[str] 兼容。
- **金标准零改动声明**：`benchmarks/references/` 未新增/修改任何文件，B1–B16 的 prompt 与断言语义逐字未动（仅追加 tier/min_jobs 两字段），按工单规则无需开工单；判据：`git diff benchmarks/references/` 为空。
- **验收**：pytest 全绿（含 v2 断言/报表单测）；canned B1–B19 全 PASS 且 judge_report 含三轴块；live 无 key 正确 skip。

## D8-2 Benchmark v3：GMTKN55 反应能垒套件（PLAN §3.3）

- **定位**：B1–B19（v2，D8-1）重新定位为**框架级 canned 回归集**；agent 能力的三档知识梯度评测由 v3 反应套件承担。v3 对标 arXiv:2609.13357 的完整评测形态。
- **数据集选型**：Transition1x 被否决（参考值仅 DFT 级自洽值，不够高精度）。改选 **GMTKN55 BH76 子集**（Goerigk et al., PCCP 2017）：① AIMNet2 官方论文（Chem. Sci. 2025, d4sc08572h）的原生 benchmark，MAE 可与已发表数据对照；② 参考值为高级别 WFT；③ 含反应物/产物/TS 三套几何；④ 参考值是能量差（能垒/反应能，kcal/mol），跨理论级别比较有物理意义（绝对能量参考态差异无意义）。提取 6 反应（`scripts/make_gmtkn_refs.py`，数据源 GitHub grimme-lab/GMTKN55 clone 190MB 不入库，提取子集入库）：R1 HCN→HNC 异构化、R2 H+N2O、R3 OH+CH4、R4 F−+CH3Cl(S_N2)、R5 CH3+C2H4、R6 O(3P)+CH4——覆盖单分子/双分子、中性/阴离子、mult=1/2/3，体系 ≤6 原子控制 freq/TS 成本。
- **套件模型 aimnet2**：BH76 含 Cl 与带电/自由基体系，ani2x 不支持；aimnet2 支持 charge/mult 且是其原生 benchmark。
- **三档在 prompt 层**（baseline/decision/replication 为同一反应同一参考值的三个 prompt 变体，decision 档由 runner 注入 ladder.md 全文），不做 instructions 消融——与论文「信息条件变化」对齐且不动框架行为。
- **指标语义**：MAE 类参考量「总是计分」（记录 abs_error 进聚合，不因超差 fail，作业缺失才 unscored）；coverage  unscored 计入覆盖损失不污染 MAE（对齐论文 verified/unidentified）；attempts = actual_jobs vs min_jobs（R1–R6 为 7/10/11/9/9/10）；cost = Σduration_s + total_jobs。failure_analysis 归因类别映射 ladder Rule A–I。
- **物种归属机制**：suite_judge 按（元素组成、charge、与参考 xyz 同序 RMSD ≤1.0 Å）把作业归属到物种/TS——R1 三个物种同为 HCN 异构体（元素组成相同），RMSD 判别是必要机制。
- **框架 ts 支持**：JobType.TS + TsSpec(method=prfo 等 6 种) + `#ts(method=)` 渲染 + parse_ts（fixture MAPLE/examples/ts/prfo/inp1.out，收敛判定取最后准则块四行全 Yes）。
- **决策阶梯**：`benchmarks/reaction/ladder.md` Rule A–I（执行层/输入校验/opt 不收敛/极小虚频/scan 异常/TS 搜索/单位制/方法学偏差/成功确认）。
- **冒烟验证（2026-10-03，canned R1-repl，真实 GPU aimnet2）**：7 作业链全收敛，prfo 在 aimnet2 上一次收敛；coverage 9/9；**MAE 3.54（max 5.32）kcal/mol vs WFT 参考**——AIMNet2 论文 GMTKN55 口径下能垒类子集的典型误差量级，证明指标链产出物理上有意义的数值。
- **复查义务**：GMTKN55 参考值静态不变无需复查；aimnet2 升级时复跑 R1-repl 冒烟确认 MAE 量级；live 三档评测需 LLM key，运行手册见 benchmarks/reaction/README.md。

## D9-1 周期性体系（PBC）落地：直算三路 + UMA 经 MAPLE（2026-10-03）

- **背景**：此前项目假设「MAPLE/各后端不支持 PBC，v1 不做周期体系」（PROJECT.md 边界）。2026-10-03 对 MAPLE 仓库与各框架源码/在用 checkpoint 逐一核实后推翻：MAPLE 原生有 `#pbc(a,b[,c][,α,β,γ])` 关键字与 extxyz Lattice 解析；NequIP/TACE/DPA-4 框架与在用 checkpoint（OAM-XL、OAM-L、DPA4-beta 的 ~20 个周期域分支）本就面向周期材料训练。
- **决定**：
  1. **不做 cell relaxation**（用户裁定）→ 不接 stress 输出，opt 一律只动原子坐标；MAPLE/UMA 无 stress 的限制（_uma_calculator.py:376-379）因此不构成阻塞。
  2. **直算三路接 PBC**（零工单，mlip_agent 内部）：周期结构统一用带 `Lattice=` 头的 extxyz（ase.io.read 自动设 cell+pbc）；`dpa4_ase.py` 传 cells（deepmd 一律按全三维周期处理）；`nequip_ase.py` 原生可用仅加校验；`tace_ase.py` 邻居表按周期性分派（分子 ase / 周期 matscipy）并硬报错两类危险输入（带胞非周期、混合 pbc——TACE 的 get_neighbourhood 会把非平凡 cell 静默升级为 3D 周期，neighbour_list.py:61-67）。
  3. **UMA 经 MAPLE**：JobSpec `structure.cell`（2/3/6 值）渲染 `#pbc(...)`；PBC 门禁 fail-fast（cell 存在时模型必须 uma 且 uma_task ∈ omat/oc20/oc22/oc25/omc/odac，omol 与 PBC 互斥——前置 MAPLE 自身的双重校验）。
  4. **build_structure 支持 cell**：3/6 值晶胞参数 → extxyz（Lattice+pbc="T T T"）；source_type=xyz 且源文件自带 Lattice 头时原样复制不丢周期信息。
  5. **ANI 不改 PBC**（用户裁定）；MACE/AIMNet2 的 MAPLE wrapper PBC 走工单（TICKET-0008/0009）。
- **pin 口径**：B23–B26（Si 金刚石原胞 sp，fcc 初基矢量 a=5.431 Å，references/si_diamond.extxyz）：DPA-4/OMat24 CPU −10.8385356665 eV（逐位）、NequIP GPU −10.8384650332 eV（噪声 4.4e-7，容差内）、TACE GPU −10.8458156586 eV（逐位）、UMA/omat −0.3981593280 Ha（2 次差 2.2e-9 Ha）。同结构四后端互证能量差 <0.01 eV、对称点 |F|max<1e-7 eV/Å。
- **教训记录**：首次试算误用「立方胞 + 2 原子」的非法 Si 金刚石构型（正确原胞是 fcc 初基矢量），三个后端同时给出对称点非零力（~0.4 eV/Å）暴露了结构错误——对称点力检查是周期结构输入的有效自检手段。
- **验收**：pytest 200+ 全绿（新增 tests/test_pbc_scripts.py 8 项、inp_builder TestPbcCell 9 项、build_structure cell 5 项）；canned B1–B26 23/23 PASS。

## D9-2 MACE-MPA-0 medium 经 MAPLE 插件接入（B20，2026-10-03）

- **决定**：新增模型 macempa0（MACE-MPA-0 medium，9.1M 参数，MPTrj+sAlex 晶体域，MIT 许可），走 MAPLE 插件机制（plugins/maple_mace_plugin.py，TICKET-0002 先例），零 MAPLE 源码改动。
- **转换**：scripts/make_macempa0_wrapper.py（scratch venv torch 2.8.0+mace-torch 0.3.10，e3nn jit.compile）；源 mace-mpa-0-medium.model SHA256 75428afe…9638；产物 /root/soft/ml_potential/mace/macempa0.pt SHA256 03c1b56e…31ef（保留官方 float64）。wrapper vs 官方 ASE calculator 能量差 7.5e-14 eV。
- **接线泛化**：maple_tools.py 的单插件常量泛化为 `_PLUGIN_REGISTRY`（模型名 → 插件模块/env）；jobspec 加 DEFAULT_MACEMPA0_MODEL_PATH 与缺省填充；`#model=macempa0(model_path=...)` 渲染。
- **pin**：B20 H2O sp = −0.5064442892 Ha（3 次逐位；材料域模型算分子，纯数值 pin 不代表物理精度）。仅 sp 已验收，forces/opt/freq/md 验收前勿用（同 D7-5 口径）。
- **验收**：pytest 218 全绿；canned 24/24 PASS。

## D9-3 DPA4C-Air（family=dpa4c）接入 DPA-4 直算路径（B21，2026-10-03）

- **背景**：DPA4C 是 DPA4 族的紧凑高吞吐分支（无 message passing，压缩 CUDA 路径）。DPA4C-OMat 尚未发布（HF 仓库为空），DPA4C-OMol 已发布 5 档；接入 Air 档（0.63M 参数，用户裁定）。
- **决定**：job.json 新增显式 `family` 字段（缺省 `dpa4-beta` 保持现状；`dpa4c` 走 `DeepPot(model)` 不带 head）。**关键陷阱**：DeepPot 对 DPA4C 会静默忽略 head 参数（连不存在的 head 名都照常加载出逐位相同结果），自动嗅探不可靠，显式 family 是防线；dpa4c 族下显式传 head 直接报错。
- **数值口径**：DPA4C-OMol25 输出**总能量**（H2O ≈ −2080 eV），与 DPA4-beta 的参考能约定（B16 −15.71 eV）不互比；B21 pin = −2079.8665150032502 eV（CPU+OMP=1，2 次逐位），GPU 对照差 1.2e-7 eV（不 pin）。许可 CC BY-NC 4.0（非商业）。
- **验收**：tests/test_dpa4c_script.py 7 项（pin 读 references/B21.json）+ B16 回归未破；canned B21 PASS。

## D9-4 MAPLE MACE calculator PBC 支持（TICKET-0008，2026-10-03 实施）

- **改动**：`_mace_calculator.py` 新增 `_periodic_radius_graph`（matscipy `neighbour_list('ijS')`，方向约定逐行镜像上游 mace-torch 0.3.10 `get_neighborhood`：`edge_index=[sender;receiver]`、`shifts=unit_shifts@cell`）；唯一分支条件 `atoms.pbc.any()`，非周期路径逐字节不变；`SUPPORTS_PBC=True`（插件 macempa0 继承并显式翻转）。checkpoint 零改动（周期前向本就在 scripted wrapper 内）。
- **验证**：Si 原胞 MAPLE 路径 vs 官方 mace_mp ASE 路径能量差 0.000e+00 eV；B14/B20 逐位回归；B27 pin −0.3978120290 Ha（3 次逐位）。
- **边界**：PBC 仅承诺 sp/opt（freq/hessian 无 Γ 点投影未验证）；MACE-OFF23/MPA 训练域分子/晶体有别，分子晶体用 MACE-OFF、无机晶体用 MACE-MPA。
- **副作用修复**：B26_sp.inp 参考文件的 XYZ 行行内注释会被 MAPLE `_parse_charge_mult` 误判为多路径——已修正（MAPLE 的 XYZ 引用行不支持行内注释）。

## D9-5 AIMNet2-NSE TorchScript 转换置入 MAPLE（TICKET-0007，2026-10-03 实施）

- **产物**：`MAPLE/maple/function/calculator/model/aimnet2nse.pt`（SHA256 `69f687b8841bfc8ef1a5fe35e371cdc0354757990e500a9b470aef4a5b3e95ab`），由 `scripts/make_aimnet2nse_ts.py` 一次性生成；源权重 HF `isayevlab/aimnet2-nse` ensemble_0.safetensors（SHA256 `c65c7e1c...d723b6`，脚本内置校验），上游 aimnetcentral MIT 许可。MAPLE 零源码改动。
- **转换要点**：v2 格式把 LRCoulomb/DFTD3 剥离为 calculator 外挂；按 MAPLE 契约重建进 JIT 图——outputs 序 = energy_mlp → atomic_shift(f64) → atomic_sum → srcoulomb（checkpoint 自带）→ lrcoulomb（FULL，`subtract_sr=False`）→ dftd3（纯 torch 复刻 nvalchemiops 0.2.0 kernel 数学，自定义算子无法嵌入因 COMPUTE_VENV 无 aimnet/nvalchemiops）。`ConvSV.forward` 的 CUDA warp 算子分支脚本化时替换为上游等价纯 torch 分支。归档断言自包含（无 `aimnet::` 引用）。
- **数值验收**：scripted 重载 vs 上游 v2 eager（CPU），H2O/CH3/O2 三体系 ΔE ≤ 2.8e-9 eV、ΔF ≤ 6e-8 eV/Å（容差 1e-6/1e-5）。B22 pin：**O2 三重态 sp = −150.4159493214 Ha**（GPU 3 次逐位）；H2O 对照 −76.4767924907 Ha（vs aimnet2 B13 −76.4770339061 Ha，Δ2.4e-4 Ha 属模型代差）。MAPLE 与 aimnet 的 Hartree 常数相差 8e-9 相对量（27.211386245988 vs 27.211386024367243），比对时须用同一常数。
- **复查义务（同 TICKET-0003 条件 2 口径）**：MAPLE AIMNet2 calculator 契约（`_aimnet2_calculator.py` 输入键/属性要求）变动，或 COMPUTE_VENV torch 升级时，必须用 `scripts/make_aimnet2nse_ts.py` 重新生成并复验 B22（O2 三重态 pin −150.4159493214 Ha）。TorchScript zip 非字节确定性，复验以数值对照为准而非文件哈希。
- **能力边界**：开壳层/带电仅 sp 路径经 B22 验收，opt/freq/md 验收前勿用；nbmat_lr 在 coulomb_method='simple' 下按 SR cutoff 5.0 Å 构建（MAPLE 既有口径，与 aimnet2 一致），大分子 LR 项 5 Å 外截断非本次引入。

## D9-6 MAPLE AIMNet2 calculator PBC 支持 + ewald 解禁（TICKET-0009，2026-10-03 实施）

- **改动**（唯一文件 `_aimnet2_calculator.py`，未 commit）：`nblist_periodic_padded` 用 matscipy `neighbour_list('ijS')` 产出与上游 aimnetcentral/nvalchemiops 同约定的 MIC 周期邻居表（padded `(N+1, M)` int32 nbmat + `(N+1, M, 3)` float32 单位晶格 shifts），周期时 data 追加 `shifts`/`shifts_lr`/`cell` 三键喂 checkpoint 内已脚本化的 `calc_distances` MIC 分支与 `coul_ewald`；`atoms.pbc.any()` 为唯一分支条件，非周期路径逐字节不变（B13/B22 逐位回归为证）。`SUPPORTS_PBC=True`；`SUPPORTED_COULOMB_METHODS=('simple','dsf','ewald')`；ewald 仅限周期体系（非周期显式报错）；`coulomb_method` 缺省 None：分子 = simple（旧行为），周期自动切 dsf（对齐上游 `prepare_input` 检测到 cell 的 simple→dsf 自动切换），显式 simple 可强制。
- **对照数值**（H2O 20 Å 立方盒，CPU，MAPLE matscipy 路径 vs 上游 nvalchemiops 路径跑同一 checkpoint，方法直设 JIT 模块）：simple −2081.0461085663 / dsf −2079.8826618866 / ewald −2081.0467415527 eV，三法**逐位一致（Δ=0.0 eV）**；上游 ASE 全流程 −2081.0461092965 eV（其 legacy 内嵌 Coulomb 不可运行时切换，实际跑 simple），差 7.3e-7 eV < 容差 1e-6。
- **pin**：B28（H2O 20 Å 盒 + `#pbc(20,20,20)`，缺省 dsf）= **−76.4342780468 Ha**（GPU 3 次逐位；金标准暂存 /tmp/golden_staging/B28/）；ewald 对照 −76.4770571854 Ha；显式 simple 强制 = −76.4770339061 Ha（与气相 B13 逐位一致）。冒烟：4×H2O 10 Å 胞 −305.7398672163 Ha 量级合理；aimnet2nse PBC 正常。
- **附带 bug 修复**：checkpoint 的 lrcoulomb 挂载点为 `outputs.lrcoulomb`，旧 `_set_lrcoulomb_method` 的 `name == 'lrcoulomb'` 精确匹配永不命中——旧 `coulomb_method='dsf'` 是静默 no-op（模型仍 simple、nbmat_lr 却按 15 Å 构建）。修为 `endswith('.lrcoulomb')` 后 dsf 真正生效；非周期缺省 simple 路径不受影响。
- **能力边界**：AIMNet2 训练域为气相分子，官方明示未对体相/表面验证——周期计算仅分子晶体类体系可信；PBC 仅承诺 sp/opt（freq/hessian 无 Γ 点投影未验证）；混合维度 PBC 显式拒绝（MAPLE `#pbc` 只产生全三维周期）；ewald accuracy=1e-8 为 checkpoint 导出写死；周期邻居表 cutoff > cell/2 时 matscipy MIC 语义已与上游逐位互证（H2O 算例 dsf 15 Å > 盒半宽场景包含在内）。
- **复查义务**：COMPUTE_VENV torch / matscipy 升级时复跑 B13/B22 回归 + B28 pin 对照；回滚 = git revert MAPLE 侧单文件改动。

## D10-1 全部模型 checkpoint 集中迁移至 checkpoints/（TICKET-0010，2026-10-03 实施）

- **背景**：checkpoint 原散落 6 处（`uma/uma_folder/checkpoints/`、`nequip/`、`mace/`、`TACE/`、`DPA-4/(dpa4c/)`、MAPLE 包内 `model/`），用户要求创建单一文件夹单独存放。
- **决定**：新建 `/root/soft/ml_potential/checkpoints/{uma,nequip,mace,tace,dpa4,maple}/`，15 个文件同文件系统 `mv`（rename，字节不变；13 个文件迁移前后 SHA256 逐一一致，uma-m-1p1/uma-s-1 以 inode+size 校验）。MAPLE 侧零源码改动：`MAPLE/maple/function/calculator/model` 由实体目录（git 未跟踪）替换为指向 `checkpoints/maple/` 的**目录级符号链接**——其硬编码 `Path(__file__).parent/'model'` 约定与未来的自动下载（经 symlink 直接落入集中目录）均不受影响。不迁移 aimnetcentral 包内数据（库内相对路径加载，迁移会破坏包结构）。
- **引用更新**：mlip_agent 内全部路径常量（config.py 6 个 Settings 默认、jobspec.py 3 个 DEFAULT_*、mlip_tools/maple_tools、6 个测试文件兜底、make_golden_refs.sh、两个 wrapper 生成脚本、tasks.yaml B21 prompt、canned_scripts B21、.env.example）+ benchmarks/references/ 31 个 golden 文件的内嵌路径（仅路径字符串，数值零改动；judge 与测试均不从中解析路径）+ README/PLAN 现行描述。历史工单、resource/dpa4_stage1_report.md、benchmarks/archive/ 保持原样（历史记录不改写）。
- **验证**：pytest 231 项全绿；canned B1–B28 28/28 PASS（执行成功率与数值正确率均 100%，12 个模型条目全过，含经 symlink 解析的 ani1x/ani2x/aimnet2/aimnet2nse/maceoff23s）。
- **回滚**：按映射表 `mv` 回原路径 + 删 symlink 恢复实体目录 + 反向编辑引用；无金标准重 pin 需求。
- **清单**：`checkpoints/README.md` 维护逐文件来源/许可/SHA256/再生脚本；新增 checkpoint 须先开 tickets/ 工单。
