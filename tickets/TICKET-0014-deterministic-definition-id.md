# TICKET-0014：服务重启后会话无法继续（subagent definition_id 随机化导致 StateError）

- 状态：待审批 ｜ 提出：2026-10-04 ｜ 审批人：____

### 1. 改动对象

`app/agent/harness.py` `build_harness`：为 orchestrator 与 inline 子 agent 的
`AgentDefinition` 传入确定性 `definition_id`。不属于冻结资产。

### 2. 动机与问题描述

web UI 实战（会话 a0b58838）：服务重启后，在用过子 agent（task_parser /
result_validator 委派）的会话里发任何新消息立即 network error。后端日志：

```
a13n_harness.errors.StateError: Inline subagent State child definition
is incompatible with the current Agent.
```

根因：a13n_harness `AgentDefinition.definition_id` 缺省
`field(default_factory=lambda: str(uuid4()))`（builder.py:220），每次
build_harness 生成随机 UUID；持久化的 inline subagent 状态里记录了旧进程的
`child_definition_id`，重启后 `_validate_inline_subagent_state`
（subagents.py:495）比对不一致即拒绝整个 run。

实测：同一进程内两次 `build_harness(Settings(), [])`，orchestrator 与
三个子 agent 的 definition_id 全部不同。

### 3. 影响面分析

B. 框架内新增：只把随机 UUID 换成确定性字符串
（`mlip.orchestrator` / `mlip.subagent.<name>`），不改变任何运行时数值行为。
对既有会话：旧状态里的随机 id 仍与新 id 不匹配（本就不可用），无回退损失；
新产生的会话状态重启后可正常续跑。

### 4. 风险

- definition_id 需全局唯一：本项目的 4 个定义各一个固定串，无碰撞。
- a13n 升级若改变校验语义，确定性 id 同样兼容（只会更宽松）。
- 数值漂移风险：无。

### 5. 回滚方案

`git revert` 对应提交即可（退回随机 UUID 行为）。

### 6. 验证计划

- 新增单测：两次 build_harness 的 orchestrator/子 agent definition_id 完全一致。
- `.venv/bin/python -m pytest tests/ -q` 全绿。
- UI 复验：重启 uvicorn 后在含子 agent 委派的会话里继续发消息，不再 StateError。

### 7. 审批结论

批准（用户指令 2026-10-04，web UI 任务中现场发现并修复）。
实施回填：`build_harness` 为 orchestrator/子 agent 传确定性 definition_id
（`mlip.orchestrator` / `mlip.subagent.<name>`）；新增
`tests/test_harness_definitions.py`；全量 `pytest tests/` 242 项通过。
注意：修复前已持久化的旧会话状态（随机 id）仍不可续跑，需新会话；
修复后新会话重启可续跑（UI 已复验）。
