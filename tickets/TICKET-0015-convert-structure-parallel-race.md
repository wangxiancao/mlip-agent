# TICKET-0015：convert_structure 并行调用竞争（convert_job.json / conversion.json 固定文件名）

- 状态：待审批 ｜ 提出：2026-10-04 ｜ 审批人：____

### 1. 改动对象

- `app/tools/structure_tools.py` `convert_structure_impl`：job/meta 文件名改为带
  随机 token 的唯一名。
- `app/compute/scripts/convert_structure.py`：支持 job.json 可选 `meta_path`
  字段（缺省保持 `dst.parent/conversion.json` 旧行为）。
不属于冻结资产。

### 2. 动机与问题描述

web UI 实战（会话 bee4cd91）：agent 并行发起两个 convert_structure 调用，
第二个返回的 `src_path` 竟是第一个调用的源路径。根因：

1. 工具把任务写入**固定** `workspace/convert_job.json`，并行调用互相覆盖；
2. 子进程把结果写到**固定** `dst.parent/conversion.json`，先完成的覆盖
   后完成的，调用方读到的 meta 可能属于另一个任务。

 TICKET-0012 修复的是同一代码点的路径问题，并行竞争是该设计的第二个缺陷。
（本次任务中 agent 自行发现数值对不上并串行重试，结果正确，但纯属侥幸。）

### 3. 影响面分析

B. 框架内修正：只把中间文件名改为 `convert_job_<token>.json` /
`conversion_<token>.json`（token 为 uuid4 前 8 位），语义不变；
脚本侧 `meta_path` 为可选字段，缺省行为与现状逐位一致（向后兼容）。
不改变任何计算数值。

### 4. 风险

- 每次调用多留两个带 token 的中间文件（审计轨迹，与既有行为一致不清理）。
- 数值漂移风险：无。

### 5. 回滚方案

`git revert` 对应提交即可。

### 6. 验证计划

- 新增单测：asyncio.gather 并行两个 convert_structure_impl，双方返回的
  src_path/dst_path 各归各、无串扰。
- `.venv/bin/python -m pytest tests/ -q` 全绿。

### 7. 审批结论

批准（用户指令 2026-10-04，web UI 任务中现场发现并修复）。
实施回填：工具侧 job/meta 文件名带 uuid token（`convert_job_<tok>.json` /
`conversion_<tok>.json`），脚本侧支持可选 `meta_path`（缺省旧行为兼容）；
新增并行无串扰单测；全量 `pytest tests/` 243 项通过。
