# TICKET-0017：轻量工具中间件移入每次调用独立子目录（治本并行竞争）

- 状态：待审批 ｜ 提出：2026-10-04 ｜ 审批人：____

### 1. 改动对象

- `app/tools/structure_tools.py` `convert_structure_impl`：中间件从
  workspace 根的 `convert_job_<tok>.json`/`conversion_<tok>.json` 移入
  `workspace/convert/<tok>/{job.json, conversion.json}`。
- 同文件 `build_structure_impl`：固定名 `workspace/meta.json`（同款并行
  竞争隐患）移入 `workspace/build/<tok>/meta.json`。
不属于冻结资产。

### 2. 动机与问题描述

用户裁定：TICKET-0015 的"文件名加 token"是治标；正确做法与计算作业
（`jobs/job-000N/` 独立子目录）一致——**每次调用一个独立子目录**，
中间件用普通文件名，从结构上消除并行覆盖。

排查确认 build_structure 的 `meta.json` 存在同一缺陷（并行两个
build_structure 会互相覆盖 meta），一并修。

计算作业本身无此问题：`JobAllocator.allocate()` 在单事件循环内同步
分配 `jobs/job-000N` 子目录，无 await 间隙，并行提交安全。

### 3. 影响面分析

B. 框架内修正：仅中间件落盘位置变化，工具返回值与产物路径不变，
不改变任何数值行为。`convert_structure.py` 脚本的 `meta_path` 机制
（TICKET-0015）保持不变，只是传入路径换到子目录。

### 4. 风险

- 无下游依赖中间件路径（grep 确认仅本文件与脚本引用）。
- 数值漂移风险：无。

### 5. 回滚方案

`git revert` 对应提交即可。

### 6. 验证计划

- 既有并行无串扰单测（TICKET-0015）照跑；为 build_structure 补并行用例。
- `.venv/bin/python -m pytest tests/ -q` 全绿。

### 7. 审批结论

批准（用户裁定 2026-10-04：独立子目录是治本方案，取代 0015 的 token 文件名）。
实施回填（mlip_agent commit `10cf224`）：`convert_structure_impl` 中间件移入
`workspace/convert/<tok>/`；`build_structure_impl`（name 来源）移入
`workspace/build/<tok>/` 且 deliverable 同落子目录（消除并行 deliverable
互覆）。全量 `pytest tests/` 243 绿；canned benchmark 28/28 复验通过。
