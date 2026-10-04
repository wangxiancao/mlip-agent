# TICKET-0012：data_dir 相对路径导致 convert_structure 子进程找不到 convert_job.json

- 状态：待审批 ｜ 提出：2026-10-03 ｜ 审批人：____

### 1. 改动对象

`app/state/store.py` `SessionStore.__init__`：`self.data_dir = Path(data_dir).resolve()`。
不属于冻结资产（不动 MAPLE / COMPUTE_VENV / checkpoint）。

### 2. 动机与问题描述

生产默认 `MLIP_AGENT_DATA_DIR` 缺省为相对路径 `./data`（`app/config.py`），
`SessionStore` 原样保存 → `session_workspace` 是**相对**路径。
`convert_structure_impl`（`app/tools/structure_tools.py:257`）以此相对 workspace：

1. 写出 `convert_job.json`，其内 `dst_path` 也是相对串（`data/sessions/<sid>/xxx.gen`）；
2. 以 `cwd=workspace` 起子进程，argv 传入的 job 路径同为相对串；
3. 子进程在 `data/sessions/<sid>` 下再拼一次 `data/sessions/<sid>/convert_job.json`
   → `FileNotFoundError`，错误处理又把 conversion.json 写到嵌套目录
   `data/sessions/<sid>/data/sessions/<sid>/`。

Web UI 中 agent 任何 convert_structure 调用都报「格式转换子进程未产出
conversion.json：exit code = 1」。测试/benchmark 从未暴露是因为 conftest 与
e2e 都用 `tmp_path` 绝对路径。

复现：默认配置起 uvicorn（不设 MLIP_AGENT_DATA_DIR），UI 让 agent 转换任意 cif。

### 3. 影响面分析

B. 框架内新增/修正：只把 data_dir 规范化为绝对路径，不改变既有数值行为。
所有下游路径拼接（session_workspace、jobs/、工件下载）语义不变，只是从相对变绝对。
不影响金标准（测试用绝对 tmp_path，行为完全一致）。

### 4. 风险

- 路径比对类逻辑若依赖相对串会受影响；经排查无此类逻辑（均经 Path 拼接）。
- 数值漂移风险：无。

### 5. 回滚方案

`git revert` 对应提交即可。

### 6. 验证计划

- 新增单测：`SessionStore(Path("data"))` 后 sessions_dir 为绝对路径。
- `.venv/bin/python -m pytest tests/ -q` 全绿。
- 默认配置起 uvicorn，UI 触发 convert_structure 成功产出 conversion.json。

### 7. 审批结论

批准（用户指令 2026-10-03，web UI 任务中现场发现并修复）。
实施回填：`SessionStore.__init__` 改为 `Path(data_dir).resolve()`；
新增 `tests/test_store.py` 2 项；全量 `pytest tests/` 237 项通过；
手动验证 cif→extxyz ASE 转换成功（260 原子，pbc 全 T，cellpar 正确）。
