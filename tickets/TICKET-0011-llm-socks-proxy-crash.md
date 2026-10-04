# TICKET-0011：LLM 客户端在有 socks 代理环境下崩溃（SSE network error）

- 状态：待审批 ｜ 提出：2026-10-03 ｜ 审批人：____

### 1. 改动对象

`app/agent/harness.py`：新增 `sanitize_proxy_env()`，在 `build_harness` 构建 LLM 模型前调用。
不属于冻结资产（不动 MAPLE / COMPUTE_VENV / checkpoint）。

### 2. 动机与问题描述

Web UI 发送任何消息后立即显示 `network error`，SSE 流中断（`ERR_INCOMPLETE_CHUNKED_ENCODING`）。
uvicorn 日志显示 `build_harness` 在构造 OpenAIProvider 时崩溃：

```
File ".../pydantic_ai/_http.py", line 70, in create_async_httpx2_client
File ".../httpx2/_config.py", line 190, in __init__
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7897/')
```

本机 shell 环境带 clash 代理变量 `ALL_PROXY=socks://127.0.0.1:7897/`，uvicorn 继承后，
pydantic_ai 内部 httpx2 客户端读取环境代理，但不认 `socks://` 裸 scheme（`socks5://` 也要求
未安装的 `socksio` 包），整个 chat 请求 500，前端表现为 network error。

复现：`ALL_PROXY=socks://127.0.0.1:7897/ .venv/bin/uvicorn app.main:app`，
UI 发任意消息即触发。直连该 LLM 端点可达（401 auth，网络层正常）。

### 3. 影响面分析

B. 框架内新增：只在 mlip_agent/ 内新增代码，不改变既有数值行为。
仅影响 LLM 客户端初始化前的环境变量预处理，不涉及任何计算路径与金标准。

### 4. 风险

- 对依赖 socks 代理才能访问 LLM 端点的部署：本修复在有 `socksio` 时改写 `socks://` →
  `socks5h://`；无 `socksio` 时删除该变量退化为直连。原本该场景直接 500 崩溃，不存在
  "可用但被改坏" 的回退。
- 数值漂移风险：无。

### 5. 回滚方案

`git revert` 对应提交即可，无数据/金标准副作用。

### 6. 验证计划

- 新增单测覆盖四种场景（http 保留 / socks+无 socksio 删除 / socks+有 socksio 改写 / 未设置不动）。
- `.venv/bin/python -m pytest tests/ -q` 全绿。
- 带 `ALL_PROXY=socks://...` 重启 uvicorn，Web UI 发消息不再 network error，agent 正常出审批卡。

### 7. 审批结论

批准（用户指令 2026-10-03，在 web UI 任务中现场发现并修复）。
实施回填：`app/agent/harness.py` 新增 `sanitize_proxy_env()`（`build_harness` 入口调用）；
新增 `tests/test_proxy_sanitize.py` 4 项；全量 `pytest tests/` 235 项通过。
带 `ALL_PROXY=socks://...` 重启 uvicorn 后 UI 发消息不再 network error。
