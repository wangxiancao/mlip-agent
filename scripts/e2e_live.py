"""e2e live 验收：真实 LLM 经 HTTP API（即 Web UI 实际链路）的端到端证明。

与 e2e_smoke.py 的差别：不使用 model_override——后端从 .env / 环境变量读取
MLIP_AGENT_LLM_*，由真实 LLM 完成解析编排。链路：

    GET  /api/health                （断言 llm_configured=true）
    POST /api/sessions
    POST /api/sessions/{sid}/messages   （SSE；真实 LLM 驱动，可能多次挂起）
    POST /api/sessions/{sid}/resume     （审批一律批准；提问给通用答复）
    GET  /api/sessions/{sid}/state      （断言 jobs 非空）
    GET  /api/jobs/{jid}/artifacts/input.out  （断言含 B1 pin 能量行）

任务用 B1（自包含 22 原子内联坐标）。全部断言通过退出码 0，否则 1。
运行：env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy \
      -u all_proxy .venv/bin/python scripts/e2e_live.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from benchmarks import judge  # noqa: E402

OVERALL_TIMEOUT_S = 900
MAX_SUSPENSIONS = 8


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  [ok] {label}", flush=True)


async def _consume_sse(client: httpx.AsyncClient, path: str, body: dict) -> list[dict]:
    events: list[dict] = []
    async with client.stream("POST", path, json=body) as resp:
        if resp.status_code != 200:
            raise AssertionError(f"POST {path} → HTTP {resp.status_code}: {await resp.aread()!r}")
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


async def _main() -> None:
    port = _free_port()
    data_dir = Path(tempfile.mkdtemp(prefix="mlip_e2e_live_"))
    print(f"data_dir={data_dir}  port={port}", flush=True)

    settings = Settings(data_dir=data_dir)
    if not settings.llm_configured:
        raise AssertionError("未配置 MLIP_AGENT_LLM_*（.env 或环境变量），live e2e 无法运行")
    app = create_app(settings)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    base = f"http://127.0.0.1:{port}"
    timeout = httpx.Timeout(OVERALL_TIMEOUT_S, connect=10.0)
    try:
        async with httpx.AsyncClient(base_url=base, timeout=timeout, trust_env=False) as client:
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("uvicorn 10s 内未就绪")

            health = (await client.get("/api/health")).json()
            _check(health["ok"] is True, "GET /api/health")
            _check(health["llm_configured"] is True, "真实 LLM 已配置（.env）")

            resp = await client.post("/api/sessions", json={"title": "e2e-live-B1"})
            _check(resp.status_code == 201, "POST /api/sessions → 201")
            sid = resp.json()["session_id"]

            prompt = next(t["prompt"] for t in judge.load_tasks() if t["id"] == "B1")
            events = await _consume_sse(client, f"/api/sessions/{sid}/messages", {"content": prompt})
            types = [e["type"] for e in events]
            _check("TOOL_CALL_START" in types, "SSE 含 TOOL_CALL_START")
            final = events[-1]
            _check(final["type"] == "MLIP_RUN_RESULT", "SSE 以 MLIP_RUN_RESULT 终止")

            # 挂起循环：审批一律批准；提问给通用确认答复
            n_susp = 0
            while final["status"] == "suspended":
                n_susp += 1
                if n_susp > MAX_SUSPENSIONS:
                    raise AssertionError(f"挂起超过 {MAX_SUSPENSIONS} 次")
                deferred = final.get("deferred") or {}
                approvals = deferred.get("approvals", [])
                calls = deferred.get("calls", [])
                print(f"  [suspend #{n_susp}] approvals={[a['tool_name'] for a in approvals]} "
                      f"calls={[c.get('tool_name') for c in calls]}", flush=True)
                body: dict = {"approve_all": True}
                if calls:
                    body["answers"] = {
                        c["tool_call_id"]: "按你认为最合适的默认方案继续（水分子 H2O，电荷 0，多重度 1）。"
                        for c in calls
                    }
                events = await _consume_sse(client, f"/api/sessions/{sid}/resume", body)
                final = events[-1]
                _check(final["type"] == "MLIP_RUN_RESULT", "resume SSE 终止事件")

            _check(n_susp >= 1, "至少触发一次审批挂起（run_maple ask 规则）")
            _check(final["status"] == "completed", f"run 最终完成（实际 {final['status']}）")
            output = final["output"] or ""
            _check("Hartree" in output and "eV" in output, "output 含能量（Ha + eV）")
            print(f"  agent 回答: {output[:400]}", flush=True)

            state = (await client.get(f"/api/sessions/{sid}/state")).json()
            _check(state["status"] == "completed", "GET /state 状态 completed")
            _check(len(state["jobs"]) >= 1, "GET /state jobs 非空")
            job_id = state["jobs"][0]["job_id"]

            resp = await client.get(f"/api/jobs/{job_id}/artifacts/input.out")
            _check(resp.status_code == 200, f"下载工件 {job_id}/input.out")
            _check(
                "Energy: -383.472238" in resp.text,
                "input.out 含 pin 能量 -383.472238 Ha",
            )
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=15)

    print("E2E LIVE PASS", flush=True)


def main() -> int:
    try:
        asyncio.run(asyncio.wait_for(_main(), timeout=OVERALL_TIMEOUT_S + 60))
    except Exception:  # noqa: BLE001
        print("E2E LIVE FAIL", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
