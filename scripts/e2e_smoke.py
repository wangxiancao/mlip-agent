"""e2e 冒烟（M6）：无 LLM key 的全链路端到端证明。

起真实 uvicorn 服务（create_app + model_override=FunctionModel 回放
benchmarks/canned_scripts.py 的 B1 剧本），用 httpx 走完整 API 流：

    GET  /api/health
    POST /api/sessions
    POST /api/sessions/{sid}/messages   （SSE 消费到 suspended，断言 run_maple 待审批）
    POST /api/sessions/{sid}/resume     （批准 → SSE 到 completed，断言 output 含能量）
    GET  /api/sessions/{sid}/state      （断言 jobs 非空）
    GET  /api/jobs/{jid}/artifacts/input.out  （断言含 pin 能量行）

除真实 LLM 外全链路真实执行（含 GPU 上的真实 ANI-2x 单点计算，需 COMPUTE_VENV
与 GPU 可用）。全部断言通过退出码 0，否则 1。用法见 scripts/e2e_smoke.sh。
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
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from benchmarks import canned_scripts, judge  # noqa: E402

OVERALL_TIMEOUT_S = 600


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  [ok] {label}", flush=True)


async def _consume_sse(client: httpx.AsyncClient, path: str, body: dict) -> list[dict]:
    """POST 并消费 SSE 流到结束，返回事件列表（忽略 :hb 心跳注释行）。"""
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
    data_dir = Path(tempfile.mkdtemp(prefix="mlip_e2e_"))
    print(f"data_dir={data_dir}  port={port}", flush=True)

    settings = Settings(data_dir=data_dir)
    script = canned_scripts.SCRIPTS["B1"]
    models = {
        "mlip_orchestrator": FunctionModel(stream_function=script.orchestrator),
        "task_parser": FunctionModel(stream_function=script.task_parser),
        "result_validator": FunctionModel(stream_function=script.result_validator),
    }
    app = create_app(settings, model_override=models)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    base = f"http://127.0.0.1:{port}"
    timeout = httpx.Timeout(OVERALL_TIMEOUT_S, connect=10.0)
    try:
        async with httpx.AsyncClient(base_url=base, timeout=timeout, trust_env=False) as client:
            # 等服务就绪
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("uvicorn 10s 内未就绪")

            health = (await client.get("/api/health")).json()
            _check(health["ok"] is True, "GET /api/health")
            _check(health["llm_configured"] is False, "无 LLM key（FunctionModel 回放）")

            resp = await client.post("/api/sessions", json={"title": "e2e-smoke"})
            _check(resp.status_code == 201, "POST /api/sessions → 201")
            sid = resp.json()["session_id"]

            prompt = next(t["prompt"] for t in judge.load_tasks() if t["id"] == "B1")
            events = await _consume_sse(
                client, f"/api/sessions/{sid}/messages", {"content": prompt}
            )
            types = [e["type"] for e in events]
            _check("TOOL_CALL_START" in types, "SSE 含 TOOL_CALL_START")
            _check(types[-1] == "MLIP_RUN_RESULT", "SSE 以 MLIP_RUN_RESULT 终止")
            final = events[-1]
            _check(final["status"] == "suspended", "run_maple 前会话挂起")
            approvals = final["deferred"]["approvals"]
            _check(
                [a["tool_name"] for a in approvals] == ["run_maple"],
                "待审批项为 run_maple",
            )

            events2 = await _consume_sse(
                client, f"/api/sessions/{sid}/resume", {"approve_all": True}
            )
            final2 = events2[-1]
            _check(final2["type"] == "MLIP_RUN_RESULT", "resume SSE 终止事件")
            _check(final2["status"] == "completed", "批准后 run 完成")
            output = final2["output"] or ""
            _check("Hartree" in output and "eV" in output, "output 含能量（Ha + eV）")
            print(f"  agent 回答: {output}", flush=True)

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

    print("E2E SMOKE PASS", flush=True)


def main() -> int:
    try:
        asyncio.run(asyncio.wait_for(_main(), timeout=OVERALL_TIMEOUT_S + 60))
    except Exception:  # noqa: BLE001 - 冒烟脚本：任何失败都打印并退出 1
        print("E2E SMOKE FAIL", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
