"""M3 验收：httpx ASGI 测试（PLAN §4 M3）。

FunctionModel 注入（create_app(model_override=...)）离线跑全链路：
建会话 → 发消息（SSE：TOOL_CALL_* + 挂起）→ resume 批准（SSE：TEXT_MESSAGE_*）
→ artifacts 可下载 → 新 app 实例（模拟重启）后 state 恢复。
剧本约定同 tests/test_agent_offline.py：按消息历史中 ModelResponse 条数推进。

剧本（无需 GPU）：
- n=0: write_maple_inp（单 H 原子 sp，自动审批，落盘 job-0001/input.inp）
- n=1: ask_user_question → 权限规则 ask → run 挂起
- n≥2: 最终文本（引用 job_id 与用户答复）
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from app.config import Settings
from app.main import create_app
from app.state.store import CURRENT_STATE_VERSION

_JOB_SPEC = {
    "job_type": "sp",
    "model": {"name": "ani2x"},
    "structure": {"coords": [{"symbol": "H", "x": 0.0, "y": 0.0, "z": 0.0}]},
    "charge": 0,
    "multiplicity": 2,
}


def _n_responses(messages) -> int:
    return sum(1 for m in messages if isinstance(m, ModelResponse))


def _returns(messages, tool_name: str) -> list:
    out = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart) and p.tool_name == tool_name:
                    out.append(p.content)
    return out


def _call(name: str, args: dict) -> dict[int, DeltaToolCall]:
    return {0: DeltaToolCall(name=name, json_args=json.dumps(args))}


async def _orch_stream(messages, info: AgentInfo):
    n = _n_responses(messages)
    if n == 0:
        yield _call("write_maple_inp", {"job_spec": _JOB_SPEC})
    elif n == 1:
        yield _call("ask_user_question", {"question": "请确认模型、电荷与自旋多重度"})
    else:
        job = _returns(messages, "write_maple_inp")[0]["data"]
        answer = _returns(messages, "ask_user_question")[0]["data"]["answer"]
        yield f"作业 {job['job_id']} 已创建（inp 落盘）。用户答复：{answer}"


async def _dummy_subagent_stream(messages, info: AgentInfo):
    yield "{}"


def _make_app(data_dir: Path):
    settings = Settings(data_dir=data_dir)
    models = {
        "mlip_orchestrator": FunctionModel(stream_function=_orch_stream),
        "task_parser": FunctionModel(stream_function=_dummy_subagent_stream),
        "result_validator": FunctionModel(stream_function=_dummy_subagent_stream),
    }
    return create_app(settings=settings, model_override=models)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30
    )


async def _collect_sse(client: httpx.AsyncClient, url: str, body: dict) -> list[dict]:
    events: list[dict] = []
    async with client.stream("POST", url, json=body) as resp:
        assert resp.status_code == 200, await resp.aread()
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
            elif line:
                assert line.startswith(":"), f"非 SSE 行: {line!r}"
    return events


async def _create_session(client: httpx.AsyncClient) -> str:
    resp = await client.post("/api/sessions", json={})
    assert resp.status_code == 201
    return resp.json()["session_id"]


# ---------- 会话 CRUD / state ----------


async def test_create_session_and_get_state(tmp_path):
    async with _client(_make_app(tmp_path / "data")) as client:
        resp = await client.post("/api/sessions", json={"title": "冒烟"})
        assert resp.status_code == 201
        info = resp.json()
        assert info["state_version"] == CURRENT_STATE_VERSION

        resp = await client.get(f"/api/sessions/{info['session_id']}/state")
        assert resp.status_code == 200
        state = resp.json()
        assert state["status"] == "idle"
        assert state["stale_state"] is False
        assert state["jobs"] == []
        assert state["pending_approvals"] == []
        # 磁盘落盘
        assert (
            tmp_path / "data" / "sessions" / info["session_id"] / "state.json"
        ).is_file()


async def test_unknown_session_404(tmp_path):
    async with _client(_make_app(tmp_path / "data")) as client:
        resp = await client.get(f"/api/sessions/{'0' * 32}/state")
        assert resp.status_code == 404
        resp = await client.post(
            f"/api/sessions/{'0' * 32}/messages", json={"content": "hi"}
        )
        assert resp.status_code == 404


# ---------- 核心链路：消息 SSE → 挂起 → resume → artifacts ----------


async def test_message_sse_suspend_resume_artifacts(tmp_path):
    app = _make_app(tmp_path / "data")
    async with _client(app) as client:
        sid = await _create_session(client)

        events = await _collect_sse(
            client, f"/api/sessions/{sid}/messages", {"content": "帮我算个单点能"}
        )
        types = [e["type"] for e in events]
        assert "RUN_STARTED" in types
        assert "TOOL_CALL_START" in types
        assert "TOOL_CALL_END" in types  # write_maple_inp 自动审批执行完毕
        assert types[-1] == "MLIP_RUN_RESULT"
        final = events[-1]
        assert final["status"] == "suspended"
        approvals = final["deferred"]["approvals"]
        assert [a["tool_name"] for a in approvals] == ["ask_user_question"]
        call_id = approvals[0]["tool_call_id"]
        assert final["state_json"] and final["deferred_json"]

        # 挂起态快照：pending_approvals + 作业列表
        state = (await client.get(f"/api/sessions/{sid}/state")).json()
        assert state["status"] == "suspended"
        assert state["pending_approvals"][0]["tool_name"] == "ask_user_question"
        assert state["thread_id"] == final["thread_id"]
        assert [j["job_id"] for j in state["jobs"]] == ["job-0001"]
        assert state["jobs"][0]["job_type"] == "sp"
        assert state["jobs"][0]["has_output"] is False

        # 挂起时再发消息 → 409
        resp = await client.post(
            f"/api/sessions/{sid}/messages", json={"content": "再来一条"}
        )
        assert resp.status_code == 409

        # resume：注入用户答复
        events2 = await _collect_sse(
            client,
            f"/api/sessions/{sid}/resume",
            {"answers": {call_id: "ANI-2x，电荷 0，多重度 2"}},
        )
        types2 = [e["type"] for e in events2]
        assert "TEXT_MESSAGE_START" in types2
        assert "TEXT_MESSAGE_CONTENT" in types2
        assert "TEXT_MESSAGE_END" in types2
        assert events2[-1]["type"] == "MLIP_RUN_RESULT"
        assert events2[-1]["status"] == "completed"
        assert "ANI-2x，电荷 0，多重度 2" in events2[-1]["output"]
        assert "job-0001" in events2[-1]["output"]

        # 终态快照
        state = (await client.get(f"/api/sessions/{sid}/state")).json()
        assert state["status"] == "completed"
        assert state["pending_approvals"] == []
        assert state["last_run"]["status"] == "completed"
        roles = {(m["role"], m["kind"]) for m in state["messages"]}
        assert ("user", "text") in roles
        assert ("assistant", "text") in roles
        assert ("assistant", "tool_call") in roles

        # artifacts 下载
        resp = await client.get("/api/jobs/job-0001")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid
        assert set(resp.json()["artifacts"]) == {
            "input.inp",
            "job_spec.json",
            "structure.xyz",
        }

        resp = await client.get("/api/jobs/job-0001/artifacts/input.inp")
        assert resp.status_code == 200
        assert "#model=ani2x" in resp.text

        # 起始结构工件（StructureView 数据源）
        resp = await client.get("/api/jobs/job-0001/artifacts/structure.xyz")
        assert resp.status_code == 200
        assert resp.text.splitlines()[0].strip().isdigit()


async def test_resume_rejects_when_not_suspended(tmp_path):
    async with _client(_make_app(tmp_path / "data")) as client:
        sid = await _create_session(client)
        resp = await client.post(f"/api/sessions/{sid}/resume", json={})
        assert resp.status_code == 409


# ---------- 重启恢复 ----------


async def test_restart_recovers_suspended_session(tmp_path):
    data_dir = tmp_path / "data"
    async with _client(_make_app(data_dir)) as client:
        sid = await _create_session(client)
        events = await _collect_sse(
            client, f"/api/sessions/{sid}/messages", {"content": "帮我算个单点能"}
        )
        assert events[-1]["status"] == "suspended"
        call_id = events[-1]["deferred"]["approvals"][0]["tool_call_id"]
        thread_id = events[-1]["thread_id"]

    # 模拟服务重启：新 app 实例（harness 缓存为空），同一 data_dir
    async with _client(_make_app(data_dir)) as client2:
        state = (await client2.get(f"/api/sessions/{sid}/state")).json()
        assert state["status"] == "suspended"
        assert state["stale_state"] is False
        assert state["thread_id"] == thread_id
        assert state["pending_approvals"][0]["tool_call_id"] == call_id
        assert [j["job_id"] for j in state["jobs"]] == ["job-0001"]

        events2 = await _collect_sse(
            client2, f"/api/sessions/{sid}/resume", {"approve_all": True}
        )
        assert events2[-1]["type"] == "MLIP_RUN_RESULT"
        assert events2[-1]["status"] == "completed"
        assert events2[-1]["thread_id"] == thread_id


async def test_stale_state_version_discarded(tmp_path):
    data_dir = tmp_path / "data"
    async with _client(_make_app(data_dir)) as client:
        sid = await _create_session(client)
        await _collect_sse(
            client, f"/api/sessions/{sid}/messages", {"content": "帮我算个单点能"}
        )

    # 篡改 state_version 模拟旧版 state；重启后弃旧开新并标记
    state_file = data_dir / "sessions" / sid / "state.json"
    raw = json.loads(state_file.read_text())
    raw["state_version"] = 999
    state_file.write_text(json.dumps(raw, ensure_ascii=False))

    async with _client(_make_app(data_dir)) as client2:
        state = (await client2.get(f"/api/sessions/{sid}/state")).json()
        assert state["stale_state"] is True
        assert state["status"] == "idle"
        assert state["pending_approvals"] == []
        assert state["thread_id"] is None
        # 作业工件保留（弃的是对话状态，不是数据）
        assert [j["job_id"] for j in state["jobs"]] == ["job-0001"]
        # 旧文件归档
        assert (data_dir / "sessions" / sid / "state.v999.stale.json").is_file()


# ---------- 杂项端点 ----------


async def test_models_endpoint(tmp_path):
    async with _client(_make_app(tmp_path / "data")) as client:
        resp = await client.get("/api/models")
        assert resp.status_code == 200
        models = resp.json()
        names = {m["name"] for m in models}
        assert {"ani2x", "uma", "nequip-oam-xl"} <= names
        assert all({"name", "label", "path", "backend", "status"} <= set(m) for m in models)


async def test_job_404_and_path_traversal(tmp_path):
    app = _make_app(tmp_path / "data")
    async with _client(app) as client:
        sid = await _create_session(client)
        await _collect_sse(
            client, f"/api/sessions/{sid}/messages", {"content": "帮我算个单点能"}
        )

        resp = await client.get("/api/jobs/job-9999")
        assert resp.status_code == 404
        resp = await client.get("/api/jobs/job-0001/artifacts/nope.txt")
        assert resp.status_code == 404
        # 路径穿越：编码后的 ../ 不得逃出作业目录（400 或路由不命中 404）
        resp = await client.get("/api/jobs/job-0001/artifacts/..%2F..%2Fstate.json")
        assert resp.status_code in (400, 404)
        resp = await client.get("/api/jobs/..%2F..%2Fetc")
        assert resp.status_code in (400, 404)
