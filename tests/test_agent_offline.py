"""M2 验收：FunctionModel 离线全链路（PLAN §3 B1/B6/B9/B10 + §4 M2）。

剧本设计：pydantic-ai FunctionModel(stream_function=...) 按「消息历史中
ModelResponse 条数」推进确定性剧本；工具真实执行（run_maple 走 GPU 子进程），
子 agent（task_parser / result_validator）也是 FunctionModel 回放。
审批挂起用 permissions.build_approval_results + harness.make_resume 恢复。

- B1：解析 → write_maple_inp → run_maple 真实执行（先 suspended，approve 后 resume）
  → result_validator → convert_units → 中文回答含 Hartree 与 eV 能量；
- B6：build_structure(C2H5OH) → opt → 取 final_xyz → freq 链式两作业；
- B9：信息不全 → ask_user_question 挂起，注入答案后 resume；
- B10：run_maple 审批挂起时无作业产物，approve resume 后产物出现。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from app.agent.harness import (
    build_harness,
    make_resume,
    make_resume_from_json,
    state_from_json,
)
from app.agent.streaming import RUN_RESULT_EVENT_TYPE, stream_agui_events
from app.config import Settings
from app.tools.common import JobAllocator
from app.tools.registry import build_tools

REFERENCES_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "references"
MAPLE_SP_INP = (
    Path(__file__).resolve().parents[2] / "MAPLE" / "examples" / "sp" / "sp.inp"
)

HARTREE_TO_EV = 27.211386245988


# ---------- 剧本辅助 ----------


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


def _as_data(content):
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return {"raw": content}


def _tool_data(messages, tool_name: str, index: int = 0):
    return _as_data(_returns(messages, tool_name)[index])


def _user_text(messages) -> str:
    parts = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    parts.append(p.content)
    return "\n".join(parts)


def _call(name: str, args: dict) -> dict[int, DeltaToolCall]:
    return {0: DeltaToolCall(name=name, json_args=json.dumps(args))}


def _b1_jobspec() -> dict:
    """从 MAPLE/examples/sp/sp.inp 解析 22 原子坐标，构造 B1 JobSpec。"""
    coords = []
    for line in MAPLE_SP_INP.read_text().splitlines():
        m = re.match(
            r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
            line,
        )
        if m:
            coords.append(
                {
                    "symbol": m.group(1),
                    "x": float(m.group(2)),
                    "y": float(m.group(3)),
                    "z": float(m.group(4)),
                }
            )
    assert len(coords) == 22
    return {
        "job_type": "sp",
        "model": {"name": "ani2x"},
        "structure": {"coords": coords},
        "charge": 0,
        "multiplicity": 1,
    }


async def _validator_stream(messages, info: AgentInfo):
    yield json.dumps(
        {"ok": True, "issues": [], "summary": "退出码 0，能量量级合理"}, ensure_ascii=False
    )


async def _parser_b1_stream(messages, info: AgentInfo):
    yield json.dumps(_b1_jobspec())


async def _orch_b1_stream(messages, info: AgentInfo):
    n = _n_responses(messages)
    if n == 0:
        yield _call("delegate", {"subagent": "task_parser", "prompt": "用 ANI-2x 算该 22 原子分子的单点能，坐标见用户消息"})
    elif n == 1:
        spec = json.loads(_tool_data(messages, "delegate")["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 2:
        inp = _tool_data(messages, "write_maple_inp")["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 600})
    elif n == 3:
        summary = _tool_data(messages, "run_maple")["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    elif n == 4:
        energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
        yield _call("convert_units", {"value": energy, "from_unit": "hartree", "to_unit": "eV"})
    else:
        energy = _tool_data(messages, "run_maple")["data"]["energy_hartree"]
        ev = _tool_data(messages, "convert_units")["data"]["value"]
        verdict = json.loads(_tool_data(messages, "delegate", 1)["output"])
        yield (
            f"计算完成。单点能 = {energy:.10f} Hartree = {ev:.4f} eV。"
            f"校验结论：{verdict['summary']}"
        )


async def _orch_b9_stream(messages, info: AgentInfo):
    n = _n_responses(messages)
    if n == 0:
        yield _call(
            "ask_user_question",
            {"question": "请提供：1) 使用哪个 MLIP 模型；2) 体系电荷与自旋多重度"},
        )
    else:
        rets = _returns(messages, "ask_user_question")
        answer = _as_data(rets[0]).get("data", {}).get("answer", "") if rets else ""
        yield f"已收到您的补充信息：{answer or '（无文字答复）'}。信息齐全后即可提交计算。"


async def _parser_b6_stream(messages, info: AgentInfo):
    text = _user_text(messages)
    # 只看 delegated_task 字段本身（路径等上下文里可能出现 "freq" 子串，
    # 如 pytest tmp 目录名 test_b6_chained_opt_freq0）
    task = json.loads(re.search(r"(\{.*delegated_task.*\})", text).group(1))[
        "delegated_task"
    ]
    xyz = re.search(r"(\S+\.xyz)", task).group(1)
    job_type = "freq" if task.startswith("freq") else "opt"
    yield json.dumps(
        {
            "job_type": job_type,
            "model": {"name": "ani2x"},
            "structure": {"xyz_file": xyz},
            "charge": 0,
            "multiplicity": 1,
        }
    )


async def _orch_b6_stream(messages, info: AgentInfo):
    n = _n_responses(messages)
    if n == 0:
        yield _call("build_structure", {"source": "C2H5OH", "source_type": "name"})
    elif n == 1:
        xyz = _tool_data(messages, "build_structure")["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"opt：用 ani2x 优化结构文件 {xyz}"})
    elif n == 2:
        spec = json.loads(_tool_data(messages, "delegate", 0)["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 3:
        inp = _tool_data(messages, "write_maple_inp", 0)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 4:
        final_xyz = _tool_data(messages, "run_maple", 0)["data"]["parsed"]["final_xyz"]
        coords = "\n".join(final_xyz.splitlines()[2:])  # 去掉 xyz 头两行
        yield _call("build_structure", {"source": coords, "source_type": "inline_coords"})
    elif n == 5:
        xyz = _tool_data(messages, "build_structure", 1)["data"]["xyz_path"]
        yield _call("delegate", {"subagent": "task_parser", "prompt": f"freq：用 ani2x 对结构文件 {xyz} 做频率分析"})
    elif n == 6:
        spec = json.loads(_tool_data(messages, "delegate", 1)["output"])
        yield _call("write_maple_inp", {"job_spec": spec})
    elif n == 7:
        inp = _tool_data(messages, "write_maple_inp", 1)["data"]["inp_path"]
        yield _call("run_maple", {"inp_path": inp, "timeout_s": 900})
    elif n == 8:
        summary = _tool_data(messages, "run_maple", 1)["data"]
        yield _call("delegate", {"subagent": "result_validator", "prompt": json.dumps(summary, ensure_ascii=False, default=str)})
    else:
        opt_e = _tool_data(messages, "run_maple", 0)["data"]["energy_hartree"]
        freq = _tool_data(messages, "run_maple", 1)["data"]["parsed"]["frequencies_cm1"]
        yield f"乙醇优化收敛，能量 {opt_e:.10f} Hartree；频率分析共 {len(freq)} 个模式，无虚频。"


# ---------- fixtures ----------


def _make_harness(tmp_path, orch_stream, parser_stream, validator_stream=None, advisor_stream=None):
    settings = Settings()
    capabilities = build_tools(tmp_path / "session", settings=settings)
    models = {
        "mlip_orchestrator": FunctionModel(stream_function=orch_stream),
        "task_parser": FunctionModel(stream_function=parser_stream),
        "result_validator": FunctionModel(
            stream_function=validator_stream or _validator_stream
        ),
    }
    if advisor_stream is not None:
        models["approval_advisor"] = FunctionModel(stream_function=advisor_stream)
    return build_harness(settings, capabilities, model_override=models)


async def _run_auto_approve(executable, prompt: str, max_rounds: int = 5):
    """run → suspended 则 approve_all resume，直到终态。返回 (result, suspensions)。"""
    result = await executable.run(prompt)
    suspensions = []
    for _ in range(max_rounds):
        if result.status != "suspended":
            return result, suspensions
        suspensions.append(result.deferred)
        result = await executable.run(
            previous_state=result.state, deferred_resume=make_resume(result)
        )
    raise AssertionError(f"超过 {max_rounds} 轮挂起仍未终态")


# ---------- 构建与 B9（无需 GPU） ----------


def test_build_harness_without_llm_key(tmp_path, monkeypatch):
    # setenv "" 同时覆盖环境变量与本地 .env（pydantic-settings 优先级：env > .env）
    monkeypatch.setenv("MLIP_AGENT_LLM_API_KEY", "")
    monkeypatch.setenv("MLIP_AGENT_LLM_MODEL", "")
    settings = Settings()
    assert not settings.llm_configured
    executable = build_harness(settings, build_tools(tmp_path / "session", settings=settings))
    assert executable is not None


async def test_b9_ask_user_question_suspends(tmp_path):
    """B9：信息不全 → ask_user_question 挂起，不得提交任何计算。"""
    executable = _make_harness(tmp_path, _orch_b9_stream, _parser_b1_stream)
    result = await executable.run("帮我算一下这个体系的能量")
    assert result.status == "suspended"
    approvals = result.deferred.approvals
    assert [c.tool_name for c in approvals] == ["ask_user_question"]
    # 未擅自提交计算
    assert not _returns(result.state.message_history, "run_maple")
    assert not _returns(result.state.message_history, "write_maple_inp")

    # 注入用户答复后 resume
    call_id = approvals[0].tool_call_id
    result2 = await executable.run(
        previous_state=result.state,
        deferred_resume=make_resume(result, answers={call_id: "用 ANI-2x，电荷 0，多重度 1"}),
    )
    assert result2.status == "completed"
    assert "ANI-2x" in result2.output


async def test_b9_denial_path(tmp_path):
    executable = _make_harness(tmp_path, _orch_b9_stream, _parser_b1_stream)
    result = await executable.run("帮我算一下这个体系的能量")
    assert result.status == "suspended"
    call_id = result.deferred.approvals[0].tool_call_id
    result2 = await executable.run(
        previous_state=result.state,
        deferred_resume=make_resume(
            result, approve_all=False, denials={call_id: "用户暂不回答"}
        ),
    )
    assert result2.status == "completed"


# ---------- B1 / B10 / B6（真实 GPU 计算） ----------


@pytest.mark.gpu
async def test_b1_offline_script_end_to_end(tmp_path):
    """B1 离线剧本端到端真实计算：能量 −383.4722385376 ±1e-6 Ha 出现在回答中。"""
    executable = _make_harness(tmp_path, _orch_b1_stream, _parser_b1_stream)
    result, suspensions = await _run_auto_approve(
        executable, "用 ANI-2x 计算这个分子的单点能：C -0.778126 …（22 原子内联坐标）"
    )
    assert result.status == "completed"
    assert len(suspensions) == 1  # run_maple 审批
    assert suspensions[0].approvals[0].tool_name == "run_maple"

    pinned = json.loads((REFERENCES_DIR / "B1.json").read_text())["values"]
    m = re.search(r"单点能 = (-?\d+\.\d+) Hartree = (-?\d+\.\d+) eV", result.output)
    assert m, result.output
    energy_ha, energy_ev = float(m.group(1)), float(m.group(2))
    assert energy_ha == pytest.approx(pinned["energy_hartree"], abs=1e-6)
    assert energy_ev == pytest.approx(energy_ha * HARTREE_TO_EV, abs=1e-3)
    assert "校验结论" in result.output


@pytest.mark.gpu
async def test_b10_approval_gate(tmp_path):
    """B10：run_maple 调用前会话 suspended 且无产物；approve resume 后产物出现。"""
    executable = _make_harness(tmp_path, _orch_b1_stream, _parser_b1_stream)
    result = await executable.run("用 ANI-2x 计算这个分子的单点能（内联坐标）")
    assert result.status == "suspended"
    pending = [c.tool_name for c in result.deferred.approvals]
    assert pending == ["run_maple"]

    # 挂起时：inp 已写好（审批前动作），但 .out 尚未产生（计算未执行）
    job_dir = tmp_path / "session" / "jobs" / "job-0001"
    assert (job_dir / "input.inp").is_file()
    assert not (job_dir / "input.out").exists()

    result2 = await executable.run(
        previous_state=result.state, deferred_resume=make_resume(result)
    )
    assert result2.status == "completed"
    assert (job_dir / "input.out").is_file()
    assert (job_dir / "stdout.log").is_file()
    out_text = (job_dir / "input.out").read_text()
    assert "Energy: -383.472238" in out_text


@pytest.mark.gpu
async def test_b6_chained_opt_freq(tmp_path):
    """B6：乙醇 build_structure(name) → opt → final_xyz → freq 两作业链。"""
    executable = _make_harness(tmp_path, _orch_b6_stream, _parser_b6_stream)
    result, suspensions = await _run_auto_approve(
        executable, "先优化乙醇的结构，再对优化后的结构做频率分析"
    )
    if result.status != "completed":  # 诊断：打印全部工具返回
        for m in result.state.message_history:
            if isinstance(m, ModelRequest):
                for p in m.parts:
                    if isinstance(p, ToolReturnPart):
                        print("RETURN", p.tool_name, str(p.content)[:500])
        if result.failure:
            print("FAILURE", result.failure.message, dict(result.failure.details))
    assert result.status == "completed"
    assert len(suspensions) == 2  # 两次 run_maple 各一次审批
    jobs_dir = tmp_path / "session" / "jobs"
    assert (jobs_dir / "job-0001" / "job_spec.json").is_file()
    assert (jobs_dir / "job-0002" / "job_spec.json").is_file()
    spec1 = json.loads((jobs_dir / "job-0001" / "job_spec.json").read_text())
    spec2 = json.loads((jobs_dir / "job-0002" / "job_spec.json").read_text())
    assert spec1["job_type"] == "opt" and spec2["job_type"] == "freq"

    from app.compute.parsers import parse_out_file

    freq_result = parse_out_file(jobs_dir / "job-0002" / "input.out", "freq")
    assert freq_result.ok
    assert freq_result.frequencies_cm1  # MAPLE 打印的模式表（大分子可能截断）
    assert freq_result.n_imaginary == 0
    # 权威计数看汇总行：乙醇 9 原子 → 3N-6 = 21 个实频、0 虚频
    out_text = (jobs_dir / "job-0002" / "input.out").read_text()
    assert "Real frequencies (ν > 5.0 cm⁻¹):       21" in out_text
    assert "Imaginary frequencies (ν < -5.0 cm⁻¹): 0" in out_text


# ---------- 流式适配器（M3 SSE 契约） ----------


async def test_streaming_adapter_events(tmp_path):
    """streaming.stream_agui_events：AG-UI 事件序列 + 终止 MLIP_RUN_RESULT。"""
    executable = _make_harness(tmp_path, _orch_b9_stream, _parser_b1_stream)
    events = [
        e async for e in stream_agui_events(executable, "帮我算一下这个体系的能量")
    ]
    types = [e["type"] for e in events]
    assert "RUN_STARTED" in types
    assert "TOOL_CALL_START" in types
    assert types[-1] == RUN_RESULT_EVENT_TYPE
    final = events[-1]
    assert final["status"] == "suspended"
    assert final["deferred"]["approvals"][0]["tool_name"] == "ask_user_question"
    assert final["state_json"]

    # state_json 可反序列化（M3 持久化契约）
    from a13n_harness import HarnessState

    HarnessState.model_validate_json(final["state_json"])

    # 流式 resume 走同一适配器：deferred 与原 run 同属一个 thread
    result = await executable.run("再挂起一次拿 deferred")
    assert result.status == "suspended"
    resume = make_resume(result)
    resumed_events = [
        e
        async for e in stream_agui_events(
            executable, None, previous_state=result.state, deferred_resume=resume
        )
    ]
    assert resumed_events[-1]["type"] == RUN_RESULT_EVENT_TYPE
    assert resumed_events[-1]["status"] == "completed"


# ---------- M6：approval_advisor 子 agent ----------

_ADVISOR_RISK_TEXT = (
    "高风险操作：uma-m-1p1 为 11GB 大模型，加载耗时且显存占用高；"
    "5000 步 NVT MD 预计耗时数小时，能量漂移可能累积。"
    "建议先用 uma-s-1p1 短轨迹试跑。确认要提交该计算吗？"
)


async def _advisor_stream(messages, info: AgentInfo):
    yield _ADVISOR_RISK_TEXT


async def _orch_highrisk_stream(messages, info: AgentInfo):
    """高危操作剧本：先委派 approval_advisor 出风险说明，并入追问请用户确认。"""
    n = _n_responses(messages)
    if n == 0:
        yield _call(
            "delegate",
            {
                "subagent": "approval_advisor",
                "prompt": "计划任务：用 uma-m-1p1（11GB 大模型）对水分子跑 5000 步 NVT MD，步长 0.5 fs",
            },
        )
    elif n == 1:
        risk = _tool_data(messages, "delegate")["output"]
        yield _call("ask_user_question", {"question": f"{risk}\n是否确认继续？"})
    else:
        risk = _tool_data(messages, "delegate")["output"]
        yield f"已收到确认。风险提示已告知：{risk}"


def test_subagent_specs_include_approval_advisor():
    """SUBAGENT_SPECS 声明三个子 agent（含 M6 的 approval_advisor）。"""
    from app.agent.subagents import SUBAGENT_SPECS

    names = [s["name"] for s in SUBAGENT_SPECS]
    assert names == ["task_parser", "result_validator", "approval_advisor"]


def test_permission_rules_fully_qualified_tool_ids():
    """D2-1 / TICKET-0004 前置条件 3：权限规则 key 必须是全限定 tool_id
    （tool/<capability_id>/<tool_name>），裸工具名永远匹配不到。"""
    from app.agent import permissions

    rules = permissions.PERMISSION_RULES
    for tool_id, action in rules.items():
        assert tool_id.startswith("tool/"), f"非全限定 tool_id: {tool_id}"
        assert action == "ask"
    # 裸工具名不得出现（D2-1 教训：裸名匹配不到，规则形同虚设）
    assert "run_maple" not in rules
    assert "run_nequip" not in rules
    assert "run_tace" not in rules
    assert "run_dpa4" not in rules
    # 全部直算/MAPLE 工具按全限定 id 置 ask（含 TICKET-0004 的 run_tace、
    # TICKET-0006 的 run_dpa4）
    assert rules[f"tool/{permissions.CAP_TACE}/run_tace"] == "ask"
    assert rules[f"tool/{permissions.CAP_NEQUIP}/run_nequip"] == "ask"
    assert rules[f"tool/{permissions.CAP_MAPLE}/run_maple"] == "ask"
    assert rules[f"tool/{permissions.CAP_DPA4}/run_dpa4"] == "ask"
    # capability id 常量与 registry 共享（registry 从 permissions import）
    from app.tools import registry

    assert registry.CAP_TACE == "mlip.tace"
    assert registry.CAP_DPA4 == "mlip.dpa4"


async def test_approval_advisor_high_risk_consultation(tmp_path):
    """M6：高危操作触发 approval_advisor 委派，风险文本进入审批追问与最终回答。"""
    executable = _make_harness(
        tmp_path,
        _orch_highrisk_stream,
        _parser_b1_stream,
        advisor_stream=_advisor_stream,
    )
    events = [
        e
        async for e in stream_agui_events(
            executable, "用 uma-m-1p1 对水分子跑 5000 步 NVT 分子动力学"
        )
    ]
    types = [e["type"] for e in events]
    assert "SUBAGENT_STARTED" in types
    assert "SUBAGENT_FINISHED" in types
    started = [e for e in events if e["type"] == "SUBAGENT_STARTED"]
    assert any(e.get("name") == "approval_advisor" for e in started)

    final = events[-1]
    assert final["type"] == RUN_RESULT_EVENT_TYPE
    assert final["status"] == "suspended"
    approvals = final["deferred"]["approvals"]
    assert [a["tool_name"] for a in approvals] == ["ask_user_question"]
    question = json.loads(approvals[0]["args"])["question"]
    assert "高风险" in question  # 风险说明并入审批追问
    # 顾问介入前不得提交任何计算
    assert not _returns(
        state_from_json(final["state_json"]).message_history, "run_maple"
    )

    # 用户确认后 resume：最终回答包含顾问的风险文本
    resumed = [
        e
        async for e in stream_agui_events(
            executable,
            None,
            previous_state=state_from_json(final["state_json"]),
            deferred_resume=make_resume_from_json(
                final["deferred_json"],
                answers={approvals[0]["tool_call_id"]: "确认"},
            ),
        )
    ]
    assert resumed[-1]["type"] == RUN_RESULT_EVENT_TYPE
    assert resumed[-1]["status"] == "completed"
    assert "uma-m-1p1" in resumed[-1]["output"]
    assert "高风险" in resumed[-1]["output"]
