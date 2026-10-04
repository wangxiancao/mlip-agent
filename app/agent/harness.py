"""Harness 组装：全项目唯一 import a13n-harness 的模块（PLAN §2.1 / §5.1）。

build_harness(config, capabilities, model_override=None) 一处组装主 agent
+ inline 子 agent + 工具 capability + 审批规则。SDK 升级只改本文件。

模型解析策略（PLAN §0.4：无 key 可构建，run 时才解析）：
- model_override 给出（测试）→ 按 agent 名取 FunctionModel 等离线模型；
- config.llm_configured（model + api_key 齐备）→ 每个 agent 独立构造
  OpenAIChatModel(OpenAIProvider(base_url, api_key))；
- 否则 → AgentSpec.model 写字段串、definition.model=None，
  模型解析延迟到 run 时（无 key 时 run 报 ModelResolutionError，build 不报错）。
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Mapping, Sequence

from a13n_harness import (
    AgentDefinition,
    AgentSpec,
    DeferredToolResume,
    ExecutableAgent,
    HarnessBuilder,
    HarnessRunResult,
    HarnessState,
    SubagentDefinition,
)
from a13n_harness.capabilities.subagents import SubagentCapability
from a13n_harness.tools.permissions import ToolPermissions, ToolPermissionsCapability
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent.instructions import ORCHESTRATOR_PROMPT
from app.agent.permissions import (
    PERMISSION_RULES,
    build_approval_results,
    load_deferred_json,
)
from app.agent.subagents import ORCHESTRATOR_NAME, SUBAGENT_SPECS
from app.config import Settings

ModelOverride = Model | Mapping[str, Model] | None

_logger = logging.getLogger(__name__)


def sanitize_proxy_env() -> None:
    """清理 LLM 客户端不支持的代理环境变量（TICKET-0011）。

    pydantic_ai 的 httpx2 客户端按环境变量建代理：`socks://` 裸 scheme 直接
    ValueError；`socks5://` 需要可选包 socksio，缺失时 ImportError —— 两种情况
    都让 chat 请求整体 500。崩溃不如直连，故无 socksio 时移除该变量；有 socksio
    时把裸 socks:// 规范化为 socks5h://。http/https 代理原样保留。
    """
    for var in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(var)
        if not value or "://" not in value:
            continue
        scheme, rest = value.split("://", 1)
        scheme = scheme.lower()
        if scheme in ("http", "https"):
            continue
        if scheme.startswith("socks"):
            if importlib.util.find_spec("socksio") is None:
                del os.environ[var]
                _logger.warning(
                    "%s=%s 使用 socks 代理但未安装 socksio，已移除该变量退化为直连",
                    var,
                    value,
                )
            elif scheme == "socks":
                os.environ[var] = f"socks5h://{rest}"


def _live_model(config: Settings) -> Model:
    return OpenAIChatModel(
        config.llm_model,
        provider=OpenAIProvider(
            base_url=config.llm_base_url or None,
            api_key=config.llm_api_key,
        ),
    )


def _resolve_models(config: Settings, model_override: ModelOverride) -> dict[str, Model | None]:
    names = (ORCHESTRATOR_NAME, *(s["name"] for s in SUBAGENT_SPECS))
    if model_override is None:
        if config.llm_configured:
            return {name: _live_model(config) for name in names}
        return dict.fromkeys(names)
    if isinstance(model_override, Model):
        return dict.fromkeys(names, model_override)
    return {name: model_override.get(name) for name in names}


def build_harness(
    config: Settings,
    capabilities: Sequence[AbstractCapability],
    model_override: ModelOverride = None,
) -> ExecutableAgent[str]:
    """组装 mlip_orchestrator 可执行 agent。

    Args:
        config: Settings（LLM 端点、路径）。
        capabilities: 工具 capability 列表（registry.build_tools 产物）。
        model_override: 单个 Model（所有 agent 共用）或 {agent_name: Model}；
            测试传 FunctionModel，生产传 None。
    """
    sanitize_proxy_env()
    models = _resolve_models(config, model_override)
    # definition.model 与 AgentSpec.model 互斥（AgentDefinition 校验）：
    # 有实例模型时 AgentSpec.model 置 None
    defer_model = models[ORCHESTRATOR_NAME] is None

    subagents = tuple(
        SubagentDefinition(
            name=spec["name"],
            description=spec["description"],
            agent=AgentDefinition(
                agent=AgentSpec(
                    instructions=spec["instructions"],
                    model=(config.llm_model or None)
                    if models[spec["name"]] is None
                    else None,
                ),
                output_type=str,
                # TICKET-0014：确定性 id——缺省随机 UUID 会让重启后持久化的
                # inline subagent 状态校验失败（StateError，会话无法续跑）
                definition_id=f"mlip.subagent.{spec['name']}",
                model=models[spec["name"]],
            ),
        )
        for spec in SUBAGENT_SPECS
    )

    definition = AgentDefinition(
        agent=AgentSpec(
            instructions=ORCHESTRATOR_PROMPT,
            model=(config.llm_model or None) if defer_model else None,
        ),
        output_type=str,
        definition_id="mlip.orchestrator",  # TICKET-0014，同上
        model=models[ORCHESTRATOR_NAME],
        capabilities=(
            *capabilities,
            ToolPermissionsCapability(ToolPermissions(rules=PERMISSION_RULES)),
            SubagentCapability(),
        ),
        subagents=subagents,
    )
    return HarnessBuilder().build(definition)


def make_resume(
    result: HarnessRunResult[str],
    *,
    approve_all: bool = True,
    answers: Mapping[str, str] | None = None,
    denials: Mapping[str, str] | None = None,
) -> DeferredToolResume:
    """把 suspended run 的 deferred 请求包装成可传给 run(deferred_resume=...) 的恢复值。

    Args:
        result: status == "suspended" 的 HarnessRunResult。
        approve_all: 未显式处理的审批一律通过。
        answers: tool_call_id → 用户文字答复（注入 ask_user_question）。
        denials: tool_call_id → 拒绝理由。
    """
    if result.status != "suspended" or result.deferred is None:
        raise ValueError(f"run 未处于挂起态（status={result.status}），无法恢复")
    return DeferredToolResume(
        result.deferred,
        build_approval_results(
            result.deferred, approve_all=approve_all, answers=answers, denials=denials
        ),
    )


def state_from_json(state_json: str) -> HarnessState:
    """反序列化持久化的 HarnessState（stream_agui_events 的 state_json 字段）。"""
    return HarnessState.model_validate_json(state_json)


def make_resume_from_json(
    deferred_json: str,
    *,
    approve_all: bool = True,
    answers: Mapping[str, str] | None = None,
    denials: Mapping[str, str] | None = None,
) -> DeferredToolResume:
    """从持久化的 deferred JSON 重建恢复值（M3 服务重启后的 resume 路径）。

    Args:
        deferred_json: stream_agui_events RUN_RESULT 事件的 deferred_json 字段。
        approve_all / answers / denials: 同 make_resume。
    """
    deferred = load_deferred_json(deferred_json)
    return DeferredToolResume(
        deferred,
        build_approval_results(
            deferred, approve_all=approve_all, answers=answers, denials=denials
        ),
    )


def summarize_messages(state_json: str) -> list[dict]:
    """把持久化 state 的消息历史压缩成 API 快照用的轻量条目列表。

    每条 {role, kind, text?, tool_name?}；工具返回内容不展开（体积大，
    前端要看细节走 jobs/artifacts 端点）。
    """
    state = HarnessState.model_validate_json(state_json)
    items: list[dict] = []
    for message in state.message_history:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                text = part.content if isinstance(part.content, str) else str(part.content)
                items.append({"role": "user", "kind": "text", "text": text})
            elif isinstance(part, TextPart):
                items.append({"role": "assistant", "kind": "text", "text": part.content})
            elif isinstance(part, ToolCallPart):
                items.append(
                    {
                        "role": "assistant",
                        "kind": "tool_call",
                        "tool_name": part.tool_name,
                        "text": part.args_as_json_str(),
                    }
                )
            elif isinstance(part, ToolReturnPart):
                items.append(
                    {"role": "tool", "kind": "tool_return", "tool_name": part.tool_name}
                )
    return items
