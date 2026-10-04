"""工具审批规则与审批结果构造（PLAN §2.1）。

SDK 隔离约定：本模块只 import pydantic-ai 原语（DeferredToolRequests /
ToolApproved / ToolDenied），ToolPermissionsCapability 的组装在 harness.py，
DeferredToolResume 的包装也在 harness.py。

工具身份：a13n-harness 把 Capability(id=...) 内的函数工具命名为
`tool/<capability_id>/<tool_name>`（tools/identity.py: source_tool_id），
所以 rules 的 key 必须写全限定 id，裸工具名（如 "run_maple"）匹配不到。

审批语义：
- run_maple / run_nequip / run_tace（GPU 计算）/ run_dpa4（CPU 直算，TICKET-0006）：
  批准 = 原样执行（ToolApproved()）。
- ask_user_question：批准 = 用户已看到问题；用户文字答复经
  ToolApproved(override_args={..., "answer": ...}) 注入（M3 resume 端点使用）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter
from pydantic_ai import ToolApproved, ToolDenied
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

CAP_STRUCTURE = "mlip.structure"
CAP_MAPLE = "mlip.maple"
CAP_NEQUIP = "mlip.nequip"
CAP_TACE = "mlip.tace"
CAP_DPA4 = "mlip.dpa4"
CAP_DIALOG = "mlip.dialog"

PERMISSION_RULES: dict[str, str] = {
    f"tool/{CAP_MAPLE}/run_maple": "ask",
    f"tool/{CAP_NEQUIP}/run_nequip": "ask",
    f"tool/{CAP_TACE}/run_tace": "ask",
    f"tool/{CAP_DPA4}/run_dpa4": "ask",
    f"tool/{CAP_DIALOG}/ask_user_question": "ask",
}

ASK_USER_QUESTION = "ask_user_question"

# DeferredToolRequests 的 JSON 序列化适配器（M3 持久化挂起现场用，
# 服务重启后据此重建 DeferredToolResume）
_DEFERRED_ADAPTER: TypeAdapter[DeferredToolRequests] = TypeAdapter(DeferredToolRequests)


def dump_deferred_json(deferred: DeferredToolRequests) -> str:
    """DeferredToolRequests → JSON 字符串（可 load_deferred_json 往返）。"""
    return _DEFERRED_ADAPTER.dump_json(deferred).decode("utf-8")


def load_deferred_json(payload: str) -> DeferredToolRequests:
    """dump_deferred_json 的逆操作。"""
    return _DEFERRED_ADAPTER.validate_json(payload)


def build_approval_results(
    deferred: DeferredToolRequests,
    *,
    approve_all: bool = True,
    answers: Mapping[str, str] | None = None,
    denials: Mapping[str, str] | None = None,
) -> DeferredToolResults:
    """把挂起的 deferred 请求转成结果批次。

    Args:
        deferred: 挂起 run 的 DeferredToolRequests（result.deferred）。
        approve_all: 未显式列出的审批一律通过。
        answers: tool_call_id → 用户文字答复（仅对 ask_user_question 生效，
            通过 override_args 注入 answer 参数）。
        denials: tool_call_id → 拒绝理由（ToolDenied）。
    """
    answers = dict(answers or {})
    denials = dict(denials or {})
    approvals: dict[str, Any] = {}
    for call in deferred.approvals:
        if call.tool_call_id in denials:
            approvals[call.tool_call_id] = ToolDenied(message=denials[call.tool_call_id])
            continue
        answer = answers.get(call.tool_call_id)
        if answer and call.tool_name == ASK_USER_QUESTION:
            args = dict(call.args_as_dict())
            args["answer"] = answer
            approvals[call.tool_call_id] = ToolApproved(override_args=args)
    return deferred.build_results(approvals=approvals, approve_all=approve_all)
