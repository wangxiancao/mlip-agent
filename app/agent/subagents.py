"""inline 子 agent 声明（PLAN §2.1）。

本模块只放纯数据（不 import a13n-harness，SDK 隔离见 harness.py）；
SubagentDefinition 的实际构造在 harness.py 完成。
"""

from __future__ import annotations

from typing import TypedDict

from app.agent.instructions import (
    APPROVAL_ADVISOR_PROMPT,
    RESULT_VALIDATOR_PROMPT,
    TASK_PARSER_PROMPT,
)


class SubagentSpec(TypedDict):
    name: str
    description: str
    instructions: str


ORCHESTRATOR_NAME = "mlip_orchestrator"

SUBAGENT_SPECS: tuple[SubagentSpec, ...] = (
    {
        "name": "task_parser",
        "description": "把自然语言计算请求转成严格 JSON JobSpec；缺字段时列 missing_fields",
        "instructions": TASK_PARSER_PROMPT,
    },
    {
        "name": "result_validator",
        "description": "校验计算摘要（收敛/虚频/能量量级/MD 漂移），输出 {ok, issues, summary}",
        "instructions": RESULT_VALIDATOR_PROMPT,
    },
    {
        "name": "approval_advisor",
        "description": "对高危操作（TS 搜索、长 MD、11GB 大模型）生成给用户的风险说明文本",
        "instructions": APPROVAL_ADVISOR_PROMPT,
    },
)
