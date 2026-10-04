"""工具闭包工厂（PLAN §2.2）：注入 session workspace 与 job 目录分配器。

build_tools(session_workspace, settings=None, job_allocator=None) -> list[Capability]
返回 pydantic-ai Capability 列表，交给 app/agent/harness.py 组装进主 agent。

capability id 决定权限规则的 tool_id（`tool/<capability_id>/<tool_name>`，
见 app/agent/permissions.py）。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic_ai.capabilities import Capability

from app.agent.permissions import (
    CAP_DIALOG,
    CAP_DPA4,
    CAP_MAPLE,
    CAP_NEQUIP,
    CAP_STRUCTURE,
    CAP_TACE,
)
from app.config import Settings, get_settings
from app.tools.common import JobAllocator, ToolServices, ok
from app.tools.maple_tools import build_maple_tools
from app.tools.mlip_tools import build_dpa4_tools, build_nequip_tools, build_tace_tools
from app.tools.structure_tools import build_structure_tools


def _build_dialog_tools() -> list[Callable]:
    def ask_user_question(question: str, answer: str = "") -> dict:
        """向用户追问缺失信息（模型选择、电荷、自旋多重度、结构等）。
        调用后会话挂起等待用户；用户答复由系统在 answer 参数注入，调用时不要填。

        Args:
            question: 要问用户的问题（中文，具体、可一次性回答）。
            answer: 系统注入的用户答复，模型调用时保持默认空串。
        """
        if answer:
            return ok({"answer": answer})
        return ok({"answer": "", "note": "用户已看到问题但未提供文字答复"})

    return [ask_user_question]


def build_tools(
    session_workspace: Path,
    settings: Settings | None = None,
    job_allocator: JobAllocator | None = None,
) -> list[Capability]:
    """构建一个 session 的全部工具 capability。

    Args:
        session_workspace: 会话工作目录（结构文件、jobs/ 都在其下）。
        settings: 缺省取 get_settings()。
        job_allocator: 缺省在 session_workspace/jobs 下新建。
    """
    settings = settings or get_settings()
    session_workspace = Path(session_workspace)
    session_workspace.mkdir(parents=True, exist_ok=True)
    allocator = job_allocator or JobAllocator(session_workspace / "jobs")
    services = ToolServices(
        settings=settings,
        session_workspace=session_workspace,
        job_allocator=allocator,
    )
    return [
        Capability(id=CAP_STRUCTURE, tools=build_structure_tools(services)),
        Capability(id=CAP_MAPLE, tools=build_maple_tools(services)),
        Capability(id=CAP_NEQUIP, tools=build_nequip_tools(services)),
        Capability(id=CAP_TACE, tools=build_tace_tools(services)),
        Capability(id=CAP_DPA4, tools=build_dpa4_tools(services)),
        Capability(id=CAP_DIALOG, tools=_build_dialog_tools()),
    ]
