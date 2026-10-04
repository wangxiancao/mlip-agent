"""依赖注入（PLAN §2.5）：AppServices 单例挂 app.state，路由经 get_services 取。

AppServices 持有：
- settings / session store；
- harness 缓存：每会话一个 ExecutableAgent（工具闭包绑定该会话 workspace），
  服务重启后缓存为空，首个请求触发 lazy 重建；
- model_override：测试注入 FunctionModel 的口子（生产为 None）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request

from app.agent.harness import ModelOverride, build_harness
from app.config import Settings
from app.state.store import SessionStore
from app.tools.registry import build_tools


@dataclass
class AppServices:
    settings: Settings
    store: SessionStore
    model_override: ModelOverride = None
    _harness_cache: dict[str, object] = field(default_factory=dict)

    def get_harness(self, session_id: str):
        """取（或 lazy 构建）某会话的 ExecutableAgent。"""
        if session_id not in self._harness_cache:
            capabilities = build_tools(
                self.store.workspace(session_id), settings=self.settings
            )
            self._harness_cache[session_id] = build_harness(
                self.settings, capabilities, model_override=self.model_override
            )
        return self._harness_cache[session_id]

    def close(self) -> None:
        self._harness_cache.clear()


def get_services(request: Request) -> AppServices:
    return request.app.state.services
