"""模型清单端点（PLAN §2.5）：复用工具层 MODEL_CATALOG（单一事实来源）。"""

from __future__ import annotations

from fastapi import APIRouter

from app.state.schemas import ModelInfo
from app.tools.maple_tools import MODEL_CATALOG

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models() -> list[ModelInfo]:
    return [ModelInfo(**m) for m in MODEL_CATALOG]
