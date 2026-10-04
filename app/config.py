"""后端配置（PLAN §0.4）：全部 LLM 端点与路径走环境变量。

环境变量：
    MLIP_AGENT_LLM_MODEL / MLIP_AGENT_LLM_BASE_URL / MLIP_AGENT_LLM_API_KEY
    COMPUTE_VENV            计算环境根目录（默认 /root/soft/ml_potential/.venv）
    UMA_CHECKPOINT_PATH     UMA 本地 checkpoint 绝对路径
    NEQUIP_MODEL_PATH       NequIP-OAM-XL 本地 zip 绝对路径
    MACEMPA0_MODEL_PATH     MACE-MPA-0 本地 scripted wrapper 绝对路径
    TACE_MODEL_PATH         TACE-OAM-L 本地 checkpoint 绝对路径
    DPA4_VENV               DPA-4 常驻 venv 根目录（TICKET-0006，默认 /root/soft/ml_potential/dpa4_venv）
    DPA4_MODEL_PATH         DPA-4 本地 checkpoint 绝对路径
    DPA4C_MODEL_PATH        DPA4C（OMol25 单域族）本地 checkpoint 绝对路径
    MLIP_AGENT_DATA_DIR     会话/作业数据目录（默认 ./data）
    MLIP_AGENT_DEFAULT_TIMEOUT_S  计算子进程默认超时（默认 1800）

无 LLM key 时 Settings 可正常实例化；模型解析延迟到 harness run 时（harness.py）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLIP_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM 端点（OpenAI 兼容）；空字符串 = 未配置
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""

    # 计算环境（COMPUTE_VENV 无前缀，与 conftest / scripts 一致）
    compute_venv: Path = Field(
        default=_REPO_ROOT / ".venv", validation_alias="COMPUTE_VENV"
    )
    uma_checkpoint_path: str = Field(
        default="/root/soft/ml_potential/checkpoints/uma/uma-s-1p1.pt",
        validation_alias="UMA_CHECKPOINT_PATH",
    )
    nequip_model_path: str = Field(
        default="/root/soft/ml_potential/checkpoints/nequip/NequIP-OAM-XL-0.1.nequip.zip",
        validation_alias="NEQUIP_MODEL_PATH",
    )
    macempa0_model_path: str = Field(
        default="/root/soft/ml_potential/checkpoints/mace/macempa0.pt",
        validation_alias="MACEMPA0_MODEL_PATH",
    )
    tace_model_path: str = Field(
        default="/root/soft/ml_potential/checkpoints/tace/TACE-OAM-L.pt",
        validation_alias="TACE_MODEL_PATH",
    )
    # DPA-4 常驻 venv（TICKET-0006 裁定 1：mlip_agent 之外、与 COMPUTE_VENV 平级）
    dpa4_venv: Path = Field(
        default="/root/soft/ml_potential/dpa4_venv", validation_alias="DPA4_VENV"
    )
    dpa4_model_path: str = Field(
        default="/root/soft/ml_potential/checkpoints/dpa4/DPA4-beta.pt",
        validation_alias="DPA4_MODEL_PATH",
    )
    # DPA4C 单域族（OMol25，无 multitask head；job.json family="dpa4c" 选用）
    dpa4c_model_path: str = Field(
        default="/root/soft/ml_potential/checkpoints/dpa4/DPA4C-Air-OMol25-100M-v20260820.pt",
        validation_alias="DPA4C_MODEL_PATH",
    )

    data_dir: Path = Path("data")
    default_timeout_s: int = Field(default=1800, gt=0)
    # 结构构建等轻量子进程的超时（不走 GPU 锁）
    light_timeout_s: int = Field(default=120, gt=0)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_model and self.llm_api_key)

    @property
    def maple_bin(self) -> Path:
        return self.compute_venv / "bin" / "maple"

    @property
    def compute_python(self) -> Path:
        return self.compute_venv / "bin" / "python"

    @property
    def dpa4_python(self) -> Path:
        """DPA-4 子进程解释器（runner argv 首元素，TICKET-0006）。"""
        return self.dpa4_venv / "bin" / "python"


@lru_cache
def get_settings() -> Settings:
    return Settings()
