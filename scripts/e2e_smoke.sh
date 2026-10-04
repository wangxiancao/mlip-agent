#!/usr/bin/env bash
# e2e 冒烟（M6）：无 LLM key 的全链路端到端证明（除真实 LLM 外全真实执行）。
# 需要 GPU + COMPUTE_VENV 可用（跑真实 ANI-2x 单点）；退出码反映成败。
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/e2e_smoke.py "$@"
