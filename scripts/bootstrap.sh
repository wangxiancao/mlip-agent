#!/usr/bin/env bash
# mlip_agent 一键环境引导（幂等）。
# 只做 agent/后端环境（mlip_agent/.venv, Python 3.13）。
# 计算环境 COMPUTE_VENV 是既有环境，本脚本只做存在性校验，绝不修改。
set -euo pipefail

cd "$(dirname "$0")/.."

UV="${UV:-/root/.local/bin/uv}"
COMPUTE_VENV="${COMPUTE_VENV:-/root/soft/ml_potential/.venv}"

echo "[bootstrap] uv: $("$UV" --version)"

# 1. Python 3.13（a13n-harness 硬性要求）+ 后端 venv
"$UV" python install 3.13
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.13
fi

# 2. 依赖按 uv.lock 精确复现（锁文件入库，钉死 a13n-harness==0.5.2）
"$UV" sync --locked

# 3. 计算环境只校验不修改
for exe in "$COMPUTE_VENV/bin/maple" "$COMPUTE_VENV/bin/python"; do
  if [ ! -x "$exe" ]; then
    echo "[bootstrap] ERROR: $exe 不存在或不可执行" >&2
    exit 1
  fi
done

# 4. 前端工具链校验（M4 需要）
node --version
npm --version

# 5. 冒烟：后端包可导入；计算环境自检脚本可跑
.venv/bin/python -c "import a13n_harness, fastapi, uvicorn, pydantic_settings; print('[bootstrap] backend imports OK')"
"$COMPUTE_VENV/bin/python" app/compute/scripts/pinspect.py

echo "[bootstrap] done"
