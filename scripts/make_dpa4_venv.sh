#!/usr/bin/env bash
# 幂等重建 DPA-4 常驻 venv（TICKET-0006 阶段 2 / D7-8）。
# 配方完全复刻 docs/dpa4_stage1_report.md §1（TICKET-0005 阶段 1 零足迹验证）：
#   CPython 3.12 + torch 2.8.0+cu128 + deepmd-kit 3.2.0 源码重编译（CUDA 12.8, sm_86）
#   + vesin/vesin-torch 0.6.2 + mpich 5.0.2 + e3nn 0.6.0 + ase
# 与阶段 1 的差异（裁定 1 / 前置条件 2）：
#   - 目标为常驻 venv：/root/soft/ml_potential/dpa4_venv（mlip_agent 之外，与 COMPUTE_VENV 平级）；
#   - 纯构建期依赖（cmake/ninja/scikit-build-core/setuptools_scm/hatch_fancy_pypi_readme
#     /dependency_groups 等）只进 /tmp scratch 构建 venv，产出 deepmd-kit wheel 后即删除，
#     不装入常驻 venv；
#   - 运行任何 deepmd.pt 进程需 LD_LIBRARY_PATH=/usr/local/lib64（gcc 11.3 产物要
#     GLIBCXX_3.4.29），由 run_dpa4 工具经 run_compute(extra_env=...) 无条件注入（前置条件 1）。
# 安全：脚本显式校验目标路径，绝不触碰 COMPUTE_VENV / mlip_agent/.venv。
# 网络：本机 socks 代理会拦 uv/pip，所有网络命令经 env -u *_PROXY 前缀剥离代理变量。
# 用法：bash scripts/make_dpa4_venv.sh    （重复运行 = 删旧重建，幂等；约 15 min）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # /root/soft/ml_potential
DPA4_VENV="${DPA4_VENV:-$REPO_ROOT/dpa4_venv}"
COMPUTE_VENV="${COMPUTE_VENV:-$REPO_ROOT/.venv}"
MLIP_AGENT_VENV="$REPO_ROOT/mlip_agent/.venv"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
DEEPMD_VERSION="3.2.0"
TORCH_VERSION="2.8.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

# 剥离本机代理变量（socks 代理对 uv/pip 不通）
NETENV=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy)

die() { echo "ERROR: $*" >&2; exit 1; }

# --- 安全校验：绝不触碰 COMPUTE_VENV / mlip_agent/.venv ---
DPA4_REAL="$(readlink -f "$DPA4_VENV")"
for forbidden in "$COMPUTE_VENV" "$MLIP_AGENT_VENV"; do
  [ "$DPA4_REAL" = "$(readlink -f "$forbidden")" ] && die "DPA4_VENV 禁止等于 $forbidden"
done
case "$DPA4_REAL" in
  "$REPO_ROOT/.venv"* | "$REPO_ROOT/mlip_agent"*)
    die "DPA4_VENV 必须位于 mlip_agent 之外且非 COMPUTE_VENV: $DPA4_VENV" ;;
esac
[ -x "$CUDA_HOME/bin/nvcc" ] || die "找不到 nvcc（需 CUDA 12.8）: $CUDA_HOME"
command -v uv >/dev/null || die "找不到 uv"

echo "== 目标常驻 venv: $DPA4_VENV（COMPUTE_VENV=$COMPUTE_VENV 不触碰）=="

BUILD="$(mktemp -d /tmp/dpa4_build.XXXXXX)"
echo "== scratch 构建目录: $BUILD =="

# --- 1. 构建 venv（torch + 构建期依赖），产出 deepmd-kit wheel ---
echo "== [1/5] 构建 venv: torch $TORCH_VERSION+cu128 =="
"${NETENV[@]}" uv venv "$BUILD/venv" --python 3.12 --seed
BPY="$BUILD/venv/bin/python"
"${NETENV[@]}" uv pip install --python "$BPY" --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
"${NETENV[@]}" uv pip install --python "$BPY" \
  "scikit-build-core>=1" "cmake>=3.25.2" ninja dependency_groups packaging \
  setuptools_scm hatch_fancy_pypi_readme

echo "== [2/5] 源码重编译 deepmd-kit $DEEPMD_VERSION（DP_VARIANT=cuda, sm_86，约 5 min）=="
PATH="$CUDA_HOME/bin:$PATH" \
DP_VARIANT=cuda CUDAToolkit_ROOT="$CUDA_HOME" \
DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 TORCH_CUDA_ARCH_LIST="8.6" \
"${NETENV[@]}" "$BPY" -m pip wheel \
  --no-deps --no-build-isolation --no-binary deepmd-kit \
  -w "$BUILD/wheelhouse" "deepmd-kit==$DEEPMD_VERSION"
ls "$BUILD/wheelhouse"

# --- 2. 常驻 venv（rm -rf 后重建，幂等）---
echo "== [3/5] 重建常驻 venv（先删旧）: $DPA4_VENV =="
rm -rf "$DPA4_VENV"
"${NETENV[@]}" uv venv "$DPA4_VENV" --python 3.12
DPY="$DPA4_VENV/bin/python"
"${NETENV[@]}" uv pip install --python "$DPY" --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
# deepmd-kit 用刚编译的 wheel，--no-deps + 显式运行期依赖（严禁 [torch] extra：其 pin torch==2.11.0）
"${NETENV[@]}" uv pip install --python "$DPY" --no-deps "$BUILD/wheelhouse"/deepmd_kit-"$DEEPMD_VERSION"-*.whl

echo "== [4/5] 运行期依赖（e3nn 0.6.0 / vesin 0.6.2 / mpich / ase 等）=="
"${NETENV[@]}" uv pip install --python "$DPY" \
  numpy scipy pyyaml dargs h5py wcmatch einops packaging ml_dtypes mendeleev \
  array-api-compat lmdb msgpack \
  e3nn==0.6.0 mpich==5.0.2 vesin==0.6.2 vesin-torch==0.6.2 ase

# --- 3. 冒烟验证 + pip freeze ---
echo "== [5/5] import 冒烟（LD_LIBRARY_PATH=/usr/local/lib64）=="
LD_LIBRARY_PATH=/usr/local/lib64 "$DPY" - <<'PY'
import importlib.metadata as im
import torch
import deepmd.pt  # noqa: F401  # 触发 libdeepmd_op_pt 加载，验证 ABI
from deepmd.infer import DeepPot  # noqa: F401
import ase, e3nn, vesin  # noqa: F401
for pkg in ("deepmd-kit", "torch", "e3nn", "vesin", "vesin-torch", "mpich", "ase", "numpy"):
    print(f"  {pkg}=={im.version(pkg)}")
print(f"  torch cuda: {torch.version.cuda}, threads={torch.get_num_threads()}")
PY

echo "== pip freeze（常驻 venv 全文，需抄录到 docs/decisions.md D7-8）=="
"${NETENV[@]}" uv pip freeze --python "$DPY"

echo "== 清理 scratch 构建目录 =="
rm -rf "$BUILD"

echo "== 磁盘占用 =="
du -sh "$DPA4_VENV"
echo "== 完成：$DPA4_VENV（run_dpa4 运行需 LD_LIBRARY_PATH=/usr/local/lib64，由工具层注入）=="
