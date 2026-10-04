#!/usr/bin/env bash
# 生成 benchmarks/references/ 金标准（B1/B2/B3/B4/B8 MAPLE 路径；B5 NequIP 直算路径，M1-B 补入；
# B11 NequIP-OAM-XL MAPLE 插件路径，TICKET-0002 补入；
# B12/B13/B14 = ani1x/aimnet2/maceoff23s H2O sp，模型覆盖扩展，D7-4 补入；
# B15 TACE-OAM-L 直算路径，TICKET-0004 补入；
# B16 DPA-4 直算路径 CPU+OMP=1 pin 口径，TICKET-0006 补入；
# B20 MACE-MPA-0 medium MAPLE 插件路径（macempa0，材料域模型算分子，纯数值 pin）；
# B21 DPA4C-Air（family=dpa4c）H2O sp，2026-10-03 补入；
# B22 aimnet2nse（AIMNet2-NSE 开壳层）O2 三重态 sp，TICKET-0007 补入；
# B23/B24/B25/B26 = Si 金刚石原胞 PBC sp（DPA-4 OMat24 / NequIP / TACE / UMA-omat），
# 2026-10-03 PBC 落地补入；B27 = Si PBC sp（macempa0 插件，TICKET-0008 验收）；
# B28 = H2O 20 Å 周期盒 sp（aimnet2，TICKET-0009 验收）。
# 全部计算通过 $COMPUTE_VENV 子进程完成（B16/B21/B23 例外：用 $DPA4_VENV 解释器）；可重复运行（覆盖式再生成）。
set -uo pipefail

cd "$(dirname "$0")/.."
MLIP_AGENT_DIR="$PWD"

COMPUTE_VENV="${COMPUTE_VENV:-/root/soft/ml_potential/.venv}"
MAPLE="$COMPUTE_VENV/bin/maple"
CPY="$COMPUTE_VENV/bin/python"
MAPLE_EXAMPLES="${MAPLE_EXAMPLES:-/root/soft/ml_potential/MAPLE/examples}"
UMA_CKPT="${UMA_CHECKPOINT_PATH:-/root/soft/ml_potential/checkpoints/uma/uma-s-1p1.pt}"
NEQUIP_ZIP="${NEQUIP_MODEL_PATH:-/root/soft/ml_potential/checkpoints/nequip/NequIP-OAM-XL-0.1.nequip.zip}"
MACEMPA0_PT="${MACEMPA0_MODEL_PATH:-/root/soft/ml_potential/checkpoints/mace/macempa0.pt}"
TACE_PT="${TACE_MODEL_PATH:-/root/soft/ml_potential/checkpoints/tace/TACE-OAM-L.pt}"
DPA4_VENV="${DPA4_VENV:-/root/soft/ml_potential/dpa4_venv}"
DPY="$DPA4_VENV/bin/python"
DPA4_PT="${DPA4_MODEL_PATH:-/root/soft/ml_potential/checkpoints/dpa4/DPA4-beta.pt}"
REFDIR="$MLIP_AGENT_DIR/benchmarks/references"
WORK="$(mktemp -d /tmp/mlip_golden.XXXXXX)"
GEN_DATE="$(date +%F)"

mkdir -p "$REFDIR"
# 与 runner.py 的子进程 env 白名单一致：杜绝联网下载
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

run_maple() { # <inp> <out>；返回 maple 退出码，不中断脚本
  (cd "$WORK" && timeout 1800 "$MAPLE" "$1" "$2" >"$2.stdout" 2>"$2.stderr")
}

echo "== B1: sp (ANI-2x, MAPLE examples/sp/sp.inp 复制件) =="
cp "$MAPLE_EXAMPLES/sp/sp.inp" "$WORK/b1_sp.inp"
run_maple b1_sp.inp b1_sp.out && cp "$WORK/b1_sp.out" "$REFDIR/B1_sp.out" || echo "B1 FAILED rc=$?"

echo "== B2: H2O opt (ANI-2x) =="
cat > "$WORK/b2_opt.inp" <<'EOF'
#model=ANI-2x
#opt()
#device=gpu0

O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000
EOF
run_maple b2_opt.inp b2_opt.out && cp "$WORK/b2_opt.out" "$WORK/b2_opt_opt.xyz" "$REFDIR/" && \
  mv "$REFDIR/b2_opt.out" "$REFDIR/B2_opt.out" && mv "$REFDIR/b2_opt_opt.xyz" "$REFDIR/B2_opt.xyz" \
  || echo "B2 FAILED rc=$?"

echo "== B3: H2O freq (ANI-2x, 基于 B2 优化结构) =="
if [ -f "$WORK/b2_opt_opt.xyz" ]; then
  cat > "$WORK/b3_freq.inp" <<'EOF'
#model=ANI-2x
#freq
#device=gpu0

XYZ b2_opt_opt.xyz
EOF
  run_maple b3_freq.inp b3_freq.out && cp "$WORK/b3_freq.out" "$REFDIR/B3_freq.out" || echo "B3 FAILED rc=$?"
else
  echo "B3 SKIPPED: B2 未产出优化结构"
fi

echo "== B4: H2O sp (UMA uma-s-1p1/omol, 本地 checkpoint) =="
cat > "$WORK/b4_uma_sp.inp" <<EOF
#model=uma(size=uma-s-1p1,task=omol,checkpoint_path=$UMA_CKPT)
#sp(verbose=0)
#device=gpu0

O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000
EOF
run_maple b4_uma_sp.inp b4_uma_sp.out && cp "$WORK/b4_uma_sp.out" "$REFDIR/B4_sp.out" || echo "B4 FAILED rc=$?"

echo "== B5: H2O sp (NequIP-OAM-XL, 直算 ASE 路径) =="
cat > "$WORK/b5_h2o.xyz" <<'EOF'
3
H2O for NequIP-OAM-XL B5 golden（与 B1/B2/B4 输入几何一致，O-H 0.96 Å）
O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000
EOF
mkdir -p "$WORK/b5_out"
cat > "$WORK/b5_job.json" <<EOF
{
  "xyz_path": "$WORK/b5_h2o.xyz",
  "task": "sp",
  "model_path": "$NEQUIP_ZIP",
  "device": "cuda",
  "output_dir": "$WORK/b5_out"
}
EOF
if timeout 1800 "$CPY" "$MLIP_AGENT_DIR/app/compute/scripts/nequip_ase.py" "$WORK/b5_job.json" >"$WORK/b5.stdout" 2>"$WORK/b5.stderr"; then
  cp "$WORK/b5_h2o.xyz" "$REFDIR/B5_h2o.xyz"
  cp "$WORK/b5_job.json" "$REFDIR/B5_job.json"
  cp "$WORK/b5_out/result.json" "$REFDIR/B5_result.json"
else
  echo "B5 FAILED rc=$?"
fi

echo "== B15: H2O sp (TACE-OAM-L, 直算 ASE 路径, TICKET-0004) =="
# 几何复用 B5 金标准（B5_h2o.xyz，O-H 0.96 Å）；neighborlist_backend 由脚本固定 'ase'
# （前置条件 1：matscipy 对零胞分子 Singular matrix）。pin 前置条件 2：同机连续
# ≥2 次逐位一致——本段只跑一次，逐位一致性证据见 docs/decisions.md D7-6。
cp "$WORK/b5_h2o.xyz" "$WORK/b15_h2o.xyz" 2>/dev/null || cp "$REFDIR/B5_h2o.xyz" "$WORK/b15_h2o.xyz"
mkdir -p "$WORK/b15_out"
cat > "$WORK/b15_job.json" <<EOF
{
  "xyz_path": "$WORK/b15_h2o.xyz",
  "task": "sp",
  "model_path": "$TACE_PT",
  "device": "cuda",
  "output_dir": "$WORK/b15_out"
}
EOF
if timeout 1800 "$CPY" "$MLIP_AGENT_DIR/app/compute/scripts/tace_ase.py" "$WORK/b15_job.json" >"$WORK/b15.stdout" 2>"$WORK/b15.stderr"; then
  cp "$WORK/b15_job.json" "$REFDIR/B15_job.json"
  cp "$WORK/b15_out/result.json" "$REFDIR/B15_result.json"
else
  echo "B15 FAILED rc=$?"
fi

echo "== B16: H2O sp (DPA-4 直算路径, CPU+OMP=1 pin 口径, TICKET-0006) =="
# 几何复用 B5 金标准（B5_h2o.xyz）；device=cpu，脚本进程内强制 OMP_NUM_THREADS=1 +
# torch.set_num_threads(1)（裁定 2）；LD_LIBRARY_PATH=/usr/local/lib64 必需（前置条件 1）。
# pin 前置条件：同机连续 ≥2 次逐位一致——本段只跑一次，逐位一致性证据见 docs/decisions.md D7-8。
cp "$WORK/b5_h2o.xyz" "$WORK/b16_h2o.xyz" 2>/dev/null || cp "$REFDIR/B5_h2o.xyz" "$WORK/b16_h2o.xyz"
mkdir -p "$WORK/b16_out"
cat > "$WORK/b16_job.json" <<EOF
{
  "xyz_path": "$WORK/b16_h2o.xyz",
  "task": "sp",
  "model_path": "$DPA4_PT",
  "device": "cpu",
  "output_dir": "$WORK/b16_out"
}
EOF
if LD_LIBRARY_PATH=/usr/local/lib64 timeout 1800 "$DPY" "$MLIP_AGENT_DIR/app/compute/scripts/dpa4_ase.py" "$WORK/b16_job.json" >"$WORK/b16.stdout" 2>"$WORK/b16.stderr"; then
  cp "$WORK/b16_job.json" "$REFDIR/B16_job.json"
  cp "$WORK/b16_out/result.json" "$REFDIR/B16_result.json"
else
  echo "B16 FAILED rc=$?"
fi

echo "== B11: H2O sp (NequIP-OAM-XL, MAPLE 插件后端, TICKET-0002) =="
# 插件路径：env 注入 MAPLE_CALCULATOR_PLUGINS + PYTHONPATH（与 runner extra_env 一致）；
# B5 直算路径与其 eV pin 保持原样，此处新建独立 Hartree pin（前置条件 2）
cat > "$WORK/b11_sp.inp" <<EOF
#model=nequip-oam-xl(model_path=$NEQUIP_ZIP)
#sp(verbose=0)
#device=gpu0

O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000
EOF
if (cd "$WORK" && env MAPLE_CALCULATOR_PLUGINS=maple_nequip_plugin \
      PYTHONPATH="$MLIP_AGENT_DIR/plugins" \
      timeout 1800 "$MAPLE" b11_sp.inp b11_sp.out >b11_sp.out.stdout 2>b11_sp.out.stderr); then
  cp "$WORK/b11_sp.inp" "$REFDIR/B11_sp.inp"
  cp "$WORK/b11_sp.out" "$REFDIR/B11_sp.out"
else
  echo "B11 FAILED rc=$?"
fi

echo "== B20: H2O sp (MACE-MPA-0 medium, MAPLE 插件后端 macempa0) =="
# 插件路径同 B11：env 注入 MAPLE_CALCULATOR_PLUGINS + PYTHONPATH。
# 注意：MPTrj+sAlex 晶体训练域模型算单分子属域外使用，本 pin 仅为数值复现锚点，
# 不代表分子体系物理精度（与 mace 官方 ASE calculator 对照差 7.5e-14 eV，
# eV→Hartree 链差 4.9e-11 Ha）；wrapper 由 scripts/make_macempa0_wrapper.py 生成。
cat > "$WORK/b20_sp.inp" <<EOF
#model=macempa0(model_path=$MACEMPA0_PT)
#sp(verbose=0)
#device=gpu0

O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000
EOF
if (cd "$WORK" && env MAPLE_CALCULATOR_PLUGINS=maple_mace_plugin \
      PYTHONPATH="$MLIP_AGENT_DIR/plugins" \
      timeout 1800 "$MAPLE" b20_sp.inp b20_sp.out >b20_sp.out.stdout 2>b20_sp.out.stderr); then
  cp "$WORK/b20_sp.inp" "$REFDIR/B20_sp.inp"
  cp "$WORK/b20_sp.out" "$REFDIR/B20_sp.out"
else
  echo "B20 FAILED rc=$?"
fi

echo "== B12/B13/B14: H2O sp (ani1x / aimnet2 / maceoff23s, 模型覆盖扩展 D7-4) =="
for m in ani1x aimnet2 maceoff23s; do
  case "$m" in
    ani1x) BID=B12 ;;
    aimnet2) BID=B13 ;;
    maceoff23s) BID=B14 ;;
  esac
  # 三模型同用 B4/B5 金标准 H2O 几何；ani1x/aimnet2 首次使用由 MAPLE 自动下载
  # checkpoint（本段要求 checkpoint 已缓存或网络可达）；maceoff23s 无自动下载，
  # 需预置本地 scripted wrapper（scripts/make_maceoff23s_wrapper.py，见 D7-4）
  cat > "$WORK/b_${m}_sp.inp" <<EOF
#model=${m}
#sp(verbose=0)
#device=gpu0

O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000
EOF
  run_maple "b_${m}_sp.inp" "b_${m}_sp.out" && \
    cp "$WORK/b_${m}_sp.inp" "$REFDIR/${BID}_sp.inp" && \
    cp "$WORK/b_${m}_sp.out" "$REFDIR/${BID}_sp.out" || echo "${BID} FAILED rc=$?"
done

echo "== B8: H2O O-H 键刚性扫描 0.9 -> 1.1 Å, 5 点 (ANI-2x) =="
cat > "$WORK/b8_scan.inp" <<'EOF'
#model=ANI-2x
#scan(mode=rigid)
#device=gpu0

O      0.00000000      0.00000000      0.00000000
H      0.90000000      0.00000000      0.00000000
H     -0.23986000      0.92749000      0.00000000

S 1 2 0.05 4
EOF
run_maple b8_scan.inp b8_scan.out && cp "$WORK/b8_scan.out" "$REFDIR/B8_scan.out" || echo "B8 FAILED rc=$?"

echo "== B23/B24/B25: Si 金刚石原胞 sp（PBC，DPA-4/NequIP/TACE 直算路径，2026-10-03 PBC 落地）=="
# 结构：fcc 初基矢量菱形原胞（a=5.431 Å，2 原子），extxyz Lattice 头，pbc="T T T"。
# pin 前置条件：同机 ≥2 次复现（tace/dpa4-cpu 逐位一致；nequip GPU 差 4.4e-7 eV，容差内）。
cat > "$WORK/si_diamond.extxyz" <<'EOF'
2
Lattice="0.0000000000 2.7155000000 2.7155000000 2.7155000000 0.0000000000 2.7155000000 2.7155000000 2.7155000000 0.0000000000" pbc="T T T" Si diamond primitive cell (fcc primitive vectors) a=5.431
Si      0.00000000      0.00000000      0.00000000
Si      1.35775000      1.35775000      1.35775000
EOF
cp "$WORK/si_diamond.extxyz" "$REFDIR/si_diamond.extxyz"

mkdir -p "$WORK/b23_out"
cat > "$WORK/b23_job.json" <<EOF
{
  "xyz_path": "$WORK/si_diamond.extxyz",
  "task": "sp",
  "model_path": "$DPA4_PT",
  "device": "cpu",
  "head": "OMat24",
  "output_dir": "$WORK/b23_out"
}
EOF
if LD_LIBRARY_PATH=/usr/local/lib64 timeout 1800 "$DPY" "$MLIP_AGENT_DIR/app/compute/scripts/dpa4_ase.py" "$WORK/b23_job.json" >"$WORK/b23.stdout" 2>"$WORK/b23.stderr"; then
  cp "$WORK/b23_job.json" "$REFDIR/B23_job.json"
  cp "$WORK/b23_out/result.json" "$REFDIR/B23_result.json"
else
  echo "B23 FAILED rc=$?"
fi

mkdir -p "$WORK/b24_out"
cat > "$WORK/b24_job.json" <<EOF
{
  "xyz_path": "$WORK/si_diamond.extxyz",
  "task": "sp",
  "model_path": "$NEQUIP_ZIP",
  "device": "cuda",
  "output_dir": "$WORK/b24_out"
}
EOF
if timeout 1800 "$CPY" "$MLIP_AGENT_DIR/app/compute/scripts/nequip_ase.py" "$WORK/b24_job.json" >"$WORK/b24.stdout" 2>"$WORK/b24.stderr"; then
  cp "$WORK/b24_job.json" "$REFDIR/B24_job.json"
  cp "$WORK/b24_out/result.json" "$REFDIR/B24_result.json"
else
  echo "B24 FAILED rc=$?"
fi

mkdir -p "$WORK/b25_out"
cat > "$WORK/b25_job.json" <<EOF
{
  "xyz_path": "$WORK/si_diamond.extxyz",
  "task": "sp",
  "model_path": "$TACE_PT",
  "device": "cuda",
  "output_dir": "$WORK/b25_out"
}
EOF
if timeout 1800 "$CPY" "$MLIP_AGENT_DIR/app/compute/scripts/tace_ase.py" "$WORK/b25_job.json" >"$WORK/b25.stdout" 2>"$WORK/b25.stderr"; then
  cp "$WORK/b25_job.json" "$REFDIR/B25_job.json"
  cp "$WORK/b25_out/result.json" "$REFDIR/B25_result.json"
else
  echo "B25 FAILED rc=$?"
fi

echo "== B26: Si 金刚石原胞 sp（UMA uma-s-1p1/omat 经 MAPLE，XYZ 引用 extxyz-Lattice）=="
cat > "$WORK/b26_uma_sp.inp" <<EOF
#model=uma(size=uma-s-1p1,task=omat,checkpoint_path=$UMA_CKPT)
#sp(verbose=0)
#device=gpu0

XYZ si_diamond.extxyz
EOF
run_maple b26_uma_sp.inp b26_uma_sp.out && \
  cp "$WORK/b26_uma_sp.inp" "$REFDIR/B26_sp.inp" && \
  cp "$WORK/b26_uma_sp.out" "$REFDIR/B26_sp.out" || echo "B26 FAILED rc=$?"

echo "== B21: H2O sp（DPA4C-Air-OMol25，family=dpa4c，CPU+OMP=1 pin 口径，2026-10-03 接入）=="
# 几何复用 B5 金标准；DPA4C 无 multitask head（显式传 head 会报错，family 字段即防线）。
# 注意：DPA4C-OMol25 输出总能量，与 DPA4-beta（B16）参考能约定不同，不互比。
DPA4C_PT="${DPA4C_MODEL_PATH:-/root/soft/ml_potential/checkpoints/dpa4/DPA4C-Air-OMol25-100M-v20260820.pt}"
cp "$WORK/b5_h2o.xyz" "$WORK/b21_h2o.xyz" 2>/dev/null || cp "$REFDIR/B5_h2o.xyz" "$WORK/b21_h2o.xyz"
mkdir -p "$WORK/b21_out"
cat > "$WORK/b21_job.json" <<EOF
{
  "xyz_path": "$WORK/b21_h2o.xyz",
  "task": "sp",
  "family": "dpa4c",
  "model_path": "$DPA4C_PT",
  "device": "cpu",
  "output_dir": "$WORK/b21_out"
}
EOF
if LD_LIBRARY_PATH=/usr/local/lib64 timeout 1800 "$DPY" "$MLIP_AGENT_DIR/app/compute/scripts/dpa4_ase.py" "$WORK/b21_job.json" >"$WORK/b21.stdout" 2>"$WORK/b21.stderr"; then
  cp "$WORK/b21_job.json" "$REFDIR/B21_job.json"
  cp "$WORK/b21_out/result.json" "$REFDIR/B21_result.json"
else
  echo "B21 FAILED rc=$?"
fi

echo "== B27: Si 金刚石原胞 sp（MACE-MPA-0 经 MAPLE 插件，PBC，TICKET-0008 验收）=="
# 插件 env 注入（同 B11/B20 机制）；结构用 XYZ 引用带 Lattice 的 extxyz
MACEMPA0_PT="${MACEMPA0_MODEL_PATH:-/root/soft/ml_potential/checkpoints/mace/macempa0.pt}"
cat > "$WORK/b27_sp.inp" <<EOF
#model=macempa0(model_path=$MACEMPA0_PT)
#sp(verbose=0)
#device=gpu0

XYZ si_diamond.extxyz
EOF
if (cd "$WORK" && env MAPLE_CALCULATOR_PLUGINS=maple_mace_plugin \
      PYTHONPATH="$MLIP_AGENT_DIR/plugins" \
      timeout 1800 "$MAPLE" b27_sp.inp b27_sp.out >b27_sp.out.stdout 2>b27_sp.out.stderr); then
  cp "$WORK/b27_sp.inp" "$REFDIR/B27_sp.inp"
  cp "$WORK/b27_sp.out" "$REFDIR/B27_sp.out"
else
  echo "B27 FAILED rc=$?"
fi

echo "== B22: O2 三重态 sp（aimnet2nse，开壳层扩展，TICKET-0007）=="
# aimnet2nse.pt 为本地自转 TorchScript（scripts/make_aimnet2nse_ts.py，
# 已预置 MAPLE 模型目录）；坐标段首行 `0 3` 传 charge/mult
cat > "$WORK/b22_sp.inp" <<'EOF'
#model=aimnet2nse
#sp(verbose=0)
#device=gpu0

0 3
O      0.00000000      0.00000000      0.00000000
O      1.21000000      0.00000000      0.00000000
EOF
run_maple b22_sp.inp b22_sp.out && \
  cp "$WORK/b22_sp.inp" "$REFDIR/B22_sp.inp" && \
  cp "$WORK/b22_sp.out" "$REFDIR/B22_sp.out" || echo "B22 FAILED rc=$?"

echo "== B28: H2O 20 Å 周期盒 sp（aimnet2，PBC，TICKET-0009；库仑缺省自动 dsf）=="
cat > "$WORK/b28_sp.inp" <<'EOF'
#model=aimnet2
#sp(verbose=0)
#device=gpu0
#pbc(20,20,20)

O      0.00000000      0.00000000      0.00000000
H      0.96000000      0.00000000      0.00000000
H     -0.24000000      0.93000000      0.00000000
EOF
run_maple b28_sp.inp b28_sp.out && \
  cp "$WORK/b28_sp.inp" "$REFDIR/B28_sp.inp" && \
  cp "$WORK/b28_sp.out" "$REFDIR/B28_sp.out" || echo "B28 FAILED rc=$?"

echo "== 提取数值并写 Bx.json =="
"$CPY" - "$WORK" "$REFDIR" "$GEN_DATE" "$UMA_CKPT" "$NEQUIP_ZIP" "$TACE_PT" "$DPA4_PT" "$MACEMPA0_PT" "$DPA4C_PT" <<'PYEOF'
import json
import math
import re
import sys
from pathlib import Path

work, refdir, gen_date, uma_ckpt, nequip_zip, tace_pt, dpa4_pt, macempa0_pt, dpa4c_pt = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]

RE_SP = re.compile(r"^Energy:\s+(-?\d+\.\d+)\s+Hartree", re.M)
RE_ANY_E = re.compile(r"^Energy:\s+(-?\d+\.\d+)", re.M)
RE_FREQ = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", re.M)

def read(p):
    p = Path(p)
    return p.read_text() if p.is_file() else None

def dump(name, payload):
    payload["generated"] = gen_date
    payload["generator"] = "scripts/make_golden_refs.sh"
    (refdir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {name}")

# B1
txt = read(refdir / "B1_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B1.json", {
        "task": "sp", "model": "ANI-2x",
        "command": "maple b1_sp.inp b1_sp.out  (b1_sp.inp = MAPLE/examples/sp/sp.inp 复制件)",
        "artifacts": ["B1_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
        "note": "同机同库复现值；MAPLE 仓库自带 sp.out 为 -383.4722385600，差 <1e-6（GPU 数值噪声）",
    })
else:
    print("  B1.json SKIPPED: no B1_sp.out or energy not found")

# B2
txt = read(refdir / "B2_opt.out")
xyz = read(refdir / "B2_opt.xyz")
if txt and xyz:
    energies = RE_ANY_E.findall(txt)
    converged = "converged at iteration" in txt
    from ase.io import read as ase_read
    import io
    atoms = ase_read(io.StringIO(xyz), format="xyz")
    d_oh1 = atoms.get_distance(0, 1)
    d_oh2 = atoms.get_distance(0, 2)
    ang = atoms.get_angle(1, 0, 2)
    dump("B2.json", {
        "task": "opt", "model": "ANI-2x",
        "command": "maple b2_opt.inp b2_opt.out  (H2O, 起始 O-H 0.96 Å)",
        "artifacts": ["B2_opt.out", "B2_opt.xyz"],
        "values": {
            "converged": converged,
            "final_energy_hartree": float(energies[-1]) if energies else None,
            "oh_bond_angstrom": [round(d_oh1, 6), round(d_oh2, 6)],
            "hoh_angle_deg": round(ang, 4),
        },
        "tolerance": {"final_energy_hartree": 1e-6, "oh_bond_angstrom": 0.005, "hoh_angle_deg": 1.0},
    })
else:
    print("  B2.json SKIPPED")

# B3
txt = read(refdir / "B3_freq.out")
if txt:
    freqs = [float(x) for x in RE_FREQ.findall(txt)]
    real = sorted(f for f in freqs if f > 5.0)
    imag = [f for f in freqs if f < -5.0]
    if real:
        dump("B3.json", {
            "task": "freq", "model": "ANI-2x",
            "command": "maple b3_freq.inp b3_freq.out  (XYZ = B2 优化结构)",
            "artifacts": ["B3_freq.out"],
            "values": {"frequencies_cm-1": real, "n_imaginary": len(imag)},
            "tolerance": {"frequencies_cm-1": 20.0},
        })
    else:
        print("  B3.json SKIPPED: no real frequencies parsed")
else:
    print("  B3.json SKIPPED")

# B4
txt = read(refdir / "B4_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B4.json", {
        "task": "sp", "model": "uma(size=uma-s-1p1,task=omol)",
        "checkpoint": uma_ckpt,
        "command": "maple b4_uma_sp.inp b4_uma_sp.out  (H2O, 本地 checkpoint 绝对路径)",
        "artifacts": ["B4_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
    })
else:
    err_tail = ""
    for cand in (work / "b4_uma_sp.out.stderr", work / "b4_uma_sp.out.stdout"):
        if cand.is_file():
            err_tail += cand.read_text(errors="replace")[-2000:]
    dump("B4.json", {
        "task": "sp", "model": "uma(size=uma-s-1p1,task=omol)", "status": "failed",
        "reason": "UMA checkpoint 加载或计算失败，见 error_tail；judge 应 skip B4",
        "error_tail": err_tail,
        "values": {}, "tolerance": {},
    })

# B5
txt = read(refdir / "B5_result.json")
if txt:
    res = json.loads(txt)
    if res.get("ok"):
        dump("B5.json", {
            "task": "sp", "model": "NequIP-OAM-XL",
            "model_path": nequip_zip,
            "command": "$COMPUTE_VENV/bin/python app/compute/scripts/nequip_ase.py b5_job.json  (H2O, 起始 O-H 0.96 Å)",
            "artifacts": ["B5_h2o.xyz", "B5_job.json", "B5_result.json"],
            "values": {"energy_eV": res["energy_eV"]},
            "tolerance": {"energy_eV": 1e-6},
            "units": res["units"],
            "note": "NequIP ASE 计算器返回标准 ASE 单位（eV/Å），与 MAPLE Hartree 语义不同，勿混用",
        })
    else:
        dump("B5.json", {
            "task": "sp", "model": "NequIP-OAM-XL", "status": "failed",
            "reason": "NequIP 直算失败，见 B5_result.json 的 error；judge 应 skip B5",
            "error": res.get("error"),
            "values": {}, "tolerance": {},
        })
else:
    print("  B5.json SKIPPED")

# B15
txt = read(refdir / "B15_result.json")
if txt:
    res = json.loads(txt)
    if res.get("ok"):
        dump("B15.json", {
            "task": "sp", "model": "TACE-OAM-L",
            "model_path": tace_pt,
            "command": "$COMPUTE_VENV/bin/python app/compute/scripts/tace_ase.py b15_job.json  (H2O 金标准几何同 B5)",
            "artifacts": ["B5_h2o.xyz", "B15_job.json", "B15_result.json"],
            "values": {"energy_eV": res["energy_eV"]},
            "tolerance": {"energy_eV": 1e-6},
            "units": res["units"],
            "neighborlist_backend": res["neighborlist_backend"],
            "versions": res["versions"],
            "note": "TACE ASE 计算器返回标准 ASE 单位（eV/Å），与 MAPLE Hartree 语义不同，勿混用；"
                    "neighborlist_backend 固定 'ase'（matscipy 对零胞分子 Singular matrix，TICKET-0004 前置条件 1）；"
                    "pin 值经同机连续 2 次逐位一致验证（D7-6），与 TICKET-0004 提案的预估值不同，以本 pin 为准",
        })
    else:
        dump("B15.json", {
            "task": "sp", "model": "TACE-OAM-L", "status": "failed",
            "reason": "TACE 直算失败，见 B15_result.json 的 error；judge 应 skip B15",
            "error": res.get("error"),
            "values": {}, "tolerance": {},
        })
else:
    print("  B15.json SKIPPED")

# B16
txt = read(refdir / "B16_result.json")
if txt:
    res = json.loads(txt)
    if res.get("ok"):
        dump("B16.json", {
            "task": "sp", "model": "DPA-4 (DPA4-beta, head=H2O_H2O_PD)",
            "model_path": dpa4_pt,
            "command": "LD_LIBRARY_PATH=/usr/local/lib64 $DPA4_VENV/bin/python app/compute/scripts/dpa4_ase.py b16_job.json  (H2O 金标准几何同 B5，device=cpu)",
            "artifacts": ["B5_h2o.xyz", "B16_job.json", "B16_result.json"],
            "values": {"energy_eV": res["energy_eV"]},
            "tolerance": {"energy_eV": 1e-6},
            "units": res["units"],
            "head": res["head"],
            "charge_spin": res["charge_spin"],
            "versions": res["versions"],
            "note": "CPU+OMP=1 口径，GPU 值约 -15.7107726 eV 不可混用；"
                    "pin 值经同机连续 2 次逐位一致验证（D7-8）；只 pin 能量，不 pin 力"
                    "（力跨进程不逐位稳定 ~1e-7 eV/Å，根因未定位）；"
                    "head 默认 H2O_H2O_PD 可覆盖（checkpoint 23 分支，以 checkpoint 分支名为准）；"
                    "charge_spin 固定 [0.0,1.0]（中性闭壳层）；解释器为常驻 venv "
                    "/root/soft/ml_potential/dpa4_venv（TICKET-0006 裁定 1），"
                    "运行需 LD_LIBRARY_PATH=/usr/local/lib64",
        })
    else:
        dump("B16.json", {
            "task": "sp", "model": "DPA-4", "status": "failed",
            "reason": "DPA-4 直算失败，见 B16_result.json 的 error；judge 应 skip B16",
            "error": res.get("error"),
            "values": {}, "tolerance": {},
        })
else:
    print("  B16.json SKIPPED")

# B11
txt = read(refdir / "B11_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B11.json", {
        "task": "sp", "model": "nequip-oam-xl (MAPLE 插件后端, plugins/maple_nequip_plugin.py)",
        "model_path": nequip_zip,
        "command": "maple b11_sp.inp b11_sp.out  (env: MAPLE_CALCULATOR_PLUGINS=maple_nequip_plugin, PYTHONPATH=mlip_agent/plugins；H2O 几何同 B5)",
        "artifacts": ["B11_sp.inp", "B11_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
        "units": {"energy": "Hartree"},
        "note": "插件 MODEL_ENERGY_UNIT='eV'，eV→Hartree 由 _finalize_results 统一换算；"
                "应与 B5 直算 pin / 27.211386245988 一致（同机容差 1e-6 Ha），防双重换算",
    })
else:
    print("  B11.json SKIPPED: no B11_sp.out or energy not found")

# B20（MACE-MPA-0 medium，MAPLE 插件后端 macempa0）
txt = read(refdir / "B20_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B20.json", {
        "task": "sp", "model": "macempa0 (MACE-MPA-0 medium, MAPLE 插件后端, plugins/maple_mace_plugin.py)",
        "model_path": macempa0_pt,
        "command": "maple b20_sp.inp b20_sp.out  (env: MAPLE_CALCULATOR_PLUGINS=maple_mace_plugin, PYTHONPATH=mlip_agent/plugins；H2O 几何同 B4/B5)",
        "artifacts": ["B20_sp.inp", "B20_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
        "units": {"energy": "Hartree"},
        "note": "材料域模型（MPTrj+sAlex 晶体）算分子：纯数值 pin，不代表分子体系物理精度。"
                "wrapper 与 mace 官方 ASE calculator（mace_mp 直载源 .model）H2O 能量差 7.5e-14 eV；"
                "pin 与官方 eV 值 / 27.211386245988 差 4.9e-11 Ha，eV→Hartree 换算链正确"
                "（_finalize_results 统一换算，防双重换算）。同机连续 3 次复现逐位一致。"
                "wrapper 由 scripts/make_macempa0_wrapper.py 生成（MIT 许可源权重）",
    })
else:
    print("  B20.json SKIPPED: no B20_sp.out or energy not found")

# B12/B13/B14（D7-4 模型覆盖扩展；三模型同构，循环提取）
_B1214_NOTES = {
    "B12": "首次使用由 MAPLE 自动下载 ani1x.pt（Wayne7815/MAPLE_models，pinned revision）；"
           "checkpoint 落盘后在 HF_HUB_OFFLINE=1 下同机复现值逐位一致",
    "B13": "AIMNet2 支持电荷/多重度（charge 0 / mult 1 合法）；首次使用自动下载 aimnet2.pt，"
           "落盘后 HF_HUB_OFFLINE=1 下同机复现值逐位一致",
    "B14": "maceoff23s 不在 Wayne7815/MAPLE_models（REQUIRES_LOCAL_MODEL_FILE，无自动下载）；"
           "checkpoint 为本地自制 scripted wrapper（ACEsuit/mace-off 官方 MACE-OFF23_small.model "
           "经 e3nn jit.compile 封装，见 docs/decisions.md D7-4）；同机三次复现逐位一致",
}
for bid, model in (("B12", "ani1x"), ("B13", "aimnet2"), ("B14", "maceoff23s")):
    txt = read(refdir / f"{bid}_sp.out")
    if txt and (m := RE_SP.search(txt)):
        dump(f"{bid}.json", {
            "task": "sp", "model": model,
            "command": f"maple b_{model}_sp.inp b_{model}_sp.out  (H2O 金标准几何，同 B4/B5)",
            "artifacts": [f"{bid}_sp.inp", f"{bid}_sp.out"],
            "values": {"energy_hartree": float(m.group(1))},
            "tolerance": {"energy_hartree": 1e-6},
            "note": _B1214_NOTES[bid],
        })
    else:
        print(f"  {bid}.json SKIPPED: no {bid}_sp.out or energy not found")

# B23/B24/B25（2026-10-03 PBC 落地；Si 金刚石原胞，三直算后端同构，循环提取）
_SI_NOTE = ("Si 金刚石原胞（fcc 初基矢量，a=5.431 Å，2 原子，si_diamond.extxyz，"
            "extxyz Lattice 头；pbc=T T T）；同结构四后端互证：能量差 < 0.01 eV，"
            "对称点 |F|max < 1e-7 eV/Å")
for bid, model, label in (
    ("B23", "DPA-4 (DPA4-beta, head=OMat24)", "dpa4_ase.py b23_job.json（device=cpu，CPU+OMP=1 口径；同机 2 次逐位一致）"),
    ("B24", "NequIP-OAM-XL", "nequip_ase.py b24_job.json（device=cuda；GPU 噪声同机 2 次差 4.4e-7 eV，容差内）"),
    ("B25", "TACE-OAM-L", "tace_ase.py b25_job.json（device=cuda，邻居表自动切 matscipy；同机 2 次逐位一致）"),
):
    txt = read(refdir / f"{bid}_result.json")
    if txt:
        res = json.loads(txt)
        if res.get("ok"):
            dump(f"{bid}.json", {
                "task": "sp", "model": model,
                "command": f"{label}",
                "artifacts": ["si_diamond.extxyz", f"{bid}_job.json", f"{bid}_result.json"],
                "values": {"energy_eV": res["energy_eV"]},
                "tolerance": {"energy_eV": 1e-6},
                "units": res["units"],
                "versions": res["versions"],
                "note": "周期性 PBC pin（2026-10-03 落地）。" + _SI_NOTE,
            })
        else:
            dump(f"{bid}.json", {
                "task": "sp", "model": model, "status": "failed",
                "reason": f"{model} 直算失败，见 {bid}_result.json 的 error；judge 应 skip {bid}",
                "error": res.get("error"),
                "values": {}, "tolerance": {},
            })
    else:
        print(f"  {bid}.json SKIPPED")

# B26（UMA omat 经 MAPLE，周期）
txt = read(refdir / "B26_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B26.json", {
        "task": "sp", "model": "uma(size=uma-s-1p1,task=omat)",
        "checkpoint": uma_ckpt,
        "command": "maple b26_uma_sp.inp b26_uma_sp.out  (Si 金刚石原胞，XYZ 引用带 Lattice 的 extxyz)",
        "artifacts": ["si_diamond.extxyz", "B26_sp.inp", "B26_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
        "units": {"energy": "Hartree"},
        "note": "周期性 PBC pin（2026-10-03 落地）。UMA 是 MAPLE 内唯一 SUPPORTS_PBC 后端；"
                "omat 为周期 task（omol 与 PBC 互斥）。同机 2 次差 2.2e-9 Ha（容差内）。" + _SI_NOTE,
    })
else:
    print("  B26.json SKIPPED: no B26_sp.out or energy not found")

# B21（DPA4C 族，family=dpa4c；2026-10-03 接入）
txt = read(refdir / "B21_result.json")
if txt:
    res = json.loads(txt)
    if res.get("ok"):
        dump("B21.json", {
            "task": "sp", "model": "DPA4C-Air (DPA4C-Air-OMol25-100M-v20260820)",
            "model_path": dpa4c_pt,
            "command": "LD_LIBRARY_PATH=/usr/local/lib64 $DPA4_VENV/bin/python app/compute/scripts/dpa4_ase.py b21_job.json  (H2O 金标准几何同 B5，device=cpu，family=dpa4c)",
            "artifacts": ["B5_h2o.xyz", "B21_job.json", "B21_result.json"],
            "values": {"energy_eV": res["energy_eV"]},
            "tolerance": {"energy_eV": 1e-6},
            "units": res["units"],
            "family": res["family"],
            "charge_spin": res["charge_spin"],
            "versions": res["versions"],
            "note": "DPA4C 族无 multitask head（DeepPot 不带 head 加载；显式传 head 会静默忽略，"
                    "family 字段即防线）。CPU+OMP=1 口径同机 2 次逐位一致；GPU 对照差 ~1.2e-7 eV "
                    "不可混用。DPA4C-OMol25 输出总能量，与 DPA4-beta B16 pin 参考能约定不同不互比。"
                    "许可 CC BY-NC 4.0（非商业）。",
        })
    else:
        dump("B21.json", {
            "task": "sp", "model": "DPA4C-Air", "status": "failed",
            "reason": "DPA4C 直算失败，见 B21_result.json 的 error；judge 应 skip B21",
            "error": res.get("error"),
            "values": {}, "tolerance": {},
        })
else:
    print("  B21.json SKIPPED")

# B22（AIMNet2-NSE 开壳层，TICKET-0007；O2 三重态）
txt = read(refdir / "B22_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B22.json", {
        "task": "sp", "model": "aimnet2nse (AIMNet2-NSE, ωB97M-D3/def2-TZVPP UKS)",
        "command": "maple b22_sp.inp b22_sp.out  (O2 三重态，O-O 1.21 Å，charge 0 mult 3)",
        "artifacts": ["B22_sp.inp", "B22_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
        "note": "aimnet2nse.pt 为 scripts/make_aimnet2nse_ts.py 自转 TorchScript（HF "
                "isayevlab/aimnet2-nse ensemble_0）；与 v2 eager 对照 ΔE=2.8e-9 eV。"
                "仅 sp 已验收，opt/freq/md 验收前勿用。许可 MIT。",
    })
else:
    print("  B22.json SKIPPED: no B22_sp.out or energy not found")

# B27（MACE-MPA-0 插件，Si 周期，TICKET-0008）
txt = read(refdir / "B27_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B27.json", {
        "task": "sp", "model": "macempa0 (MACE-MPA-0 medium, MAPLE 插件后端 plugins/maple_mace_plugin.py)",
        "model_path": macempa0_pt,
        "command": "maple b27_sp.inp b27_sp.out  (env: MAPLE_CALCULATOR_PLUGINS=maple_mace_plugin, PYTHONPATH=mlip_agent/plugins；Si 金刚石原胞，XYZ 引用带 Lattice 的 extxyz)",
        "artifacts": ["si_diamond.extxyz", "B27_sp.inp", "B27_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
        "units": {"energy": "Hartree"},
        "note": "周期性 PBC pin（TICKET-0008）。同机 3 次逐位一致；与官方 mace_mp ASE 路径"
                "对照差 0.000e+00 eV。" + _SI_NOTE,
    })
else:
    print("  B27.json SKIPPED: no B27_sp.out or energy not found")

# B28（AIMNet2 PBC，TICKET-0009；H2O 20 Å 立方盒，库仑缺省自动 dsf）
txt = read(refdir / "B28_sp.out")
if txt and (m := RE_SP.search(txt)):
    dump("B28.json", {
        "task": "sp", "model": "aimnet2 (PBC, coulomb 缺省自动 dsf)",
        "command": "maple b28_sp.inp b28_sp.out  (H2O 金标准几何，#pbc(20,20,20) 立方盒)",
        "artifacts": ["B28_sp.inp", "B28_sp.out"],
        "values": {"energy_hartree": float(m.group(1))},
        "tolerance": {"energy_hartree": 1e-6},
        "units": {"energy": "Hartree"},
        "note": "AIMNet2 PBC（TICKET-0009）：matscipy MIC 周期邻居表 + cell/shifts 入图；"
                "与上游 nvalchemiops 逐位零差互证。能力边界：PBC 仅承诺 sp/opt；"
                "训练域气相分子，周期计算仅分子晶体类可信。",
    })
else:
    print("  B28.json SKIPPED: no B28_sp.out or energy not found")

# B8
txt = read(refdir / "B8_scan.out")
if txt:
    pts = [float(x) for x in RE_ANY_E.findall(txt)]
    if pts:
        dump("B8.json", {
            "task": "scan", "model": "ANI-2x",
            "command": "maple b8_scan.inp b8_scan.out  (rigid, S 1 2 0.05 4, O-H 0.9->1.1 Å)",
            "artifacts": ["B8_scan.out"],
            "values": {
                "oh_distances_angstrom": [0.9, 0.95, 1.0, 1.05, 1.1],
                "energies_hartree": pts,
                "argmin_index": pts.index(min(pts)),
            },
            "tolerance": {"energies_hartree": 1e-5},
            "note": "scan .out 能量仅打印 6 位小数，容差放宽到 1e-5",
        })
    else:
        print("  B8.json SKIPPED: no scan energies parsed")
else:
    print("  B8.json SKIPPED")
PYEOF

echo "== done, work dir: $WORK =="
ls -la "$REFDIR"
