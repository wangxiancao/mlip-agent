#!/usr/bin/env bash
# 把一次 benchmark 运行的关键记录归档进 git（判定报告 + 每任务 run_record.json）。
# 用法: scripts/archive_run.sh <run_dir> [label]
# 例: scripts/archive_run.sh benchmarks/runs/20261002T052106 canned-11x11
set -euo pipefail

RUN_DIR="${1:?usage: archive_run.sh <run_dir> [label]}"
LABEL="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(basename "$RUN_DIR")"
DEST="$ROOT/benchmarks/archive/${STAMP}${LABEL:+-$LABEL}"

mkdir -p "$DEST"
cp "$RUN_DIR/judge_report.json" "$DEST/"
for task_dir in "$RUN_DIR"/B*; do
    task="$(basename "$task_dir")"
    mkdir -p "$DEST/$task"
    cp "$task_dir/run_record.json" "$DEST/$task/"
done

python3 - "$DEST" <<'PY'
import json, sys
from pathlib import Path
dest = Path(sys.argv[1])
report = json.loads((dest / "judge_report.json").read_text())
summary = report.get("summary", {})
lines = [
    "# Benchmark 运行归档",
    "",
    f"- 来源运行目录: `{report.get('run_dir', '?')}`",
    f"- 汇总: `{json.dumps(summary, ensure_ascii=False)}`",
    "",
    "归档内容：judge_report.json（逐任务断言判定）+ 每任务 run_record.json（事件流摘要/作业/解析结果）。",
    "原始事件流与计算工件在 benchmarks/runs/（不入库，可再生）。",
    f"复现: `.venv/bin/python benchmarks/runner.py --mode=canned`，详见 README。",
]
(dest / "README.md").write_text("\n".join(lines) + "\n")
PY

echo "archived -> $DEST"
