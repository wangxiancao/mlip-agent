# TICKET-0004：TACE 直算后端接入（B15）

- 状态：已批准，已实施
- 提案日期：2026-10-02
- 审批：审批 agent，2026-10-02

## 1. 背景与目标

MODEL_CATALOG 缺 TACE。TACE 0.2.0 已 editable 安装在 COMPUTE_VENV，本地已有预训练 checkpoint `/root/soft/ml_potential/TACE/TACE-OAM-L.pt`（333MB）。ASE 接口 `tace.interface.ase.TACEAseCalc` 已实测：H2O 单点能 E = −13.933330535888672 eV（RTX 3090，neighborlist_backend='ase'）。

提案：复刻 nequip_direct 模式——`app/compute/scripts/tace_ase.py` 子进程脚本 + `run_tace` 工具 + MODEL_CATALOG 条目 + benchmark 任务 B15（pin −13.9333305359 eV ±1e-6）。MAPLE/ 零改动、COMPUTE_VENV 零装包。

## 2. 影响面分析

既有金标准 B1–B14 零影响（新后端、新任务）。单位语义沿用 ASE eV/Å 惯例（result.json 显式 units 字段）。前端 result.json 分支已有 nequip 先例，无新格式负担。

## 3. 风险与回滚

pin 值需同机 ≥2 次逐位一致验证（容差 1e-6 eV 的前提）；`TACEAseCalc` 是公开 ASE 接口，无半私有 API 风险。回滚 = git revert。

## 4. 审批结论

**批准** — B 类框架内新增（与 TICKET-0002 同类且更简单，连 env 注入都不需要）。

## 5. 前置条件（批准附带）

1. `tace_ase.py` 显式固定 `neighborlist_backend='ase'`，并在 result.json 记录该选项；
2. pin 前同机连续 ≥2 次确认逐位一致，device/dtype/tace/torch 版本写入 result.json versions 字段；
3. `run_tace` 权限规则用全限定 tool_id（`tool/mlip.tace/run_tace`，"ask"）——D2-1 教训；
4. B15 断言对齐 B5；canned 剧本用金标准内联坐标（D5-3）；
5. MODEL_CATALOG 标注：分子体系（非周期）、元素覆盖范围、仅 sp/opt 已验收；
6. 完成后 pytest + canned B1–B15 全绿，新增 D7-x 决策记录。

## 6. 实施结果回填

实施完成（2026-10-02），前置条件 1–6 逐条落实：

1. `app/compute/scripts/tace_ase.py` 显式固定 `neighborlist_backend='ase'`，并写入 result.json 顶层 `neighborlist_backend` 字段（B15.json 同步记录）。
2. **pin 值修正为 −13.926433563232422 eV（±1e-6）**——与提案预估值 −13.933330535888672 eV 不同（Δ≈6.9e-3 eV，预估值在本机无法复现：cuda/float32、cuda/float64、cpu、matscipy/ase+center(vacuum) 各变体均落在 ~−13.9264 eV）。按前置条件 2 以同机连续 2 次逐位一致的实测值为准（能量 repr 与 forces 列表均逐位一致，证据见 D7-6）。`versions` 含 tace 0.2.0 / torch 2.8.0+cu128 / ase 3.27.0 / device / dtype。
3. 权限规则 `tool/mlip.tace/run_tace` = "ask"（全限定 tool_id），`tests/test_agent_offline.py::test_permission_rules_fully_qualified_tool_ids` 断言。
4. B15 断言对齐 B5（final_status / tool_called=run_tace / suspended_with_tool=run_tace / job_count=1 / backend=tace / artifact_present=result.json / energy_eV pin），canned 剧本用金标准内联坐标（`_orch_b15_stream`，与 B5 同构）。
5. MODEL_CATALOG `tace-oam-l`：分子体系（非周期）；元素覆盖按 checkpoint type map 实测 89 种（Z=1–83 + Z=89–94，缺 Po/At/Rn/Fr/Ra）；标注「仅 sp/opt 已验收（B15）」。
6. pytest 全绿 + canned B1–B15 15/15；决策记录 D7-6（docs/decisions.md）。

其他落点：`run_tace` 工具在 `app/tools/mlip_tools.py`（与 run_nequip 共用 `_run_direct_impl`）；`benchmarks/runner.py` 作业收集按 result.json versions 有无 `tace` 键区分 tace/nequip 直算作业；B9 断言补 `tool_not_called: run_tace`；Settings 新增 `tace_model_path`（env `TACE_MODEL_PATH`）。
