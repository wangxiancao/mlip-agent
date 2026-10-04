# TICKET-0006：DPA-4 正式接入（阶段 2：独立常驻 venv + 子进程脚本）

- 状态：已完成（2026-10-02 批准并实施，结果见 §6）
- 提案日期：2026-10-02
- 依据：TICKET-0005 阶段 1 验证通过（`mlip_agent/docs/dpa4_stage1_report.md`，D7-7）

## 1. 背景与目标

阶段 1 已证实：deepmd-kit 3.2.0 源码重编译后在 torch 2.8.0+cu128 上可加载 DPA4-beta.pt（23 分支，选 H2O 专属分支 `H2O_H2O_PD`，chg_spin 固定 [0.0,1.0]），H2O（B5_h2o.xyz 金标准几何）能量 CPU+OMP=1 跨进程逐位一致：**−15.710773446015082 eV**；SeZM 与 e3nn 0.6.0 在 torch 2.8 兼容。

提案：按 TICKET-0005 预设形态正式接入——
1. 独立常驻 venv（**非** COMPUTE_VENV）装 deepmd-kit 3.2.0（源码重编译配方见阶段 1 报告，含 vesin/vesin-torch/mpich 与 LD_LIBRARY_PATH 固化），重建时间约 15 min；
2. `app/compute/scripts/dpa4_ase.py` 子进程脚本（job.json/result.json 契约对齐 tace_ase.py；runner argv 首元素即解释器路径，新增 settings 字段 `dpa4_python` 指向常驻 venv）；
3. `run_dpa4` 工具 + MODEL_CATALOG 条目（dpa-4，仅能量已验收——力不能跨进程逐位复现，**不 pin 力**，标注 opt/freq 勿用）；
4. benchmark B16：H2O sp，pin 能量（CPU+OMP=1 金标准 −15.710773446015082 eV ±1e-6，或 GPU ±1e-5 容差，实施时按前置条件裁定）。

## 2. 影响面分析

COMPUTE_VENV 零改动、MAPLE/ 零改动；新增磁盘占用约 6.9 GB（常驻 venv）；既有 B1–B15 零影响。GPU 锁：DPA-4 作业经 run_compute 的同一把 GPU semaphore 串行。

## 3. 风险与回滚

- 常驻 venv 位置与磁盘占用需评审裁定；
- 力不可逐位复现（邻居排序/scatter 累加顺序，根因未定位）→ 只 pin 能量；
- GPU↔CPU 能量差 ~8.5e-7（TF32/AMP）→ pin 需固定设备口径；
- 回滚 = git revert + 删常驻 venv，完全可逆。

## 4. 审批结论

**批准** — 审批 agent，2026-10-02。B 类框架内新增，阶段 1 证据链完整。

**四项裁定**：
1. **常驻 venv 位置**：`/root/soft/ml_potential/dpa4_venv`（mlip_agent 之外、与 COMPUTE_VENV 平级；根 .gitignore 白名单天然排除，不批准放 mlip_agent/ 内）；
2. **pin 口径**：CPU + OMP_NUM_THREADS=1 逐位金标准 **−15.710773446015082 eV ±1e-6**（确定性优先、与 CUDA 栈解耦），只 pin 能量，不 pin 力；`dpa4_ase.py` CPU 模式自行强制 `OMP_NUM_THREADS=1` + `torch.set_num_threads(1)`；
3. **MODEL_CATALOG 标注**：sp 已验收（CPU pin）；opt 可交互使用、力跨进程不逐位稳定（~1e-7 eV/Å）不作 pin；freq/md 不支持；
4. GPU 锁：B16 走 CPU pin 时 DPA-4 作业以 `gpu=False` 提交。

## 5. 前置条件（批准附带）

1. `LD_LIBRARY_PATH=/usr/local/lib64` 经 run_compute 的 `extra_env` 通道无条件注入（可选加固：重建加 `-static-libstdc++`，做了记入 D7-x）；
2. `scripts/make_dpa4_venv.sh` 幂等入库（剔除纯构建期依赖）+ 常驻 venv pip freeze 全文记入 decisions.md；脚本显式校验不触碰 COMPUTE_VENV；
3. job.json：`head` 默认 `H2O_H2O_PD` 允许覆盖、`charge_spin` 固定 `[0.0,1.0]`；result.json versions 记录 deepmd-kit/torch/e3nn/vesin/device/线程数；
4. 权限规则全限定 `tool/mlip.dpa4/run_dpa4`（"ask"），capability 常量在 registry.py 共享（D2-1）；
5. B16 对齐 B5/B15；canned 剧本用 B5_h2o.xyz 同款内联坐标；references/B16.json note 注明「CPU+OMP=1 口径，GPU 值约 −15.7107726 eV 不可混用」；
6. 新增 D7-x 记录 venv 位置/pin 口径裁定、head 默认、力的已知非确定性（根因未定位）；分支命名差异（Alloy_tongqi→Alloy_APEX、Omat24→OMat24，以 checkpoint 为准）写入接入说明；
7. 完成后 pytest 全绿 + canned B1–B16 全绿（B16 连续两次逐位 hit pin），结果回填第 6 节。

## 6. 实施结果回填

实施完成：2026-10-02，mlip-agent。

- **状态：已完成**。状态行由「待审批」经 2026-10-02 批准（§4）后实施完毕。
- **常驻 venv**：`/root/soft/ml_potential/dpa4_venv`（裁定 1），`scripts/make_dpa4_venv.sh` 幂等重建实测约 5 min，磁盘占用 **6.8 GB**；pip freeze 全文记入 `docs/decisions.md` D7-8。COMPUTE_VENV / mlip_agent/.venv / MAPLE 全程零改动（脚本含显式路径校验）。
- **代码**：`app/compute/scripts/dpa4_ase.py`（契约对齐 tace_ase.py；head 默认 H2O_H2O_PD 可覆盖、charge_spin 固定 [0.0,1.0]、CPU 模式进程内强制 OMP=1 + torch.set_num_threads(1)、versions 含 deepmd-kit/torch/e3nn/vesin/device/线程数、units 显式 eV/Å）；`run_dpa4` 工具（`tool/mlip.dpa4/run_dpa4`，"ask"）+ registry capability + Settings `dpa4_venv`/`dpa4_model_path`/`dpa4_python`；`LD_LIBRARY_PATH=/usr/local/lib64` 经 `run_compute(extra_env=...)` 无条件注入（前置条件 1；`-static-libstdc++` 加固未做，可选，D7-8 已记）；DPA-4 作业 `gpu=False`（CPU pin 口径，裁定 4）；MODEL_CATALOG `dpa-4` 条目按裁定 3 标注。
- **B16 pin（裁定 2）**：CPU+OMP=1 逐位金标准 **−15.710773446015082 eV ±1e-6**，只 pin 能量。逐位一致证据：① 脚本级同机连续 2 次 sp，能量 repr 均 `-15.710773446015082`；② canned B16 同机连续 2 次（runs/20261002T221516、20261002T221526）judge 全 PASS、run_record 能量 repr 逐位相同、backend=dpa4。工件：`benchmarks/references/{B16.json,B16_job.json,B16_result.json}`（note 注明 CPU+OMP=1 口径、GPU 值约 −15.7107726 eV 不可混用）；`scripts/make_golden_refs.sh` 已补 B16 段。
- **任务/测试**：tasks.yaml B16（对齐 B15）+ canned 剧本（B5_h2o.xyz 同款内联坐标）；`tests/test_benchmarks.py` 任务数断言 B1–B16；`tests/test_dpa4_script.py` 同款新增（pin/schema/单位/head 默认/线程固定/错误路径，6 用例）；`tests/test_agent_offline.py` 权限规则补 run_dpa4 全限定断言；agent instructions 补 dpa-4（sp 为主，分子体系）；B9 断言补 tool_not_called run_dpa4；runner.py 作业收集 backend 判定扩展 deepmd-kit→dpa4；README/PLAN/根 AGENTS.md 口径同步 B1–B16。
- **验证（前置条件 7）**：`pytest tests/ -q` 全绿（含 test_dpa4_script 6 项）；canned B1–B16 16/16 全 PASS（B16 连续两次逐位 hit pin，见上）。决策记录：D7-8（venv 位置/pin 口径裁定、head 默认、力的非确定性根因未定位、分支命名差异以 checkpoint 为准）。