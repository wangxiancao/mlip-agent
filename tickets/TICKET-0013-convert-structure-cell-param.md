# TICKET-0013：convert_structure 增加可选 cell 参数（ASE set_cell 补胞）

- 状态：待审批 ｜ 提出：2026-10-04 ｜ 审批人：____

### 1. 改动对象

- `app/compute/scripts/convert_structure.py` `run_job`：job.json 新增可选 `cell` 字段。
- `app/tools/structure_tools.py` `convert_structure_impl` / `convert_structure` 工具签名：
  新增可选 `cell` 参数与校验。
不属于冻结资产（不动 MAPLE / COMPUTE_VENV / checkpoint）。

### 2. 动机与问题描述

Web UI 实战（会话 a0b58838）暴露工具链缺口：MAPLE 周期 opt 作业（MACE-MPA-0
job-0001、UMA job-0002）的输出 `input_opt.xyz` 是纯 xyz，**不含 Lattice/pbc**，
经 convert_structure 转出的 cif 因此丢失周期信息（pbc=false），而用户要求
"优化后的周期结构保存为 cif"。同期 TACE 直算输出 final.xyz 带完整 Lattice，
cif 正常。

补胞这一步按用户硬性约束必须由 ASE 执行，agent 不得自行拼接坐标与晶胞。
当前 convert_structure 无任何注入晶胞的入口，唯一"合规"路径是重跑作业
（浪费算力且不保证 MAPLE 支持），故给工具加 `cell` 参数。

### 3. 影响面分析

B. 框架内新增：只在 mlip_agent/ 内新增可选参数，不改变既有数值行为。
缺省 `cell=None` 时行为与现状逐位一致；给定 cell 时由 ASE
`set_cell(cellpar_to_cell(...))` + `set_pbc(True)` 执行，3 值（正交 a,b,c）
或 6 值（a,b,c,α,β,γ）约定与 build_structure 的 cell 参数一致。
源结构已有胞且给出 cell 时记 warning（覆盖语义）。

### 4. 风险

- 误用风险：给非周期分子乱加胞 → 由 warnings 与 conversion.json 的
  pbc/cellpar 回报，agent 可校验。
- 数值漂移风险：无（不改坐标，只设晶胞）。

### 5. 回滚方案

`git revert` 对应提交即可。

### 6. 验证计划

- 新增单测：无胞 xyz + cell 参数 → extxyz/cif 带 Lattice、pbc=true、cellpar 正确；
  非法 cell（长度/非正值）报错；缺省路径行为不变（既有 4 项 convert 测试照跑）。
- `.venv/bin/python -m pytest tests/ -q` 全绿。
- UI 复验：agent 用新参数把 job-0001/0002 的 input_opt.xyz 转成带胞 cif。

### 7. 审批结论

批准（用户指令 2026-10-04，web UI 任务中现场发现并修复）。
实施回填：`convert_structure.py`/`structure_tools.py` 新增可选 `cell` 参数
（ASE set_cell + set_pbc(True)，3/6 值校验，覆盖既有胞记 warning）；
新增 4 项单测；全量 `pytest tests/` 241 项通过。
