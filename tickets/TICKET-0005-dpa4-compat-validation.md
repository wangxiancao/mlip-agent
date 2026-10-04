# TICKET-0005：DPA-4 兼容性验证与接入（阶段 1：scratch venv 验证）

- 状态：阶段 1 已完成（2026-10-02，结论支持，见第 6 节）；阶段 2 须凭阶段 1 记录另行工单
- 提案日期：2026-10-02
- 审批：审批 agent，2026-10-02

## 1. 背景与目标

`/root/soft/ml_potential/DPA-4/DPA4-beta.pt` 是 deepmd-kit PyTorch 训练 checkpoint（DPA4/SeZM，23 个 multitask 分支，`default_chg_spin: [0.0, 1.0]` 需固定）。无免安装加载路径，必须 deepmd-kit ≥3.2.0。SeZM 在 torch 2.8 上的兼容性未经验证（官方 [torch] extra 期望 2.11，且该 extra 会 pin 死 torch==2.11.0，严禁安装）。

分两阶段：
- **阶段 1（本工单）**：一次性 scratch venv（/tmp/dpa4_venv，Python 3.12）零足迹验证：`deepmd-kit==3.2.0`（不带 extras）能否 DeepEval 加载 checkpoint 并对 H2O 出能量；固定 model_branch / fparam；记录数值与逐位可复现性。
- **阶段 2（另开工单）**：通过后正式接入——推荐独立常驻 venv（非 COMPUTE_VENV）+ 子进程脚本（runner argv 首元素即解释器路径，无改造障碍）；失败则 DPA-4 在 MODEL_CATALOG 标注「环境不兼容，不支持」。

## 2. 影响面分析

阶段 1：零。不动 MAPLE/、COMPUTE_VENV、mlip_agent 代码、checkpoint；唯一动作是 uv 建临时 venv + 网络装包到临时目录，删除即彻底回滚。

## 3. 风险与回滚

风险仅为实验失败的时间成本；回滚 = 删 venv。

## 4. 审批结论

**批准（仅限阶段 1）** — 零足迹验证活动，方法学正确（以实验自身回答 torch 2.8 不确定性）。

## 5. 前置条件（批准附带）

1. scratch venv 建在冻结路径之外（/tmp/dpa4_venv），严禁向 COMPUTE_VENV 或 mlip_agent/.venv 装任何包；Python 3.12（与 COMPUTE_VENV 对齐，保证结论对阶段 2 有效）；
2. 安装严格 `deepmd-kit==3.2.0`（--no-deps 或显式纯 py 依赖清单），torch 用 2.8.0 wheel；任何情况下不装 `[torch]` extra；装完记录完整 `pip freeze`；
3. 实验必须书面记录（decisions.md 新条目）：① DeepEval 能否加载；② model_branch 选择与理由、fparam/chg_spin 固定值；③ H2O 能量与同机两次逐位一致性；④ torch 2.8 上 SeZM 兼容性结论；
4. 阶段 1 不得 pin benchmark、不得把 DPA-4 加入 MODEL_CATALOG；
5. 阶段 2 工单必须附阶段 1 完整记录；venv 存放位置、磁盘占用、GPU 锁共享语义届时一并评审。

## 6. 实施结果回填

阶段 1 于 2026-10-02 完成，结论：**支持，阶段 2 可行**。完整记录见 `mlip_agent/docs/dpa4_stage1_report.md`（pip freeze、源码重编译配方、四问答案、数值表）与 decisions.md D7-7。

- 前置条件 1：scratch venv 建于 /tmp/dpa4_venv（CPython 3.12.12，与 COMPUTE_VENV 同版），COMPUTE_VENV 与 mlip_agent/.venv 零改动 ✅
- 前置条件 2：`deepmd-kit==3.2.0` 严格该版本；未装 [torch] extra；torch 2.8.0+cu128 wheel。关键发现：PyPI wheel 绑死 torch 2.11 ABI，必须 `--no-binary` 源码重编译（配方见报告 §1），另补 mpich / vesin+vesin-torch / e3nn 0.6.0；pip freeze 已录入报告 ✅
- 前置条件 3：四问均已书面记录（报告 §3 + decisions.md D7-7）：① DeepEval 直接加载成功（无需 restart/导出）；② head=H2O_H2O_PD（H2O 领域分支），dim_fparam=0 无 fparam，chg_spin 固定 [0.0,1.0]（与默认逐位一致）；③ H2O 能量 CPU+OMP=1 两次逐位一致（−15.710773446015082 eV），GPU 能量 −15.710772596299648 eV（跨进程漂 ~2e-7），力任何配置下跨进程不逐位一致；④ torch 2.8 上 SeZM 重编译后零 warning 零报错，数值正常 ✅
- 前置条件 4：未 pin benchmark、未加入 MODEL_CATALOG ✅
- 前置条件 5（移交阶段 2）：venv 配方与磁盘占用（6.9 GB，scratch 已删）已记录；GPU 锁共享语义、常驻 venv 位置待阶段 2 工单评审 ✅
