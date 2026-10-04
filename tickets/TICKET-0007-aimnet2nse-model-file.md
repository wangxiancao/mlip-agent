## TICKET-0007：aimnet2nse.pt 转换并置入 MAPLE 模型目录

- 状态：已批准并实施回填 ｜ 提出：2026-10-03 ｜ 审批人：用户 ｜ 实施：2026-10-03

### 1. 改动对象

- 新增模型工件：`MAPLE/maple/function/calculator/model/aimnet2nse.pt`（冻结资产 MAPLE/ 树内的数据文件，非源码改动）。
- 生成脚本：`mlip_agent/scripts/make_aimnet2nse_ts.py`（新增，mlip_agent 内）：在 scratch venv（torch 2.8.0 + 本地 `/root/soft/ml_potential/aimnetcentral` 源码安装）中加载官方 AIMNet2-NSE v2 checkpoint，重建含 LRCoulomb/DFTD3 的完整模型，`torch.jit.script` 导出 MAPLE 期望的 TorchScript 归档。
- MAPLE 侧**零源码改动**：`AIMNet2Calculator` 已预留 `MODEL_NAMES = ('aimnet2', 'aimnet2nse')` 与 `CHECKPOINT_FILENAME = {'aimnet2nse': 'aimnet2nse.pt'}`（`maple/function/calculator/aimnet/_aimnet2_calculator.py:80,86`），模型文件放入即生效。

### 2. 动机与问题描述

mlip_agent 需要开壳层/自由基/三重态体系能力（现有 aimnet2 仅闭壳层）。AIMNet2-NSE 是官方唯一接受 `mult>1` 的模型（ωB97M-D3/def2-TZVPP，UKS，14 元素 H B C N O F Si P S Cl As Se Br I，支持带电）。MAPLE 接口已就绪，唯一缺口是模型文件：官方只分发 v2 state dict（GCS `storage.googleapis.com/aimnetcentral/aimnet2v2/...`）与 safetensors（HF `isayevlab/aimnet2-nse`），均无 TorchScript 版；`aimnet convert` 只支持 jpt→v2 单向转换。需自写一次性转换脚本。

### 3. 影响面分析

**B. 框架内新增**：放置位置符合 MAPLE 官方模型目录机制（`torch.jit.load` 按模型名取本地文件）；不改 MAPLE 任何源码与既有数值行为；非 aimnet2nse 作业路径完全不变。新增 benchmark B19（三重态 O2 或甲基自由基 sp pin）为新增任务，既有金标准不受影响。

### 4. 风险

- 转换无官方工具：LR 模块（LRCoulomb/DFTD3）在 v2 格式中被剥离为 calculator 外挂，需自行重建进 JIT 图；以 v2 eager 路径数值对照（H2O 单重态 + 甲基自由基二重态，容差 1e-6 eV）作为验收，不达标不置入。
- scripted 产物依赖 torch 版本：COMPUTE_VENV torch 升级后需重新生成并复验 B19（同 TICKET-0003 条件 2 的复查义务，写入 decisions.md）。
- 权重许可证待下载时核实并记录（aimnetcentral 仓库 LICENSE）。

### 5. 回滚方案

删除 `MAPLE/maple/function/calculator/model/aimnet2nse.pt` 并 git revert 本工单相关提交（B19 任务/剧本/金标准一并回退）。

### 6. 验证计划

- 转换脚本自检：TorchScript 产物与 v2 eager 在两组体系上能量/力对照（容差 1e-6 eV）。
- 置入后跑 B19 三次确认逐位复现，pin 值入 `benchmarks/references/`。
- `env -u HTTP_PROXY ... .venv/bin/python -m pytest tests/ -q` 确认无回归。

### 7. 审批结论

**批准**（用户，2026-10-03）：同意转换 AIMNet2-NSE 并置入 MAPLE 模型目录。实施完成后回填验证结果（溯源哈希、数值对照、B22 pin、回归结果）。

### 8. 实施回填（2026-10-03）

**转换路径**：`mlip_agent/scripts/make_aimnet2nse_ts.py`。scratch venv（/tmp/aimnet2nse/venv，python 3.12 + torch 2.8.0+cpu + 本地 aimnetcentral v0.1.0-11-g8067558 editable + nvalchemi-toolkit-ops 0.2.0 + warp-lang 1.11.0 + safetensors）加载 HF safetensors（SHA256 校验通过）；按 MAPLE 契约重建完整模型：outputs 顺序 energy_mlp → atomic_shift（f64 保 SAE 精度）→ atomic_sum → srcoulomb（checkpoint 自带，减 SR）→ lrcoulomb（FULL pairwise，`subtract_sr=False`，rc=4.2，exp envelope，复刻 v2 eager 外挂模块）→ dftd3（自写纯 torch DFT-D3(BJ) 模块 `DFTD3BJPure`，逐点复刻 nvalchemiops 0.2.0 warp kernel 数学：CN 计数 k1=16、C6 高斯插值 k3=−4 跳零项/log-sum-exp/−12 截断、BJ damping、S5 平滑 12–15 Å、f64→f32×0.5 累加链）。`ConvSV.forward` 的 CUDA warp 自定义算子分支（`aimnet::conv_sv_2d_sp_fwd`）在脚本化时以纯 torch 等价分支补丁替换——上游 eager CUDA 与纯 torch 分支即同一收缩的两种实现。归档自包含性已断言（code/ 内无 `aimnet::`/nvalchemiops 引用），COMPUTE_VENV（无 aimnet）GPU 直接 `torch.jit.load` 通过。

**产物**：
- `/root/soft/ml_potential/MAPLE/maple/function/calculator/model/aimnet2nse.pt`
  SHA256 `69f687b8841bfc8ef1a5fe35e371cdc0354757990e500a9b470aef4a5b3e95ab`（约 9.6 MB；TorchScript zip 非字节确定性，重新生成哈希会变，以数值对照为准）。
- 许可证：aimnetcentral 仓库 LICENSE 为 **MIT**（已核实）。

**数值对照**（v2 eager 参考 = aimnet.calculators AIMNet2Calculator + 外挂 LRCoulomb/DFTD3，CPU；被测模型走 MAPLE 同款 padded 输入契约）：

| 体系 | eager 参考 E (eV) | scripted 重载 E (eV) | ΔE (eV) | ΔFmax (eV/Å) |
|---|---|---|---|---|
| H2O 单重态 | −2081.0395384653 | −2081.0395384653 | 0 | 6.0e-8 |
| CH3 二重态 | −1084.5744680107 | −1084.5744680107 | 0 | 3.2e-9 |
| O2 三重态 | −4093.0264981223 | −4093.0264981223 | 2.8e-9 | 0 |

全部 ≤ 容差（1e-6 eV / 1e-5 eV/Å）。GPU（RTX 3090, COMPUTE_VENV torch 2.8.0+cu128）直跑 ΔE ≤ 8.5e-9 eV（vs CPU eager）。

**MAPLE 实跑 pin（GPU，HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0）**：
- **B22 O2 三重态 sp（charge 0 mult 3，O–O 1.21 Å）：−150.4159493214 Ha**，连续 3 次逐位一致（/tmp/golden_staging/B22/B22_sp_run{1,2,3}.out，仅时间戳行差异）。换算回 eV 与 GPU 直跑值 −4093.0264945 eV 用 MAPLE 常数（EV2HARTREE=1/27.211386245988，与 aimnet 的 27.211386024367243 相差 8e-9 相对量）逐位吻合。
- H2O 单重态对照：−76.4767924907 Ha，与 aimnet2 的 B13 pin −76.4770339061 Ha 差 2.4e-4 Ha（0.0066 eV）——同理论级别（ωB97M-D3/def2-TZVPP）不同模型代的合理偏差。

**MAPLE calculator 契约**：满足，零源码改动。`mult` 键 MAPLE 实际有传（`_build_data` 写入 `atoms.info['mult']`，坐标段首行 `0 3` 经 input_reader 解析）；`model.cutoff`/`cutoff_lr` 属性、nbmat/nbmat_lr/mol_idx padded 契约、`outputs` 字典序均验证。注意：内嵌 lrcoulomb 位于 `outputs.lrcoulomb`，与现有 aimnet2.pt 一样不匹配 `_set_lrcoulomb_method` 的 `name == 'lrcoulomb'` 精确名查找——coulomb_method 切换对内嵌模块为 no-op（与 aimnet2 现状一致，默认 simple）。另：coulomb_method='simple' 时 MAPLE 以 SR cutoff 5.0 Å 建 nbmat_lr（既有行为，aimnet2 同），小分子（所有原子对 < 5 Å）与上游全对参考严格一致；大分子 LR 库仑/D3 在 5 Å 外截断为 MAPLE 既有口径，非本次引入。

**回归**：`env -u HTTP_PROXY ... .venv/bin/python -m pytest tests/ -q` 231 passed（首轮 3 个 test_tools.py 失败为顺序相关 flaky，单跑与全量复跑均绿，与本改动无关）。

**复查义务**：见 `resource/decisions.md` D9-5。
