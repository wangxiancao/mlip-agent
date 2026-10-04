## TICKET-0003：maceoff23s.pt 自制 wrapper 置入 MAPLE 模型目录（先执行后补票，追认）

- 状态：已追认（附条件，5 项条件已全部落实） ｜ 提出：2026-10-02 ｜ 审批人：审批 agent（追认）

### 1. 改动对象

- 新增模型工件：`MAPLE/maple/function/calculator/model/maceoff23s.pt`（冻结资产 MAPLE/ 树内的数据文件，非源码改动）。
- 生成脚本：`mlip_agent/scripts/make_maceoff23s_wrapper.py`（在 scratch 环境 torch 2.8.0 CPU + mace-torch 0.3.10 下，把 ACEsuit/mace-off 官方权重 `MACE-OFF23_small.model` 经 `e3nn.util.jit.compile` 封装为 MAPLE `MACECalculator` 期望的 scripted wrapper）。

### 2. 事实经过（先执行后补票）

maceoff23s 是 `REQUIRES_LOCAL_MODEL_FILE=True`，HF 仓库 Wayne7815/MAPLE_models 只提供 maceoff23m/egret1s，MAPLE 无法自动下载。为支撑 D7-4 的 B14 验收任务，实施时直接将自制 wrapper 放入 MAPLE 模型目录并通过 B14（H2O sp，−76.4797837970 Ha，同机三次复现逐位一致），**未先开工单**，属流程瑕疵。本工单为事后补办，审批以追认了结。

### 3. 影响面分析

**B. 框架内新增**：放置位置与生成方式符合 MAPLE 官方 `_require_local_model_file` 机制（`MACECalculator` 按模型名在该目录取本地文件），不改 MAPLE 任何源码与既有数值行为；非 maceoff23s 作业路径完全不变。B14 为新增 benchmark 任务，既有金标准不受影响。

### 4. 风险

- scripted 产物依赖 torch/mace-torch 版本：COMPUTE_VENV torch 升级后旧 .pt 可能无法加载或数值漂移（→ 条件 2 复查义务）。
- wrapper forward 固定 `compute_force=False`，forces 路径未经对照验证（→ 条件 3 能力边界）。
- 权重许可证为非商业许可（→ 条件 4）。

### 5. 回滚方案

删除 `MAPLE/maple/function/calculator/model/maceoff23s.pt` 并 `git revert` 本工单相关提交（B14 任务/剧本/金标准一并回退）。

### 6. 验证计划

- 条件落实后跑 `env -u HTTP_PROXY ... .venv/bin/python -m pytest tests/ -q` 确认不回归。
- B14 数值已在 D7-4 验收（本工单不改变任何数值）。

### 7. 审批结论

**追认**（审批 agent，2026-10-02）：定性 B 类（框架内新增、未触库逻辑）；放置位置与生成方式符合 MAPLE 官方 `_require_local_model_file` 机制；流程瑕疵（先执行后补票）以追认了结，**今后向 MAPLE/ 树内写任何文件（含数据工件）须先开工单**。附 5 项条件，落实记录如下：

#### 条件 1（溯源）——已落实

- 产物 `MAPLE/maple/function/calculator/model/maceoff23s.pt` SHA256：
  `a234308d11de00fbf2ca1a83f29c870cf429e1ee16ada0e11cf9ebc155c3c5f6`
- 源权重 `MACE-OFF23_small.model` SHA256：
  `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`
  （本地 `/tmp/mace_off/MACE-OFF23_small.model` 实测；与 pinned URL 重新下载版本哈希一致，已复验）
- 生成脚本下载源原为 `/raw/main/` 浮动引用，已 pin 到 commit
  `91a78c5a9c300d1104700d9352c8bfe449227737`（ACEsuit/mace-off 当前 main，
  2026-10-02 查询 GitHub API 确认），并在用法中增加 `sha256sum -c` 校验步骤。

#### 条件 2（复查义务）——已落实

`docs/decisions.md` 新增 D7-5：MAPLE mace calculator 契约变动或 COMPUTE_VENV torch
升级时，必须重新生成 wrapper 并复验 B14。

#### 条件 3（能力边界）——已落实

- `app/tools/maple_tools.py` MODEL_CATALOG 的 maceoff23s notes 改为
  「有机小分子；仅 sp 已验收（forces 路径未验证，opt/freq/md 勿用）；另有 maceoff23m / maceoff23l」。
- `docs/decisions.md` D7-5 同步注明：forces 路径对照验证前不得新增 maceoff23s 的
  opt/freq/md benchmark 任务。

#### 条件 4（许可证）——已落实

查证 <https://github.com/ACEsuit/mace-off>（README 与 `LICENSE.md`）：MACE-OFF23
权重采用 **Academic Software License（ASL）**——**仅限学术研究用途**（academic
non-commercial use，教育机构/非营利/政府机构的学术研究），**禁止商业使用**；
商业使用须向原始权利人另购商业许可。ASL 为 reciprocal（类 GPLv2 + 非商业限制）：
再分发衍生作品须同许可发布并附源码。结论：**本项目对 maceoff23s 的使用仅限学术
研究用途**。生成脚本注释与本工单均已标注。

#### 条件 5（文档）——已落实

`README.md` 环境搭建一节注明：maceoff23s.pt 不在全新 MAPLE checkout 中，需走
`scripts/make_maceoff23s_wrapper.py` 重新生成（含 pinned 下载地址与哈希校验）。

#### 验证结果

- `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy .venv/bin/python -m pytest tests/ -q`：通过，无回归（MODEL_CATALOG notes 改动无测试断言依赖）。
