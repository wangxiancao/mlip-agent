# 审批工单模板

复制本文件为 `tickets/TICKET-<序号>-<短标题>.md` 并填写。任何触及 **MAPLE 源码 / COMPUTE_VENV 装包 / 其他冻结资产** 的改动必须先有「批准」结论的工单。

---

## TICKET-0010：全部模型 checkpoint 集中迁移至 `/root/soft/ml_potential/checkpoints/`

- 状态：已批准并实施回填 ｜ 提出：2026-10-03 ｜ 审批人：用户（2026-10-03 对话直接指令「把checkpoint创建一个文件夹，单独存放」）

### 1. 改动对象

本地模型 checkpoint 文件的**存放位置**（冻结资产中的 checkpoint 类，共 15 个文件）：

| 现位置 | 文件 |
|---|---|
| `uma/uma_folder/checkpoints/` | uma-s-1p1.pt、uma-s-1.pt、uma-m-1p1.pt |
| `nequip/` | NequIP-OAM-XL-0.1.nequip.zip |
| `mace/` | macempa0.pt、mace-mpa-0-medium.model |
| `TACE/` | TACE-OAM-L.pt、TECE-OAM-RRA-1.0.pt |
| `DPA-4/`、`DPA-4/dpa4c/` | DPA4-beta.pt、DPA4C-Air-OMol25-100M-v20260820.pt |
| `MAPLE/maple/function/calculator/model/`（git 未跟踪，`*.pt` 在 .gitignore） | ani1x.pt、ani2x.pt、aimnet2.pt、aimnet2nse.pt、maceoff23s.pt |

**不改**：MAPLE 源码零改动；COMPUTE_VENV 零改动；所有文件字节不变（同文件系统 `mv` = rename）；aimnetcentral 包内数据（aimnet2_b973c_d3_0.pt 等，库内解析）与 dpa4_venv 不动。

目标布局：

```
/root/soft/ml_potential/checkpoints/
├── README.md   # 清单：来源/许可/SHA256/再生脚本
├── uma/    nequip/    mace/    tace/    dpa4/    # 显式绝对路径消费的模型
└── maple/  # MAPLE model 目录硬编码约定的全部模型
```

MAPLE 侧用**目录级符号链接**保持其硬编码 `Path(__file__).parent/'model'` 约定：
`MAPLE/maple/function/calculator/model -> /root/soft/ml_potential/checkpoints/maple`
（该目录 git ls-files 为空，替换对 git 不可见；未来 MAPLE 自动下载经 symlink 直接落入集中目录。）

### 2. 动机与问题描述

checkpoint 散落在 6 个顶层目录和 MAPLE 包内，无统一视图；用户要求创建一个文件夹单独存放全部 checkpoint。集中后清单一目了然，且 mlip_agent/MAPLE 两个 git 仓库都不再窝藏大二进制。

### 3. 影响面分析

B 类变体：**不改任何文件内容与数值行为**，仅更新 mlip_agent 内的路径常量与文档。引用遗漏的后果是「文件找不到」型功能故障（显式报错），非数值漂移。需更新的引用点（已全量盘点）：

- 代码：`app/config.py`（6 个 Settings 默认值）、`app/compute/jobspec.py`（3 个 DEFAULT_* 常量）、`app/tools/mlip_tools.py`、`app/tools/maple_tools.py`（MODEL_CATALOG path 字段）、`plugins/maple_mace_plugin.py`（docstring）
- 脚本：`scripts/make_golden_refs.sh`、`scripts/make_maceoff23s_wrapper.py`、`scripts/make_macempa0_wrapper.py`（含源 .model 路径）
- 测试：tests/test_{tace,dpa4,dpa4c,nequip}_script.py、test_{nequip,mace}_plugin.py 的 env 兜底值
- benchmark：`benchmarks/tasks.yaml`（B21 prompt 内嵌路径）、`benchmarks/canned_scripts.py`（B21 tool args）
- 文档：README、PLAN、STATUS、resource/decisions.md（现行状态描述；历史工单/阶段报告不改）

### 4. 风险

- 路径引用遗漏 → 运行时 FileNotFound；由全量 pytest（231 项）+ canned B1–B28 兜底验证。
- symlink 兼容性：torch.load / nequip from_package 均经 os 层透明解析，无已知问题；`_model_dir_description()` 本就有 resolve() 显示逻辑。
- 数值风险：零（rename 不改字节，SHA256 不变）。

### 5. 回滚方案

`mv` 按映射表搬回原路径、删除 symlink 恢复实体目录、还原 mlip_agent 内引用编辑（逐文件反向 Edit）。全部操作可逆，无金标准重 pin 需求。

### 6. 验证计划

1. 迁移前后各文件 SHA256 抽检一致。
2. `env -u *_PROXY .venv/bin/python -m pytest tests/ -q` 全绿（231 项基线）。
3. `.venv/bin/python benchmarks/runner.py --mode=canned` B1–B28 全 PASS（覆盖 uma/nequip/tace/dpa4/dpa4c/macempa0/ani1x/aimnet2/aimnet2nse/maceoff23s 全部模型路径解析）。
4. MAPLE CLI 直接冒烟一次（经 symlink 解析 ani1x）。

### 7. 审批结论

批准（用户 2026-10-03 直接指令「把checkpoint创建一个文件夹，单独存放」）。

**实施回填（2026-10-03）**：

- 15 个文件全部迁至 `/root/soft/ml_potential/checkpoints/{uma,nequip,mace,tace,dpa4,maple}/`；迁移前后 13 个文件 SHA256 逐一比对一致，uma-m-1p1（11 GB）/uma-s-1 以 inode+size 校验（rename 字节不变）。`MAPLE/maple/function/calculator/model` 已替换为指向 `checkpoints/maple/` 的目录级符号链接，MAPLE 源码零改动。
- 引用更新范围在 §3 清单基础上追加：`benchmarks/references/` 31 个 golden 文件的内嵌路径字符串（仅路径、数值零改动，judge/测试均不从中解析路径）与 `.env.example` 模板。
- 验证：迁移后 `pytest tests/` **231 项全绿**；`benchmarks/runner.py --mode=canned` **B1–B28 28/28 PASS**（执行成功率与数值正确率均 100%，12 个模型条目全过，含经 symlink 解析的 ani1x/ani2x/aimnet2/aimnet2nse/maceoff23s）。MAPLE CLI 路径由 B1/B12/B13/B14 等 MAPLE 后端任务实跑覆盖，symlink 解析正常。
- 决策记录：resource/decisions.md D10-1；目录清单：checkpoints/README.md。
