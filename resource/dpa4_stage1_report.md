# DPA-4 阶段 1 验证报告（TICKET-0005，零足迹 scratch venv）

> 后序：阶段 2 已完成（TICKET-0006），正式接入形态与 B16 pin 值见 `decisions.md` D7-8。本报告为阶段 1 历史记录，配方与数值仍有效。

- 日期：2026-10-02
- 执行：mlip-agent（阶段 1 已批准范围：只建 /tmp/dpa4_venv，不动 COMPUTE_VENV / mlip_agent/.venv / MAPLE / mlip_agent 代码）
- 模型：`/root/soft/ml_potential/DPA-4/DPA4-beta.pt`（deepmd-kit PyTorch 训练 checkpoint，SeZM/DPA4，23 分支 multitask）
- 结构：`mlip_agent/benchmarks/references/B5_h2o.xyz`（O−H 0.96 Å，3 原子非周期）

## 1. 环境构建（关键发现：PyPI 二进制 wheel 与 torch 2.8 二进制不兼容，须源码重编译）

1. `uv venv /tmp/dpa4_venv --python 3.12` → CPython 3.12.12（与 COMPUTE_VENV 完全同版）。
2. `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0` → torch 2.8.0+cu128（与 COMPUTE_VENV 同版本线）。
3. `uv pip install --no-deps deepmd-kit==3.2.0` + 显式纯 py 依赖（numpy / scipy / pyyaml / dargs / h5py / wcmatch / einops / packaging / ml_dtypes / mendeleev / array-api-compat / lmdb / msgpack）+ `e3nn==0.6.0`（SeZM 描述符必需；COMPUTE_VENV 实际并未安装 e3nn，0.6.0 为按任务指定版本新装）。**未装 `[torch]` extra**（其 pin torch==2.11.0，严禁）。
4. 首次 `import deepmd.pt` 失败两处，均为 PyPI 二进制 wheel 的构建期假设：
   - `mpich` 元数据缺失（cibuildwheel wheel 在 import 时 CDLL mpich 的 libmpi）→ 补装 `mpich==5.0.2`（纯运行库，非 [torch] extra）；
   - **致命**：`libdeepmd_op_pt.so: undefined symbol: torch::autograd::_wrap_outputs(...)`——wheel 针对 **torch 2.11.0** 编译（见 wheel 内 `deepmd/lib/run_config.ini` 的 `PT_VERSIOn = 2.11.0`），与 torch 2.8.0 ABI 不兼容，`cxx_op.load_library` 直接 raise，import 无法继续。
5. 按 deepmd-kit 自身错误提示的机制源码重编译（约 4.5 min，48 核 RTX 3090 机器）：
   ```
   uv pip install "scikit-build-core>=1" "cmake>=3.25.2" ninja dependency_groups packaging \
       setuptools_scm hatch_fancy_pypi_readme   # 后两个是 sdist 未声明的构建期依赖
   PATH=/usr/local/cuda-12.8/bin:$PATH \
   DP_VARIANT=cuda CUDAToolkit_ROOT=/usr/local/cuda-12.8 \
   DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 TORCH_CUDA_ARCH_LIST="8.6" \
   uv pip install --no-deps --no-build-isolation --no-binary deepmd-kit \
       --reinstall-package deepmd-kit deepmd-kit==3.2.0
   ```
   注意：系统 PATH 上的 nvcc 是 CUDA 10.1（过旧），必须显式指向 `/usr/local/cuda-12.8`（与 torch cu128 对齐）；cmake ≥3.25.2 由 pip cmake 提供（系统 cmake 3.16 不够）。
6. 运行期注意：本机 gcc 11.3（/usr/local）编译出的 .so 需要 `GLIBCXX_3.4.29`，系统 libstdc++ 过旧，运行任何 deepmd.pt 进程需 `LD_LIBRARY_PATH=/usr/local/lib64`（阶段 2 常驻方案需在 wrapper 脚本里固化，或改用 -static-libstdc++ 重链）。
7. SeZM 推理时邻居表构造强制要求 `vesin` 或 `nvalchemiops`（`_select_neighbor_builder`，两者均属 [torch] extra 但自身不 pin torch 版本）→ 补装 `vesin==0.6.2` + `vesin-torch==0.6.2`（单独装，不触发 extra）。此后全流程通过。

### pip freeze（uv pip freeze，76 行）

```
annotated-types==0.8.0
array-api-compat==1.15.0
bracex==3.0.1
cmake==4.4.3
colorama==0.4.6
dargs==0.5.1
deepmd-kit==3.2.0
dependency-groups==1.3.2
deprecated==1.3.1
e3nn==0.6.0
einops==0.8.2
filelock==3.32.3
flexcache==0.3
flexparser==0.4
fsspec==2026.7.0
h5py==3.16.0
hatch-fancy-pypi-readme==25.1.0
hatchling==1.32.4
jinja2==3.1.6
lmdb==2.3.0
markupsafe==3.0.3
mendeleev==1.3.0
ml-dtypes==0.6.0
mpich==5.0.2
mpmath==1.3.0
msgpack==1.2.3
networkx==3.6.1
ninja==1.13.2
numpy==2.5.3
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
opt-einsum==3.4.0
opt-einsum-fx==0.1.4
packaging==26.3
pandas==3.0.6
pathspec==1.1.1
pint==0.24.4
platformdirs==4.12.2
pluggy==1.6.0
pydantic==2.13.5
pydantic-core==2.46.5
pyfiglet==0.8.post1
pygments==2.21.0
python-dateutil==2.9.0.post0
pyyaml==6.0.3
scikit-build-core==1.1.0
scipy==1.18.1
setuptools==78.1.0
setuptools-scm==10.3.4
six==1.17.0
sqlalchemy==2.1.2
sympy==1.14.0
tomlkit==0.15.1
torch==2.8.0+cu128
triton==3.4.0
trove-classifiers==2026.9.21.13
typeguard==4.6.0
typing-extensions==4.16.0
typing-inspection==0.4.4
vcs-versioning==2.5.0
vesin==0.6.2
vesin-torch==0.6.2
wcmatch==11.0.1
wrapt==2.5.0
```

（其中 cmake / ninja / scikit-build-core / setuptools_scm / hatch_fancy_pypi_readme / dependency_groups / hatchling 等仅构建期需要，常驻 venv 可精简；fsspec/jinja2/networkx/opt-einsum(-fx)/sympy/mpmath 等随 torch 2.8.0 wheel 带入。）

## 2. 加载代码（/tmp/dpa4_eval.py，临时脚本，已随 venv 一并删除；核心逻辑如下）

```python
from deepmd.infer import DeepPot
dp = DeepPot("/root/soft/ml_potential/DPA-4/DPA4-beta.pt", head="H2O_H2O_PD")
e, f, v = dp.eval(coords.reshape(1, -1, 3), None, atom_types,
                  charge_spin=np.array([0.0, 1.0]))
```

atom_types 按模型 118 元素 type_map（H=0, O=7 → `[7, 0, 0]`）；cells=None（非周期）。运行前缀：`LD_LIBRARY_PATH=/usr/local/lib64`，CPU 实验加 `DEVICE=cpu`。

## 3. 四问答案

### ① DeepEval 能否加载？

**能，直接加载，无需 `dp train --restart` 或模型导出**。deepmd-kit 3.2.0 的 PT 后端 DeepEval 对 `.pt` 训练 checkpoint 原生支持：`torch.load(weights_only=True)` 后取 `state_dict["model"]["_extra_state"]["model_params"]` 重建模型，multitask 时按 `head` 抽取分支权重（`deepmd/pt/infer/deep_eval.py`）。前提只有一条：**deepmd-kit 必须针对运行期 torch 版本源码重编译**（PyPI wheel 绑死 torch 2.11，见 §1）。重编译后 `ENABLE_CUSTOMIZED_OP=True`，加载零 warning。

### ② model_branch 选择与 fparam/chg_spin 固定值

- checkpoint 实际 23 分支（`torch.load` 实测）：Domains_Alloy, Domains_Anode, Domains_Cluster, Domains_FerroEle, Domains_SSE_PBE, Domains_SemiCond, **H2O_H2O_PD**, Metals_AlMgCu, Metals_AgAu_PBED3, Others_In2Se3, MPGen_OpenCSP, **Alloy_APEX**, SSE_ABACUS, Hybrid_Perovskite, Electrolyte, ODAC23, Alex2D, **OMat24**, OC20M, OC22, Organic_Reactions, **OMol25**, MPTrj。注意与 `DPA-4/input.json` 有两处命名差异（`Alloy_tongqi`→`Alloy_APEX`、`Omat24`→`OMat24`），checkpoint 训练所用 input 比目录内留存版本略新；分支选择以 checkpoint 实际名单为准。
- **推荐 head=`H2O_H2O_PD`**：input.json 的 `data_dict` 显示该分支是专门的 H2O 数据集（`H2O_H2O_PD/trn.lmdb`，训练采样权重 0.89），是纯水/液水场景的领域专属分支；`OMol25` 是通用有机分子分支（采样权重 15.2，wB97M-V 级）。单分子 H2O 用领域专属分支语义最正确；两者能量参考态/理论级别不同，绝对值不可互换（实测差 ~2064 eV）。
- **fparam**：实测 `dim_fparam=0`、`has_default_fparam=False`——该模型无 fparam 维度，无需也不能传 fparam（input.json 统计文件名里的 `with_fparam_0_1` 指 chg_spin 两通道经 fparam 路径统计，推理侧体现为 charge_spin 而非 fparam）。
- **chg_spin**：`has_chg_spin_ebd=True`、`has_default_chg_spin=True`，默认即 `[0.0, 1.0]`（电荷 0、自旋通道 1，中性闭壳层 H2O 正确）。显式传 `charge_spin=[0.0,1.0]` 与省略（走默认）**逐位一致**（`default_equals_explicit: True`），故接入时可固定传 `[0.0,1.0]` 或省略均可。

### ③ H2O 能量与逐位一致性

能量单位 eV（deepmd-kit 推理原生单位）。

| 配置 | head | 能量 (eV) | 同机两次逐位一致？ |
|---|---|---|---|
| GPU (cuda:0, 默认) | H2O_H2O_PD | 进程1: **−15.710772596299648**（sha256 ba6058e2…）<br>进程2: −15.710772827267647 | **能量**：进程内逐位一致，**跨进程否**（Δ≈2.3e-7）；**力**：进程内即不逐位一致 |
| GPU | OMol25 | 进程1: **−2079.8514928892255**<br>进程2: −2079.8514929637313 | 同上（Δ≈7.5e-8） |
| CPU（DEVICE=cpu，默认线程） | H2O_H2O_PD | **−15.710773142986**（两进程 sha256 均 2aa74925…） | 能量跨进程**逐位一致**；力跨进程不逐位一致（Δ~1e-7） |
| CPU（OMP_NUM_THREADS=1） | H2O_H2O_PD | **−15.710773446015082**（两进程 sha256 均 b82700b7…） | 能量跨进程**逐位一致**；力仍不逐位一致（Δ~1e-7，PYTHONHASHSEED=0 亦不能消除） |

要点：
- **能量在固定设备+固定线程配置下逐位可复现**（跨进程 sha256 相同）；GPU 与 CPU 之间差 ~8.5e-7 eV（TF32/AMP 路径差异，input.json `enable_tf32: true`、`use_amp: true`）；不同线程数配置之间能量本身也有 ~3e-7 漂移。
- **力在任何配置下均不能跨进程逐位复现**（进程内 CPU 一致、GPU 不一致），偏差量级 ~1e-7 eV/Å（相对 ~1e-5），疑似邻居排序/scatter 累加顺序非确定所致。阶段 2 若 pin benchmark，应只 pin 能量且带容差（建议 ±1e-5 eV），或固定 CPU+OMP_NUM_THREADS=1 出金标准能量。
- 力数值本身正常（O 上合力 ~0.05 eV/Å 量级，三个原子力之和 ~1e-7 接近零动量），无 NaN/Inf。

### ④ torch 2.8 上 SeZM 兼容性结论

**兼容（源码重编译后），无 warning、无报错、无数值异常**。细目：
- PyPI 二进制 wheel 不兼容 torch 2.8（绑 torch 2.11 ABI），必须 `--no-binary` 源码编译；编译一次通过（CUDA 12.8 + gcc 11.3 + sm_86）。
- e3nn 0.6.0 在 torch 2.8.0+cu128 上运行 SeZM（lmax=4, SO(2) conv）无任何 deprecation/用户 warning（实验脚本以 `warnings.simplefilter("always")` 全程捕获，import/load/eval 三个阶段 warnings 列表均为空）。
- 唯一额外运行依赖：vesin+vesin-torch（邻居表）、mpich（import 期 libmpi）、`LD_LIBRARY_PATH=/usr/local/lib64`（libstdc++）。
- 数值：能量/力量级正常、可复现性如上；GPU 首次 eval 无 kernel 报错。
- 次要噪音：stderr 有一条 OMP 线程调优提示（logger info 级，非 warning）。

## 4. 阶段 2 建议：**可行**

- 机制验证全部通过，无阻断项。建议按工单预设走**独立常驻 venv + 子进程脚本**（runner argv 首元素即解释器路径，无需改 mlip_agent 框架）：
  - venv 配方完全复用本报告 §1（Python 3.12.12、torch 2.8.0+cu128、deepmd-kit 3.2.0 源码编译、vesin、mpich、e3nn 0.6.0；构建期依赖可不装入常驻环境）；
  - wrapper 脚本固化 `LD_LIBRARY_PATH=/usr/local/lib64`、`head=H2O_H2O_PD`（水体系）/`charge_spin=[0.0,1.0]`；
  - benchmark pin：能量 pin 用 CPU+OMP=1 金标准 **−15.710773446015082 eV**（逐位可复现）或 GPU 值 ±1e-5 容差；**不要 pin 力**（跨进程不逐位稳定）。
- 阶段 2 工单待评审项（按前置条件 5）：常驻 venv 位置（本 scratch 占 **6.9 GB**，其中 torch+CUDA 库占大头，构建期依赖剔除后可瘦身）、GPU 锁共享语义（与 MAPLE/直算后端共用 GPU 0 的排队策略）。
- 风险残留：① 力不可逐位复现的根因未定位（不影响能量验收）；② 其他 22 个分支未逐一验证（机制相同，风险低）；③ input.json 与 checkpoint 分支命名差异提示目录内 input.json 非最终训练版，接入文档应注明以 checkpoint 内嵌参数为准。

## 5. 清理

`/tmp/dpa4_venv`（6.9 GB）按零足迹原则**已删除**；全部结论与配方记录于本报告，阶段 2 可按 §1 配方 ~15 min 内重建。未触碰 COMPUTE_VENV、mlip_agent/.venv、MAPLE、checkpoint 及 mlip_agent 代码。实验脚本 /tmp/dpa4_eval.py 与运行日志（/tmp/run_h2o_{1,2}.log、/tmp/run_omol_{1,2}.log、/tmp/run_h2o_cpu_{1,2}.log）一并删除，关键数值已录入 §3。
