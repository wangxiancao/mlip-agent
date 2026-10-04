"""把 mace-mpa-0-medium.model 封装成 MAPLE MACECalculator 期望的 scripted Wrapper。

背景（仿 scripts/make_maceoff23s_wrapper.py 的 D7-4 范式）：MACE-MPA-0 medium
（mace-foundations，MPTrj+sAlex 晶体训练域，约 9.1M 参数，MIT 许可）官方权重是
mace-torch pickle（torch.load），MAPLE MACECalculator 只接受 scripted wrapper。
本脚本在 scratch 环境（torch 2.8.0 + mace-torch 0.3.10，与 COMPUTE_VENV 的
torch 2.8.0 同版本）经 e3nn jit.compile 生成
/root/soft/ml_potential/checkpoints/mace/macempa0.pt，由 plugins/maple_mace_plugin.py
（MODEL_NAMES=("macempa0",)）经显式 model_path 加载。

MAPLE 调用契约（MAPLE/maple/function/calculator/mace/_mace_calculator.py）：
- 属性：model.r_max / model.atomic_numbers（buffer）
- forward(data: Dict[str, Tensor], local_or_ghost: Tensor, compute_virials: bool=False)
  返回 per-graph local energy（MAPLE 侧 .sum() 得总能量，eV）
dtype 链：MACE-MPA-0 官方 ckpt 为 float64；MAPLE 侧 model_float_dtype 从
scripted 模型参数推断输入 dtype（_common.py），保持 float64 与 maceoff23s.pt 一致。

溯源（2026-10-03）：
- 源权重：mace-foundations/mace-mpa-0（HuggingFace）的 mace-mpa-0-medium.model，
  https://huggingface.co/mace-foundations/mace-mpa-0/resolve/main/mace-mpa-0-medium.model
  许可证 MIT。
- 源文件 /root/soft/ml_potential/checkpoints/mace/mace-mpa-0-medium.model SHA256:
  75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638
- 产物 macempa0.pt SHA256（torch 2.8.0+cpu + mace-torch 0.3.10 + e3nn 0.4.4 环境下）:
  03c1b56effe5aeb21de5a9966986f7371ec53f5818873963a8edf9968a5831ef
  （torch.jit.save 产物哈希依赖 torch/mace-torch 版本，重生成后以本注释为准复核）
- heads 分支说明：mace-torch 0.3.10 的 ScaleShiftMACE 一律带 heads 属性，
  本 ckpt heads=['default']（单头，非多头 checkpoint）；wrapper 的
  hasattr(model, "heads") 分支因此触发并取 head=0（唯一头），与官方
  MACECalculator 单头路径一致——已由自检 2 数值证实（差 7.5e-14 eV）。

用法：
    uv venv /tmp/mace_mpa_venv --python 3.12
    uv pip install --python /tmp/mace_mpa_venv/bin/python torch==2.8.0 \\
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/mace_mpa_venv/bin/python mace-torch==0.3.10 ase
    curl -sL -o /root/soft/ml_potential/checkpoints/mace/mace-mpa-0-medium.model \\
        https://huggingface.co/mace-foundations/mace-mpa-0/resolve/main/mace-mpa-0-medium.model
    echo '75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638  /root/soft/ml_potential/checkpoints/mace/mace-mpa-0-medium.model' | sha256sum -c -
    /tmp/mace_mpa_venv/bin/python scripts/make_macempa0_wrapper.py

自检（main 内置）：reload 后对 H2O（B4/B5 金标准几何）跑一次模拟 MAPLE
data dict 的 forward，并与同 venv 内 mace 官方 ASE calculator（mace_mp 直载
源 .model）的 H2O 能量对照，容差 1e-6 eV。
"""
from typing import Dict

import torch
from e3nn.util.jit import compile_mode
from mace.tools.scatter import scatter_sum

SRC = "/root/soft/ml_potential/checkpoints/mace/mace-mpa-0-medium.model"
OUT = "/root/soft/ml_potential/checkpoints/mace/macempa0.pt"

# B4/B5 金标准 H2O 几何（O-H 0.96 Å）
H2O_POSITIONS = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
H2O_Z = [8, 1, 1]


@compile_mode("script")
class MapleMACEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
    ) -> torch.Tensor:
        num_graphs = data["ptr"].numel() - 1
        if hasattr(self.model, "heads"):
            data["head"] = torch.tensor(
                [0], dtype=torch.long, device=data["positions"].device
            )
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
        )
        node_energy = out["node_energy"]
        assert node_energy is not None
        node_energy = node_energy * local_or_ghost
        return scatter_sum(
            src=node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
        )


def build_h2o_data(model, dtype):
    """模拟 MAPLE build_data_from_atoms 的 H2O data dict（no-PBC 半径图）。"""
    pos = torch.tensor(H2O_POSITIONS, dtype=dtype)
    Z = torch.tensor(H2O_Z, dtype=torch.long)
    table = model.atomic_numbers
    node_attrs = (Z[:, None] == table[None, :]).to(dtype)
    r_max = float(model.r_max)
    N = pos.size(0)
    rij = pos[:, None, :] - pos[None, :, :]
    d2 = (rij * rij).sum(dim=-1)
    mask = torch.ones((N, N), dtype=torch.bool)
    mask.fill_diagonal_(False)
    mask &= d2 <= (r_max + 1e-12) ** 2
    iu, ju = torch.nonzero(torch.triu(mask), as_tuple=True)
    edge_index = torch.stack(
        [torch.cat([iu, ju]), torch.cat([ju, iu])]
    ).to(torch.long)
    E = edge_index.size(1)
    data = {
        "batch": torch.zeros(N, dtype=torch.int64),
        "cell": torch.zeros(3, 3, dtype=dtype),
        "charges": torch.zeros(N, dtype=dtype),
        "dipole": torch.zeros(1, 3, dtype=dtype),
        "edge_index": edge_index,
        "energy": torch.tensor([0.0], dtype=dtype),
        "energy_weight": torch.tensor([0.0], dtype=dtype),
        "forces": torch.zeros(N, 3, dtype=dtype),
        "forces_weight": torch.tensor([0.0], dtype=dtype),
        "node_attrs": node_attrs,
        "positions": pos,
        "ptr": torch.tensor([0, N], dtype=torch.int64),
        "shifts": torch.zeros(E, 3, dtype=dtype),
        "stress": torch.zeros(1, 3, 3, dtype=dtype),
        "stress_weight": torch.tensor([0.0], dtype=dtype),
        "unit_shifts": torch.zeros(E, 3, dtype=dtype),
        "virials": torch.zeros(1, 3, 3, dtype=dtype),
        "virials_weight": torch.tensor([0.0], dtype=dtype),
        "weight": torch.tensor([1.0], dtype=dtype),
    }
    local_or_ghost = torch.ones(N, dtype=dtype)
    return data, local_or_ghost


def main():
    import ase
    import e3nn
    import mace

    print(
        f"versions: torch={torch.__version__} mace-torch={mace.__version__} "
        f"e3nn={e3nn.__version__} ase={ase.__version__}"
    )
    raw = torch.load(SRC, map_location="cpu")
    print("loaded:", type(raw).__name__)
    print("r_max:", raw.r_max, "atomic_numbers:", raw.atomic_numbers)
    dtype = next(raw.parameters()).dtype
    print("dtype:", dtype, "has heads:", hasattr(raw, "heads"))

    from e3nn.util import jit

    wrapper = MapleMACEWrapper(raw)
    scripted = jit.compile(wrapper)
    torch.jit.save(scripted, OUT)
    print("saved:", OUT)

    # 自检 1：重新加载，对 H2O 跑一次 forward（模拟 MAPLE 的 data 构造）
    m = torch.jit.load(OUT)
    data, local_or_ghost = build_h2o_data(m, dtype)
    e_wrap = m.forward(data=data, local_or_ghost=local_or_ghost, compute_virials=False)
    e_wrap_ev = e_wrap.sum().item()
    print("H2O energy (wrapper, eV):", e_wrap_ev)

    # 自检 2：与 mace 官方 ASE calculator（同 venv 直载源 .model）对照
    from ase import Atoms
    from mace.calculators import mace_mp

    atoms = Atoms(
        numbers=H2O_Z, positions=H2O_POSITIONS, pbc=False
    )
    atoms.calc = mace_mp(model=SRC, device="cpu", default_dtype="float64")
    e_ase_ev = float(atoms.get_potential_energy())
    print("H2O energy (mace_mp ASE, eV):", e_ase_ev)
    diff = abs(e_wrap_ev - e_ase_ev)
    print("abs diff (eV):", diff)
    assert diff < 1e-6, f"wrapper 与官方 ASE calculator 不一致: {diff} eV"
    print("OK: wrapper == mace_mp within 1e-6 eV")


if __name__ == "__main__":
    main()
