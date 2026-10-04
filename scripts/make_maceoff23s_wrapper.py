"""把 MACE-OFF23_small.model 封装成 MAPLE MACECalculator 期望的 scripted Wrapper。

背景（docs/decisions.md D7-4）：maceoff23s 是 REQUIRES_LOCAL_MODEL_FILE，
Wayne7815/MAPLE_models 不提供，MAPLE 无法自动下载。本脚本用官方权重
（ACEsuit/mace-off 仓库的 mace_off23/MACE-OFF23_small.model）在 scratch 环境
（torch 2.8.0 + mace-torch 0.3.10，与 COMPUTE_VENV 的 torch 2.8.0 同版本）经
e3nn jit.compile 生成 checkpoints/maple/maceoff23s.pt
（MAPLE/maple/function/calculator/model 是该目录的符号链接，TICKET-0010）。

MAPLE 调用契约（MAPLE/maple/function/calculator/mace/_mace_calculator.py）：
- 属性：model.r_max / model.atomic_numbers（buffer）
- forward(data: Dict[str, Tensor], local_or_ghost: Tensor, compute_virials: bool=False)
  返回 per-graph local energy（MAPLE 侧 .sum() 得总能量，eV）
与 mace.calculators.lammps_mace.LAMMPS_MACE 的能量路径同构。

溯源（TICKET-0003 条件 1，2026-10-02）：
- 源权重 pin 到 ACEsuit/mace-off commit 91a78c5a9c300d1104700d9352c8bfe449227737
  （原 /raw/main/ 浮动引用已固定；pin 版本与初版下载的源文件哈希一致，已复验）。
- 源文件 MACE-OFF23_small.model SHA256:
  165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f
- 产物 maceoff23s.pt SHA256（torch 2.8.0 + mace-torch 0.3.10 环境下）:
  a234308d11de00fbf2ca1a83f29c870cf429e1ee16ada0e11cf9ebc155c3c5f6
  （torch.jit.save 产物哈希依赖 torch/mace-torch 版本，重生成后以工单记录为准复核）
- 许可证：MACE-OFF23 权重为 Academic Software License（ASL），
  仅限学术研究用途、禁止商业使用（见 tickets/TICKET-0003 条件 4）。

用法：
    uv venv /tmp/mace_venv --python 3.12
    uv pip install --python /tmp/mace_venv/bin/python torch==2.8.0 \\
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/mace_venv/bin/python mace-torch==0.3.10
    curl -sL -o /tmp/mace_off/MACE-OFF23_small.model \\
        https://github.com/ACEsuit/mace-off/raw/91a78c5a9c300d1104700d9352c8bfe449227737/mace_off23/MACE-OFF23_small.model
    echo '165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f  /tmp/mace_off/MACE-OFF23_small.model' | sha256sum -c -
    /tmp/mace_venv/bin/python scripts/make_maceoff23s_wrapper.py
    cp /tmp/mace_off/maceoff23s.pt \\
        /root/soft/ml_potential/checkpoints/maple/
    # （MAPLE/maple/function/calculator/model 是该目录的符号链接，TICKET-0010）
"""
from typing import Dict

import torch
from e3nn.util.jit import compile_mode
from mace.tools.scatter import scatter_sum


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


def main():
    raw = torch.load(
        "/tmp/mace_off/MACE-OFF23_small.model", map_location="cpu"
    )
    print("loaded:", type(raw).__name__)
    print("r_max:", raw.r_max, "atomic_numbers:", raw.atomic_numbers)
    dtype = next(raw.parameters()).dtype
    print("dtype:", dtype, "has heads:", hasattr(raw, "heads"))

    from e3nn.util import jit

    wrapper = MapleMACEWrapper(raw)
    scripted = jit.compile(wrapper)
    out = "/tmp/mace_off/maceoff23s.pt"
    torch.jit.save(scripted, out)
    print("saved:", out)

    # 自检：重新加载，对 H2O 跑一次 forward（模拟 MAPLE 的 data 构造）
    m = torch.jit.load(out)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=dtype
    )
    Z = torch.tensor([8, 1, 1], dtype=torch.long)
    table = m.atomic_numbers
    node_attrs = (Z[:, None] == table[None, :]).to(dtype)
    r_max = float(m.r_max)
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
    e = m.forward(data=data, local_or_ghost=local_or_ghost, compute_virials=False)
    print("H2O energy (eV):", e.sum().item())


if __name__ == "__main__":
    main()
