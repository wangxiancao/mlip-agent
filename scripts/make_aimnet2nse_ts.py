#!/usr/bin/env python
"""One-shot converter: AIMNet2-NSE v2 checkpoint (HF safetensors) -> MAPLE TorchScript model.

Provenance
----------
- Source weights: HuggingFace isayevlab/aimnet2-nse
  - ensemble_0.safetensors  SHA256 c65c7e1c3dc15b97f473f544d0c6835bd8d0c3e6b294523a9311b83eb0d723b6
  - config.json             (AIMNet2 v2 metadata: model_yaml, d3_params, coulomb_sr_rc ...)
- Theory level: wB97M-D3/def2-TZVPP, UKS open-shell (mult via NSE, num_charge_channels=2),
  14 elements H B C N O F Si P S Cl As Se Br I, charged systems supported.
- Upstream code: /root/soft/ml_potential/aimnetcentral (git v0.1.0-11-g8067558, MIT license)
- Conversion env: python 3.12, torch 2.8.0+cpu (matches COMPUTE_VENV torch 2.8.0),
  aimnet (editable local snapshot), nvalchemi-toolkit-ops 0.2.0, warp-lang 1.11.0, ase 3.27.0

Why this script exists
----------------------
Official AIMNet2-NSE ships only v2 format (state dict + config); v2 strips the long-range
modules (LRCoulomb / DFTD3) out of the model into the calculator. MAPLE's AIMNet2 calculator
(/root/soft/ml_potential/MAPLE/.../aimnet/_aimnet2_calculator.py, READ-ONLY) instead requires
a single TorchScript archive with LR modules embedded in the JIT graph (like aimnet2.pt).
This script rebuilds the full model per the MAPLE contract:

    outputs order: energy_mlp -> atomic_shift -> atomic_sum -> srcoulomb (embedded SR,
    from checkpoint) -> lrcoulomb (FULL pairwise, subtract_sr=False, replicating the v2
    eager external module) -> dftd3 (pure-torch DFT-D3(BJ) replicating the nvalchemiops
    0.2.0 kernel math; the upstream custom op cannot be embedded because MAPLE's runtime
    has no aimnet/nvalchemiops installed).

Numerical acceptance (checked here, hard fail if violated):
    scripted reloaded model vs upstream v2 eager path (aimnet.calculators AIMNet2Calculator
    + external LRCoulomb/DFTD3), on H2O singlet and CH3 doublet:
    |dE| <= 1e-6 eV, |dF| <= 1e-5 eV/Angstrom.

Usage:
    /tmp/aimnet2nse/venv/bin/python scripts/make_aimnet2nse_ts.py [--out /tmp/aimnet2nse/aimnet2nse.ts.pt]
"""

import argparse
import hashlib
import json

import torch
import yaml
from safetensors.torch import load_file
from torch import Tensor, nn

from aimnet import nbops, ops
from aimnet.config import build_module
from aimnet.modules import LRCoulomb
from aimnet.modules.aev import ConvSV

WORK_DIR = "/tmp/aimnet2nse"
SAFETENSORS = f"{WORK_DIR}/ensemble_0.safetensors"
CONFIG_JSON = f"{WORK_DIR}/config.json"
EXPECTED_SHA256 = "c65c7e1c3dc15b97f473f544d0c6835bd8d0c3e6b294523a9311b83eb0d723b6"

DFTD3_DATA = "/root/soft/ml_potential/aimnetcentral/aimnet/dftd3_data.pt"

# nvalchemiops 0.2.0 dftd3 defaults (aimnet wrapper does not override them)
D3_K1 = 16.0
D3_K3 = -4.0
# aimnet.calculators.calculator external DFTD3 defaults
D3_CUTOFF = 15.0
D3_SMOOTHING_FRACTION = 0.2

BOHR_INV = 1.0 / 0.5291772105638411  # matches aimnet.constants.Bohr_inv


def _conv_sv_forward_pure_torch(self: ConvSV, data: dict[str, Tensor], a: Tensor) -> Tensor:
    """Pure-torch replacement for aimnet.modules.aev.ConvSV.forward.

    The upstream forward dispatches to a warp custom op (aimnet::conv_sv_2d_sp_fwd)
    when running on CUDA. TorchScript compiles BOTH branches into the archive, and
    MAPLE's runtime (no aimnet/nvalchemiops installed) cannot resolve the custom op
    at torch.jit.load time. The custom op computes exactly the same contraction as
    the pure-torch fallback branch (verified by upstream tests), so we script the
    fallback only. Patch is applied around torch.jit.script in main().
    """
    g_sv = data["g_sv"]
    mode = nbops.get_nb_mode(data)
    if self.d2features:
        if mode > 0:
            a_j = a.index_select(0, data["nbmat"].flatten()).unflatten(0, data["nbmat"].shape)
            avf_sv = torch.einsum("...mag,...mgd->...agd", a_j, g_sv)
        else:
            avf_sv = torch.einsum("...mag,...mgd->...agd", a.unsqueeze(1), g_sv)
    else:
        if mode > 0:
            a_j = a.index_select(0, data["nbmat"].flatten()).unflatten(0, data["nbmat"].shape)
            avf_sv = torch.einsum("...ma,...mgd->...agd", a_j, g_sv)
        else:
            avf_sv = torch.einsum("...ma,...mgd->...agd", a.unsqueeze(1), g_sv)
    avf_s, avf_v = avf_sv.split([1, 3], dim=-1)
    avf_v = torch.einsum("agh,...agd->...ahd", self.agh, avf_v).pow(2).sum(-1)
    return torch.cat([avf_s.squeeze(-1).flatten(-2, -1), avf_v.flatten(-2, -1)], dim=-1)


class DFTD3BJPure(nn.Module):
    """Pure-PyTorch DFT-D3(BJ) replicating nvalchemiops 0.2.0 `dftd3` numerics.

    Replicates the warp kernel (_cn_kernel_nl + _direct_forces_and_dE_dCN_kernel_nl):
    - CN counting: f = 1/(1+exp(-k1*((rcov_i+rcov_j)/r - 1))), f32, padding/zero-r skipped
    - C6 Gaussian interpolation over the 5x5 grid, skipping c6==0 entries,
      log-sum-exp stabilization, L=exp(k3*(di^2+dj^2)-max) dropped below -12,
      c6 = z/w for w > 1e-12, pair skipped when c6 < 1e-12
    - BJ damping: r4r2_ij = 3*r4r2_i*r4r2_j, r0 = a1*sqrt(r4r2_ij)+a2,
      damp = s6/(r^6+r0^6) + s8*r4r2_ij/(r^8+r0^8), e_ij = -c6*damp
    - S5 switching on [smoothing_on, smoothing_off]
    - accumulation: per-atom f64 sum -> f32 -> *0.5 -> per-system f32 sum (kernel order)
    Distances handled in Bohr inside (kernel works on positions converted to Bohr);
    output energy in Hartree is converted to eV by the caller-side factor Hartree.
    """

    s6: float
    s8: float
    a1: float
    a2: float
    k1: float
    k3: float
    smoothing_on: float  # Bohr
    smoothing_off: float  # Bohr
    bohr_inv: float
    key_out: str

    def __init__(
        self,
        rcov: Tensor,
        r4r2: Tensor,
        c6ab: Tensor,
        cnref_i: Tensor,
        cnref_j: Tensor,
        s8: float,
        a1: float,
        a2: float,
        s6: float = 1.0,
        cutoff: float = D3_CUTOFF,
        smoothing_fraction: float = D3_SMOOTHING_FRACTION,
        key_out: str = "energy",
    ):
        super().__init__()
        self.register_buffer("rcov", rcov.float())
        self.register_buffer("r4r2", r4r2.float())
        self.register_buffer("c6ab", c6ab.float())  # [Z, Z, 5, 5]
        self.register_buffer("cnref_i", cnref_i.float())  # [Z, Z, 5, 5]
        self.register_buffer("cnref_j", cnref_j.float())  # [Z, Z, 5, 5]
        self.s6 = float(s6)
        self.s8 = float(s8)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.k1 = D3_K1
        self.k3 = D3_K3
        self.smoothing_on = cutoff * (1.0 - smoothing_fraction) * BOHR_INV
        self.smoothing_off = cutoff * BOHR_INV
        self.bohr_inv = BOHR_INV
        self.key_out = key_out

    def _s5_switch(self, r: Tensor) -> Tensor:
        # r in Bohr; returns Sw(r) in [0, 1]
        t = (r - self.smoothing_on) / (self.smoothing_off - self.smoothing_on)
        t2 = t * t
        s5 = 10.0 * t2 * t - 15.0 * t2 * t2 + 6.0 * t2 * t2 * t
        sw_mid = 1.0 - s5
        return torch.where(
            r <= self.smoothing_on,
            torch.ones_like(r),
            torch.where(r >= self.smoothing_off, torch.zeros_like(r), sw_mid),
        )

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        suffix = nbops.resolve_suffix(data, ["_dftd3", "_lr"])
        data = ops.lazy_calc_dij(data, suffix)
        d_bohr = data[f"d_ij{suffix}"] * self.bohr_inv

        numbers = data["numbers"]
        z_i, z_j = nbops.get_ij(numbers, data, suffix)

        # --- coordination numbers (f32) ---
        rcov_i, rcov_j = nbops.get_ij(self.rcov[numbers], data, suffix)
        r_ok = d_bohr >= 1e-12
        f_cn = 1.0 / (1.0 + torch.exp(-self.k1 * ((rcov_i + rcov_j) / torch.clamp(d_bohr, min=1e-12) - 1.0)))
        f_cn = torch.where(r_ok, f_cn, torch.zeros_like(f_cn))
        f_cn = nbops.mask_ij_(f_cn, data, 0.0, suffix=suffix)
        cn = f_cn.sum(-1)
        cn_i, cn_j = nbops.get_ij(cn, data, suffix)

        # --- C6 interpolation (f32, kernel-identical branching) ---
        c6ab_pair = self.c6ab[z_i, z_j]  # (..., 5, 5)
        cnref_i = self.cnref_i[z_i, z_j]
        cnref_j = self.cnref_j[z_i, z_j]
        di = cn_i.unsqueeze(-1).unsqueeze(-1) - cnref_i
        dj = cn_j.unsqueeze(-1).unsqueeze(-1) - cnref_j
        pre = self.k3 * (di * di + dj * dj)
        valid = c6ab_pair != 0.0
        neg_big = torch.full_like(pre, -1.0e30)
        max_exp = torch.where(valid, pre, neg_big).amax(dim=(-1, -2), keepdim=True)
        exp_arg = pre - max_exp
        l_ij = torch.where(
            valid & (exp_arg >= -12.0),
            torch.exp(torch.clamp(exp_arg, max=50.0)),
            torch.zeros_like(exp_arg),
        )
        w = l_ij.sum(dim=(-1, -2))
        z = (c6ab_pair * l_ij).sum(dim=(-1, -2))
        c6 = torch.where(w > 1e-12, z / torch.clamp(w, min=1e-12), torch.zeros_like(w))
        c6 = torch.where(c6 >= 1e-12, c6, torch.zeros_like(c6))

        # --- BJ damping (f32, kernel op order) ---
        r4r2_i, r4r2_j = nbops.get_ij(self.r4r2[numbers], data, suffix)
        r4r2_ij = 3.0 * r4r2_i * r4r2_j
        r0 = self.a1 * torch.sqrt(r4r2_ij) + self.a2
        r2 = d_bohr * d_bohr
        r4 = r2 * r2
        r6 = r4 * r2
        r8 = r4 * r4
        r0_2 = r0 * r0
        r0_4 = r0_2 * r0_2
        r0_6 = r0_4 * r0_2
        r0_8 = r0_4 * r0_4
        damp = self.s6 / (r6 + r0_6) + self.s8 * r4r2_ij / (r8 + r0_8)

        e_ij = -c6 * damp
        e_ij = e_ij * self._s5_switch(d_bohr)
        e_ij = torch.where(r_ok, e_ij, torch.zeros_like(e_ij))
        e_ij = nbops.mask_ij_(e_ij, data, 0.0, suffix=suffix)

        # kernel: per-atom f64 accumulate -> f32 -> 0.5* -> atomic_add f32 per system
        e_atom = (e_ij.sum(-1, dtype=torch.float64)).float() * 0.5
        e = nbops.mol_sum(e_atom, data)

        # Hartree -> eV and add to output (matches aimnet wrapper: energy * Hartree)
        e_ev = e * 27.211386024367243
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e_ev.to(data[self.key_out].dtype)
        else:
            data[self.key_out] = e_ev
        return data


# ----------------------------------------------------------------------------
# MAPLE input contract replication (from MAPLE _aimnet2_calculator.py, read-only
# reference) — builds exactly the tensors MAPLE feeds to the scripted model.
# ----------------------------------------------------------------------------
def nblist_dense_padded(coord: Tensor, cutoff: float) -> Tensor:
    device = coord.device
    N = coord.shape[0]
    diff = coord[:, None, :] - coord[None, :, :]
    dist2 = torch.sum(diff**2, dim=-1)
    dist2[torch.eye(N, dtype=torch.bool, device=device)] = float("inf")
    mask = dist2 <= cutoff**2
    M = max(int(mask.sum(dim=1).max().item()), 1)
    nbmat = torch.full((N + 1, M), N, dtype=torch.int32, device=device)
    for i in range(N):
        nb_i = torch.nonzero(mask[i], as_tuple=False).flatten()
        if nb_i.numel() > 0:
            nbmat[i, : min(nb_i.numel(), M)] = nb_i[: min(nb_i.numel(), M)]
    return nbmat


def pad_dim0(a: Tensor, value: float = 0.0) -> Tensor:
    pad_shape = list(a.shape)
    pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)


def maple_build_data(coords, numbers, charge, mult, cutoff, cutoff_lr, device="cpu"):
    """Replicates MAPLE AIMNet2Calculator._build_data for coulomb_method='simple'
    (cutoff_lr=inf -> nbmat_lr built with the SR cutoff)."""
    import numpy as np

    coord = torch.tensor(coords, dtype=torch.float32, device=device, requires_grad=True)
    Z = torch.tensor(numbers, dtype=torch.int32, device=device)
    N = coord.shape[0]
    mol_idx = torch.zeros(N, dtype=torch.int32, device=device)
    data = {
        "coord": pad_dim0(coord, value=0.0),
        "numbers": pad_dim0(Z, value=0),
        "charge": torch.tensor([float(charge)], dtype=torch.float32, device=device),
        "mult": torch.tensor([float(mult)], dtype=torch.float32, device=device),
        "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
        "nbmat": nblist_dense_padded(coord, cutoff),
    }
    lr_cutoff = cutoff_lr if np.isfinite(cutoff_lr) else cutoff
    data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
    data["cutoff_lr"] = torch.tensor(lr_cutoff, device=device)
    return data, coord


SYSTEMS = {
    "h2o_singlet": (
        [8, 1, 1],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        0,
        1,
    ),
    "ch3_doublet": (
        [6, 1, 1, 1],
        [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [-0.545, 0.943959, 0.0], [-0.545, -0.943959, 0.0]],
        0,
        2,
    ),
    "o2_triplet": (
        [8, 8],
        [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0]],
        0,
        3,
    ),
}

E_TOL = 1e-6  # eV
F_TOL = 1e-5  # eV/Angstrom


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_full_model() -> nn.Module:
    """Build AIMNet2-NSE with embedded SR coulomb (checkpoint) + FULL LR coulomb + D3(BJ)."""
    cfg = json.load(open(CONFIG_JSON))
    sd = load_file(SAFETENSORS)

    model = build_module(yaml.safe_load(cfg["model_yaml"]))
    # float64 atomic shifts BEFORE weight loading to preserve SAE precision
    # (mirrors aimnet.models.utils.load_v1_model / models.base.load_model)
    model.outputs.atomic_shift.double()
    model.load_state_dict(sd, strict=True)

    # External-module equivalents, embedded in the same order the v2 eager path runs them:
    # model(SRCoulomb subtracts SR) -> LRCoulomb FULL (subtract_sr=False) -> DFTD3.
    model.outputs["lrcoulomb"] = LRCoulomb(
        key_in="charges",
        key_out="energy",
        rc=float(cfg["coulomb_sr_rc"]),
        method="simple",
        subtract_sr=False,
        envelope=str(cfg["coulomb_sr_envelope"]),
    )
    param = torch.load(DFTD3_DATA, map_location="cpu", weights_only=True)
    c6ab_packed = param["c6ab"]
    d3 = cfg["d3_params"]
    model.outputs["dftd3"] = DFTD3BJPure(
        rcov=param["rcov"],
        r4r2=param["r4r2"],
        c6ab=c6ab_packed[..., 0].contiguous(),
        cnref_i=c6ab_packed[..., 1].contiguous(),
        cnref_j=c6ab_packed[..., 2].contiguous(),
        s8=float(d3["s8"]),
        a1=float(d3["a1"]),
        a2=float(d3["a2"]),
        s6=float(d3.get("s6", 1.0)),
    )
    # MAPLE reads getattr(model, 'cutoff') and getattr(model, 'cutoff_lr', inf)
    model.cutoff = float(cfg["cutoff"])
    model.cutoff_lr = float("inf")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def eager_reference(cfg):
    """Upstream v2 eager reference: AIMNet2Calculator with external LRCoulomb/DFTD3."""
    from aimnet.calculators.calculator import AIMNet2Calculator

    sd = load_file(SAFETENSORS)
    model = build_module(yaml.safe_load(cfg["model_yaml"]))
    model.outputs.atomic_shift.double()
    model.load_state_dict(sd, strict=True)
    model.eval()
    model._metadata = {
        "format_version": 2,
        "cutoff": cfg["cutoff"],
        "needs_coulomb": cfg["needs_coulomb"],
        "needs_dispersion": cfg["needs_dispersion"],
        "coulomb_mode": cfg["coulomb_mode"],
        "coulomb_sr_rc": cfg["coulomb_sr_rc"],
        "coulomb_sr_envelope": cfg["coulomb_sr_envelope"],
        "d3_params": cfg["d3_params"],
        "has_embedded_lr": cfg["has_embedded_lr"],
        "implemented_species": cfg["implemented_species"],
    }
    calc = AIMNet2Calculator(model=model, device="cpu")
    ref = {}
    for name, (numbers, coords, charge, mult) in SYSTEMS.items():
        _in = {
            "coord": torch.tensor(coords, dtype=torch.float32),
            "numbers": torch.tensor(numbers, dtype=torch.int64),
            "charge": torch.tensor(float(charge), dtype=torch.float32),
            "mult": torch.tensor(float(mult), dtype=torch.float32),
        }
        res = calc(_in, forces=True)
        ref[name] = {
            "energy": res["energy"].item(),
            "forces": res["forces"].detach().numpy(),
        }
    return ref


def run_maple_style(model, numbers, coords, charge, mult, cutoff, exec_optimized=False):
    """Run a (scripted or eager) model exactly the way MAPLE does and return E[eV], F[eV/A]."""
    data, coord = maple_build_data(coords, numbers, charge, mult, cutoff, float("inf"))
    ctx = torch.jit.optimized_execution(False)
    ctx.__enter__()
    try:
        out = model(data)
    finally:
        ctx.__exit__(None, None, None)
    energy = out["energy"].sum()
    grad = torch.autograd.grad(energy, data["coord"])[0]
    forces = -grad[: coord.shape[0]]
    return energy.item(), forces.detach().numpy()


def compare(tag, got, ref):
    ok = True
    for name in SYSTEMS:
        dE = abs(got[name][0] - ref[name]["energy"])
        dF = abs(got[name][1] - ref[name]["forces"]).max()
        status = "OK " if (dE <= E_TOL and dF <= F_TOL) else "FAIL"
        ok &= status == "OK "
        print(f"  [{status}] {tag} {name}: E={got[name][0]:.10f} eV  dE={dE:.3e} eV  dFmax={dF:.3e} eV/A")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK_DIR}/aimnet2nse.ts.pt")
    args = ap.parse_args()

    digest = sha256_of(SAFETENSORS)
    print(f"source safetensors sha256: {digest}")
    assert digest == EXPECTED_SHA256, "source weights hash mismatch!"

    model, cfg = build_full_model()
    print(f"built full model; cutoff={model.cutoff}, species={cfg['implemented_species']}")

    print("computing upstream v2 eager reference ...")
    ref = eager_reference(cfg)

    print("checking assembled eager model (MAPLE-style inputs) ...")
    got_eager = {}
    for name, (numbers, coords, charge, mult) in SYSTEMS.items():
        got_eager[name] = run_maple_style(model, numbers, coords, charge, mult, model.cutoff)
    compare("assembled-eager", got_eager, ref)

    print("scripting ...")
    _orig_forward = ConvSV.forward
    ConvSV.forward = _conv_sv_forward_pure_torch
    try:
        scripted = torch.jit.script(model)
    finally:
        ConvSV.forward = _orig_forward
    assert hasattr(scripted, "cutoff"), "scripted model lost 'cutoff' attribute"
    assert hasattr(scripted, "cutoff_lr"), "scripted model lost 'cutoff_lr' attribute"
    torch.jit.save(scripted, args.out)
    print(f"saved {args.out}")

    # archive self-containment check: no aimnet::/nvalchemiops custom op references
    import zipfile

    with zipfile.ZipFile(args.out) as zf:
        code_blobs = [zf.read(n).decode() for n in zf.namelist() if "/code/" in n and n.endswith(".py")]
    bad = [s for s in ("aimnet::", "nvalchemiops", "conv_sv_2d_sp") if any(s in b for b in code_blobs)]
    assert not bad, f"archive references runtime-registered ops: {bad}"
    print("archive self-containment: OK (no custom ops, no nvalchemiops)")

    loaded = torch.jit.load(args.out, map_location="cpu").eval()
    print(f"reloaded; cutoff={float(getattr(loaded, 'cutoff'))}, "
          f"cutoff_lr={float(getattr(loaded, 'cutoff_lr'))}")
    lrc = dict(loaded.named_modules()).get("outputs.lrcoulomb")
    assert lrc is not None, "embedded lrcoulomb module missing"
    print(f"embedded lrcoulomb: method={lrc.method}, rc={float(lrc.rc):.10f}, dsf_alpha={lrc.dsf_alpha}")

    print("checking scripted+reloaded model (MAPLE-style inputs) ...")
    got_ts = {}
    for name, (numbers, coords, charge, mult) in SYSTEMS.items():
        got_ts[name] = run_maple_style(loaded, numbers, coords, charge, mult, float(getattr(loaded, "cutoff")))
    ok = compare("torchscript", got_ts, ref)

    out_sha = sha256_of(args.out)
    print(f"artifact sha256: {out_sha}")
    if not ok:
        raise SystemExit("TOLERANCE CHECK FAILED - do not deploy this artifact")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
