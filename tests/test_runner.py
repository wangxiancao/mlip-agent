"""runner 测试：真实子进程（COMPUTE_VENV maple）跑 sp 并复现 B1 pinned 能量。"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time

import pytest

from app.compute.parsers import parse_out_file
from app.compute.runner import GPU_LOCK, run_compute


@pytest.mark.gpu
async def test_run_maple_sp_reproduces_b1(
    tmp_path, maple_bin, maple_examples_dir, references_dir
):
    """真实跑 MAPLE/examples/sp/sp.inp 复制件，能量复现 B1 pin 值 ±1e-6。"""
    shutil.copy(maple_examples_dir / "sp" / "sp.inp", tmp_path / "sp.inp")
    job = await run_compute(
        [str(maple_bin), "sp.inp", "sp.out"], cwd=tmp_path, timeout_s=600, gpu=True
    )
    assert job.ok, job.error
    assert job.exit_code == 0
    assert not job.timed_out
    assert (tmp_path / "stdout.log").is_file()
    assert (tmp_path / "stderr.log").is_file()
    assert job.out_path == str(tmp_path / "sp.out")

    pinned = json.loads((references_dir / "B1.json").read_text())["values"]
    result = parse_out_file(tmp_path / "sp.out", "sp")
    assert result.ok
    assert result.energy_hartree == pytest.approx(
        pinned["energy_hartree"], abs=1e-6
    )


@pytest.mark.gpu
async def test_env_whitelist_offline(tmp_path, maple_bin):
    """子进程 env 白名单生效：HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1、CUDA_VISIBLE_DEVICES=0。"""
    (tmp_path / "env.inp").write_text("placeholder\n")
    code = (
        "import os,sys;"
        "open(sys.argv[1],'w').write("
        "os.environ.get('HF_HUB_OFFLINE','')+'\\n'+"
        "os.environ.get('TRANSFORMERS_OFFLINE','')+'\\n'+"
        "os.environ.get('CUDA_VISIBLE_DEVICES','')+'\\n'+"
        "('MAPLE_MODEL_REVISION' in os.environ and 'leaked' or 'clean')+'\\n')"
    )
    job = await run_compute(
        [sys.executable, "-c", code, "probe.out"],
        cwd=tmp_path,
        timeout_s=60,
        gpu=False,
    )
    assert job.ok, job.error
    lines = (tmp_path / "probe.out").read_text().splitlines()
    assert lines[:3] == ["1", "1", "0"]


async def test_timeout_kills_process_group(tmp_path):
    started = time.monotonic()
    job = await run_compute(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        timeout_s=2,
        gpu=False,
    )
    assert job.timed_out
    assert not job.ok
    assert "超时" in (job.error or "")
    assert time.monotonic() - started < 30


async def test_error_marker_in_out_fails_job(tmp_path):
    """exit code 0 但 .out 含 ERROR: 行 → 判定失败。"""
    code = "import sys; open(sys.argv[1],'w').write('ERROR: something broke\\n')"
    job = await run_compute(
        [sys.executable, "-c", code, "fake.out"],
        cwd=tmp_path,
        timeout_s=30,
        gpu=False,
    )
    assert job.exit_code == 0
    assert not job.ok
    assert "ERROR:" in (job.error or "")


async def test_nonzero_exit_code_fails(tmp_path):
    job = await run_compute(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        cwd=tmp_path,
        timeout_s=30,
        gpu=False,
    )
    assert not job.ok
    assert job.exit_code == 3
    assert "exit code = 3" in (job.error or "")


async def test_gpu_lock_serializes(tmp_path, monkeypatch):
    """gpu=True 时两个并发任务不重叠执行（Semaphore(1)）。"""
    import app.compute.runner as runner_mod

    active = 0
    overlap_detected = False

    async def fake_run(argv, cwd, timeout_s, extra_env=None):
        nonlocal active, overlap_detected
        if active:
            overlap_detected = True
        active += 1
        await asyncio.sleep(0.05)
        active -= 1
        return runner_mod.CompletedJob(
            argv=argv,
            cwd=str(cwd),
            ok=True,
            exit_code=0,
            duration_s=0.0,
            stdout_path=str(cwd / "stdout.log"),
            stderr_path=str(cwd / "stderr.log"),
        )

    monkeypatch.setattr(runner_mod, "_run", fake_run)
    await asyncio.gather(
        run_compute(["a"], cwd=tmp_path, gpu=True),
        run_compute(["b"], cwd=tmp_path, gpu=True),
    )
    assert not overlap_detected
    assert GPU_LOCK._value == 1  # 锁已释放
