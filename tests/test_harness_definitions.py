"""TICKET-0014：build_harness 的 definition_id 必须跨构建确定性（重启续跑前提）。"""

from __future__ import annotations

from app.agent.harness import build_harness
from app.agent.subagents import SUBAGENT_SPECS
from app.config import Settings


def test_definition_ids_deterministic_across_builds():
    h1 = build_harness(Settings(), [])
    h2 = build_harness(Settings(), [])
    d1, d2 = h1.definition, h2.definition
    assert d1.definition_id == d2.definition_id == "mlip.orchestrator"
    ids1 = {s.name: s.agent.definition_id for s in d1.subagents}
    ids2 = {s.name: s.agent.definition_id for s in d2.subagents}
    assert ids1 == ids2
    assert ids1 == {
        spec["name"]: f"mlip.subagent.{spec['name']}" for spec in SUBAGENT_SPECS
    }
