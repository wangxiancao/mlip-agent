"""TICKET-0011：sanitize_proxy_env 对 ALL_PROXY socks 代理的处理。"""

from __future__ import annotations

import pytest

from app.agent.harness import sanitize_proxy_env


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch):
    for var in ("ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


def test_http_proxy_kept(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:7897/")
    sanitize_proxy_env()
    assert "ALL_PROXY" in __import__("os").environ


def test_socks_removed_without_socksio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:7897/")
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    sanitize_proxy_env()
    import os

    assert "ALL_PROXY" not in os.environ
    assert "all_proxy" not in os.environ


def test_socks_rewritten_with_socksio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: True if name == "socksio" else real_find_spec(name),
    )
    sanitize_proxy_env()
    import os

    assert os.environ["ALL_PROXY"] == "socks5h://127.0.0.1:7897/"


def test_unset_is_noop():
    sanitize_proxy_env()
    import os

    assert "ALL_PROXY" not in os.environ
