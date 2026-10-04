"""TICKET-0012：SessionStore 必须把相对 data_dir 规范化为绝对路径。"""

from __future__ import annotations

from pathlib import Path

from app.state.store import SessionStore


def test_session_store_resolves_relative_data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = SessionStore(Path("data"))
    assert store.data_dir.is_absolute()
    assert store.sessions_dir == (tmp_path / "data" / "sessions").resolve()


def test_session_store_keeps_absolute_data_dir(tmp_path):
    store = SessionStore(tmp_path / "data")
    assert store.data_dir == (tmp_path / "data").resolve()
