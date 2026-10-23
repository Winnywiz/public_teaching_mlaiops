"""Tests for parsing the copy-ready cloud.env.example format."""
from __future__ import annotations

import os

from src.config import _load_env_file


def test_env_loader_strips_whitespace_separated_inline_comment(tmp_path, monkeypatch):
    key = "ITCS355_LAB5_ENV_PARSER_TEST"
    monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / "cloud.env"
    env_file.write_text(
        f"{key}=region-name # explanatory note\n"
        f"{key}_URL=https://example.test/path#fragment # note\n",
        encoding="utf-8",
    )

    _load_env_file(env_file)

    assert os.environ[key] == "region-name"
    assert os.environ[f"{key}_URL"] == "https://example.test/path#fragment"
    monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(f"{key}_URL", raising=False)
