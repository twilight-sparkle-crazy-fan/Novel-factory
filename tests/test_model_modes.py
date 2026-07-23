from __future__ import annotations

import asyncio
from dataclasses import replace

from fastapi.testclient import TestClient

import backend.app as app_module
from backend.config import get_settings
from backend.database import Database
from backend.llama_client import LlamaClient
from backend.novel_repository import NovelRepository


def test_deepseek_client_keeps_key_only_in_memory_and_estimates_tokens() -> None:
    settings = replace(get_settings(), model_mode="deepseek")
    client = LlamaClient(settings)

    assert client.online_mode is True
    assert client.has_api_key is False
    client.set_api_key("sk-temporary")
    assert client.has_api_key is True
    assert asyncio.run(
        client.count_chat_tokens([{"role": "user", "content": "林舟走进车站。"}])
    ) >= len("林舟走进车站。")
    client.clear_api_key()
    assert client.has_api_key is False


def test_api_mode_runtime_requires_ephemeral_key(monkeypatch, tmp_path) -> None:
    database = Database(tmp_path / "api-mode.db")
    database.initialize()
    online_settings = replace(
        app_module.settings,
        model_mode="deepseek",
        deepseek_model="deepseek-v4-flash",
        api_context_size=81920,
    )
    monkeypatch.setattr(app_module, "settings", online_settings)
    monkeypatch.setattr(app_module.llama_client, "settings", online_settings)
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "novels", NovelRepository(database))
    app_module.llama_client.clear_api_key()

    async def validate(_api_key: str) -> None:
        return None

    monkeypatch.setattr(app_module.llama_client, "validate_api_key", validate)

    with TestClient(app_module.app) as client:
        before = client.get("/api/runtime").json()
        assert before["mode"] == "deepseek"
        assert before["status"] == "needs_key"

        oversized_secret = "sk-" + ("x" * 600)
        rejected = client.post(
            "/api/runtime/api-key", json={"api_key": oversized_secret}
        )
        assert rejected.status_code == 400
        assert oversized_secret not in rejected.text

        secret = "sk-one-launch-only"
        connected = client.post(
            "/api/runtime/api-key", json={"api_key": secret}
        )
        assert connected.status_code == 200
        assert connected.json()["status"] == "ready"
        assert connected.json()["api_key_present"] is True
        assert secret not in connected.text

        cleared = client.delete("/api/runtime/api-key").json()
        assert cleared["status"] == "needs_key"
        assert cleared["api_key_present"] is False
        assert database.get_app_setting("deepseek_api_key") is None
