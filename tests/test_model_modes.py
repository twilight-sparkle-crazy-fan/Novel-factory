from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import backend.app as app_module
import backend.llama_client as llama_client_module
from backend.config import get_settings
from backend.database import Database
from backend.llama_client import GenerationTruncated, LlamaClient, LlamaClientError
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


def test_deepseek_generation_sends_supported_sampling_parameters(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"正文"}}]}'
            yield "data: [DONE]"

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def stream(self, _method, _url, **kwargs):
            captured.update(kwargs["json"])
            return FakeStream()

    monkeypatch.setattr(llama_client_module.httpx, "AsyncClient", FakeAsyncClient)
    client = LlamaClient(replace(get_settings(), model_mode="deepseek"))
    client.set_api_key("sk-temporary")

    async def collect() -> list[dict]:
        return [
            event
            async for event in client.stream_chat(
                [{"role": "user", "content": "续写"}],
                {
                    "temperature": 0.7,
                    "top_p": 0.85,
                    "max_tokens": 2400,
                    "repeat_penalty": 1.2,
                    "seed": 42,
                },
                asyncio.Event(),
            )
        ]

    events = asyncio.run(collect())
    assert events[0] == {"type": "content_delta", "text": "正文"}
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.85
    assert captured["max_tokens"] == 2400
    assert captured["thinking"] == {"type": "disabled"}
    assert "repeat_penalty" not in captured
    assert "seed" not in captured


def test_deepseek_length_finish_reason_is_reported_as_truncation(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"半句正文"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}'
            yield "data: [DONE]"

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def stream(self, _method, _url, **_kwargs):
            return FakeStream()

    monkeypatch.setattr(llama_client_module.httpx, "AsyncClient", FakeAsyncClient)
    client = LlamaClient(replace(get_settings(), model_mode="deepseek"))
    client.set_api_key("sk-temporary")
    received: list[dict] = []

    async def collect() -> None:
        async for event in client.stream_chat(
            [{"role": "user", "content": "续写"}],
            {
                "temperature": 0.9,
                "top_p": 0.95,
                "max_tokens": 2400,
                "repeat_penalty": 1.08,
                "seed": 1,
            },
            asyncio.Event(),
        ):
            received.append(event)

    with pytest.raises(GenerationTruncated):
        asyncio.run(collect())
    assert received == [
        {"type": "content_delta", "text": "半句正文"},
        {"type": "finish_reason", "value": "length"},
    ]


def test_deepseek_buffered_call_discards_partial_attempt_before_retry(monkeypatch) -> None:
    attempts = 0

    class FakeResponse:
        status_code = 200

        def __init__(self, attempt: int):
            self.attempt = attempt

        async def aiter_lines(self):
            if self.attempt == 1:
                yield 'data: {"choices":[{"delta":{"content":"残缺 JSON"}}]}'
                raise llama_client_module.httpx.RemoteProtocolError("peer disconnected")
            yield 'data: {"choices":[{"delta":{"content":"完整 JSON"}}]}'
            yield "data: [DONE]"

    class FakeStream:
        def __init__(self, attempt: int):
            self.attempt = attempt

        async def __aenter__(self):
            return FakeResponse(self.attempt)

        async def __aexit__(self, *_args):
            return False

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def stream(self, _method, _url, **_kwargs):
            nonlocal attempts
            attempts += 1
            return FakeStream(attempts)

    monkeypatch.setattr(llama_client_module.httpx, "AsyncClient", FakeAsyncClient)
    client = LlamaClient(
        replace(
            get_settings(),
            model_mode="deepseek",
            deepseek_max_retries=2,
            deepseek_retry_base_seconds=0.001,
        )
    )
    client.set_api_key("sk-temporary")

    async def collect() -> list[dict]:
        return [
            event
            async for event in client.stream_chat(
                [{"role": "user", "content": "检查场景"}],
                {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 700,
                    "repeat_penalty": 1.05,
                    "seed": 1,
                },
                asyncio.Event(),
                buffer_for_retry=True,
            )
        ]

    events = asyncio.run(collect())
    assert attempts == 2
    assert events[0]["type"] == "retry"
    assert events[0]["attempt"] == 2
    assert {"type": "content_delta", "text": "完整 JSON"} in events
    assert all(event.get("text") != "残缺 JSON" for event in events)


def test_deepseek_visible_partial_output_is_not_replayed(monkeypatch) -> None:
    attempts = 0

    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"已经显示的正文"}}]}'
            raise llama_client_module.httpx.RemoteProtocolError("peer disconnected")

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *_args):
            return False

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def stream(self, _method, _url, **_kwargs):
            nonlocal attempts
            attempts += 1
            return FakeStream()

    monkeypatch.setattr(llama_client_module.httpx, "AsyncClient", FakeAsyncClient)
    client = LlamaClient(
        replace(
            get_settings(),
            model_mode="deepseek",
            deepseek_max_retries=2,
            deepseek_retry_base_seconds=0.001,
        )
    )
    client.set_api_key("sk-temporary")
    received: list[dict] = []

    async def collect() -> None:
        async for event in client.stream_chat(
            [{"role": "user", "content": "续写"}],
            {
                "temperature": 0.9,
                "top_p": 0.95,
                "max_tokens": 2400,
                "repeat_penalty": 1.08,
                "seed": 1,
            },
            asyncio.Event(),
        ):
            received.append(event)

    with pytest.raises(LlamaClientError):
        asyncio.run(collect())
    assert attempts == 1
    assert received == [{"type": "content_delta", "text": "已经显示的正文"}]


def test_generation_max_tokens_are_clamped_by_active_mode(monkeypatch) -> None:
    local_settings = replace(app_module.settings, model_mode="local")
    monkeypatch.setattr(app_module, "settings", local_settings)
    conversation = {"generation_settings": {"max_tokens": 200_000}}
    assert app_module.resolve_generation_settings(conversation, None)["max_tokens"] == 16_384

    api_settings = replace(
        app_module.settings,
        model_mode="deepseek",
        api_max_output_tokens=384_000,
    )
    monkeypatch.setattr(app_module, "settings", api_settings)
    assert app_module.resolve_generation_settings(conversation, None)["max_tokens"] == 200_000


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
        assert before["max_output_tokens"] == online_settings.api_max_output_tokens
        assert before["retry_policy"]["max_retries"] == online_settings.deepseek_max_retries

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
