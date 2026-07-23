from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Settings


class LlamaClientError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


class LlamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._api_key: str | None = None

    @property
    def online_mode(self) -> bool:
        return self.settings.model_mode == "deepseek"

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def set_api_key(self, api_key: str) -> None:
        value = str(api_key or "").strip()
        if not value:
            raise ValueError("DeepSeek API Key 不能为空")
        if len(value) > 500:
            raise ValueError("DeepSeek API Key 格式不正确")
        self._api_key = value

    def clear_api_key(self) -> None:
        self._api_key = None

    async def validate_api_key(self, api_key: str) -> None:
        value = str(api_key or "").strip()
        if not value:
            raise LlamaClientError("DeepSeek API Key 不能为空")
        timeout = httpx.Timeout(connect=10, read=20, write=20, pool=5)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"{self.settings.deepseek_base_url}/models",
                    headers={"Authorization": f"Bearer {value}"},
                )
            if response.status_code >= 400:
                detail = response.text[:300]
                raise LlamaClientError(
                    f"DeepSeek API Key 验证失败（{response.status_code}）：{detail}"
                )
        except httpx.HTTPError as exc:
            raise LlamaClientError(f"无法连接 DeepSeek API：{exc}") from exc

    async def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        if self.online_mode:
            # DeepSeek does not expose llama.cpp's input_tokens endpoint. Chinese
            # prose is close to one token per character; add per-message overhead
            # and a small safety margin so context trimming remains conservative.
            characters = sum(len(message.get("content", "")) for message in messages)
            return max(1, math.ceil(characters * 1.08) + len(messages) * 8)
        payload = {"model": "local-model", "messages": messages}
        timeout = httpx.Timeout(connect=5, read=30, write=30, pool=5)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.settings.llama_base_url}/v1/chat/completions/input_tokens",
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            return int(data["input_tokens"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise LlamaClientError(f"无法统计上下文 token：{exc}") from exc

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        generation_settings: dict[str, Any],
        stop_event: asyncio.Event,
        extra_payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if self.online_mode and not self._api_key:
            raise LlamaClientError("请先在页面右上角填写 DeepSeek API Key")
        payload: dict[str, Any] = {
            "model": self.settings.deepseek_model if self.online_mode else "local-model",
            "messages": messages,
            "stream": True,
            "temperature": generation_settings["temperature"],
            "top_p": generation_settings["top_p"],
            "max_tokens": generation_settings["max_tokens"],
        }
        if self.online_mode:
            payload["thinking"] = {"type": "disabled"}
            payload["stream_options"] = {"include_usage": True}
        else:
            payload["repeat_penalty"] = generation_settings["repeat_penalty"]
            payload["seed"] = generation_settings["seed"]
        if extra_payload:
            payload.update(extra_payload)
        timeout = httpx.Timeout(connect=10, read=None, write=30, pool=10)
        base_url = (
            self.settings.deepseek_base_url
            if self.online_mode
            else self.settings.llama_base_url
        )
        headers = (
            {"Authorization": f"Bearer {self._api_key}"}
            if self.online_mode
            else None
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions"
                    if self.online_mode
                    else f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", errors="replace")[:1000]
                        raise LlamaClientError(
                            f"{'DeepSeek API' if self.online_mode else 'llama-server'}"
                            f" 返回 {response.status_code}：{detail}"
                        )
                    async for line in response.aiter_lines():
                        if stop_event.is_set():
                            raise GenerationCancelled("用户停止了生成")
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            yield {"type": "done"}
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise LlamaClientError(str(chunk["error"]))
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            reasoning = delta.get("reasoning_content")
                            if content:
                                yield {"type": "content_delta", "text": content}
                            if reasoning:
                                yield {"type": "reasoning_delta", "text": reasoning}
                            finish_reason = choices[0].get("finish_reason")
                            if finish_reason:
                                yield {"type": "finish_reason", "value": finish_reason}
                        if chunk.get("usage"):
                            yield {"type": "usage", "value": chunk["usage"]}
                        if chunk.get("timings"):
                            timings = chunk["timings"]
                            yield {
                                "type": "timings",
                                "value": {
                                    "prompt_tokens": timings.get("prompt_n"),
                                    "completion_tokens": timings.get("predicted_n"),
                                },
                            }
        except GenerationCancelled:
            raise
        except httpx.HTTPError as exc:
            target = "DeepSeek API" if self.online_mode else "本地模型服务"
            raise LlamaClientError(f"无法连接{target}：{exc}") from exc
