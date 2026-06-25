from __future__ import annotations

import asyncio
import json
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

    async def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
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
        payload = {
            "model": "local-model",
            "messages": messages,
            "stream": True,
            "temperature": generation_settings["temperature"],
            "top_p": generation_settings["top_p"],
            "max_tokens": generation_settings["max_tokens"],
            "repeat_penalty": generation_settings["repeat_penalty"],
            "seed": generation_settings["seed"],
        }
        if extra_payload:
            payload.update(extra_payload)
        timeout = httpx.Timeout(connect=10, read=None, write=30, pool=10)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.settings.llama_base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", errors="replace")[:1000]
                        raise LlamaClientError(
                            f"llama-server 返回 {response.status_code}：{detail}"
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
            raise LlamaClientError(f"无法连接本地模型服务：{exc}") from exc
