from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Settings


logger = logging.getLogger("llm4chat.model")


class LlamaClientError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


class GenerationTruncated(LlamaClientError):
    def __init__(self, finish_reason: str = "length") -> None:
        self.finish_reason = finish_reason
        super().__init__(
            "模型输出达到长度上限，正文可能停在半句。已保留本次输出，但不会把它标记为完整版本；请提高最大输出 token 后重新生成。"
        )


class RetryableLlamaClientError(LlamaClientError):
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

    async def _wait_before_retry(
        self,
        stop_event: asyncio.Event,
        delay_seconds: float,
    ) -> None:
        if stop_event.is_set():
            raise GenerationCancelled("用户停止了生成")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay_seconds)
        except TimeoutError:
            return
        raise GenerationCancelled("用户停止了生成")

    async def _stream_chat_attempt(
        self,
        *,
        request_id: str,
        attempt: int,
        max_attempts: int,
        started: float,
        payload: dict[str, Any],
        stop_event: asyncio.Event,
        base_url: str,
        headers: dict[str, str] | None,
        timeout: httpx.Timeout,
    ) -> AsyncIterator[dict[str, Any]]:
        terminal_finish_reason: str | None = None
        logger.info(
            "model_stream_started request_id=%s attempt=%s/%s mode=%s model=%s messages=%s input_chars=%s max_tokens=%s temperature=%s top_p=%s",
            request_id,
            attempt,
            max_attempts,
            "deepseek" if self.online_mode else "local",
            payload["model"],
            len(payload["messages"]),
            sum(len(message.get("content", "")) for message in payload["messages"]),
            payload["max_tokens"],
            payload["temperature"],
            payload["top_p"],
        )
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
                    message = (
                        f"{'DeepSeek API' if self.online_mode else 'llama-server'}"
                        f" 返回 {response.status_code}：{detail}"
                    )
                    if self.online_mode and response.status_code in {
                        408, 425, 429, 500, 502, 503, 504,
                    }:
                        raise RetryableLlamaClientError(message)
                    raise LlamaClientError(message)
                async for line in response.aiter_lines():
                    if stop_event.is_set():
                        raise GenerationCancelled("用户停止了生成")
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        if terminal_finish_reason == "length":
                            logger.warning(
                                "model_stream_truncated request_id=%s attempt=%s/%s duration_ms=%s max_tokens=%s",
                                request_id,
                                attempt,
                                max_attempts,
                                int((time.monotonic() - started) * 1000),
                                payload["max_tokens"],
                            )
                            raise GenerationTruncated(terminal_finish_reason)
                        if terminal_finish_reason == "insufficient_system_resource":
                            raise RetryableLlamaClientError(
                                "DeepSeek 暂时没有足够的服务资源完成请求"
                            )
                        if terminal_finish_reason not in {None, "stop"}:
                            raise LlamaClientError(
                                f"模型未正常完成生成（finish_reason={terminal_finish_reason}）"
                            )
                        logger.info(
                            "model_stream_completed request_id=%s attempt=%s/%s duration_ms=%s",
                            request_id,
                            attempt,
                            max_attempts,
                            int((time.monotonic() - started) * 1000),
                        )
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
                            terminal_finish_reason = str(finish_reason)
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
                if terminal_finish_reason == "length":
                    raise GenerationTruncated(terminal_finish_reason)
                if terminal_finish_reason == "insufficient_system_resource":
                    raise RetryableLlamaClientError(
                        "DeepSeek 暂时没有足够的服务资源完成请求"
                    )
                if terminal_finish_reason == "stop":
                    logger.info(
                        "model_stream_completed_without_done_marker request_id=%s attempt=%s/%s duration_ms=%s",
                        request_id,
                        attempt,
                        max_attempts,
                        int((time.monotonic() - started) * 1000),
                    )
                    yield {"type": "done"}
                    return
                message = "模型流式连接在完成标记前结束"
                if self.online_mode:
                    raise RetryableLlamaClientError(message)
                raise LlamaClientError(
                    f"{message}；已保留收到的内容，但不会把它标记为完整版本。"
                )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        generation_settings: dict[str, Any],
        stop_event: asyncio.Event,
        extra_payload: dict[str, Any] | None = None,
        *,
        buffer_for_retry: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        if self.online_mode and not self._api_key:
            raise LlamaClientError("请先在页面右上角填写 DeepSeek API Key")
        request_id = secrets.token_hex(6)
        started = time.monotonic()
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
        timeout = httpx.Timeout(
            connect=10,
            read=(self.settings.deepseek_read_timeout_seconds if self.online_mode else None),
            write=30,
            pool=10,
        )
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
        max_attempts = 1 + (
            self.settings.deepseek_max_retries if self.online_mode else 0
        )
        try:
            for attempt_index in range(max_attempts):
                emitted_model_output = False
                attempt_events: list[dict[str, Any]] = []
                try:
                    async for event in self._stream_chat_attempt(
                        request_id=request_id,
                        attempt=attempt_index + 1,
                        max_attempts=max_attempts,
                        started=started,
                        payload=payload,
                        stop_event=stop_event,
                        base_url=base_url,
                        headers=headers,
                        timeout=timeout,
                    ):
                        if event["type"] in {"content_delta", "reasoning_delta"}:
                            emitted_model_output = True
                        if buffer_for_retry:
                            attempt_events.append(event)
                        else:
                            yield event
                    if buffer_for_retry:
                        for event in attempt_events:
                            yield event
                    return
                except (RetryableLlamaClientError, httpx.HTTPError) as exc:
                    can_retry = (
                        self.online_mode
                        and attempt_index + 1 < max_attempts
                        and (buffer_for_retry or not emitted_model_output)
                    )
                    if not can_retry:
                        attempts_used = attempt_index + 1
                        skip_reason = (
                            "partial_output_already_emitted"
                            if emitted_model_output and not buffer_for_retry
                            else "attempts_exhausted_or_retry_disabled"
                        )
                        logger.warning(
                            "model_stream_retry_skipped request_id=%s attempt=%s/%s reason=%s buffered=%s error_type=%s",
                            request_id,
                            attempts_used,
                            max_attempts,
                            skip_reason,
                            buffer_for_retry,
                            type(exc).__name__,
                        )
                        if isinstance(exc, httpx.HTTPError):
                            target = "DeepSeek API" if self.online_mode else "本地模型服务"
                            message = f"无法连接{target}：{exc}"
                        else:
                            message = str(exc)
                        if attempts_used > 1:
                            message += f"（已尝试 {attempts_used} 次）"
                        raise LlamaClientError(message) from exc
                    delay_seconds = self.settings.deepseek_retry_base_seconds * (
                        2 ** attempt_index
                    )
                    next_attempt = attempt_index + 2
                    logger.warning(
                        "model_stream_retry_scheduled request_id=%s failed_attempt=%s/%s next_attempt=%s delay_seconds=%.1f buffered=%s error_type=%s error=%s",
                        request_id,
                        attempt_index + 1,
                        max_attempts,
                        next_attempt,
                        delay_seconds,
                        buffer_for_retry,
                        type(exc).__name__,
                        str(exc)[:500],
                    )
                    yield {
                        "type": "retry",
                        "attempt": next_attempt,
                        "max_attempts": max_attempts,
                        "delay_seconds": delay_seconds,
                        "message": (
                            f"DeepSeek 连接中断，{delay_seconds:g} 秒后自动重试"
                            f"（第 {next_attempt}/{max_attempts} 次）"
                        ),
                    }
                    await self._wait_before_retry(stop_event, delay_seconds)
        except GenerationCancelled:
            logger.warning(
                "model_stream_stopped request_id=%s duration_ms=%s",
                request_id,
                int((time.monotonic() - started) * 1000),
            )
            raise
        except asyncio.CancelledError:
            logger.warning(
                "model_stream_cancelled request_id=%s duration_ms=%s",
                request_id,
                int((time.monotonic() - started) * 1000),
            )
            raise
        except LlamaClientError:
            logger.exception(
                "model_stream_error request_id=%s mode=%s duration_ms=%s",
                request_id,
                "deepseek" if self.online_mode else "local",
                int((time.monotonic() - started) * 1000),
            )
            raise
