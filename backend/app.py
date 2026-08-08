from __future__ import annotations

import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_GENERATION_SETTINGS, LOCAL_MAX_OUTPUT_TOKENS, get_settings
from .context_builder import ContextResult, build_messages
from .database import Database, new_id
from .analysis_service import NovelAnalysisService
from .llama_client import GenerationCancelled, GenerationTruncated, LlamaClient, LlamaClientError
from .llama_process import LlamaProcessError, LlamaProcessManager
from .material_system import (
    MATERIAL_SCHEMA_VERSION,
    PACKAGE_FORMAT_VERSION,
    MaterialPackageError,
    MaterialPackageService,
)
from .novel_repository import NovelRepository, format_chapter_summary
from .schemas import (
    BranchRequest,
    ChapterSelectionApplyRequest,
    ChapterSelectionRewriteRequest,
    ChapterUpdate,
    CharacterCreateRequest,
    CharacterExtractRequest,
    CharacterMergeRequest,
    CharacterUpdate,
    ConversationCreate,
    ConversationUpdate,
    ContextCountRequest,
    DocumentCharacterEventUpdate,
    DocumentCreateRequest,
    DocumentUpdate,
    GenerateRequest,
    MaterialAuxiliaryRecordCreate,
    MaterialAuxiliaryRecordUpdate,
    MaterialCharacterAliasCreate,
    MaterialCharacterEntityCreate,
    MaterialCharacterEntityUpdate,
    MaterialCharacterEventCreate,
    MaterialCharacterEventUpdate,
    MaterialCharacterFactCreate,
    MaterialCharacterFactUpdate,
    MaterialCharacterMergeRequest,
    MaterialCharacterProfileCreate,
    MaterialCharacterProfileUpdate,
    MaterialCharacterSplitRequest,
    MaterialPromptBudgetUpdate,
    MaterialReviewBatchRequest,
    MaterialRelationshipCreate,
    MaterialRelationshipEventCreate,
    MaterialRelationshipEventUpdate,
    MaterialRelationshipUpdate,
    MaterialSemanticObservationUpdate,
    MaterialTimelineEventCreate,
    MaterialTimelineEventUpdate,
    MaterialTimelineNodeCreate,
    MaterialTimelineNodeUpdate,
    OutlineCandidateEditRequest,
    OutlineCandidateSaveRequest,
    OutlineGenerateRequest,
    OutlineUpdateRequest,
    ProjectAppendRequest,
    ProjectUpdate,
    RegenerateRequest,
    RuntimeContextRequest,
    SceneFragmentRegenerateRequest,
    SceneWorkflowAcceptRequest,
    SceneWorkflowPolishRequest,
    SceneWorkflowRequest,
    SelectionRequest,
    SummarizeRequest,
    StoryFactUpdate,
)
from .text_import import decode_text, normalize_text


settings = get_settings()


def configure_logging() -> None:
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    log_path = settings.project_root / "data" / "novel-factory.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.WARNING)
    resolved_path = str(log_path.resolve())
    if not any(
        isinstance(handler, RotatingFileHandler)
        and str(getattr(handler, "baseFilename", "")) == resolved_path
        for handler in root_logger.handlers
    ):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max(1024, settings.app_log_max_bytes),
            backupCount=max(0, settings.app_log_backup_count),
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)


configure_logging()
logger = logging.getLogger("llm4chat")
logging.getLogger("httpx").setLevel(logging.WARNING)

database = Database(settings.database_path)
llama_process = LlamaProcessManager(settings)
llama_client = LlamaClient(settings)
novels = NovelRepository(database)
analysis_service = NovelAnalysisService(llama_client)
MAX_MATERIAL_PACKAGE_BYTES = 200 * 1024 * 1024
POLISH_CONTEXT_TOKEN_LIMIT = 20_000
POLISH_OUTPUT_RESERVE_TOKENS = 256
POLISH_MIN_OUTPUT_TOKENS = 6000


class GenerationCoordinator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.active_candidate_id: str | None = None
        self.stop_event: asyncio.Event | None = None

    async def begin(self, candidate_id: str) -> asyncio.Event:
        if self._lock.locked():
            raise RuntimeError("generation_in_progress")
        await self._lock.acquire()
        self.active_candidate_id = candidate_id
        self.stop_event = asyncio.Event()
        return self.stop_event

    def stop(self) -> bool:
        if self.stop_event is None:
            return False
        self.stop_event.set()
        return True

    def finish(self) -> None:
        self.active_candidate_id = None
        self.stop_event = None
        if self._lock.locked():
            self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()


generation = GenerationCoordinator()


class AgentActivity:
    def __init__(self) -> None:
        self.revision = 0
        self._sessions: dict[str, dict[str, Any]] = {}

    def start(self, label: str) -> dict[str, Any]:
        token = new_id()
        self._sessions[token] = {
            "label": str(label or "外部 Agent 操作").strip()[:120],
            "started": time.monotonic(),
        }
        return {"token": token, **self.snapshot()}

    def finish(self, token: str) -> dict[str, Any]:
        if self._sessions.pop(token, None) is not None:
            self.revision += 1
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        expired = [
            token
            for token, item in self._sessions.items()
            if now - float(item["started"]) > 300
        ]
        for token in expired:
            self._sessions.pop(token, None)
        labels = [str(item["label"]) for item in self._sessions.values()]
        return {
            "revision": self.revision,
            "active": bool(labels),
            "active_count": len(labels),
            "labels": labels,
        }


agent_activity = AgentActivity()


def error_response(status_code: int, code: str, message: str, detail: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status_code, content=body)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def analysis_progress_events(
    task: asyncio.Task[Any],
    queue: asyncio.Queue[tuple[str, int, int]],
    *,
    phase: str,
    context: dict[str, Any] | None = None,
):
    started = time.monotonic()
    extra = context or {}
    queue_task: asyncio.Task[tuple[str, int, int]] | None = None
    try:
        while not task.done() or not queue.empty():
            if not queue.empty():
                stage, index, total = queue.get_nowait()
                yield sse(
                    "analysis_progress",
                    {
                        "phase": phase,
                        "stage": stage,
                        "index": index,
                        "total": total,
                        "elapsed_seconds": int(time.monotonic() - started),
                        **extra,
                    },
                )
                continue
            queue_task = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {task, queue_task}, timeout=4.0, return_when=asyncio.FIRST_COMPLETED
            )
            if queue_task in done:
                stage, index, total = queue_task.result()
                yield sse(
                    "analysis_progress",
                    {
                        "phase": phase,
                        "stage": stage,
                        "index": index,
                        "total": total,
                        "elapsed_seconds": int(time.monotonic() - started),
                        **extra,
                    },
                )
                queue_task = None
            else:
                queue_task.cancel()
                try:
                    await queue_task
                except asyncio.CancelledError:
                    pass
                queue_task = None
                if not task.done():
                    yield sse(
                        "analysis_heartbeat",
                        {
                            "phase": phase,
                            "elapsed_seconds": int(time.monotonic() - started),
                            **extra,
                        },
                    )
    except asyncio.CancelledError:
        if queue_task is not None and not queue_task.done():
            queue_task.cancel()
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        raise


def resolve_generation_settings(
    conversation: dict[str, Any], override: Any | None
) -> dict[str, Any]:
    merged = {
        **DEFAULT_GENERATION_SETTINGS,
        **conversation.get("generation_settings", {}),
    }
    if override is not None:
        merged.update(override.model_dump())
    merged.pop("min_completion_tokens", None)
    max_output_tokens = (
        settings.api_max_output_tokens
        if settings.model_mode == "deepseek"
        else LOCAL_MAX_OUTPUT_TOKENS
    )
    merged["max_tokens"] = min(
        max_output_tokens,
        max(16, int(merged.get("max_tokens") or DEFAULT_GENERATION_SETTINGS["max_tokens"])),
    )
    if merged.get("seed") is None:
        merged["seed"] = secrets.randbelow(2_147_483_647)
    return merged


def material_service() -> MaterialPackageService:
    return MaterialPackageService(database)


def material_system_disabled_response() -> JSONResponse | None:
    if settings.experimental_material_system:
        return None
    return error_response(
        404,
        "EXPERIMENTAL_MATERIAL_SYSTEM_DISABLED",
        "实验资料系统默认关闭，请设置 EXPERIMENTAL_MATERIAL_SYSTEM=true 后重启。",
    )


def prompt_assets_for_conversation(
    conversation_id: str,
    *,
    include_outline: bool = True,
    query_text: str = "",
    material_budget_tokens: int = 8000,
) -> dict[str, str]:
    _ = material_budget_tokens
    return novels.get_prompt_context(
        conversation_id,
        include_outline=include_outline,
        query_text=query_text,
    )


async def read_material_package(request: Request) -> bytes | JSONResponse:
    data = await request.body()
    if not data:
        return error_response(400, "EMPTY_PACKAGE", "分析包为空")
    if len(data) > MAX_MATERIAL_PACKAGE_BYTES:
        return error_response(413, "PACKAGE_TOO_LARGE", "分析包不能超过 200 MB")
    return data


async def ensure_model_ready() -> None:
    if settings.model_mode == "deepseek":
        if not llama_client.has_api_key:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "DEEPSEEK_API_KEY_REQUIRED",
                    "message": "请先在页面右上角填写 DeepSeek API Key",
                },
            )
        return
    if not await llama_process.is_healthy():
        info = await llama_process.runtime_info(check_health=False)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "LLAMA_SERVER_UNAVAILABLE",
                "message": info["message"],
            },
        )


def active_context_size() -> int:
    if settings.model_mode == "deepseek":
        return settings.api_context_size
    return llama_process.context_size


def active_max_output_tokens() -> int:
    if settings.model_mode == "deepseek":
        return settings.api_max_output_tokens
    return LOCAL_MAX_OUTPUT_TOKENS


def polish_context_token_limit() -> int:
    if settings.model_mode == "deepseek":
        return active_context_size()
    return min(POLISH_CONTEXT_TOKEN_LIMIT, active_context_size())


async def model_runtime_info(*, check_health: bool = True) -> dict[str, Any]:
    if settings.model_mode == "deepseek":
        ready = llama_client.has_api_key
        return {
            "mode": "deepseek",
            "status": "ready" if ready else "needs_key",
            "message": (
                f"DeepSeek {settings.deepseek_model} 已就绪"
                if ready
                else "请输入本次启动使用的 DeepSeek API Key"
            ),
            "healthy": ready,
            "started_by_app": False,
            "model_name": settings.deepseek_model,
            "model_path": "",
            "llama_url": "",
            "api_base_url": settings.deepseek_base_url,
            "context_size": settings.api_context_size,
            "max_output_tokens": settings.api_max_output_tokens,
            "retry_policy": {
                "max_retries": settings.deepseek_max_retries,
                "base_seconds": settings.deepseek_retry_base_seconds,
                "read_timeout_seconds": settings.deepseek_read_timeout_seconds,
            },
            "cache_type_k": "",
            "cache_type_v": "",
            "reasoning": "off",
            "api_key_present": ready,
        }
    info = await llama_process.runtime_info(check_health=check_health)
    info["mode"] = "local"
    info["api_key_present"] = False
    info["max_output_tokens"] = LOCAL_MAX_OUTPUT_TOKENS
    return info


async def count_or_estimate(messages: list[dict[str, str]]) -> int:
    try:
        return await llama_client.count_chat_tokens(messages)
    except LlamaClientError:
        # The token-count endpoint is unavailable while the model is loading and in
        # mocked tests. Chinese prose is close enough to one token per character for
        # a conservative fallback, but all live generations use the real endpoint.
        return max(1, sum(len(message.get("content", "")) for message in messages))


async def build_fitted_context(
    *,
    conversation_id: str,
    system_prompt: str,
    pinned_context: str,
    style_guide: str = "",
    style_lexicon: str = "",
    history: list[dict[str, str]],
    current_user_content: str,
    max_output_tokens: int,
    include_outline: bool = True,
) -> ContextResult:
    original_pair_count = len(history) // 2
    working_history = list(history)
    context_size = active_context_size()
    budget = max(1024, context_size - max_output_tokens - 384)
    project_context = prompt_assets_for_conversation(
        conversation_id,
        include_outline=include_outline,
        query_text=current_user_content,
        material_budget_tokens=max(1024, min(12000, budget - 512)),
    )
    while True:
        result = build_messages(
            system_prompt=system_prompt,
            pinned_context=pinned_context,
            style_guide=style_guide,
            style_lexicon=style_lexicon,
            history=working_history,
            current_user_content=current_user_content,
            n_ctx=context_size,
            project_context=project_context,
            trim_by_characters=False,
        )
        token_count = await count_or_estimate(result.messages)
        if token_count <= budget:
            result.trimmed_exchange_count = original_pair_count - len(working_history) // 2
            result.prompt_tokens = token_count
            return result
        if len(working_history) >= 2:
            working_history = working_history[2:]
            continue
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FIXED_CONTEXT_TOO_LONG",
                "message": "固定创作资料、人物卡或场景卡超过了当前上下文预算",
            },
        )


async def context_for_exchange(
    exchange_id: str, max_output_tokens: int, *, include_outline: bool = True
) -> ContextResult:
    conversation, history, current_user_content = database.get_context_source(exchange_id)
    return await build_fitted_context(
        conversation_id=conversation["id"],
        system_prompt=conversation["system_prompt"],
        pinned_context=conversation["pinned_context"],
        style_guide=conversation.get("style_guide", ""),
        style_lexicon=conversation.get("style_lexicon", ""),
        history=history,
        current_user_content=current_user_content,
        max_output_tokens=max_output_tokens,
        include_outline=include_outline,
    )


def stream_candidate(
    *,
    exchange: dict[str, Any],
    candidate: dict[str, Any],
    context: ContextResult,
    generation_settings: dict[str, Any],
    stop_event: asyncio.Event,
):
    async def event_stream():
        content = ""
        reasoning = ""
        prompt_tokens: int | None = None
        completion_tokens = 0
        finish_reason: str | None = None
        started = time.monotonic()
        last_flush = started
        try:
            yield sse(
                "candidate_created",
                {
                    "exchange_id": exchange["id"],
                    "user_content": exchange["user_content"],
                    "candidate": candidate,
                    "trimmed_exchange_count": context.trimmed_exchange_count,
                    "prompt_tokens": context.prompt_tokens,
                    "context_size": active_context_size(),
                },
            )

            active_generation_settings = dict(generation_settings)
            active_generation_settings.pop("min_completion_tokens", None)
            async for event in llama_client.stream_chat(
                context.messages, active_generation_settings, stop_event
            ):
                event_type = event["type"]
                if event_type == "content_delta":
                    content += event["text"]
                    yield sse("content_delta", {"text": event["text"]})
                elif event_type == "reasoning_delta":
                    reasoning += event["text"]
                    yield sse("reasoning_delta", {"text": event["text"]})
                elif event_type == "usage":
                    usage = event["value"]
                    if prompt_tokens is None:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens) or 0)
                elif event_type == "timings":
                    usage = event["value"]
                    if prompt_tokens is None:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens) or 0)
                elif event_type == "finish_reason":
                    finish_reason = event["value"]
                elif event_type == "retry":
                    yield sse("model_retry", event)
                now = time.monotonic()
                if now - last_flush >= 0.8:
                    database.update_candidate_draft(candidate["id"], content, reasoning)
                    last_flush = now
            database.update_candidate_draft(candidate["id"], content, reasoning)

            duration_ms = int((time.monotonic() - started) * 1000)
            updated_exchange = database.finalize_candidate(
                candidate["id"],
                status="completed",
                content=content,
                reasoning=reasoning,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
            )
            yield sse(
                "done",
                {
                    "candidate_id": candidate["id"],
                    "exchange": updated_exchange,
                    "finish_reason": finish_reason,
                    "duration_ms": duration_ms,
                },
            )
        except GenerationCancelled:
            duration_ms = int((time.monotonic() - started) * 1000)
            updated_exchange = database.finalize_candidate(
                candidate["id"],
                status="cancelled",
                content=content,
                reasoning=reasoning,
                duration_ms=duration_ms,
                error_message="用户停止了生成",
            )
            yield sse("cancelled", {"candidate_id": candidate["id"], "exchange": updated_exchange})
        except asyncio.CancelledError:
            database.finalize_candidate(
                candidate["id"],
                status="cancelled",
                content=content,
                reasoning=reasoning,
                duration_ms=int((time.monotonic() - started) * 1000),
                error_message="浏览器连接已断开",
            )
            raise
        except GenerationTruncated as exc:
            logger.warning("generation truncated for candidate %s", candidate["id"])
            updated_exchange = database.finalize_candidate(
                candidate["id"],
                status="failed",
                content=content,
                reasoning=reasoning,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=int((time.monotonic() - started) * 1000),
                error_message=str(exc),
            )
            yield sse(
                "error",
                {
                    "code": "GENERATION_TRUNCATED",
                    "message": "输出达到长度上限，未保存为完整版本",
                    "detail": str(exc),
                    "exchange": updated_exchange,
                },
            )
        except (LlamaClientError, Exception) as exc:
            logger.exception("generation failed for candidate %s", candidate["id"])
            updated_exchange = database.finalize_candidate(
                candidate["id"],
                status="failed",
                content=content,
                reasoning=reasoning,
                duration_ms=int((time.monotonic() - started) * 1000),
                error_message=str(exc)[:1000],
            )
            yield sse(
                "error",
                {
                    "code": "GENERATION_FAILED",
                    "message": "本次生成失败，可以重新尝试",
                    "detail": str(exc)[:500],
                    "exchange": updated_exchange,
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SceneCardFormatError(ValueError):
    pass


SCENE_CARD_REQUIRED_FIELDS = ("id", "title", "purpose", "entry", "beats", "exit", "constraints")


def selected_outline_content(outline: dict[str, Any] | None) -> str:
    if not outline or not outline.get("selected_candidate_id"):
        return ""
    for candidate in outline.get("candidates", []):
        if candidate.get("id") == outline["selected_candidate_id"]:
            return str(candidate.get("edited_content") or candidate.get("content") or "").strip()
    return ""


def parse_outline_json(outline_text: str) -> dict[str, Any]:
    text = outline_text.strip()
    if not text:
        raise SceneCardFormatError("场景卡不能为空")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SceneCardFormatError(f"场景卡必须是合法 JSON：{exc.msg}") from exc
    if not isinstance(data, dict):
        raise SceneCardFormatError("场景卡 JSON 根节点必须是对象")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise SceneCardFormatError("场景卡 JSON 必须包含非空 scenes 数组")
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise SceneCardFormatError(f"scenes[{index}] 必须是对象")
        for field in SCENE_CARD_REQUIRED_FIELDS:
            if field not in scene:
                raise SceneCardFormatError(f"scenes[{index}] 必须包含 {field}")
        if not str(scene.get("id") or "").strip():
            raise SceneCardFormatError(f"scenes[{index}].id 不能为空")
        if not str(scene.get("title") or "").strip():
            raise SceneCardFormatError(f"scenes[{index}].title 不能为空")
        if not isinstance(scene.get("beats"), list) or not scene["beats"]:
            raise SceneCardFormatError(f"scenes[{index}].beats 必须是非空数组")
        if not isinstance(scene.get("constraints"), list):
            raise SceneCardFormatError(f"scenes[{index}].constraints 必须是数组")
    return data


def validate_outline_json(outline_text: str) -> None:
    parse_outline_json(outline_text)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def normalize_scene_card(scene: dict[str, Any]) -> dict[str, Any]:
    parsed_card: dict[str, Any] = {}
    card_text = str(scene.get("card") or "").strip()
    if card_text:
        try:
            loaded = json.loads(card_text)
            if isinstance(loaded, dict):
                parsed_card = loaded
        except json.JSONDecodeError:
            parsed_card = {}
    merged = {**parsed_card, **{key: value for key, value in scene.items() if value not in (None, "")}}
    label = str(merged.get("id") or merged.get("label") or "").strip() or "S?"
    title = str(merged.get("title") or "").strip()
    normalized = {
        "label": label,
        "title": title,
        "purpose": str(merged.get("purpose") or merged.get("goal") or merged.get("narrative_function") or "").strip(),
        "entry": str(merged.get("entry") or merged.get("time") or merged.get("location") or "").strip(),
        "beats": _string_list(merged.get("beats") or merged.get("actions") or merged.get("completion_checks")),
        "exit": str(merged.get("exit") or merged.get("chapter_ending") or "").strip(),
        "constraints": _string_list(merged.get("constraints") or merged.get("continuity_notes")),
        "card": card_text or json.dumps({
            "id": label,
            "title": title,
            "purpose": str(merged.get("purpose") or "").strip(),
            "entry": str(merged.get("entry") or "").strip(),
            "beats": _string_list(merged.get("beats")),
            "exit": str(merged.get("exit") or "").strip(),
            "constraints": _string_list(merged.get("constraints")),
        }, ensure_ascii=False),
        "content": str(scene.get("content") or "").strip(),
        "check": scene.get("check") or {},
    }
    return normalized


def scene_output_token_limit() -> int:
    """Use only the active provider's technical limit for scene prose.

    Legacy scene cards may still contain a budget, but workflow prose must not be
    truncated by that historical, local-model-oriented field.
    """
    return active_max_output_tokens()


def render_scene_brief(scene: dict[str, Any]) -> str:
    normalized = normalize_scene_card(scene)
    title = normalized.get("title") or "未命名场景"
    purpose = normalized.get("purpose") or "推进本章叙事。"
    return f"{normalized['label']} {title}：{purpose}"


def render_scene_for_model(scene: dict[str, Any]) -> str:
    normalized = normalize_scene_card(scene)
    beats = "\n".join(f"- {item}" for item in normalized["beats"]) or "- 按场景目标自然推进。"
    constraints = "\n".join(f"- {item}" for item in normalized["constraints"]) or "- 不违背已确认设定。"
    title = normalized.get("title") or "未命名场景"
    return f"""【当前场景：{title}】

场景目标：
{normalized.get("purpose") or "推进本章叙事。"}

开场状态：
{normalized.get("entry") or "承接上一场景自然开始。"}

必须完成：
{beats}

禁止：
{constraints}

结束状态：
{normalized.get("exit") or "到达场景目标后自然收束。"}

篇幅：
按完成场景所需自然展开。完成全部关键动作和信息后自然停止，不要为凑字数注水，也不要因固定 token 数提前收尾。

只写小说正文，自然展开，不要逐条复述任务；不要输出场景编号、场景标题或任何 Markdown 标题。"""


def render_outline_for_model(scenes: list[dict[str, Any]]) -> str:
    lines = [render_scene_brief(scene) for scene in scenes]
    return "\n".join(lines)


def parse_scene_cards(outline_text: str) -> list[dict[str, Any]]:
    data = parse_outline_json(outline_text)
    parsed: list[dict[str, str]] = []
    for scene in data["scenes"]:
        label = str(scene["id"]).strip()
        title = str(scene["title"]).strip()
        parsed.append(normalize_scene_card({**scene, "label": label, "title": title}))
    return parsed


def parse_workflow_check(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return {
                    "status": str(value.get("status") or "complete").lower(),
                    "reason": str(value.get("reason") or ""),
                    "fix_instruction": str(value.get("fix_instruction") or ""),
                }
        except json.JSONDecodeError:
            pass
    lowered = raw.lower()
    if "deviat" in lowered or "偏离" in raw:
        return {"status": "deviated", "reason": raw.strip(), "fix_instruction": raw.strip()}
    if "未完成" in raw or "incomplete" in lowered:
        return {"status": "incomplete", "reason": raw.strip(), "fix_instruction": raw.strip()}
    return {"status": "complete", "reason": raw.strip(), "fix_instruction": ""}


def scene_draft_prompt(
    scene: dict[str, str],
    scene_plan: str,
    completed_text: str,
    extra_instruction: str,
) -> str:
    return f"""你正在按场景卡逐场景写作一章小说。

本章场景顺序：
{scene_plan or render_scene_brief(scene)}

已经完成的前序场景正文：
{completed_text or '（无）'}

{render_scene_for_model(scene)}

本次补充要求：
{extra_instruction or '（无）'}

只输出当前场景的小说正文。不要写标题、解释、检查结果或 Markdown。
保持与前序场景自然衔接，但当前场景开头必须直接进入新的动作、反应或信息；不得复述前序场景已经写过的环境、动作、对白、人物状态或设定说明。"""


def scene_check_prompt(
    scene: dict[str, str],
    scene_text: str,
    previous_scene_text: str = "",
) -> str:
    return f"""请检查下面“当前场景正文”是否完成了场景卡要求。

{render_scene_for_model(scene)}

当前场景正文：
{scene_text}

前序场景正文（只用于检查重复；不得要求当前场景重新交代这些内容）：
{previous_scene_text or '（无，这是第一个场景。）'}

只返回 JSON：{{"status":"complete|incomplete|deviated","reason":"...","fix_instruction":"..."}}。
complete 表示可以进入下一个场景；incomplete 表示缺少场景卡内关键动作、信息或情绪变化；deviated 表示偏离场景卡、已确认设定，或当前场景开头重新复述了前序场景已经完成的环境、动作、对白、状态和设定说明。
如果只是自然承接所必需的极短指代，不算重复；如果用新措辞重新解释同一状态，仍算重复并返回 deviated。"""


def scene_continue_prompt(scene: dict[str, str], scene_text: str, check: dict[str, str]) -> str:
    return f"""当前场景还没有完成。请从现有正文最后一句自然续写，只补完缺失部分。

{render_scene_for_model(scene)}

已有当前场景正文：
{scene_text}

缺失说明：
{check.get('reason') or check.get('fix_instruction') or '补完场景目标。'}

只输出续写正文，不要重写已有内容，不要解释。"""


def scene_rewrite_prompt(
    scene: dict[str, str],
    scene_text: str,
    check: dict[str, str],
    previous_scene_text: str = "",
) -> str:
    return f"""当前场景偏离了场景卡。请局部重写这个场景，使它回到场景卡要求。

{render_scene_for_model(scene)}

偏离版本：
{scene_text}

前序场景正文（重写时不要复述）：
{previous_scene_text or '（无）'}

纠偏说明：
{check.get('reason') or check.get('fix_instruction') or '回到场景目标。'}

只输出重写后的当前场景正文，不要解释；开头直接进入本场景的新动作、反应或信息。"""


def continuity_prompt(chapter_draft: str) -> str:
    return f"""请检查下面整章草稿的连续性。

整章草稿：
{chapter_draft}

只输出需要修正的要点，覆盖：场景衔接、时间连续、地点变化、称呼一致、物品位置、对话信息重复、结尾完整度。不要重写正文。"""


def polish_prompt(chapter_draft: str, continuity_notes: str) -> str:
    return f"""请根据连续性检查结果，对整章草稿做首尾衔接与统一润色，输出最终章节正文。

整章草稿：
{chapter_draft}

连续性检查：
{continuity_notes or '（无明显问题）'}

要求：保留场景顺序和关键情节；修正突兀衔接、重复信息、称呼变化、时间和物品位置问题；只输出最终小说正文，不要解释，不要 Markdown。"""


def polish_system_prompt() -> str:
    return (
        "你是中文长篇小说的整章润色编辑。你的任务只是在给定草稿内部做首尾衔接、"
        "连续性修正和语言统一，不引入外部资料，不扩写新剧情，不输出解释。"
    )


def polish_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": polish_system_prompt()},
        {"role": "user", "content": prompt},
    ]


def polish_output_token_target(draft: str, generation_settings: dict[str, Any]) -> int:
    configured = int(generation_settings.get("max_tokens") or 0)
    draft_based = max(POLISH_MIN_OUTPUT_TOKENS, text_char_count(draft) * 2)
    return min(active_max_output_tokens(), max(configured, draft_based))


def assemble_scene_fragments(scenes: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        content
        for scene in scenes
        if (content := str(scene.get("content") or "").strip())
    ).strip()


SCENE_HEADING_LINE = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s*(?:场景\s*)?S\d{1,3}(?:\s+[^\n]*)?\s*$\n?"
)


def strip_scene_headings(text: str) -> str:
    return SCENE_HEADING_LINE.sub("", str(text or "")).strip()


def clean_scene_payload(scene: dict[str, Any], *, content: str = "", check: dict[str, str] | None = None) -> dict[str, Any]:
    normalized = normalize_scene_card(scene)
    return {
        "label": normalized["label"],
        "title": normalized["title"],
        "purpose": normalized["purpose"],
        "entry": normalized["entry"],
        "beats": normalized["beats"],
        "exit": normalized["exit"],
        "constraints": normalized["constraints"],
        "card": normalized["card"],
        "content": strip_scene_headings(content),
        "check": check or scene.get("check") or {},
    }


def text_char_count(text: str) -> int:
    return len(str(text or "").strip())


def chapter_experience_text(summary: dict[str, Any]) -> str:
    value = summary.get("summary") if isinstance(summary, dict) else ""
    if isinstance(value, list):
        rendered = "；".join(str(item).strip() for item in value if str(item).strip())
    elif isinstance(value, dict):
        rendered = "；".join(
            f"{key}：{item}" for key, item in value.items() if str(item).strip()
        )
    else:
        rendered = str(value or "").strip()
    return rendered or format_chapter_summary(summary)


def chapter_selection_rewrite_messages(
    chapter: dict[str, Any],
    workspace: dict[str, Any],
    start: int,
    end: int,
    instruction: str,
) -> list[dict[str, str]]:
    content = str(chapter.get("content") or "")
    selected = content[start:end]
    prefix = content[max(0, start - 6000):start]
    suffix = content[end:min(len(content), end + 6000)]
    local_text = f"{prefix}\n{selected}\n{suffix}"
    relevant_characters: list[str] = []
    for character in workspace.get("characters", []):
        names = [
            str(character.get("name") or "").strip(),
            *[str(alias).strip() for alias in character.get("aliases") or []],
        ]
        if not any(name and name in local_text for name in names):
            continue
        prompt_text = str(character.get("prompt_text") or "").strip()
        if prompt_text:
            relevant_characters.append(prompt_text[:3000])
        if len(relevant_characters) >= 6:
            break
    background = str(
        workspace.get("short_summary")
        or workspace.get("short_summary_effective")
        or workspace.get("global_summary")
        or ""
    ).strip()[:5000]
    supporting = []
    if background:
        supporting.append(f"必要背景：\n{background}")
    if relevant_characters:
        supporting.append("选区涉及人物：\n" + "\n\n".join(relevant_characters))
    supporting_text = "\n\n".join(supporting) or "（无额外资料）"
    return [
        {
            "role": "system",
            "content": (
                "你是中文小说局部重写编辑。只重写用户明确圈选的原文；"
                "保持圈选区前后的时态、视角、人物称谓、事实和衔接。"
                "不要扩写圈选区之外的情节，不要输出说明、标题、引号包装或 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": f"""章节：{chapter.get('title') or '未命名章节'}

{supporting_text}

选区前文（不可改写）：
{prefix or '（章节开头）'}

需要重写的原文：
{selected}

选区后文（不可改写）：
{suffix or '（章节结尾）'}

用户指导：
{instruction.strip() or '在不改变事实与情节作用的前提下，让表达更自然、节奏更清楚。'}

只输出替换“需要重写的原文”的新正文。""",
        },
    ]


def dedupe_scene_continuation(
    existing: str,
    continuation: str,
    *,
    min_overlap: int = 16,
    max_scan_chars: int = 8000,
) -> str:
    existing_text = str(existing or "").strip()
    continuation_text = str(continuation or "").strip()
    if not existing_text or not continuation_text:
        return continuation_text

    def normalized(value: str) -> tuple[str, list[int]]:
        characters: list[str] = []
        boundaries: list[int] = []
        for index, character in enumerate(value):
            if not character.isalnum():
                continue
            folded = character.casefold()
            for folded_character in folded:
                characters.append(folded_character)
                boundaries.append(index + 1)
        return "".join(characters), boundaries

    def strip_once(value: str) -> str:
        value = re.sub(
            r"^(?:#+\s*)?(?:续写(?:正文)?|继续写|补写(?:正文)?|接上文)\s*[：:]\s*",
            "",
            value.strip(),
        )
        existing_normalized, _ = normalized(existing_text)
        value_normalized, boundaries = normalized(value)
        if not existing_normalized or not value_normalized:
            return value
        scan_length = min(len(existing_normalized), len(value_normalized), max_scan_chars)
        removal_size = 0
        if value_normalized.startswith(existing_normalized):
            removal_size = len(existing_normalized)
        else:
            for size in range(scan_length, min_overlap - 1, -1):
                prefix = value_normalized[:size]
                if existing_normalized.endswith(prefix) or prefix in existing_normalized:
                    removal_size = size
                    break
        if removal_size and removal_size <= len(boundaries):
            return value[boundaries[removal_size - 1]:].lstrip(" \t\r\n，。！？；：、,.!?;:")
        return value

    deduped = continuation_text
    for _ in range(4):
        next_text = strip_once(deduped)
        if next_text == deduped:
            break
        deduped = next_text
        if not deduped:
            break
    return deduped.strip()


def log_scene_fragment_stats(
    event: str,
    *,
    conversation_id: str,
    candidate_id: str,
    exchange_id: str | None = None,
    fragments: list[dict[str, Any]],
) -> None:
    payload = {
        "event": event,
        "conversation_id": conversation_id,
        "exchange_id": exchange_id,
        "candidate_id": candidate_id,
        "fragment_count": len(fragments),
        "fragments": fragments,
    }
    logger.info("scene_fragment_stats %s", json.dumps(payload, ensure_ascii=False))


def stream_scene_workflow(
    *,
    exchange: dict[str, Any],
    candidate: dict[str, Any],
    conversation_id: str,
    outline_text: str,
    scenes: list[dict[str, str]],
    extra_instruction: str,
    generation_settings: dict[str, Any],
    stop_event: asyncio.Event,
):
    async def event_stream():
        content = ""
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0
        started = time.monotonic()
        last_flush = started
        scene_plan = render_outline_for_model(scenes)
        conversation, history, _current_user_content = database.get_context_source(exchange["id"])

        async def model_call(
            prompt: str,
            *,
            max_tokens: int,
            emit_delta: bool,
            output_parts: list[str],
        ):
            nonlocal content, reasoning, prompt_tokens, completion_tokens, last_flush
            call_settings = {**generation_settings, "max_tokens": max_tokens}
            call_settings.pop("min_completion_tokens", None)
            context = await build_fitted_context(
                conversation_id=conversation_id,
                system_prompt=conversation["system_prompt"],
                pinned_context=conversation["pinned_context"],
                style_guide=conversation.get("style_guide", ""),
                style_lexicon=conversation.get("style_lexicon", ""),
                history=history,
                current_user_content=prompt,
                max_output_tokens=max_tokens,
                include_outline=False,
            )
            prompt_tokens += int(context.prompt_tokens or 0)
            async for event in llama_client.stream_chat(
                context.messages,
                call_settings,
                stop_event,
                buffer_for_retry=not emit_delta,
            ):
                event_type = event["type"]
                if event_type == "content_delta":
                    text = event["text"]
                    output_parts.append(text)
                    if emit_delta:
                        content += text
                        yield sse("content_delta", {"text": text})
                elif event_type == "reasoning_delta":
                    reasoning += event["text"]
                    if emit_delta:
                        yield sse("reasoning_delta", {"text": event["text"]})
                elif event_type in {"usage", "timings"}:
                    usage = event["value"]
                    completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                elif event_type == "retry":
                    yield sse("model_retry", event)
                now = time.monotonic()
                if emit_delta and now - last_flush >= 0.8:
                    database.update_candidate_draft(candidate["id"], content, reasoning)
                    last_flush = now

        try:
            yield sse(
                "candidate_created",
                {
                    "exchange_id": exchange["id"],
                    "user_content": exchange["user_content"],
                    "candidate": candidate,
                    "trimmed_exchange_count": 0,
                    "prompt_tokens": None,
                    "context_size": active_context_size(),
                },
            )
            review_scenes: list[dict[str, Any]] = []
            fragment_stats: list[dict[str, Any]] = []
            for index, scene in enumerate(scenes, start=1):
                output_lengths: list[int] = []
                rewrite_count = 0
                yield sse(
                    "workflow_step",
                    {
                        "step": "draft",
                        "scene_index": index,
                        "scene_total": len(scenes),
                        "message": f"{scene['label']} 逐场景写作",
                    },
                )
                parts: list[str] = []
                async for outbound in model_call(
                    scene_draft_prompt(
                        scene,
                        scene_plan,
                        assemble_scene_fragments(review_scenes),
                        extra_instruction,
                    ),
                    max_tokens=scene_output_token_limit(),
                    emit_delta=True,
                    output_parts=parts,
                ):
                    yield outbound
                scene_text = "".join(parts).strip()
                previous_scene_text = "\n\n".join(
                    str(item.get("content") or "").strip()
                    for item in review_scenes
                    if str(item.get("content") or "").strip()
                )
                scene_text = dedupe_scene_continuation(previous_scene_text, scene_text)
                output_lengths.append(text_char_count(scene_text))

                yield sse(
                    "workflow_step",
                    {
                        "step": "check",
                        "scene_index": index,
                        "scene_total": len(scenes),
                        "message": f"{scene['label']} 完成度检查",
                    },
                )
                check_parts: list[str] = []
                async for outbound in model_call(
                    scene_check_prompt(scene, scene_text, previous_scene_text),
                    max_tokens=700,
                    emit_delta=False,
                    output_parts=check_parts,
                ):
                    yield outbound
                check = parse_workflow_check("".join(check_parts))
                if check["status"] == "incomplete":
                    yield sse(
                        "workflow_step",
                        {
                            "step": "draft",
                            "scene_index": index,
                            "scene_total": len(scenes),
                            "message": f"{scene['label']} 续写当前场景",
                        },
                    )
                    continuation_parts: list[str] = []
                    async for outbound in model_call(
                        scene_continue_prompt(scene, scene_text, check),
                        max_tokens=scene_output_token_limit(),
                        emit_delta=False,
                        output_parts=continuation_parts,
                    ):
                        yield outbound
                    continuation_text = dedupe_scene_continuation(
                        scene_text,
                        "".join(continuation_parts),
                    )
                    output_lengths.append(text_char_count(continuation_text))
                    if continuation_text:
                        scene_text = f"{scene_text}\n{continuation_text}".strip()
                elif check["status"] == "deviated":
                    yield sse(
                        "workflow_step",
                        {
                            "step": "check",
                            "scene_index": index,
                            "scene_total": len(scenes),
                            "message": f"{scene['label']} 局部重写",
                        },
                    )
                    rewrite_parts: list[str] = []
                    async for outbound in model_call(
                        scene_rewrite_prompt(scene, scene_text, check, previous_scene_text),
                        max_tokens=scene_output_token_limit(),
                        emit_delta=False,
                        output_parts=rewrite_parts,
                    ):
                        yield outbound
                    rewritten = "".join(rewrite_parts).strip()
                    if rewritten:
                        rewrite_count += 1
                        output_lengths.append(text_char_count(rewritten))
                        scene_text = rewritten
                review_scenes.append(clean_scene_payload(scene, content=scene_text, check=check))
                content = assemble_scene_fragments(review_scenes)
                database.update_candidate_draft(candidate["id"], content, reasoning)
                yield sse("content_replace", {"text": content})
                fragment_stats.append({
                    "scene_index": index,
                    "label": scene["label"],
                    "title": scene["title"],
                    "attempt_count": len(output_lengths),
                    "rewritten": rewrite_count > 0,
                    "rewrite_count": rewrite_count,
                    "output_char_counts": output_lengths,
                    "final_char_count": text_char_count(scene_text),
                    "check_status": check.get("status", ""),
                })

            draft = assemble_scene_fragments(review_scenes)
            if draft and draft != content.strip():
                content = draft
                database.update_candidate_draft(candidate["id"], content, reasoning)
                yield sse("content_replace", {"text": draft})
            yield sse("workflow_step", {"step": "review", "message": "场景审阅"})
            duration_ms = int((time.monotonic() - started) * 1000)
            updated_exchange = database.finalize_candidate(
                candidate["id"],
                status="completed",
                content=draft,
                reasoning=reasoning,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
            )
            review_payload = {
                "candidate_id": candidate["id"],
                "exchange": updated_exchange,
                "outline_text": outline_text,
                "instruction": extra_instruction,
                "scenes": review_scenes,
            }
            log_scene_fragment_stats(
                "workflow_review_ready",
                conversation_id=conversation_id,
                exchange_id=exchange["id"],
                candidate_id=candidate["id"],
                fragments=fragment_stats,
            )
            yield sse("workflow_review_ready", review_payload)
            yield sse(
                "done",
                {
                    "candidate_id": candidate["id"],
                    "exchange": updated_exchange,
                    "finish_reason": "scene_workflow_review_ready",
                    "duration_ms": duration_ms,
                },
            )
        except GenerationCancelled:
            duration_ms = int((time.monotonic() - started) * 1000)
            updated_exchange = database.finalize_candidate(
                candidate["id"],
                status="cancelled",
                content=content,
                reasoning=reasoning,
                duration_ms=duration_ms,
                error_message="用户停止了编排流程",
            )
            yield sse("cancelled", {"candidate_id": candidate["id"], "exchange": updated_exchange})
        except asyncio.CancelledError:
            database.finalize_candidate(
                candidate["id"],
                status="cancelled",
                content=content,
                reasoning=reasoning,
                error_message="连接已关闭",
            )
            logger.warning(
                "stream_disconnected operation=scene_workflow candidate_id=%s conversation_id=%s",
                candidate["id"], conversation_id,
            )
            raise
        except Exception as exc:
            logger.exception("scene workflow failed")
            truncated = isinstance(exc, GenerationTruncated)
            updated_exchange = database.finalize_candidate(
                candidate["id"],
                status="failed",
                content=content,
                reasoning=reasoning,
                error_message=str(exc)[:1000],
            )
            yield sse(
                "error",
                {
                    "candidate_id": candidate["id"],
                    "exchange": updated_exchange,
                    "code": "GENERATION_TRUNCATED" if truncated else "SCENE_WORKFLOW_FAILED",
                    "message": (
                        "场景输出达到长度上限，未把残缺正文当作完成版本"
                        if truncated
                        else "场景编排流程失败"
                    ),
                    "detail": str(exc)[:500],
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def stream_scene_fragment_regeneration(
    *,
    conversation_id: str,
    source_exchange_id: str,
    candidate_id: str,
    outline_text: str,
    scenes: list[dict[str, Any]],
    scene_index: int,
    extra_instruction: str,
    generation_settings: dict[str, Any],
    stop_event: asyncio.Event,
):
    async def event_stream():
        fragment_text = ""
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0
        started = time.monotonic()
        conversation, history, _current_user_content = database.get_context_source(source_exchange_id)
        scenes[:] = [normalize_scene_card(scene) for scene in scenes]
        scene = scenes[scene_index]
        original_fragment_text = str(scene.get("content") or "")
        scene_plan = render_outline_for_model(scenes)
        output_lengths: list[int] = []
        rewrite_count = 0

        async def model_call(prompt: str, *, max_tokens: int, emit_delta: bool, output_parts: list[str]):
            nonlocal fragment_text, reasoning, prompt_tokens, completion_tokens
            call_settings = {**generation_settings, "max_tokens": max_tokens}
            call_settings.pop("min_completion_tokens", None)
            context = await build_fitted_context(
                conversation_id=conversation_id,
                system_prompt=conversation["system_prompt"],
                pinned_context=conversation["pinned_context"],
                style_guide=conversation.get("style_guide", ""),
                style_lexicon=conversation.get("style_lexicon", ""),
                history=history,
                current_user_content=prompt,
                max_output_tokens=max_tokens,
                include_outline=False,
            )
            prompt_tokens += int(context.prompt_tokens or 0)
            async for event in llama_client.stream_chat(
                context.messages,
                call_settings,
                stop_event,
                buffer_for_retry=not emit_delta,
            ):
                event_type = event["type"]
                if event_type == "content_delta":
                    text = event["text"]
                    output_parts.append(text)
                    if emit_delta:
                        fragment_text += text
                        yield sse("fragment_delta", {"scene_index": scene_index, "text": text})
                elif event_type == "reasoning_delta":
                    reasoning += event["text"]
                elif event_type in {"usage", "timings"}:
                    usage = event["value"]
                    completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                elif event_type == "retry":
                    yield sse("model_retry", event)

        try:
            yield sse(
                "workflow_step",
                {
                    "step": "regenerate",
                    "scene_index": scene_index + 1,
                    "scene_total": len(scenes),
                    "message": f"{scene['label']} 重新生成",
                },
            )
            yield sse("fragment_replace", {"scene_index": scene_index, "text": ""})
            parts: list[str] = []
            async for outbound in model_call(
                scene_draft_prompt(
                    scene,
                    scene_plan,
                    assemble_scene_fragments(scenes[:scene_index]),
                    extra_instruction,
                ),
                max_tokens=scene_output_token_limit(),
                emit_delta=True,
                output_parts=parts,
            ):
                yield outbound
            fragment_text = "".join(parts).strip()
            previous_scene_text = "\n\n".join(
                str(item.get("content") or "").strip()
                for item in scenes[:scene_index]
                if str(item.get("content") or "").strip()
            )
            fragment_text = dedupe_scene_continuation(previous_scene_text, fragment_text)
            yield sse("fragment_replace", {"scene_index": scene_index, "text": fragment_text})
            output_lengths.append(text_char_count(fragment_text))

            yield sse(
                "workflow_step",
                {
                    "step": "check",
                    "scene_index": scene_index + 1,
                    "scene_total": len(scenes),
                    "message": f"{scene['label']} 完成度检查",
                },
            )
            check_parts: list[str] = []
            async for outbound in model_call(
                scene_check_prompt(scene, fragment_text, previous_scene_text),
                max_tokens=700,
                emit_delta=False,
                output_parts=check_parts,
            ):
                yield outbound
            check = parse_workflow_check("".join(check_parts))
            if check["status"] == "incomplete":
                yield sse(
                    "workflow_step",
                    {
                        "step": "regenerate",
                        "scene_index": scene_index + 1,
                        "scene_total": len(scenes),
                        "message": f"{scene['label']} 续写当前场景",
                    },
                )
                base_fragment_text = fragment_text
                continuation_parts: list[str] = []
                async for outbound in model_call(
                    scene_continue_prompt(scene, base_fragment_text, check),
                    max_tokens=scene_output_token_limit(),
                    emit_delta=False,
                    output_parts=continuation_parts,
                ):
                    yield outbound
                continuation_text = dedupe_scene_continuation(
                    base_fragment_text,
                    "".join(continuation_parts),
                )
                output_lengths.append(text_char_count(continuation_text))
                fragment_text = (
                    f"{base_fragment_text}\n{continuation_text}".strip()
                    if continuation_text
                    else base_fragment_text
                )
                yield sse("fragment_replace", {"scene_index": scene_index, "text": fragment_text})
            elif check["status"] == "deviated":
                yield sse(
                    "workflow_step",
                    {
                        "step": "regenerate",
                        "scene_index": scene_index + 1,
                        "scene_total": len(scenes),
                        "message": f"{scene['label']} 局部重写",
                    },
                )
                rewrite_parts: list[str] = []
                async for outbound in model_call(
                    scene_rewrite_prompt(scene, fragment_text, check, previous_scene_text),
                    max_tokens=scene_output_token_limit(),
                    emit_delta=False,
                    output_parts=rewrite_parts,
                ):
                    yield outbound
                rewritten = "".join(rewrite_parts).strip()
                if rewritten:
                    rewrite_count += 1
                    output_lengths.append(text_char_count(rewritten))
                    fragment_text = rewritten
                    yield sse("fragment_replace", {"scene_index": scene_index, "text": fragment_text})

            scenes[scene_index] = clean_scene_payload(scene, content=fragment_text, check=check)
            duration_ms = int((time.monotonic() - started) * 1000)
            updated_exchange = database.update_candidate_content(
                candidate_id,
                content=assemble_scene_fragments(scenes),
                reasoning=reasoning,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                expected_conversation_id=conversation_id,
            )
            log_scene_fragment_stats(
                "fragment_regenerated",
                conversation_id=conversation_id,
                exchange_id=updated_exchange["id"],
                candidate_id=candidate_id,
                fragments=[{
                    "scene_index": scene_index + 1,
                    "label": scene.get("label", ""),
                    "title": scene.get("title", ""),
                    "attempt_count": len(output_lengths),
                    "rewritten": rewrite_count > 0,
                    "rewrite_count": rewrite_count,
                    "output_char_counts": output_lengths,
                    "final_char_count": text_char_count(fragment_text),
                    "check_status": check.get("status", ""),
                }],
            )
            yield sse(
                "fragment_done",
                {
                    "candidate_id": candidate_id,
                    "exchange": updated_exchange,
                    "scene_index": scene_index,
                    "scene": scenes[scene_index],
                    "scenes": scenes,
                    "duration_ms": duration_ms,
                },
            )
            yield sse("workflow_step", {"step": "review", "message": "场景审阅"})
        except GenerationCancelled:
            yield sse("cancelled", {"candidate_id": candidate_id, "message": "已停止重生成当前场景"})
        except asyncio.CancelledError:
            logger.warning(
                "stream_disconnected operation=scene_fragment candidate_id=%s conversation_id=%s scene_index=%s",
                candidate_id, conversation_id, scene_index,
            )
            raise
        except Exception as exc:
            logger.exception("scene fragment regeneration failed")
            truncated = isinstance(exc, GenerationTruncated)
            if truncated:
                yield sse(
                    "fragment_replace",
                    {"scene_index": scene_index, "text": original_fragment_text},
                )
            yield sse(
                "error",
                {
                    "candidate_id": candidate_id,
                    "code": "GENERATION_TRUNCATED" if truncated else "SCENE_FRAGMENT_FAILED",
                    "message": (
                        "当前场景输出达到长度上限，已恢复重写前正文"
                        if truncated
                        else "场景重生成失败"
                    ),
                    "detail": str(exc)[:500],
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def stream_scene_workflow_polish(
    *,
    conversation_id: str,
    candidate_id: str,
    scenes: list[dict[str, Any]],
    generation_settings: dict[str, Any],
    stop_event: asyncio.Event,
):
    async def event_stream():
        content = ""
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0
        started = time.monotonic()
        draft = assemble_scene_fragments(scenes)
        exchange = database.update_candidate_content(
            candidate_id,
            content=draft,
            expected_conversation_id=conversation_id,
        )
        candidate = next(item for item in exchange["candidates"] if item["id"] == candidate_id)

        async def model_call(prompt: str, *, max_tokens: int, emit_delta: bool, output_parts: list[str]):
            nonlocal content, reasoning, prompt_tokens, completion_tokens
            messages = polish_messages(prompt)
            prompt_token_count = await count_or_estimate(messages)
            context_limit = polish_context_token_limit()
            available_output = max(
                512,
                context_limit - prompt_token_count - POLISH_OUTPUT_RESERVE_TOKENS,
            )
            effective_max_tokens = max(512, min(max_tokens, available_output))
            call_settings = {**generation_settings, "max_tokens": effective_max_tokens}
            call_settings.pop("min_completion_tokens", None)
            prompt_tokens += prompt_token_count
            async for event in llama_client.stream_chat(
                messages,
                call_settings,
                stop_event,
                buffer_for_retry=not emit_delta,
            ):
                event_type = event["type"]
                if event_type == "content_delta":
                    text = event["text"]
                    output_parts.append(text)
                    if emit_delta:
                        content += text
                        yield sse("content_delta", {"text": text})
                elif event_type == "reasoning_delta":
                    reasoning += event["text"]
                    if emit_delta:
                        yield sse("reasoning_delta", {"text": event["text"]})
                elif event_type in {"usage", "timings"}:
                    usage = event["value"]
                    completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                elif event_type == "retry":
                    yield sse("model_retry", event)

        try:
            yield sse(
                "candidate_created",
                {
                    "exchange_id": exchange["id"],
                    "user_content": exchange["user_content"],
                    "candidate": candidate,
                    "trimmed_exchange_count": 0,
                    "prompt_tokens": None,
                    "context_size": polish_context_token_limit(),
                },
            )
            yield sse("workflow_step", {"step": "continuity", "message": "章节连续性检查"})
            continuity_parts: list[str] = []
            async for outbound in model_call(
                continuity_prompt(draft),
                max_tokens=1600,
                emit_delta=False,
                output_parts=continuity_parts,
            ):
                yield outbound
            continuity_notes = "".join(continuity_parts).strip()

            yield sse("workflow_step", {"step": "polish", "message": "首尾衔接与润色"})
            yield sse("content_replace", {"text": ""})
            final_parts: list[str] = []
            async for outbound in model_call(
                polish_prompt(draft, continuity_notes),
                max_tokens=polish_output_token_target(draft, generation_settings),
                emit_delta=True,
                output_parts=final_parts,
            ):
                yield outbound
            final_text = strip_scene_headings("".join(final_parts)) or draft
            if not content.strip():
                content = final_text
                yield sse("content_replace", {"text": final_text})
            yield sse("workflow_step", {"step": "final", "message": "最终章节完成"})
            duration_ms = int((time.monotonic() - started) * 1000)
            updated_exchange = database.update_candidate_content(
                candidate_id,
                content=final_text,
                reasoning=reasoning,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                expected_conversation_id=conversation_id,
            )
            yield sse(
                "done",
                {
                    "candidate_id": candidate_id,
                    "exchange": updated_exchange,
                    "finish_reason": "scene_workflow_completed",
                    "duration_ms": duration_ms,
                },
            )
        except GenerationCancelled:
            updated_exchange = database.update_candidate_content(
                candidate_id,
                content=content.strip() or draft,
                reasoning=reasoning,
                expected_conversation_id=conversation_id,
            )
            yield sse("cancelled", {"candidate_id": candidate_id, "exchange": updated_exchange})
        except asyncio.CancelledError:
            logger.warning(
                "stream_disconnected operation=scene_polish candidate_id=%s conversation_id=%s",
                candidate_id, conversation_id,
            )
            raise
        except Exception as exc:
            logger.exception("scene workflow polish failed")
            truncated = isinstance(exc, GenerationTruncated)
            restored_exchange = None
            if truncated:
                content = draft
                restored_exchange = database.update_candidate_content(
                    candidate_id,
                    content=draft,
                    reasoning=reasoning,
                    prompt_tokens=prompt_tokens or None,
                    completion_tokens=completion_tokens,
                    expected_conversation_id=conversation_id,
                )
                yield sse("content_replace", {"text": draft})
            yield sse(
                "error",
                {
                    "candidate_id": candidate_id,
                    "exchange": restored_exchange,
                    "code": "GENERATION_TRUNCATED" if truncated else "SCENE_POLISH_FAILED",
                    "message": (
                        "润色输出达到长度上限，已恢复并保留润色前正文"
                        if truncated
                        else "最终润色失败"
                    ),
                    "detail": str(exc)[:500],
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def stream_outline_preview(
    *,
    preview_id: str,
    context: ContextResult,
    generation_settings: dict[str, Any],
    stop_event: asyncio.Event,
):
    async def event_stream():
        content = ""
        started = time.monotonic()
        try:
            yield sse(
                "outline_preview_created",
                {
                    "candidate": {
                        "id": preview_id,
                        "content": "",
                        "edited_content": "",
                        "status": "streaming",
                        "persisted": False,
                    },
                    "prompt_tokens": context.prompt_tokens,
                    "context_size": active_context_size(),
                    "max_tokens": generation_settings["max_tokens"],
                },
            )
            async for event in llama_client.stream_chat(
                context.messages, generation_settings, stop_event
            ):
                if event["type"] == "content_delta":
                    content += event["text"]
                    yield sse("content_delta", {"text": event["text"]})
                elif event["type"] == "retry":
                    yield sse("model_retry", event)
            yield sse(
                "done",
                {
                    "candidate": {
                        "id": preview_id,
                        "content": content,
                        "edited_content": "",
                        "status": "completed",
                        "persisted": False,
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    }
                },
            )
        except GenerationCancelled:
            yield sse(
                "cancelled",
                {
                    "candidate": {
                        "id": preview_id,
                        "content": content,
                        "edited_content": "",
                        "status": "cancelled",
                        "persisted": False,
                    }
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("outline preview generation failed")
            truncated = isinstance(exc, GenerationTruncated)
            yield sse(
                "error",
                {
                    "code": "GENERATION_TRUNCATED" if truncated else "OUTLINE_GENERATION_FAILED",
                    "message": (
                        "场景卡输出达到长度上限，请提高最大输出 token 后重试"
                        if truncated
                        else "场景卡生成失败，可以重新尝试"
                    ),
                    "detail": str(exc)[:500],
                    "candidate": {
                        "id": preview_id,
                        "content": content,
                        "edited_content": "",
                        "status": "failed",
                        "persisted": False,
                    },
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def selected_history(conversation: dict[str, Any]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for exchange in conversation.get("exchanges", []):
        selected = next(
            (
                candidate
                for candidate in exchange["candidates"]
                if candidate["id"] == exchange["selected_candidate_id"]
                and candidate["status"] == "completed"
            ),
            None,
        )
        if selected:
            history.append({"role": "user", "content": exchange["user_content"]})
            history.append({"role": "assistant", "content": selected["content"]})
    return history


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.initialize()
    if settings.model_mode == "local":
        persisted_context = database.get_app_setting("context_size")
        if persisted_context:
            try:
                llama_process.set_context_size(int(persisted_context))
            except (TypeError, ValueError):
                logger.warning("ignoring invalid persisted context size: %s", persisted_context)
    startup_task: asyncio.Task[Any] | None = None
    if settings.model_mode == "local" and settings.auto_start_llama:
        async def start_model() -> None:
            try:
                await llama_process.start()
            except LlamaProcessError as exc:
                logger.error("model startup failed: %s", exc)

        startup_task = asyncio.create_task(start_model(), name="llama-model-startup")
        app.state.model_startup_task = startup_task
    yield
    generation.stop()
    if startup_task is not None and not startup_task.done():
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass
    if settings.model_mode == "local":
        await llama_process.stop()
    llama_client.clear_api_key()


app = FastAPI(title="Novel-factory", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    request_id = secrets.token_hex(6)
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "http_request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    duration_ms = int((time.monotonic() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http_request request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(KeyError)
async def handle_key_error(_request: Request, exc: KeyError):
    return error_response(404, "NOT_FOUND", "没有找到请求的内容", str(exc))


@app.exception_handler(ValueError)
async def handle_value_error(_request: Request, exc: ValueError):
    return error_response(400, "INVALID_REQUEST", str(exc))


@app.get("/api/health")
async def health():
    return {
        "app": "Novel-factory",
        "status": "ok",
        "generation_in_progress": generation.busy,
    }


@app.get("/api/agent/activity")
async def get_agent_activity():
    return agent_activity.snapshot()


@app.post("/api/agent/activity/start")
async def start_agent_activity(request: Request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return error_response(403, "LOCAL_ONLY", "Agent 活动接口只允许本机调用")
    payload = await request.json()
    return agent_activity.start(str(payload.get("label") or "外部 Agent 操作"))


@app.post("/api/agent/activity/{token}/finish")
async def finish_agent_activity(token: str, request: Request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return error_response(403, "LOCAL_ONLY", "Agent 活动接口只允许本机调用")
    return agent_activity.finish(token)


@app.get("/api/runtime")
async def runtime():
    info = await model_runtime_info()
    info["generation_in_progress"] = generation.busy
    return info


@app.get("/api/runtime/logs", response_class=PlainTextResponse)
async def runtime_logs(
    request: Request,
    lines: int = Query(default=200, ge=1, le=2000),
):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="日志接口只允许本机访问")
    log_path = settings.project_root / "data" / "novel-factory.log"
    if not log_path.exists():
        return ""
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])


@app.post("/api/runtime/start")
async def start_runtime():
    if settings.model_mode == "deepseek":
        info = await model_runtime_info()
        info["generation_in_progress"] = generation.busy
        return info
    try:
        return await llama_process.start()
    except LlamaProcessError as exc:
        return error_response(500, "LLAMA_START_FAILED", "模型服务启动失败", str(exc))


@app.post("/api/runtime/stop")
async def stop_runtime():
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成")
    if settings.model_mode == "deepseek":
        llama_client.clear_api_key()
        return await model_runtime_info(check_health=False)
    await llama_process.stop()
    return {"status": "stopped"}


@app.post("/api/runtime/api-key")
async def set_runtime_api_key(request: Request):
    if settings.model_mode != "deepseek":
        return error_response(409, "NOT_API_MODE", "当前实例不是 DeepSeek API 模式")
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return error_response(403, "LOCAL_ONLY", "API Key 只能从本机页面提交")
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成")
    try:
        body = await request.json()
    except Exception:
        return error_response(400, "API_KEY_INVALID", "请求格式不正确")
    api_key = body.get("api_key") if isinstance(body, dict) else None
    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 500:
        return error_response(400, "API_KEY_INVALID", "DeepSeek API Key 格式不正确")
    api_key = api_key.strip()
    try:
        await llama_client.validate_api_key(api_key)
        llama_client.set_api_key(api_key)
    except LlamaClientError as exc:
        llama_client.clear_api_key()
        return error_response(401, "DEEPSEEK_AUTH_FAILED", "DeepSeek API Key 验证失败", str(exc))
    info = await model_runtime_info(check_health=False)
    info["generation_in_progress"] = generation.busy
    return info


@app.delete("/api/runtime/api-key")
async def clear_runtime_api_key():
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成")
    llama_client.clear_api_key()
    return await model_runtime_info(check_health=False)


@app.post("/api/runtime/context")
async def change_runtime_context(payload: RuntimeContextRequest):
    if settings.model_mode == "deepseek":
        return error_response(
            409,
            "CONTEXT_FIXED_IN_API_MODE",
            f"API 模式上下文固定为 {settings.api_context_size}",
        )
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    if payload.context_size == active_context_size():
        return await llama_process.runtime_info()
    await llama_process.stop()
    llama_process.set_context_size(payload.context_size)
    database.set_app_setting("context_size", str(payload.context_size))
    try:
        return await llama_process.start()
    except LlamaProcessError as exc:
        return error_response(500, "LLAMA_START_FAILED", "切换上下文后模型启动失败", str(exc))


@app.get("/api/conversations")
async def list_conversations():
    return {"items": database.list_conversations()}


@app.post("/api/conversations", status_code=201)
async def create_conversation(payload: ConversationCreate):
    return database.create_conversation(title=payload.title)


@app.post("/api/conversations/import", status_code=201)
async def import_conversation_backup(request: Request):
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return error_response(400, "INVALID_CONVERSATION_BACKUP", "JSON 备份文件无法解析")
    try:
        return database.import_conversation_backup(payload)
    except ValueError as exc:
        return error_response(400, "INVALID_CONVERSATION_BACKUP", str(exc))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    return database.get_conversation(conversation_id)


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, payload: ConversationUpdate):
    if payload.document_id:
        novels.get_document_workspace(payload.document_id)
    return database.update_conversation(conversation_id, payload.changes())


@app.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str):
    database.delete_conversation(conversation_id)
    return Response(status_code=204)


@app.post("/api/conversations/{conversation_id}/context-count")
async def count_conversation_context(
    conversation_id: str, payload: ContextCountRequest
):
    await ensure_model_ready()
    conversation = database.get_conversation(conversation_id)
    max_output = resolve_generation_settings(conversation, None)["max_tokens"]
    context = await build_fitted_context(
        conversation_id=conversation_id,
        system_prompt=conversation["system_prompt"],
        pinned_context=conversation["pinned_context"],
        style_guide=conversation.get("style_guide", ""),
        style_lexicon=conversation.get("style_lexicon", ""),
        history=selected_history(conversation),
        current_user_content=payload.content or "（下一条创作指令）",
        max_output_tokens=max_output,
    )
    reserved = max_output + 384
    return {
        "input_tokens": context.prompt_tokens,
        "context_size": active_context_size(),
        "reserved_output_tokens": reserved,
        "available_tokens": max(
            0, active_context_size() - reserved - int(context.prompt_tokens or 0)
        ),
        "trimmed_exchange_count": context.trimmed_exchange_count,
        "source_characters": {
            key: len(value) for key, value in prompt_assets_for_conversation(
                conversation_id,
                query_text=payload.content,
                material_budget_tokens=max(
                    1024,
                    min(12000, active_context_size() - reserved - 512),
                ),
            ).items()
        },
    }


@app.get("/api/conversations/{conversation_id}/prompt-preview")
async def prompt_preview(conversation_id: str, query: str = ""):
    conversation = database.get_conversation(conversation_id)
    assets = prompt_assets_for_conversation(conversation_id, query_text=query)
    max_output = resolve_generation_settings(conversation, None)["max_tokens"]
    context = await build_fitted_context(
        conversation_id=conversation_id,
        system_prompt=conversation["system_prompt"],
        pinned_context=conversation["pinned_context"],
        style_guide=conversation.get("style_guide", ""),
        style_lexicon=conversation.get("style_lexicon", ""),
        history=selected_history(conversation),
        current_user_content=query or "（下一条创作指令）",
        max_output_tokens=max_output,
    )
    return {
        "document_id": conversation.get("document_id"),
        "system_prompt": conversation["system_prompt"],
        "pinned_context": conversation["pinned_context"],
        "style_guide": conversation.get("style_guide", ""),
        "style_lexicon": conversation.get("style_lexicon", ""),
        "sources": assets,
        "messages": context.messages,
        "full_prompt": "\n\n".join(
            f"## {message['role']}\n{message['content']}" for message in context.messages
        ),
        "input_tokens": context.prompt_tokens,
        "context_size": active_context_size(),
        "trimmed_exchange_count": context.trimmed_exchange_count,
    }


@app.get("/api/projects")
async def list_projects():
    return {"items": novels.list_projects()}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    return novels.get_project(project_id)


@app.get("/api/documents/{document_id}/workspace")
async def get_document_workspace(document_id: str):
    return novels.get_document_workspace(document_id)


@app.patch("/api/documents/{document_id}")
async def update_document(document_id: str, payload: DocumentUpdate):
    return novels.update_document(document_id, payload.model_dump(exclude_none=True))


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, payload: ProjectUpdate):
    return novels.update_project(project_id, payload.model_dump(exclude_none=True))


@app.post("/api/projects/{project_id}/import-txt", status_code=201)
async def import_txt(project_id: str, request: Request):
    data = await request.body()
    if not data:
        return error_response(400, "EMPTY_FILE", "导入的 TXT 文件为空")
    if len(data) > 50 * 1024 * 1024:
        return error_response(413, "FILE_TOO_LARGE", "TXT 文件不能超过 50 MB")
    filename = unquote(request.headers.get("x-filename", "导入小说.txt"))
    imported = decode_text(data)
    text = normalize_text(imported.text)
    if not text:
        return error_response(400, "EMPTY_TEXT", "没有从文件中读取到有效文字")
    result = novels.import_document(
        project_id, filename, imported.encoding, text
    )
    result["encoding"] = imported.encoding
    return result


@app.post("/api/projects/{project_id}/documents", status_code=201)
async def create_document(project_id: str, payload: DocumentCreateRequest):
    return novels.create_document(project_id, payload.filename)


@app.get("/api/chapters/{chapter_id}")
async def get_chapter(chapter_id: str):
    return novels.get_chapter(chapter_id)


@app.post("/api/chapters/{chapter_id}/rewrite-selection")
async def rewrite_chapter_selection(
    chapter_id: str,
    payload: ChapterSelectionRewriteRequest,
):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    chapter = novels.get_chapter(chapter_id)
    content = str(chapter.get("content") or "")
    if payload.end <= payload.start or payload.end > len(content):
        return error_response(400, "INVALID_SELECTION", "请选择章节正文中的有效区域")
    selected = content[payload.start:payload.end]
    if not selected.strip():
        return error_response(400, "EMPTY_SELECTION", "选中的正文不能为空")
    if len(selected) > 100_000:
        return error_response(413, "SELECTION_TOO_LARGE", "单次局部重写不能超过 10 万字")
    workspace = novels.get_document_workspace(chapter["document_id"])
    messages = chapter_selection_rewrite_messages(
        chapter,
        workspace,
        payload.start,
        payload.end,
        payload.instruction,
    )
    generation_settings = {
        **DEFAULT_GENERATION_SETTINGS,
        **(payload.settings.model_dump() if payload.settings else {}),
    }
    if generation_settings.get("seed") is None:
        generation_settings["seed"] = secrets.randbelow(2_147_483_647)
    generation_settings["max_tokens"] = min(
        active_max_output_tokens(),
        max(
            int(generation_settings.get("max_tokens") or 1600),
            max(800, len(selected) * 2),
        ),
    )
    prompt_tokens = await count_or_estimate(messages)
    if prompt_tokens + generation_settings["max_tokens"] + 256 > active_context_size():
        return error_response(
            413,
            "REWRITE_CONTEXT_TOO_LONG",
            "局部上下文过长，请缩小选区后重试",
        )
    operation_id = new_id()
    stop_event = await generation.begin(operation_id)

    async def event_stream():
        rewritten = ""
        reasoning = ""
        completion_tokens = 0
        started = time.monotonic()
        try:
            yield sse(
                "rewrite_started",
                {
                    "operation_id": operation_id,
                    "chapter_id": chapter_id,
                    "start": payload.start,
                    "end": payload.end,
                    "source_hash": chapter["content_hash"],
                    "original_text": selected,
                    "prompt_tokens": prompt_tokens,
                },
            )
            async for event in llama_client.stream_chat(
                messages,
                generation_settings,
                stop_event,
            ):
                event_type = event["type"]
                if event_type == "content_delta":
                    rewritten += event["text"]
                    yield sse("content_delta", {"text": event["text"]})
                elif event_type == "reasoning_delta":
                    reasoning += event["text"]
                elif event_type in {"usage", "timings"}:
                    completion_tokens = int(
                        event["value"].get("completion_tokens", completion_tokens) or 0
                    )
                elif event_type == "retry":
                    yield sse("model_retry", event)
            rewritten = rewritten.strip()
            if not rewritten:
                raise LlamaClientError("模型没有返回可用的重写正文")
            yield sse(
                "done",
                {
                    "chapter_id": chapter_id,
                    "start": payload.start,
                    "end": payload.end,
                    "source_hash": chapter["content_hash"],
                    "original_text": selected,
                    "replacement": rewritten,
                    "reasoning": reasoning,
                    "completion_tokens": completion_tokens,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
        except GenerationCancelled:
            yield sse("cancelled", {"message": "已停止局部重写"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("chapter selection rewrite failed")
            truncated = isinstance(exc, GenerationTruncated)
            yield sse(
                "error",
                {
                    "code": "GENERATION_TRUNCATED" if truncated else "CHAPTER_SELECTION_REWRITE_FAILED",
                    "message": (
                        "局部重写达到长度上限，原文没有被修改"
                        if truncated
                        else "局部重写失败"
                    ),
                    "detail": str(exc)[:500],
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chapters/{chapter_id}/rewrite-selection/apply")
async def apply_chapter_selection_rewrite(
    chapter_id: str,
    payload: ChapterSelectionApplyRequest,
):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先等待当前任务结束")
    return novels.replace_chapter_selection(
        chapter_id,
        start=payload.start,
        end=payload.end,
        source_hash=payload.source_hash,
        original_text=payload.original_text,
        replacement=payload.replacement,
    )


@app.patch("/api/chapters/{chapter_id}")
async def update_chapter(chapter_id: str, payload: ChapterUpdate):
    return novels.update_chapter(chapter_id, payload.model_dump(exclude_none=True))


@app.patch("/api/characters/{character_id}")
async def update_character(character_id: str, payload: CharacterUpdate):
    return novels.update_character(character_id, payload.model_dump(exclude_none=True))


@app.post("/api/documents/{document_id}/characters", status_code=201)
async def create_character(document_id: str, payload: CharacterCreateRequest):
    return novels.create_character(document_id, payload.card)


@app.post("/api/documents/{document_id}/characters/extract")
async def extract_characters(document_id: str, payload: CharacterExtractRequest):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    workspace = novels.get_document_workspace(document_id)
    if payload.end_position < payload.start_position:
        return error_response(400, "INVALID_RANGE", "结束章节不能早于开始章节")
    selected = []
    for chapter in workspace["chapters"]:
        if not payload.start_position <= chapter["position"] <= payload.end_position:
            continue
        summary = str(chapter.get("edited_summary") or "").strip()
        if not summary and chapter.get("summary"):
            summary = format_chapter_summary(chapter["summary"])
        if summary:
            selected.append({"title": chapter["title"], "summary": summary})
    if not selected:
        return error_response(400, "NO_SUMMARIES", "所选范围没有可用于提炼人物卡的章节总结")
    operation_id = new_id()
    stop_event = await generation.begin(operation_id)

    async def event_stream():
        try:
            yield sse("characters_started", {
                "operation_id": operation_id,
                "chapter_count": len(selected),
            })
            queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
            task = asyncio.create_task(
                analysis_service.extract_character_cards_from_summaries(
                    selected,
                    workspace["characters"],
                    stop_event,
                    max_tokens=payload.max_tokens,
                    on_progress=lambda stage, item, total: queue.put_nowait((stage, item, total)),
                )
            )
            async for event in analysis_progress_events(task, queue, phase="characters"):
                yield event
            cards = await task
            characters = novels.replace_characters(document_id, cards)
            yield sse("done", {"characters": characters, "created_or_updated": len(cards)})
        except GenerationCancelled:
            yield sse("cancelled", {"message": "已停止人物卡提炼"})
        except asyncio.CancelledError:
            logger.warning(
                "stream_disconnected operation=character_extract operation_id=%s document_id=%s",
                operation_id,
                document_id,
            )
            raise
        except Exception as exc:
            logger.exception("character extraction failed: %s", operation_id)
            yield sse("error", {
                "code": "CHARACTER_EXTRACTION_FAILED",
                "message": "人物卡提炼失败",
                "detail": str(exc)[:500],
            })
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.patch("/api/character-events/{event_id}")
async def update_character_event(event_id: str, payload: DocumentCharacterEventUpdate):
    return novels.update_character_event(event_id, payload.model_dump(exclude_none=True))


@app.post("/api/characters/{character_id}/merge")
async def merge_character(character_id: str, payload: CharacterMergeRequest):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    return novels.merge_characters(
        character_id,
        payload.target_character_id,
        payload.keep_name,
    )


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    project_id = novels.delete_document(document_id)
    return novels.get_project(project_id)


@app.delete("/api/chapters/{chapter_id}")
async def delete_chapter(chapter_id: str):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    document_id = novels.delete_chapter(chapter_id)
    return novels.get_document_workspace(document_id)


@app.delete("/api/characters/{character_id}")
async def delete_character(character_id: str):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    document_id = novels.delete_character(character_id)
    return novels.get_document_workspace(document_id)


@app.patch("/api/facts/{fact_id}")
async def update_fact(fact_id: str, payload: StoryFactUpdate):
    return novels.update_story_fact(fact_id, payload.model_dump(exclude_none=True))


@app.delete("/api/facts/{fact_id}")
async def delete_fact(fact_id: str):
    document_id = novels.delete_story_fact(fact_id)
    return novels.get_document_workspace(document_id)


@app.delete("/api/projects/{project_id}/library")
async def clear_project_library(project_id: str):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    novels.clear_project_library(project_id)
    return novels.get_project(project_id)


@app.get("/api/projects/{project_id}/export.txt")
async def export_project_txt(project_id: str):
    name, content = novels.export_project_text(project_id)
    filename = quote(name or "小说")
    return PlainTextResponse(
        content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}.txt"},
    )


@app.get("/api/documents/{document_id}/export.txt")
async def export_document_txt(document_id: str):
    name, content = novels.export_document_text(document_id)
    filename = quote(name or "小说")
    return PlainTextResponse(
        content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}.txt"},
    )


@app.get("/api/experimental/material-system/health")
async def experimental_material_system_health():
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return {
        "status": "ok",
        "format_version": PACKAGE_FORMAT_VERSION,
        "schema_version": MATERIAL_SCHEMA_VERSION,
    }


@app.get("/api/experimental/material-system/documents/{document_id}/package")
async def export_material_package(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    package = material_service().export_document_package(document_id)
    workspace = novels.get_document_workspace(document_id)
    stem = Path(workspace["filename"]).stem or "project-analysis"
    filename = quote(f"{stem}.llm4pkg")
    return Response(
        package,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@app.get("/api/experimental/material-system/documents/{document_id}/package/report")
async def export_material_package_report(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().export_document_package_report(document_id)


@app.post("/api/experimental/material-system/packages/validate")
async def validate_material_package(
    request: Request,
    document_id: str | None = Query(default=None),
    material_layers: str | None = Query(default=None),
    chapter_start: int | None = Query(default=None),
    chapter_end: int | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    package = await read_material_package(request)
    if isinstance(package, JSONResponse):
        return package
    try:
        layer_list = [
            item.strip()
            for item in (material_layers or "").split(",")
            if item.strip()
        ] or None
        return material_service().validate_package(
            package,
            target_document_id=document_id,
            material_layers=layer_list,
            chapter_start=chapter_start,
            chapter_end=chapter_end,
        )
    except MaterialPackageError as exc:
        return error_response(400, "INVALID_MATERIAL_PACKAGE", str(exc))


@app.post("/api/experimental/material-system/packages/migrate")
async def migrate_material_package(request: Request):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    package = await read_material_package(request)
    if isinstance(package, JSONResponse):
        return package
    try:
        migrated = material_service().migrate_package_schema(package)
    except MaterialPackageError as exc:
        return error_response(400, "MATERIAL_PACKAGE_MIGRATION_FAILED", str(exc))
    filename = quote("migrated-analysis.llm4pkg")
    return Response(
        migrated,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@app.post("/api/experimental/material-system/packages/import", status_code=201)
async def import_material_package(
    request: Request,
    project_id: str = Query(default="default"),
    mode: str = Query(default="create_document"),
    document_id: str | None = Query(default=None),
    material_layers: str | None = Query(default=None),
    chapter_start: int | None = Query(default=None),
    chapter_end: int | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    package = await read_material_package(request)
    if isinstance(package, JSONResponse):
        return package
    try:
        layer_list = [
            item.strip()
            for item in (material_layers or "").split(",")
            if item.strip()
        ] or None
        return material_service().import_package(
            package,
            project_id=project_id,
            mode=mode,
            target_document_id=document_id,
            material_layers=layer_list,
            chapter_start=chapter_start,
            chapter_end=chapter_end,
        )
    except MaterialPackageError as exc:
        return error_response(400, "MATERIAL_PACKAGE_IMPORT_FAILED", str(exc))


@app.get("/api/experimental/material-system/documents/{document_id}/overview")
async def get_material_overview(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().get_material_overview(document_id)


@app.post("/api/experimental/material-system/documents/{document_id}/rebuild")
async def rebuild_material_overview(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    return material_service().rebuild_document_material(document_id)


@app.get("/api/experimental/material-system/documents/{document_id}/observations")
async def get_material_semantic_observations(
    document_id: str,
    limit: int = Query(default=40, ge=1, le=200),
    observation_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().semantic_observation_ledger(
        document_id,
        limit=limit,
        observation_type=observation_type,
        status=status,
    )


@app.patch("/api/experimental/material-system/observations/{observation_id}")
async def update_material_semantic_observation(
    observation_id: str,
    payload: MaterialSemanticObservationUpdate,
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_semantic_observation(
        observation_id,
        payload.model_dump(exclude_none=True),
    )


@app.get("/api/experimental/material-system/documents/{document_id}/timeline")
async def get_material_timeline(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().get_timeline(document_id)


@app.post("/api/experimental/material-system/documents/{document_id}/timeline/nodes", status_code=201)
async def create_material_timeline_node(document_id: str, payload: MaterialTimelineNodeCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_timeline_node(document_id, payload.model_dump(exclude_none=True))


@app.post("/api/experimental/material-system/documents/{document_id}/timeline/events", status_code=201)
async def create_material_timeline_event(document_id: str, payload: MaterialTimelineEventCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_timeline_event(document_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/timeline-events/{event_id}")
async def update_material_timeline_event(event_id: str, payload: MaterialTimelineEventUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_timeline_event(event_id, payload.model_dump(exclude_unset=True))


@app.delete("/api/experimental/material-system/timeline-events/{event_id}")
async def delete_material_timeline_event(event_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_timeline_event(event_id)


@app.patch("/api/experimental/material-system/timeline-nodes/{node_id}")
async def update_material_timeline_node(node_id: str, payload: MaterialTimelineNodeUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_timeline_node(node_id, payload.model_dump(exclude_unset=True))


@app.delete("/api/experimental/material-system/timeline-nodes/{node_id}")
async def delete_material_timeline_node(node_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_timeline_node(node_id)


@app.post("/api/experimental/material-system/documents/{document_id}/timeline/rebuild")
async def rebuild_material_timeline(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().rebuild_timeline(document_id)


@app.get("/api/experimental/material-system/documents/{document_id}/characters/entities")
async def get_material_character_entities(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().list_character_entities(document_id)


@app.post("/api/experimental/material-system/documents/{document_id}/characters/entities", status_code=201)
async def create_material_character_entity(document_id: str, payload: MaterialCharacterEntityCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_character_entity(document_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/characters/entities/{character_id}")
async def update_material_character_entity(character_id: str, payload: MaterialCharacterEntityUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_character_entity(character_id, payload.model_dump(exclude_none=True))


@app.delete("/api/experimental/material-system/characters/entities/{character_id}")
async def delete_material_character_entity(character_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_character_entity(character_id)


@app.get("/api/experimental/material-system/characters/entities/{character_id}/dependencies")
async def get_material_character_entity_dependencies(character_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().character_entity_dependencies(character_id)


@app.get("/api/experimental/material-system/characters/entities/{character_id}/snapshot")
async def get_material_character_snapshot(
    character_id: str,
    chapter_id: str | None = Query(default=None),
    chapter_position: int | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().character_snapshot(
        character_id,
        chapter_id=chapter_id,
        chapter_position=chapter_position,
    )


@app.post("/api/experimental/material-system/characters/entities/{character_id}/profiles", status_code=201)
async def create_material_character_profile(character_id: str, payload: MaterialCharacterProfileCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_character_profile(character_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/characters/profiles/{profile_id}")
async def update_material_character_profile(profile_id: str, payload: MaterialCharacterProfileUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_character_profile(profile_id, payload.model_dump(exclude_unset=True))


@app.delete("/api/experimental/material-system/characters/profiles/{profile_id}")
async def delete_material_character_profile(profile_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_character_profile(profile_id)


@app.post("/api/experimental/material-system/characters/entities/{character_id}/events", status_code=201)
async def create_material_character_event(character_id: str, payload: MaterialCharacterEventCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_character_event(character_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/characters/events/{event_id}")
async def update_material_character_event(event_id: str, payload: MaterialCharacterEventUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_character_event(event_id, payload.model_dump(exclude_unset=True))


@app.delete("/api/experimental/material-system/characters/events/{event_id}")
async def delete_material_character_event(event_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_character_event(event_id)


@app.post("/api/experimental/material-system/characters/entities/{character_id}/facts", status_code=201)
async def create_material_character_fact(character_id: str, payload: MaterialCharacterFactCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_character_fact(character_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/characters/facts/{fact_id}")
async def update_material_character_fact(fact_id: str, payload: MaterialCharacterFactUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_character_fact(fact_id, payload.model_dump(exclude_unset=True))


@app.delete("/api/experimental/material-system/characters/facts/{fact_id}")
async def delete_material_character_fact(fact_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_character_fact(fact_id)


@app.post("/api/experimental/material-system/characters/entities/{character_id}/aliases")
async def add_material_character_alias(character_id: str, payload: MaterialCharacterAliasCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().add_character_alias(
        character_id,
        payload.alias,
        alias_type=payload.alias_type,
    )


@app.post("/api/experimental/material-system/characters/entities/{character_id}/merge")
async def merge_material_character_entity(character_id: str, payload: MaterialCharacterMergeRequest):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().merge_character_entities(
        character_id,
        payload.target_character_id,
        keep_source_name_as_alias=payload.keep_source_name_as_alias,
    )


@app.post("/api/experimental/material-system/characters/entities/{character_id}/split")
async def split_material_character_entity(character_id: str, payload: MaterialCharacterSplitRequest):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().split_character_entity(
        character_id,
        payload.model_dump(exclude_none=True),
    )


@app.post("/api/experimental/material-system/documents/{document_id}/characters/entities/rebuild")
async def rebuild_material_character_entities(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().seed_character_entities(document_id)


@app.get("/api/experimental/material-system/documents/{document_id}/relationships")
async def get_material_relationships(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().list_relationships(document_id)


@app.get("/api/experimental/material-system/documents/{document_id}/relationships/network")
async def get_material_relationship_network(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().relationship_network(document_id)


@app.get("/api/experimental/material-system/documents/{document_id}/relationships/snapshot")
async def get_material_relationship_snapshot(
    document_id: str,
    chapter_id: str | None = Query(default=None),
    chapter_position: int | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().relationship_snapshot(
        document_id,
        chapter_id=chapter_id,
        chapter_position=chapter_position,
    )


@app.get("/api/experimental/material-system/characters/entities/{character_id}/relationships")
async def get_material_character_relationships(
    character_id: str,
    status: str | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().character_relationships(character_id, status=status)


@app.get("/api/experimental/material-system/characters/entities/{source_character_id}/relationships/{target_character_id}/history")
async def get_material_relationship_history(
    source_character_id: str,
    target_character_id: str,
    relation_type: str | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().relationship_history(
        source_character_id,
        target_character_id,
        relation_type=relation_type,
    )


@app.post("/api/experimental/material-system/documents/{document_id}/relationships", status_code=201)
async def create_material_relationship(document_id: str, payload: MaterialRelationshipCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_relationship(document_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/relationships/{relationship_id}")
async def update_material_relationship(relationship_id: str, payload: MaterialRelationshipUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_relationship(relationship_id, payload.model_dump(exclude_none=True))


@app.delete("/api/experimental/material-system/relationships/{relationship_id}")
async def delete_material_relationship(relationship_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_relationship(relationship_id)


@app.post("/api/experimental/material-system/relationships/{relationship_id}/events", status_code=201)
async def create_material_relationship_event(relationship_id: str, payload: MaterialRelationshipEventCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_relationship_event(relationship_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/relationships/events/{event_id}")
async def update_material_relationship_event(event_id: str, payload: MaterialRelationshipEventUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_relationship_event(event_id, payload.model_dump(exclude_unset=True))


@app.delete("/api/experimental/material-system/relationships/events/{event_id}")
async def delete_material_relationship_event(event_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_relationship_event(event_id)


@app.post("/api/experimental/material-system/documents/{document_id}/relationships/rebuild")
async def rebuild_material_relationships(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().rebuild_relationships(document_id)


@app.get("/api/experimental/material-system/documents/{document_id}/auxiliary-records")
async def get_material_auxiliary_records(document_id: str, record_type: str | None = Query(default=None)):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().list_auxiliary_records(document_id, record_type=record_type)


@app.post("/api/experimental/material-system/documents/{document_id}/auxiliary-records", status_code=201)
async def create_material_auxiliary_record(document_id: str, payload: MaterialAuxiliaryRecordCreate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().create_auxiliary_record(document_id, payload.model_dump(exclude_none=True))


@app.patch("/api/experimental/material-system/auxiliary-records/{record_id}")
async def update_material_auxiliary_record(record_id: str, payload: MaterialAuxiliaryRecordUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_auxiliary_record(record_id, payload.model_dump(exclude_unset=True))


@app.delete("/api/experimental/material-system/auxiliary-records/{record_id}")
async def delete_material_auxiliary_record(record_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().delete_auxiliary_record(record_id)


@app.get("/api/experimental/material-system/documents/{document_id}/prompt-budget-profile")
async def get_material_prompt_budget_profile(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().ensure_prompt_budget_profile(document_id)


@app.patch("/api/experimental/material-system/documents/{document_id}/prompt-budget-profile")
async def update_material_prompt_budget_profile(document_id: str, payload: MaterialPromptBudgetUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    values = payload.model_dump(exclude_none=True)
    return material_service().update_prompt_budget_profile(
        document_id,
        name=values.get("name"),
        config=values.get("config"),
    )


@app.get("/api/experimental/material-system/documents/{document_id}/review-items")
async def get_material_review_items(
    document_id: str,
    status: str | None = Query(default=None),
    review_type: str | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().list_review_items(
        document_id,
        status=status,
        review_type=review_type,
    )


@app.post("/api/experimental/material-system/documents/{document_id}/review-items/batch/resolve")
async def batch_resolve_material_review_items(document_id: str, payload: MaterialReviewBatchRequest):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().batch_update_review_items(
        document_id,
        payload.item_ids,
        "resolved",
        payload.resolution,
    )


@app.post("/api/experimental/material-system/documents/{document_id}/review-items/batch/reject")
async def batch_reject_material_review_items(document_id: str, payload: MaterialReviewBatchRequest):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().batch_update_review_items(
        document_id,
        payload.item_ids,
        "rejected",
        payload.resolution,
    )


@app.post("/api/experimental/material-system/review-items/{item_id}/resolve")
async def resolve_material_review_item(item_id: str, request: Request):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    payload = await request.json() if request.headers.get("content-length") not in {None, "0"} else {}
    return material_service().resolve_review_item(item_id, payload if isinstance(payload, dict) else {})


@app.post("/api/experimental/material-system/review-items/{item_id}/reject")
async def reject_material_review_item(item_id: str, request: Request):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    payload = await request.json() if request.headers.get("content-length") not in {None, "0"} else {}
    return material_service().reject_review_item(item_id, payload if isinstance(payload, dict) else {})


@app.post("/api/experimental/material-system/documents/{document_id}/prompt-plan")
async def build_material_prompt_plan(document_id: str, request: Request):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    payload = await request.json() if request.headers.get("content-length") not in {None, "0"} else {}
    if not isinstance(payload, dict):
        payload = {}
    return material_service().build_prompt_plan(
        document_id,
        query_text=str(payload.get("query_text") or payload.get("query") or ""),
        max_tokens=int(payload.get("max_tokens") or 8000),
    )


@app.get("/api/experimental/material-system/documents/{document_id}/snapshot")
async def get_material_current_snapshot(
    document_id: str,
    max_tokens: int = Query(default=8000, ge=1024, le=50000),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().current_material_snapshot(document_id, max_tokens=max_tokens)


@app.post("/api/legacy/projects/{project_id}/summarize", include_in_schema=False)
async def summarize_project_legacy(project_id: str, payload: SummarizeRequest):
    # Kept only so old bookmarks receive an explicit migration response. The
    # document-scoped endpoint below is required to prevent multiple TXT novels
    # from sharing summaries or character cards.
    return error_response(
        410,
        "LEGACY_SUMMARY_REMOVED",
        "旧版项目级总结已停用，请选择一个 TXT 后使用新版总结功能",
    )

    # Unreachable compatibility implementation retained for one release so old
    # stack traces remain readable while installations migrate.
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    project = novels.get_project(project_id)
    chapter_ids = set(payload.chapter_ids or [])
    targets = [
        chapter
        for chapter in project["chapters"]
        if (not chapter_ids or chapter["id"] in chapter_ids)
        and (payload.regenerate or chapter["status"] != "completed")
    ]
    job_id = new_id()
    stop_event = await generation.begin(job_id)

    async def event_stream():
        current_chapter_id: str | None = None
        try:
            yield sse(
                "job_started",
                {"job_id": job_id, "total": len(targets), "project_id": project_id},
            )
            for index, chapter_meta in enumerate(targets, start=1):
                if stop_event.is_set():
                    raise GenerationCancelled("用户停止了总结")
                current_chapter_id = chapter_meta["id"]
                chapter = novels.get_chapter(current_chapter_id)
                novels.set_chapter_status(current_chapter_id, "processing")
                yield sse(
                    "chapter_started",
                    {
                        "chapter_id": current_chapter_id,
                        "title": chapter["title"],
                        "index": index,
                        "total": len(targets),
                    },
                )
                try:
                    progress_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                    summary_task = asyncio.create_task(
                        analysis_service.summarize_chapter(
                            chapter["title"],
                            chapter["content"],
                            stop_event,
                            on_progress=lambda stage, item, total: progress_queue.put_nowait(
                                (stage, item, total)
                            ),
                            max_tokens=payload.max_tokens,
                        )
                    )
                    async for progress_event in analysis_progress_events(
                        summary_task,
                        progress_queue,
                        phase="chapter",
                        context={
                            "chapter_id": current_chapter_id,
                            "title": chapter["title"],
                            "chapter_index": index,
                            "chapter_total": len(targets),
                        },
                    ):
                        yield progress_event
                    summary = await summary_task
                    chunk_summaries = summary.pop("_chunk_summaries", [])
                    character_observations = summary.pop("_character_observations", [])
                    novels.save_chunk_summaries(current_chapter_id, chunk_summaries)
                    saved = novels.save_chapter_summary(
                        current_chapter_id, summary, character_observations
                    )
                    yield sse(
                        "chapter_completed",
                        {"chapter": saved, "index": index, "total": len(targets)},
                    )
                except GenerationCancelled:
                    novels.set_chapter_status(current_chapter_id, "pending")
                    raise
                except Exception as exc:
                    logger.exception("chapter summarization failed: %s", current_chapter_id)
                    novels.set_chapter_status(current_chapter_id, "failed", str(exc)[:1000])
                    yield sse(
                        "chapter_error",
                        {
                            "chapter_id": current_chapter_id,
                            "message": str(exc)[:500],
                            "index": index,
                            "total": len(targets),
                        },
                    )
                current_chapter_id = None

            refreshed = novels.get_project(project_id)
            completed_summaries = [
                chapter["summary"]
                for chapter in refreshed["chapters"]
                if chapter["status"] == "completed" and chapter["summary"]
            ]
            if completed_summaries:
                yield sse("project_summary_started", {})
                project_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                project_task = asyncio.create_task(
                    analysis_service.build_project_summary(
                        completed_summaries,
                        stop_event,
                        max_tokens=payload.max_tokens,
                        on_progress=lambda stage, item, total: project_queue.put_nowait(
                            (stage, item, total)
                        ),
                    )
                )
                async for progress_event in analysis_progress_events(
                    project_task, project_queue, phase="project_summary"
                ):
                    yield progress_event
                global_summary = await project_task
                novels.save_document_summary(project["documents"][0]["id"], global_summary)
                yield sse("project_summary_completed", {"global_summary": global_summary})

                yield sse("characters_started", {})
                character_observations = novels.get_document_character_observations(
                    project["documents"][0]["id"]
                )
                character_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                character_task = asyncio.create_task(
                    analysis_service.extract_character_cards(
                        character_observations,
                        stop_event,
                        max_tokens=max(8192, payload.max_tokens),
                        on_progress=lambda stage, item, total: character_queue.put_nowait(
                            (stage, item, total)
                        ),
                    )
                )
                async for progress_event in analysis_progress_events(
                    character_task, character_queue, phase="characters"
                ):
                    yield progress_event
                cards = await character_task
                characters = novels.replace_characters(project["documents"][0]["id"], cards)
                yield sse("characters_completed", {"characters": characters})

            yield sse("done", {"project": novels.get_project(project_id)})
        except GenerationCancelled:
            if current_chapter_id:
                novels.set_chapter_status(current_chapter_id, "pending")
            yield sse("cancelled", {"project": novels.get_project(project_id)})
        except asyncio.CancelledError:
            if current_chapter_id:
                novels.set_chapter_status(current_chapter_id, "pending")
            raise
        except Exception as exc:
            logger.exception("project summarization failed")
            yield sse(
                "error",
                {
                    "code": "SUMMARIZATION_FAILED",
                    "message": "小说总结任务失败",
                    "detail": str(exc)[:500],
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/projects/{project_id}/summarize")
async def summarize_project(project_id: str, payload: SummarizeRequest):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    document_id = payload.document_id
    if payload.resume_job_id:
        job = novels.get_analysis_job(payload.resume_job_id)
        document_id = job["document_id"]
        start_position, end_position = job["start_position"], job["end_position"]
        max_tokens, regenerate = job["max_tokens"], bool(job["regenerate"])
        novels.update_analysis_job(job["id"], status="running", error_message=None)
    else:
        if not document_id:
            return error_response(400, "DOCUMENT_REQUIRED", "请先选择要处理的 TXT 小说")
        workspace = novels.get_document_workspace(document_id)
        positions = [chapter["position"] for chapter in workspace["chapters"]]
        if not positions:
            return error_response(400, "NO_CHAPTERS", "这个 TXT 下没有章节")
        start_position = payload.start_position or min(positions)
        end_position = payload.end_position or max(positions)
        max_tokens, regenerate = payload.max_tokens, payload.regenerate
        selected = [p for p in positions if start_position <= p <= end_position]
        job = novels.create_analysis_job(
            document_id, start_position, end_position, len(selected), regenerate, max_tokens
        )
    workspace = novels.get_document_workspace(document_id)
    chapter_ids = set(payload.chapter_ids or [])
    targets = [
        chapter for chapter in workspace["chapters"]
        if start_position <= chapter["position"] <= end_position
        and (not chapter_ids or chapter["id"] in chapter_ids)
        and (regenerate or chapter["status"] != "completed")
    ]
    stop_event = await generation.begin(job["id"])

    async def event_stream():
        processed = int(job.get("processed_chapters") or 0)
        current_chapter_id: str | None = None
        had_errors = False

        try:
            yield sse("job_started", {"job": novels.get_analysis_job(job["id"]), "total": len(targets)})
            for chapter_index, chapter_meta in enumerate(targets, start=1):
                if stop_event.is_set():
                    raise GenerationCancelled("用户停止了总结")
                current_chapter_id = chapter_meta["id"]
                if regenerate:
                    novels.reset_chapter_analysis(current_chapter_id)
                chapter = novels.get_chapter(current_chapter_id)
                novels.set_chapter_status(current_chapter_id, "processing")
                novels.update_analysis_job(
                    job["id"], current_chapter_id=current_chapter_id,
                    current_chunk_position=0, processed_chapters=processed,
                )
                yield sse("chapter_started", {
                    "chapter_id": current_chapter_id, "title": chapter["title"],
                    "index": chapter_index, "total": len(targets),
                })
                try:
                    queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                    task = asyncio.create_task(analysis_service.summarize_chapter(
                        chapter["title"], chapter["content"], stop_event,
                        max_tokens=max_tokens,
                        on_progress=lambda stage, item, total: queue.put_nowait(
                            (stage, item, total)
                        ),
                    ))
                    async for event in analysis_progress_events(
                        task, queue, phase="chapter",
                        context={"title": chapter["title"], "chapter_index": chapter_index,
                                 "chapter_total": len(targets)},
                    ):
                        yield event
                    summary = await task
                    saved = novels.save_chapter_summary(current_chapter_id, summary, [])
                    processed += 1
                    novels.update_analysis_job(
                        job["id"], processed_chapters=processed,
                        current_chunk_position=1,
                    )
                    yield sse("chapter_completed", {
                        "chapter": saved, "index": chapter_index, "total": len(targets)
                    })
                except GenerationCancelled:
                    novels.set_chapter_status(current_chapter_id, "pending")
                    raise
                except Exception as exc:
                    had_errors = True
                    logger.exception("chapter analysis failed: %s", current_chapter_id)
                    novels.set_chapter_status(current_chapter_id, "failed", str(exc)[:1000])
                    yield sse("chapter_error", {
                        "chapter_id": current_chapter_id, "message": str(exc)[:500],
                        "index": chapter_index, "total": len(targets),
                    })
                current_chapter_id = None

            refreshed = novels.get_document_workspace(document_id)
            completed_summaries = [c["summary"] for c in refreshed["chapters"] if c["status"] == "completed" and c["summary"]]
            if completed_summaries:
                yield sse("project_summary_started", {})
                global_summary = await analysis_service.build_project_summary(
                    completed_summaries, stop_event, max_tokens=max_tokens
                )
                novels.save_document_summary(document_id, global_summary)
                yield sse("project_summary_completed", {"global_summary": global_summary})
            status = "failed" if had_errors else "completed"
            novels.update_analysis_job(job["id"], status=status, error_message="部分章节失败" if had_errors else None)
            yield sse("done", {
                "workspace": novels.get_document_workspace(document_id),
                "job": novels.get_analysis_job(job["id"]),
            })
        except GenerationCancelled:
            if current_chapter_id:
                novels.set_chapter_status(current_chapter_id, "pending")
            novels.update_analysis_job(job["id"], status="paused", error_message="用户暂停")
            yield sse("cancelled", {
                "workspace": novels.get_document_workspace(document_id),
                "job": novels.get_analysis_job(job["id"]),
            })
        except asyncio.CancelledError:
            if current_chapter_id:
                novels.set_chapter_status(current_chapter_id, "pending")
            novels.update_analysis_job(job["id"], status="paused", error_message="连接中断")
            logger.warning(
                "stream_disconnected operation=document_summary job_id=%s document_id=%s chapter_id=%s",
                job["id"], document_id, current_chapter_id,
            )
            raise
        except Exception as exc:
            logger.exception("document analysis job failed: %s", job["id"])
            if current_chapter_id:
                novels.set_chapter_status(current_chapter_id, "failed", str(exc)[:1000])
            novels.update_analysis_job(job["id"], status="failed", error_message=str(exc)[:1000])
            yield sse("error", {
                "code": "ANALYSIS_FAILED",
                "message": "小说资料分析失败，已保存完成的断点",
                "detail": str(exc)[:500],
                "workspace": novels.get_document_workspace(document_id),
            })
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/projects/{project_id}/append")
async def append_project_content(project_id: str, payload: ProjectAppendRequest):
    if not payload.summarize_now:
        try:
            appended = novels.append_content(
                project_id, payload.content, chapter_id=payload.chapter_id,
                document_id=payload.document_id, title=payload.title,
                source_candidate_id=payload.source_candidate_id,
            )
        except ValueError as exc:
            if str(exc) == "candidate_already_appended":
                return error_response(409, "ALREADY_APPENDED", "这版正文已经加入过资料库")
            if str(exc) == "document_required":
                return error_response(400, "DOCUMENT_REQUIRED", "请先选择目标 TXT 小说")
            raise
        return {
            "saved": True,
            "summarized": False,
            "chapter": appended["chapter"],
            "workspace": novels.get_document_workspace(appended["document_id"]),
        }
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    job_id = new_id()
    stop_event = await generation.begin(job_id)
    try:
        appended = novels.append_content(
            project_id,
            payload.content,
            chapter_id=payload.chapter_id,
            document_id=payload.document_id,
            title=payload.title,
            source_candidate_id=payload.source_candidate_id,
        )
    except ValueError as exc:
        generation.finish()
        if str(exc) == "candidate_already_appended":
            return error_response(409, "ALREADY_APPENDED", "这版正文已经加入过资料库")
        raise
    except Exception:
        generation.finish()
        raise

    async def event_stream():
        chapter = appended["chapter"]
        document_id = appended["document_id"]
        try:
            yield sse("append_saved", {"chapter": chapter, "job_id": job_id})
            increment_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
            increment_task = asyncio.create_task(
                analysis_service.summarize_increment(
                    chapter["title"],
                    appended["previous_summary"],
                    appended["new_content"],
                    stop_event,
                    max_tokens=payload.max_tokens,
                    on_progress=lambda stage, item, total: increment_queue.put_nowait(
                        (stage, item, total)
                    ),
                )
            )
            async for progress_event in analysis_progress_events(
                increment_task,
                increment_queue,
                phase="increment",
                context={"chapter_id": chapter["id"], "title": chapter["title"]},
            ):
                yield progress_event
            summary = await increment_task
            updated_chapter = novels.save_chapter_summary(
                chapter["id"],
                summary,
                [],
                append_observations=True,
            )
            yield sse("chapter_completed", {"chapter": updated_chapter})

            refreshed = novels.get_document_workspace(document_id)
            completed_summaries = [
                item["summary"]
                for item in refreshed["chapters"]
                if item["status"] == "completed" and item["summary"]
            ]
            if completed_summaries:
                yield sse("project_summary_started", {})
                project_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                project_task = asyncio.create_task(
                    analysis_service.build_project_summary(
                        completed_summaries,
                        stop_event,
                        max_tokens=payload.max_tokens,
                        on_progress=lambda stage, item, total: project_queue.put_nowait(
                            (stage, item, total)
                        ),
                    )
                )
                async for progress_event in analysis_progress_events(
                    project_task, project_queue, phase="project_summary"
                ):
                    yield progress_event
                global_summary = await project_task
                novels.save_document_summary(document_id, global_summary)
                yield sse("project_summary_completed", {"global_summary": global_summary})
            yield sse("done", {"workspace": novels.get_document_workspace(document_id)})
        except GenerationCancelled:
            novels.mark_increment_failed(
                chapter["id"], "增量总结已停止；正文已保存，可重新总结本章"
            )
            yield sse("cancelled", {"workspace": novels.get_document_workspace(document_id)})
        except asyncio.CancelledError:
            novels.mark_increment_failed(
                chapter["id"], "连接中断；正文已保存，可重新总结本章"
            )
            logger.warning(
                "stream_disconnected operation=increment_summary job_id=%s document_id=%s chapter_id=%s",
                job_id, document_id, chapter["id"],
            )
            raise
        except Exception as exc:
            logger.exception("incremental project update failed")
            novels.mark_increment_failed(
                chapter["id"], f"增量总结失败：{str(exc)[:500]}"
            )
            yield sse(
                "error",
                {
                    "code": "INCREMENT_SUMMARY_FAILED",
                    "message": "正文已经保存，但增量总结失败；可在资料库重新总结本章",
                    "detail": str(exc)[:500],
                    "workspace": novels.get_document_workspace(document_id),
                },
            )
        finally:
            generation.finish()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def outline_instruction(user_instruction: str, include_hook: bool = True) -> str:
    ending_rule = (
        "本章需要结尾钩子：最后一场必须留下明确的新问题、危险、发现或行动驱动力，吸引读者进入下一章。"
        if include_hook
        else "本章不需要结尾钩子：最后一场应完整收束本章目标，不要强行制造悬念、突发危险或未完句。"
    )
    return f"""请把用户给出的下一章大方向拆成“场景编排器”可用的场景卡。
用户的大方向：{user_instruction}
结尾要求：{ending_rule}

只输出一个合法 JSON 对象，不要 Markdown，不要代码块，不要注释，不要尾随逗号，不要解释你的工作过程。
必须承接已有前文、人物卡和最近对话，不能改动已确认设定。

JSON schema 必须如下：
{{
  "chapter_goal": "一句话概括本章推进目标",
  "scenes": [
    {{
      "id": "S01",
      "title": "场景标题",
      "purpose": "为什么要写这个场景",
      "entry": "从哪里开始，包含地点、状态和承接点",
      "beats": ["必须发生的关键动作或信息"],
      "exit": "写到什么状态结束",
      "constraints": ["不能发生什么，不能揭露什么，人物不能突然变成什么状态"]
    }}
  ],
  "chapter_ending": "最后一个场景如何形成钩子或完整落点",
  "polish_checklist": ["场景衔接", "时间连续", "角色称呼", "物品位置", "对话信息重复", "结尾完整度"]
}}

规则：
- scenes 通常 4 到 8 个。
- 每个 scenes 项必须有 id/title/purpose/entry/beats/exit/constraints。
- id 按 S01、S02、S03 递增。
- beats 是必须完成的动作链，不要写成抽象主题。
- constraints 只写硬禁令和人物状态边界。
- 不要设置 target_tokens、max_tokens 或 budget；场景正文会按内容需要自然展开。
- {ending_rule}
- 场景之间要有因果推进，避免把同一个动作拆成多个空场景。"""


async def prepare_outline_preview(
    conversation_id: str, payload: OutlineGenerateRequest
):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    conversation = database.get_conversation(conversation_id)
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    outline_max_tokens = min(
        16_384,
        max(1024, int(generation_settings["max_tokens"] * 1.5)),
    )
    outline_generation_settings = {**generation_settings, "max_tokens": outline_max_tokens}
    context = await build_fitted_context(
        conversation_id=conversation_id,
        system_prompt=conversation["system_prompt"],
        pinned_context=conversation["pinned_context"],
        style_guide=conversation.get("style_guide", ""),
        style_lexicon=conversation.get("style_lexicon", ""),
        history=selected_history(conversation),
        current_user_content=outline_instruction(payload.instruction, payload.include_hook),
        max_output_tokens=outline_max_tokens,
        include_outline=False,
    )
    preview_id = new_id()
    try:
        stop_event = await generation.begin(preview_id)
    except RuntimeError:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    return stream_outline_preview(
        preview_id=preview_id,
        context=context,
        generation_settings=outline_generation_settings,
        stop_event=stop_event,
    )


@app.get("/api/conversations/{conversation_id}/outline")
async def get_active_outline(conversation_id: str):
    database.get_conversation(conversation_id)
    return novels.find_latest_outline(conversation_id)


@app.post("/api/conversations/{conversation_id}/outline/generate")
async def generate_outline(
    conversation_id: str,
    payload: OutlineGenerateRequest,
    new_group: bool = False,
):
    await ensure_model_ready()
    database.get_conversation(conversation_id)
    return await prepare_outline_preview(conversation_id, payload)


@app.post("/api/outlines/{outline_id}/regenerate")
async def regenerate_outline(outline_id: str, payload: OutlineGenerateRequest):
    await ensure_model_ready()
    outline = novels.get_outline(outline_id)
    return await prepare_outline_preview(outline["conversation_id"], payload)


@app.post("/api/conversations/{conversation_id}/outline/candidates", status_code=201)
async def save_outline_candidate(
    conversation_id: str, payload: OutlineCandidateSaveRequest
):
    conversation = database.get_conversation(conversation_id)
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    try:
        validate_outline_json(payload.content)
    except SceneCardFormatError as exc:
        return error_response(400, "OUTLINE_JSON_INVALID", "场景卡必须是合法 JSON", str(exc))
    try:
        return novels.save_outline_candidate(
            conversation_id,
            outline_id=payload.outline_id,
            instruction=payload.instruction,
            content=payload.content,
            settings=generation_settings,
            seed=generation_settings["seed"],
            select=payload.select,
        )
    except ValueError:
        return error_response(400, "OUTLINE_MISMATCH", "场景卡不属于当前对话")


@app.patch("/api/outlines/{outline_id}")
async def update_outline(outline_id: str, payload: OutlineUpdateRequest):
    return novels.update_outline(outline_id, payload.enabled)


@app.put("/api/outlines/{outline_id}/selection")
async def select_outline(outline_id: str, payload: SelectionRequest):
    try:
        outline = novels.get_outline(outline_id)
        selected = next(
            (item for item in outline["candidates"] if item["id"] == payload.candidate_id),
            None,
        )
        if selected is None:
            return error_response(400, "OUTLINE_NOT_SELECTABLE", "这个场景卡候选不能被选用")
        validate_outline_json(selected.get("edited_content") or selected.get("content") or "")
        return novels.select_outline_candidate(outline_id, payload.candidate_id)
    except SceneCardFormatError as exc:
        return error_response(400, "OUTLINE_JSON_INVALID", "场景卡必须是合法 JSON", str(exc))
    except ValueError:
        return error_response(400, "OUTLINE_NOT_SELECTABLE", "这个场景卡候选不能被选用")


@app.patch("/api/outline-candidates/{candidate_id}")
async def edit_outline_candidate(
    candidate_id: str, payload: OutlineCandidateEditRequest
):
    try:
        validate_outline_json(payload.content)
    except SceneCardFormatError as exc:
        return error_response(400, "OUTLINE_JSON_INVALID", "场景卡必须是合法 JSON", str(exc))
    return novels.edit_outline_candidate(candidate_id, payload.content)


@app.delete("/api/outline-candidates/{candidate_id}")
async def delete_outline_candidate(candidate_id: str):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成任务")
    return novels.delete_outline_candidate(candidate_id)


@app.delete("/api/outlines/{outline_id}", status_code=204)
async def delete_outline(outline_id: str):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成任务")
    novels.delete_outline(outline_id)
    return Response(status_code=204)


@app.post("/api/conversations/{conversation_id}/scene-workflow")
async def generate_scene_workflow(conversation_id: str, payload: SceneWorkflowRequest):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    conversation = database.get_conversation(conversation_id)
    outline_text = selected_outline_content(novels.find_latest_outline(conversation_id))
    if not outline_text:
        return error_response(400, "SCENE_CARD_REQUIRED", "请先在场景编排器里选用一版场景卡")
    try:
        scenes = parse_scene_cards(outline_text)
    except SceneCardFormatError as exc:
        return error_response(400, "OUTLINE_JSON_INVALID", "场景卡必须是合法 JSON", str(exc))
    if not scenes:
        return error_response(400, "SCENE_CARD_EMPTY", "当前场景卡没有可写作的场景")
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    extra_instruction = payload.instruction.strip()
    user_content = "一键启动编排流程"
    if extra_instruction:
        user_content = f"{user_content}：{extra_instruction}"
    exchange, candidate = database.create_exchange_with_candidate(
        conversation_id,
        user_content,
        generation_settings,
        generation_settings["seed"],
    )
    try:
        stop_event = await generation.begin(candidate["id"])
    except RuntimeError:
        database.finalize_candidate(
            candidate["id"],
            status="failed",
            content="",
            reasoning="",
            error_message="已有生成任务",
        )
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    return stream_scene_workflow(
        exchange=exchange,
        candidate=candidate,
        conversation_id=conversation_id,
        outline_text=outline_text,
        scenes=scenes,
        extra_instruction=extra_instruction,
        generation_settings=generation_settings,
        stop_event=stop_event,
    )


@app.post("/api/conversations/{conversation_id}/scene-workflow/fragment")
async def regenerate_scene_fragment(
    conversation_id: str,
    payload: SceneFragmentRegenerateRequest,
):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    conversation = database.get_conversation(conversation_id)
    if payload.scene_index >= len(payload.scenes):
        return error_response(400, "SCENE_INDEX_INVALID", "要重生成的场景不存在")
    source_exchange = next(
        (
            exchange
            for exchange in conversation.get("exchanges", [])
            if any(
                candidate.get("id") == payload.candidate_id
                for candidate in exchange.get("candidates", [])
            )
        ),
        None,
    )
    if source_exchange is None:
        return error_response(404, "CANDIDATE_NOT_FOUND", "要重写的编排候选不存在")
    scenes = [scene.model_dump() for scene in payload.scenes]
    outline_text = payload.outline_text.strip() or "\n\n".join(scene["card"] for scene in scenes)
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    try:
        stop_event = await generation.begin(payload.candidate_id)
    except RuntimeError:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    return stream_scene_fragment_regeneration(
        conversation_id=conversation_id,
        source_exchange_id=source_exchange["id"],
        candidate_id=payload.candidate_id,
        outline_text=outline_text,
        scenes=scenes,
        scene_index=payload.scene_index,
        extra_instruction=payload.instruction.strip(),
        generation_settings=generation_settings,
        stop_event=stop_event,
    )


@app.post("/api/conversations/{conversation_id}/scene-workflow/polish")
async def polish_scene_workflow(
    conversation_id: str,
    payload: SceneWorkflowPolishRequest,
):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    conversation = database.get_conversation(conversation_id)
    scenes = [scene.model_dump() for scene in payload.scenes]
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    try:
        stop_event = await generation.begin(payload.candidate_id)
    except RuntimeError:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    return stream_scene_workflow_polish(
        conversation_id=conversation_id,
        candidate_id=payload.candidate_id,
        scenes=scenes,
        generation_settings=generation_settings,
        stop_event=stop_event,
    )


@app.post("/api/conversations/{conversation_id}/scene-workflow/accept")
async def accept_scene_workflow(
    conversation_id: str,
    payload: SceneWorkflowAcceptRequest,
):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    draft = strip_scene_headings(
        assemble_scene_fragments([scene.model_dump() for scene in payload.scenes])
    )
    if not draft:
        return error_response(400, "SCENE_DRAFT_EMPTY", "当前没有可直接采用的场景正文")
    conversation = database.get_conversation(conversation_id)
    candidate = next(
        (
            item
            for exchange in conversation.get("exchanges", [])
            for item in exchange.get("candidates", [])
            if item.get("id") == payload.candidate_id
        ),
        None,
    )
    if candidate is None:
        return error_response(404, "CANDIDATE_NOT_FOUND", "没有找到对应的场景正文")
    exchange = database.update_candidate_content(
        payload.candidate_id,
        content=draft,
        reasoning=str(candidate.get("reasoning_content") or ""),
        expected_conversation_id=conversation_id,
    )
    logger.info(
        "scene_workflow_accepted_without_polish conversation_id=%s candidate_id=%s chars=%s",
        conversation_id,
        payload.candidate_id,
        len(draft),
    )
    return {
        "candidate_id": payload.candidate_id,
        "exchange": exchange,
        "finish_reason": "scene_workflow_accepted",
    }


@app.post("/api/conversations/{conversation_id}/generate")
async def generate(conversation_id: str, payload: GenerateRequest):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    conversation = database.get_conversation(conversation_id)
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    exchange, candidate = database.create_exchange_with_candidate(
        conversation_id,
        payload.content,
        generation_settings,
        generation_settings["seed"],
    )
    try:
        stop_event = await generation.begin(candidate["id"])
    except RuntimeError:
        database.finalize_candidate(
            candidate["id"],
            status="failed",
            content="",
            reasoning="",
            error_message="已有生成任务",
        )
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    try:
        context = await context_for_exchange(
            exchange["id"], generation_settings["max_tokens"]
        )
    except Exception:
        database.finalize_candidate(
            candidate["id"], status="failed", content="", reasoning="",
            error_message="上下文构建失败",
        )
        generation.finish()
        raise
    return stream_candidate(
        exchange=exchange,
        candidate=candidate,
        context=context,
        generation_settings=generation_settings,
        stop_event=stop_event,
    )


@app.post("/api/exchanges/{exchange_id}/regenerate")
async def regenerate(exchange_id: str, payload: RegenerateRequest):
    await ensure_model_ready()
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    exchange = database.get_exchange(exchange_id)
    if database.count_completed_candidates(exchange_id) >= settings.max_candidates:
        return error_response(
            409,
            "CANDIDATE_LIMIT_REACHED",
            f"这个回复已经保存了 {settings.max_candidates} 个候选版本",
        )
    conversation = database.get_conversation(exchange["conversation_id"])
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    updated_exchange, candidate = database.create_candidate(
        exchange_id, generation_settings, generation_settings["seed"]
    )
    try:
        stop_event = await generation.begin(candidate["id"])
    except RuntimeError:
        database.finalize_candidate(
            candidate["id"],
            status="failed",
            content="",
            reasoning="",
            error_message="已有生成任务",
        )
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有内容正在生成")
    try:
        context = await context_for_exchange(
            exchange_id, generation_settings["max_tokens"]
        )
    except Exception:
        database.finalize_candidate(
            candidate["id"], status="failed", content="", reasoning="",
            error_message="上下文构建失败",
        )
        generation.finish()
        raise
    return stream_candidate(
        exchange=updated_exchange,
        candidate=candidate,
        context=context,
        generation_settings=generation_settings,
        stop_event=stop_event,
    )


@app.post("/api/generation/stop")
async def stop_generation():
    stopped = generation.stop()
    return {"stopping": stopped}


@app.put("/api/exchanges/{exchange_id}/selection")
async def select_candidate(exchange_id: str, payload: SelectionRequest):
    try:
        return database.select_candidate(exchange_id, payload.candidate_id)
    except RuntimeError as exc:
        if str(exc) == "branch_required":
            return error_response(
                409,
                "BRANCH_REQUIRED",
                "这条回复后面已有内容，请从此版本创建新分支",
            )
        raise
    except ValueError:
        return error_response(400, "CANDIDATE_NOT_SELECTABLE", "这个候选版本不能被选用")


@app.post("/api/exchanges/{exchange_id}/branch", status_code=201)
async def create_branch(exchange_id: str, payload: BranchRequest):
    try:
        return database.create_branch(exchange_id, payload.candidate_id)
    except ValueError:
        return error_response(400, "CANDIDATE_NOT_SELECTABLE", "这个候选版本不能用于分支")


def conversation_markdown(
    conversation: dict[str, Any],
    include_all: bool,
    outline: dict[str, Any] | None = None,
) -> str:
    lines = [f"# {conversation['title']}", ""]
    if conversation["system_prompt"]:
        lines.extend(["## 系统提示词", "", conversation["system_prompt"], ""])
    if conversation["pinned_context"]:
        lines.extend(["## 固定创作资料", "", conversation["pinned_context"], ""])
    if conversation.get("style_guide"):
        lines.extend(["## 词汇风格", "", conversation["style_guide"], ""])
    if conversation.get("style_lexicon"):
        lines.extend(["## 词表白名单 / 优先用词", "", conversation["style_lexicon"], ""])
    if outline and outline.get("selected_candidate_id"):
        selected_outline = next(
            (
                item for item in outline.get("candidates", [])
                if item["id"] == outline["selected_candidate_id"]
                and item.get("status") == "completed"
            ),
            None,
        )
        if selected_outline:
            outline_content = selected_outline.get("edited_content") or selected_outline.get("content") or ""
            if outline_content.strip():
                outline_state = "已启用" if outline.get("enabled") else "未启用"
                lines.extend([
                    "## 下一章场景卡",
                    "",
                    f"状态：{outline_state}",
                    "",
                    outline_content,
                    "",
                ])
    lines.extend(["## 对话", ""])
    for exchange in conversation["exchanges"]:
        lines.extend(["### 我", "", exchange["user_content"], ""])
        completed = [item for item in exchange["candidates"] if item["status"] == "completed"]
        if include_all:
            for index, candidate in enumerate(completed, start=1):
                marker = "（已选用）" if candidate["id"] == exchange["selected_candidate_id"] else ""
                lines.extend([f"### 候选 {index}{marker}", "", candidate["content"], ""])
        else:
            selected = next(
                (item for item in completed if item["id"] == exchange["selected_candidate_id"]),
                None,
            )
            if selected:
                lines.extend(["### Novel-factory", "", selected["content"], ""])
    return "\n".join(lines).rstrip() + "\n"


@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: str = Query(default="markdown", pattern="^(markdown|json)$"),
    include_all: bool = False,
):
    conversation = database.get_conversation(conversation_id)
    outline = novels.find_latest_outline(conversation_id)
    filename = quote(conversation["title"] or "Novel-factory")
    if format == "json":
        content = json.dumps(
            {**conversation, "outline": outline},
            ensure_ascii=False,
            indent=2,
        )
        return Response(
            content=content,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}.json"},
        )
    return PlainTextResponse(
        conversation_markdown(
            conversation,
            include_all,
            outline,
        ),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}.md"},
    )


frontend_dir = settings.project_root / "frontend"
app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-store"})
