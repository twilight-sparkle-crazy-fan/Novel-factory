from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_GENERATION_SETTINGS, get_settings
from .context_builder import ContextResult, build_messages
from .database import Database, new_id
from .analysis_service import NovelAnalysisService
from .llama_client import GenerationCancelled, LlamaClient, LlamaClientError
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
    ChapterUpdate,
    CharacterUpdate,
    ConversationCreate,
    ConversationUpdate,
    ContextCountRequest,
    DocumentUpdate,
    GenerateRequest,
    MaterialCharacterAliasCreate,
    MaterialCharacterEntityUpdate,
    MaterialCharacterMergeRequest,
    MaterialPromptBudgetUpdate,
    MaterialRelationshipUpdate,
    MaterialTimelineEventUpdate,
    OutlineCandidateEditRequest,
    OutlineCandidateSaveRequest,
    OutlineGenerateRequest,
    OutlineUpdateRequest,
    ProjectAppendRequest,
    ProjectUpdate,
    RegenerateRequest,
    RuntimeContextRequest,
    SelectionRequest,
    SummarizeRequest,
    StoryFactUpdate,
)
from .text_import import decode_text, normalize_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llm4chat")
logging.getLogger("httpx").setLevel(logging.WARNING)

settings = get_settings()
database = Database(settings.database_path)
llama_process = LlamaProcessManager(settings)
llama_client = LlamaClient(settings)
novels = NovelRepository(database)
analysis_service = NovelAnalysisService(llama_client)
MIN_COMPLETION_TOKENS = 2000
MAX_AUTO_CONTINUATIONS = 3
MAX_MATERIAL_PACKAGE_BYTES = 200 * 1024 * 1024


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
    legacy_assets = novels.get_prompt_context(
        conversation_id,
        include_outline=include_outline,
        query_text=query_text,
    )
    if not settings.experimental_material_system:
        return legacy_assets
    conversation = database.get_conversation(conversation_id)
    document_id = conversation.get("document_id")
    if not document_id:
        return legacy_assets
    try:
        with database.connect() as connection:
            document = connection.execute(
                "SELECT * FROM source_documents WHERE id = ?", (document_id,)
            ).fetchone()
        if document is None or not bool(document["library_enabled"]):
            return legacy_assets
        plan = material_service().build_prompt_plan(
            document_id,
            query_text=query_text,
            max_tokens=max(1024, material_budget_tokens),
        )
    except Exception:
        logger.exception("experimental material prompt planning failed")
        return legacy_assets

    sections = {
        section["key"]: section
        for section in plan.get("sections", [])
        if section.get("included") and str(section.get("content") or "").strip()
    }

    def render_sections(*keys: str) -> str:
        blocks = []
        for key in keys:
            section = sections.get(key)
            if not section:
                continue
            blocks.append(f"{section['label']}：\n{str(section['content']).strip()}")
        return "\n\n".join(blocks)

    return {
        "project_summary": (
            render_sections("project_summary")
            if bool(document["summary_enabled"])
            else ""
        ),
        "recent_chapters": (
            render_sections(
                "current_timeline_node",
                "recent_chapter_summaries",
                "timeline_events",
            )
            if bool(document["recent_chapters_enabled"])
            else ""
        ),
        "characters": (
            render_sections("character_snapshots", "relationships")
            if bool(document["characters_enabled"])
            else ""
        ),
        "facts": render_sections("facts") if bool(document["facts_enabled"]) else "",
        "outline": legacy_assets.get("outline", ""),
    }


async def read_material_package(request: Request) -> bytes | JSONResponse:
    data = await request.body()
    if not data:
        return error_response(400, "EMPTY_PACKAGE", "分析包为空")
    if len(data) > MAX_MATERIAL_PACKAGE_BYTES:
        return error_response(413, "PACKAGE_TOO_LARGE", "分析包不能超过 200 MB")
    return data


async def ensure_model_ready() -> None:
    if not await llama_process.is_healthy():
        info = await llama_process.runtime_info(check_health=False)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "LLAMA_SERVER_UNAVAILABLE",
                "message": info["message"],
            },
        )


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
    budget = max(1024, llama_process.context_size - max_output_tokens - 384)
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
            n_ctx=llama_process.context_size,
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
                "message": "固定创作资料、人物卡或大纲超过了当前上下文预算",
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
                    "context_size": llama_process.context_size,
                    "min_completion_tokens": MIN_COMPLETION_TOKENS,
                },
            )

            messages = context.messages
            active_generation_settings = dict(generation_settings)
            auto_continue_count = 0
            while True:
                round_completion_tokens: int | None = None
                async for event in llama_client.stream_chat(
                    messages, active_generation_settings, stop_event
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
                        round_completion_tokens = usage.get(
                            "completion_tokens", round_completion_tokens
                        )
                    elif event_type == "timings":
                        usage = event["value"]
                        if prompt_tokens is None:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        round_completion_tokens = usage.get(
                            "completion_tokens", round_completion_tokens
                        )
                    elif event_type == "finish_reason":
                        finish_reason = event["value"]
                    now = time.monotonic()
                    if now - last_flush >= 0.8:
                        database.update_candidate_draft(candidate["id"], content, reasoning)
                        last_flush = now

                completion_tokens += int(round_completion_tokens or 0)
                database.update_candidate_draft(candidate["id"], content, reasoning)
                if (
                    completion_tokens >= MIN_COMPLETION_TOKENS
                    or auto_continue_count >= MAX_AUTO_CONTINUATIONS
                    or not content.strip()
                ):
                    break
                auto_continue_count += 1
                remaining = max(0, MIN_COMPLETION_TOKENS - completion_tokens)
                continuation_settings = {
                    **active_generation_settings,
                    "max_tokens": min(
                        16_384,
                        max(
                            int(active_generation_settings["max_tokens"]),
                            remaining + 512,
                            1024,
                        ),
                    ),
                }
                yield sse(
                    "auto_continue_started",
                    {
                        "attempt": auto_continue_count,
                        "completion_tokens": completion_tokens,
                        "target_completion_tokens": MIN_COMPLETION_TOKENS,
                    },
                )
                messages = [
                    *context.messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "隐藏续写指令：上一段还未达到本次输出长度目标。"
                            "请从最后一句自然接着写正文，不要总结，不要解释，"
                            "不要重写已输出内容，不要使用“继续”“下面”等过渡提示。"
                            f"保持同一视角、文风、人物状态和场景连续性；"
                            f"当前已输出约 {completion_tokens} tokens，目标至少 "
                            f"{MIN_COMPLETION_TOKENS} tokens。"
                        ),
                    },
                ]
                active_generation_settings = continuation_settings

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
                    "context_size": llama_process.context_size,
                    "max_tokens": generation_settings["max_tokens"],
                },
            )
            async for event in llama_client.stream_chat(
                context.messages, generation_settings, stop_event
            ):
                if event["type"] == "content_delta":
                    content += event["text"]
                    yield sse("content_delta", {"text": event["text"]})
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
            yield sse(
                "error",
                {
                    "code": "OUTLINE_GENERATION_FAILED",
                    "message": "大纲生成失败，可以重新尝试",
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
    persisted_context = database.get_app_setting("context_size")
    if persisted_context:
        try:
            llama_process.set_context_size(int(persisted_context))
        except (TypeError, ValueError):
            logger.warning("ignoring invalid persisted context size: %s", persisted_context)
    startup_task: asyncio.Task[Any] | None = None
    if settings.auto_start_llama:
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
    await llama_process.stop()


app = FastAPI(title="Novel-factory", version="0.1.0", lifespan=lifespan)


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


@app.get("/api/runtime")
async def runtime():
    info = await llama_process.runtime_info()
    info["generation_in_progress"] = generation.busy
    return info


@app.post("/api/runtime/start")
async def start_runtime():
    try:
        return await llama_process.start()
    except LlamaProcessError as exc:
        return error_response(500, "LLAMA_START_FAILED", "模型服务启动失败", str(exc))


@app.post("/api/runtime/stop")
async def stop_runtime():
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成")
    await llama_process.stop()
    return {"status": "stopped"}


@app.post("/api/runtime/context")
async def change_runtime_context(payload: RuntimeContextRequest):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "请先停止当前生成或总结任务")
    if payload.context_size == llama_process.context_size:
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
    max_output = int(conversation["generation_settings"].get("max_tokens", 1600))
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
        "context_size": llama_process.context_size,
        "reserved_output_tokens": reserved,
        "available_tokens": max(
            0, llama_process.context_size - reserved - int(context.prompt_tokens or 0)
        ),
        "trimmed_exchange_count": context.trimmed_exchange_count,
        "source_characters": {
            key: len(value) for key, value in prompt_assets_for_conversation(
                conversation_id,
                query_text=payload.content,
                material_budget_tokens=max(
                    1024,
                    min(12000, llama_process.context_size - reserved - 512),
                ),
            ).items()
        },
    }


@app.get("/api/conversations/{conversation_id}/prompt-preview")
async def prompt_preview(conversation_id: str, query: str = ""):
    conversation = database.get_conversation(conversation_id)
    assets = prompt_assets_for_conversation(conversation_id, query_text=query)
    return {
        "document_id": conversation.get("document_id"),
        "system_prompt": conversation["system_prompt"],
        "pinned_context": conversation["pinned_context"],
        "style_guide": conversation.get("style_guide", ""),
        "style_lexicon": conversation.get("style_lexicon", ""),
        "sources": assets,
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


@app.get("/api/chapters/{chapter_id}")
async def get_chapter(chapter_id: str):
    return novels.get_chapter(chapter_id)


@app.patch("/api/chapters/{chapter_id}")
async def update_chapter(chapter_id: str, payload: ChapterUpdate):
    return novels.update_chapter(chapter_id, payload.model_dump(exclude_none=True))


@app.patch("/api/characters/{character_id}")
async def update_character(character_id: str, payload: CharacterUpdate):
    return novels.update_character(character_id, payload.model_dump(exclude_none=True))


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


@app.post("/api/experimental/material-system/packages/validate")
async def validate_material_package(
    request: Request,
    document_id: str | None = Query(default=None),
):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    package = await read_material_package(request)
    if isinstance(package, JSONResponse):
        return package
    try:
        return material_service().validate_package(package, target_document_id=document_id)
    except MaterialPackageError as exc:
        return error_response(400, "INVALID_MATERIAL_PACKAGE", str(exc))


@app.post("/api/experimental/material-system/packages/import", status_code=201)
async def import_material_package(
    request: Request,
    project_id: str = Query(default="default"),
    mode: str = Query(default="create_document"),
    document_id: str | None = Query(default=None),
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
        return material_service().import_package(
            package,
            project_id=project_id,
            mode=mode,
            target_document_id=document_id,
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


@app.get("/api/experimental/material-system/documents/{document_id}/timeline")
async def get_material_timeline(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().get_timeline(document_id)


@app.patch("/api/experimental/material-system/timeline-events/{event_id}")
async def update_material_timeline_event(event_id: str, payload: MaterialTimelineEventUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_timeline_event(event_id, payload.model_dump(exclude_none=True))


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


@app.patch("/api/experimental/material-system/characters/entities/{character_id}")
async def update_material_character_entity(character_id: str, payload: MaterialCharacterEntityUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_character_entity(character_id, payload.model_dump(exclude_none=True))


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


@app.patch("/api/experimental/material-system/relationships/{relationship_id}")
async def update_material_relationship(relationship_id: str, payload: MaterialRelationshipUpdate):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().update_relationship(relationship_id, payload.model_dump(exclude_none=True))


@app.post("/api/experimental/material-system/documents/{document_id}/relationships/rebuild")
async def rebuild_material_relationships(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().rebuild_relationships(document_id)


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
async def get_material_review_items(document_id: str):
    disabled = material_system_disabled_response()
    if disabled:
        return disabled
    return material_service().list_review_items(document_id)


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
        and (
            regenerate
            or chapter["status"] != "completed"
            or int(chapter.get("pending_fact_count") or 0) > 0
        )
    ]
    stop_event = await generation.begin(job["id"])

    async def event_stream():
        processed = int(job.get("processed_chapters") or 0)
        current_chapter_id: str | None = None
        current_chunk_id: str | None = None
        had_errors = False
        touched_character_observations: list[dict[str, Any]] = []
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
                    partials: list[dict[str, Any]] = []
                    observations: list[dict[str, Any]] = []
                    previous_summary = ""
                    chunks = chapter["chunks"]
                    for chunk in chunks:
                        current_chunk_id = chunk["id"]
                        chunk_index = int(chunk["position"])
                        if stop_event.is_set():
                            raise GenerationCancelled("用户停止了总结")
                        if (
                            chunk["status"] == "completed"
                            and chunk["summary"]
                            and chunk["facts_status"] == "completed"
                        ):
                            partials.append(chunk["summary"])
                            observations.extend(chunk["character_observations"])
                            previous_summary = format_chapter_summary(chunk["summary"])
                            yield sse("analysis_progress", {
                                "phase": "chapter", "stage": "chunk_resumed",
                                "index": chunk_index, "total": len(chunks),
                                "chapter_index": chapter_index, "chapter_total": len(targets),
                                "title": chapter["title"],
                            })
                            continue
                        novels.set_chunk_status(current_chunk_id, "processing")
                        summary = chunk["summary"]
                        chunk_observations = chunk["character_observations"]
                        if not summary:
                            yield sse("analysis_progress", {
                                "phase": "chapter", "stage": "summary_chunk_started",
                                "index": chunk_index, "total": len(chunks),
                                "chapter_index": chapter_index, "chapter_total": len(targets),
                                "title": chapter["title"],
                            })
                            task = asyncio.create_task(analysis_service.analyze_chunk(
                                chapter["title"], chunk["content"], previous_summary,
                                chunk_index, len(chunks), stop_event, max_tokens=max_tokens,
                            ))
                            async for event in analysis_progress_events(
                                task, asyncio.Queue(), phase="chapter",
                                context={"title": chapter["title"], "chapter_index": chapter_index,
                                         "chapter_total": len(targets), "index": chunk_index,
                                         "total": len(chunks)},
                            ):
                                yield event
                            summary, chunk_observations = await task
                            novels.save_chunk_analysis(
                                current_chunk_id, summary, chunk_observations, completed=False
                            )
                        if chunk["facts_status"] != "completed":
                            novels.set_chunk_facts_status(current_chunk_id, "processing")
                            yield sse("analysis_progress", {
                                "phase": "facts", "stage": "fact_chunk_started",
                                "index": chunk_index, "total": len(chunks),
                                "chapter_index": chapter_index, "chapter_total": len(targets),
                                "title": chapter["title"],
                            })
                            fact_task = asyncio.create_task(analysis_service.extract_story_facts(
                                chapter["title"], chunk["content"], stop_event,
                                max_tokens=max_tokens,
                            ))
                            async for event in analysis_progress_events(
                                fact_task, asyncio.Queue(), phase="facts",
                                context={"title": chapter["title"], "chapter_index": chapter_index,
                                         "chapter_total": len(targets), "index": chunk_index,
                                         "total": len(chunks)},
                            ):
                                yield event
                            facts = await fact_task
                            novels.save_story_facts(
                                document_id, current_chapter_id, current_chunk_id, facts
                            )
                            if settings.experimental_material_system:
                                try:
                                    material_service().seed_character_entities(document_id)
                                    unified_events = await analysis_service.extract_unified_events(
                                        chapter["title"], chunk["content"], stop_event,
                                        max_tokens=max_tokens,
                                    )
                                    material_service().save_unified_events(
                                        document_id,
                                        current_chapter_id,
                                        current_chunk_id,
                                        unified_events,
                                    )
                                except Exception:
                                    logger.exception("unified event extraction failed: %s", current_chunk_id)
                        novels.set_chunk_status(current_chunk_id, "completed")
                        partials.append(summary)
                        observations.extend(chunk_observations)
                        previous_summary = format_chapter_summary(summary)
                        novels.update_analysis_job(
                            job["id"], current_chunk_position=chunk_index
                        )
                        yield sse("analysis_progress", {
                            "phase": "chapter", "stage": "chunk_completed",
                            "index": chunk_index, "total": len(chunks),
                            "chapter_index": chapter_index, "chapter_total": len(targets),
                            "title": chapter["title"],
                        })
                    merge_task = asyncio.create_task(analysis_service.merge_chapter_summaries(
                        chapter["title"], partials, stop_event, max_tokens=max_tokens
                    ))
                    async for event in analysis_progress_events(
                        merge_task, asyncio.Queue(), phase="chapter_merge",
                        context={"title": chapter["title"], "chapter_index": chapter_index,
                                 "chapter_total": len(targets)},
                    ):
                        yield event
                    merged = await merge_task
                    saved = novels.save_chapter_summary(
                        current_chapter_id, merged, observations
                    )
                    touched_character_observations.extend(observations)
                    processed += 1
                    novels.update_analysis_job(
                        job["id"], processed_chapters=processed,
                        current_chunk_position=len(chunks),
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
                    if current_chunk_id:
                        novels.set_chunk_status(current_chunk_id, "failed", str(exc)[:1000])
                        novels.set_chunk_facts_status(current_chunk_id, "failed")
                    novels.set_chapter_status(current_chapter_id, "failed", str(exc)[:1000])
                    yield sse("chapter_error", {
                        "chapter_id": current_chapter_id, "message": str(exc)[:500],
                        "index": chapter_index, "total": len(targets),
                    })
                current_chapter_id = None
                current_chunk_id = None

            refreshed = novels.get_document_workspace(document_id)
            completed_summaries = [c["summary"] for c in refreshed["chapters"] if c["status"] == "completed" and c["summary"]]
            if completed_summaries:
                yield sse("project_summary_started", {})
                global_summary = await analysis_service.build_project_summary(
                    completed_summaries, stop_event, max_tokens=max_tokens
                )
                novels.save_document_summary(document_id, global_summary)
                yield sse("project_summary_completed", {"global_summary": global_summary})
                existing_characters = refreshed["characters"]
                if regenerate or not existing_characters:
                    character_observations = novels.get_document_character_observations(document_id)
                    relevant_characters = []
                    character_mode = "full"
                else:
                    character_observations = touched_character_observations
                    relevant_characters = novels.get_relevant_character_cards(
                        document_id, character_observations
                    )
                    character_mode = "incremental"
                yield sse("characters_started", {"mode": character_mode})
                character_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                if character_mode == "full":
                    character_task = asyncio.create_task(
                        analysis_service.extract_character_cards(
                            character_observations, stop_event,
                            max_tokens=max(8192, max_tokens),
                            on_progress=lambda stage, item, total: character_queue.put_nowait(
                                (stage, item, total)
                            ),
                        )
                    )
                else:
                    character_task = asyncio.create_task(
                        analysis_service.merge_character_updates(
                            relevant_characters,
                            character_observations,
                            stop_event,
                            max_tokens=max(8192, max_tokens),
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
                characters = novels.replace_characters(document_id, cards)
                yield sse("characters_completed", {"characters": characters, "mode": character_mode})
            status = "failed" if had_errors else "completed"
            novels.update_analysis_job(job["id"], status=status, error_message="部分章节失败" if had_errors else None)
            yield sse("done", {
                "workspace": novels.get_document_workspace(document_id),
                "job": novels.get_analysis_job(job["id"]),
            })
        except GenerationCancelled:
            if current_chunk_id:
                novels.set_chunk_status(current_chunk_id, "pending")
                novels.set_chunk_facts_status(current_chunk_id, "pending")
            if current_chapter_id:
                novels.set_chapter_status(current_chapter_id, "pending")
            novels.update_analysis_job(job["id"], status="paused", error_message="用户暂停")
            yield sse("cancelled", {
                "workspace": novels.get_document_workspace(document_id),
                "job": novels.get_analysis_job(job["id"]),
            })
        except asyncio.CancelledError:
            if current_chunk_id:
                novels.set_chunk_status(current_chunk_id, "pending")
                novels.set_chunk_facts_status(current_chunk_id, "pending")
            if current_chapter_id:
                novels.set_chapter_status(current_chapter_id, "pending")
            novels.update_analysis_job(job["id"], status="paused", error_message="连接中断")
            raise
        except Exception as exc:
            logger.exception("document analysis job failed: %s", job["id"])
            if current_chunk_id:
                novels.set_chunk_status(current_chunk_id, "failed", str(exc)[:1000])
                novels.set_chunk_facts_status(current_chunk_id, "failed")
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
        start_position = appended["chunk_start_position"]
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
            chunk_summaries = summary.pop("_chunk_summaries", [])
            character_observations = summary.pop("_character_observations", [])
            novels.save_chunk_summaries(
                chapter["id"], chunk_summaries, start_position=start_position
            )
            refreshed_chapter = novels.get_chapter(chapter["id"])
            new_chunks = [
                chunk for chunk in refreshed_chapter["chunks"]
                if int(chunk["position"]) >= start_position
            ]
            for fact_index, chunk in enumerate(new_chunks, start=1):
                novels.set_chunk_facts_status(chunk["id"], "processing")
                yield sse("analysis_progress", {
                    "phase": "facts", "stage": "fact_chunk_started",
                    "index": fact_index, "total": len(new_chunks),
                    "title": chapter["title"],
                })
                fact_task = asyncio.create_task(
                    analysis_service.extract_story_facts(
                        chapter["title"], chunk["content"], stop_event,
                        max_tokens=payload.max_tokens,
                    )
                )
                async for progress_event in analysis_progress_events(
                    fact_task,
                    asyncio.Queue(),
                    phase="facts",
                    context={
                        "title": chapter["title"],
                        "index": fact_index,
                        "total": len(new_chunks),
                    },
                ):
                    yield progress_event
                facts = await fact_task
                novels.save_story_facts(document_id, chapter["id"], chunk["id"], facts)
                if settings.experimental_material_system:
                    try:
                        material_service().seed_character_entities(document_id)
                        unified_events = await analysis_service.extract_unified_events(
                            chapter["title"], chunk["content"], stop_event,
                            max_tokens=payload.max_tokens,
                        )
                        material_service().save_unified_events(
                            document_id,
                            chapter["id"],
                            chunk["id"],
                            unified_events,
                        )
                    except Exception:
                        logger.exception("unified event extraction failed: %s", chunk["id"])
            updated_chapter = novels.save_chapter_summary(
                chapter["id"],
                summary,
                character_observations,
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
                relevant_characters = novels.get_relevant_character_cards(
                    document_id, character_observations
                )
                character_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
                character_task = asyncio.create_task(
                    analysis_service.merge_character_updates(
                        relevant_characters,
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
                characters = novels.replace_characters(document_id, cards)
                yield sse("characters_completed", {"characters": characters, "mode": "incremental"})
            yield sse("done", {"workspace": novels.get_document_workspace(document_id)})
        except GenerationCancelled:
            novels.mark_increment_failed(
                chapter["id"], start_position, "增量总结已停止；正文已保存，可重新总结本章"
            )
            yield sse("cancelled", {"workspace": novels.get_document_workspace(document_id)})
        except asyncio.CancelledError:
            novels.mark_increment_failed(
                chapter["id"], start_position, "连接中断；正文已保存，可重新总结本章"
            )
            raise
        except Exception as exc:
            logger.exception("incremental project update failed")
            novels.mark_increment_failed(
                chapter["id"], start_position, f"增量总结失败：{str(exc)[:500]}"
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


def outline_instruction(user_instruction: str) -> str:
    return f"""请为小说紧接当前进度的下一章设计可直接用于写作的结构化大纲。
用户的额外要求：{user_instruction}

使用中文 Markdown，必须包含：
1. 章节标题候选
2. POV、时间、地点
3. 本章目标与核心冲突
4. 5—10 个按顺序排列的情节节点
5. 情绪曲线
6. 核心人物的状态变化
7. 伏笔、揭示与回收
8. 高潮场面
9. 结尾钩子
10. 禁止偏离的设定与建议字数

大纲必须承接已有前文、人物卡和最近对话。不要写正文，不要解释你的工作过程。"""


async def prepare_outline_preview(
    conversation_id: str, payload: OutlineGenerateRequest
):
    if generation.busy:
        return error_response(409, "GENERATION_IN_PROGRESS", "当前已有生成或总结任务")
    conversation = database.get_conversation(conversation_id)
    generation_settings = resolve_generation_settings(conversation, payload.settings)
    context = await build_fitted_context(
        conversation_id=conversation_id,
        system_prompt=conversation["system_prompt"],
        pinned_context=conversation["pinned_context"],
        style_guide=conversation.get("style_guide", ""),
        style_lexicon=conversation.get("style_lexicon", ""),
        history=selected_history(conversation),
        current_user_content=outline_instruction(payload.instruction),
        max_output_tokens=generation_settings["max_tokens"],
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
        generation_settings=generation_settings,
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
        return error_response(400, "OUTLINE_MISMATCH", "大纲不属于当前对话")


@app.patch("/api/outlines/{outline_id}")
async def update_outline(outline_id: str, payload: OutlineUpdateRequest):
    return novels.update_outline(outline_id, payload.enabled)


@app.put("/api/outlines/{outline_id}/selection")
async def select_outline(outline_id: str, payload: SelectionRequest):
    try:
        return novels.select_outline_candidate(outline_id, payload.candidate_id)
    except ValueError:
        return error_response(400, "OUTLINE_NOT_SELECTABLE", "这个大纲候选不能被选用")


@app.patch("/api/outline-candidates/{candidate_id}")
async def edit_outline_candidate(
    candidate_id: str, payload: OutlineCandidateEditRequest
):
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


def conversation_markdown(conversation: dict[str, Any], include_all: bool) -> str:
    lines = [f"# {conversation['title']}", ""]
    if conversation["system_prompt"]:
        lines.extend(["## 系统提示词", "", conversation["system_prompt"], ""])
    if conversation["pinned_context"]:
        lines.extend(["## 固定创作资料", "", conversation["pinned_context"], ""])
    if conversation.get("style_guide"):
        lines.extend(["## 词汇风格", "", conversation["style_guide"], ""])
    if conversation.get("style_lexicon"):
        lines.extend(["## 词表白名单 / 优先用词", "", conversation["style_lexicon"], ""])
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
    filename = quote(conversation["title"] or "Novel-factory")
    if format == "json":
        content = json.dumps(conversation, ensure_ascii=False, indent=2)
        return Response(
            content=content,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}.json"},
        )
    return PlainTextResponse(
        conversation_markdown(conversation, include_all),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}.md"},
    )


frontend_dir = settings.project_root / "frontend"
app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-store"})
