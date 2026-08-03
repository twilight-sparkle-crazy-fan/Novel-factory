#!/usr/bin/env python3
"""MCP stdio bridge for a running Novel-factory application."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import quote

import httpx


BASE_URL = os.getenv("NOVEL_FACTORY_URL", "http://127.0.0.1:8000").rstrip("/")
SERVER_NAME = "novel-factory"
SERVER_VERSION = "0.2.0"


def object_schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


STRING = {"type": "string"}
SETTINGS = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
        "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "max_tokens": {"type": "integer", "minimum": 16, "maximum": 384000},
        "repeat_penalty": {"type": "number", "minimum": 0.5, "maximum": 2},
        "seed": {"type": ["integer", "null"]},
    },
    "additionalProperties": False,
}
SCENE_FRAGMENT = {
    "type": "object",
    "properties": {
        "label": STRING,
        "title": STRING,
        "card": STRING,
        "content": STRING,
        "check": {"type": ["object", "null"]},
    },
    "required": ["label", "card"],
    "additionalProperties": False,
}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "novel_status",
        "description": "Check the Novel-factory app, model mode, readiness, and current generation state.",
        "inputSchema": object_schema({}),
    },
    {
        "name": "novel_list_conversations",
        "description": "List writing conversations.",
        "inputSchema": object_schema({}),
    },
    {
        "name": "novel_get_conversation",
        "description": "Read one conversation with exchanges, candidates, settings, and selected versions.",
        "inputSchema": object_schema({"conversation_id": STRING}, ["conversation_id"]),
    },
    {
        "name": "novel_create_conversation",
        "description": "Create a writing conversation.",
        "inputSchema": object_schema({"title": STRING}),
    },
    {
        "name": "novel_update_conversation",
        "description": "Update conversation title, prompts, style, settings, or attached document.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "changes": {"type": "object"},
            },
            ["conversation_id", "changes"],
        ),
    },
    {
        "name": "novel_branch_candidate",
        "description": "Create a new conversation branch from a selected candidate on an earlier exchange.",
        "inputSchema": object_schema(
            {"exchange_id": STRING, "candidate_id": STRING},
            ["exchange_id", "candidate_id"],
        ),
    },
    {
        "name": "novel_generate",
        "description": "Generate a candidate in a conversation and return the completed exchange.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "instruction": STRING,
                "settings": SETTINGS,
            },
            ["conversation_id", "instruction"],
        ),
    },
    {
        "name": "novel_regenerate_candidate",
        "description": "Generate another candidate for an existing exchange without replacing prior versions.",
        "inputSchema": object_schema(
            {"exchange_id": STRING, "settings": SETTINGS},
            ["exchange_id"],
        ),
    },
    {
        "name": "novel_select_candidate",
        "description": "Select a completed candidate for an exchange. Earlier exchanges may require branching.",
        "inputSchema": object_schema(
            {"exchange_id": STRING, "candidate_id": STRING},
            ["exchange_id", "candidate_id"],
        ),
    },
    {
        "name": "novel_list_projects",
        "description": "List novel library projects.",
        "inputSchema": object_schema({}),
    },
    {
        "name": "novel_get_project",
        "description": "Read a project and its imported documents.",
        "inputSchema": object_schema({"project_id": STRING}, ["project_id"]),
    },
    {
        "name": "novel_update_project",
        "description": "Update a project name or manually edited global summary.",
        "inputSchema": object_schema(
            {"project_id": STRING, "changes": {"type": "object"}},
            ["project_id", "changes"],
        ),
    },
    {
        "name": "novel_get_document",
        "description": "Read a document workspace including chapters, character cards, events, and summaries.",
        "inputSchema": object_schema({"document_id": STRING}, ["document_id"]),
    },
    {
        "name": "novel_update_document",
        "description": "Update a document name, material switches, or manually edited short summary.",
        "inputSchema": object_schema(
            {"document_id": STRING, "changes": {"type": "object"}},
            ["document_id", "changes"],
        ),
    },
    {
        "name": "novel_import_text",
        "description": "Import UTF-8 novel text into a project and split it into chapters.",
        "inputSchema": object_schema(
            {"project_id": STRING, "filename": STRING, "content": STRING},
            ["project_id", "filename", "content"],
        ),
    },
    {
        "name": "novel_create_document",
        "description": "Create a new TXT document with an empty first chapter.",
        "inputSchema": object_schema(
            {"project_id": STRING, "filename": STRING},
            ["project_id"],
        ),
    },
    {
        "name": "novel_get_chapter",
        "description": "Read a chapter with its full正文, single complete summary, and status.",
        "inputSchema": object_schema({"chapter_id": STRING}, ["chapter_id"]),
    },
    {
        "name": "novel_update_chapter",
        "description": "Update chapter title, full content, or edited summary. Updating content invalidates its derived summary.",
        "inputSchema": object_schema(
            {"chapter_id": STRING, "changes": {"type": "object"}},
            ["chapter_id", "changes"],
        ),
    },
    {
        "name": "novel_preview_selection_rewrite",
        "description": "Rewrite only a character range in a chapter using local context. Returns a preview and does not persist it.",
        "inputSchema": object_schema(
            {
                "chapter_id": STRING,
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 1},
                "instruction": STRING,
                "settings": SETTINGS,
            },
            ["chapter_id", "start", "end"],
        ),
    },
    {
        "name": "novel_apply_selection_rewrite",
        "description": "Apply a previously previewed selection rewrite with optimistic conflict checks.",
        "inputSchema": object_schema(
            {
                "chapter_id": STRING,
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 1},
                "source_hash": STRING,
                "original_text": STRING,
                "replacement": STRING,
            },
            [
                "chapter_id",
                "start",
                "end",
                "source_hash",
                "original_text",
                "replacement",
            ],
        ),
    },
    {
        "name": "novel_summarize_chapters",
        "description": "Summarize selected chapter IDs or a chapter-position range and update document materials.",
        "inputSchema": object_schema(
            {
                "project_id": STRING,
                "document_id": STRING,
                "chapter_ids": {"type": "array", "items": STRING},
                "start_position": {"type": "integer", "minimum": 1},
                "end_position": {"type": "integer", "minimum": 1},
                "regenerate": {"type": "boolean"},
                "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 384000},
            },
            ["project_id", "document_id"],
        ),
    },
    {
        "name": "novel_append_chapter",
        "description": "Append正文 to an existing chapter or create a new chapter in a document.",
        "inputSchema": object_schema(
            {
                "project_id": STRING,
                "document_id": STRING,
                "chapter_id": STRING,
                "title": STRING,
                "content": STRING,
                "summarize_now": {"type": "boolean"},
                "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 384000},
            },
            ["project_id", "content"],
        ),
    },
    {
        "name": "novel_get_prompt_preview",
        "description": "Inspect the exact prompt and material sources for a conversation without generating.",
        "inputSchema": object_schema(
            {"conversation_id": STRING, "query": STRING},
            ["conversation_id"],
        ),
    },
    {
        "name": "novel_get_outline",
        "description": "Read the scene outline group and candidates for a conversation.",
        "inputSchema": object_schema({"conversation_id": STRING}, ["conversation_id"]),
    },
    {
        "name": "novel_generate_outline",
        "description": "Generate a JSON scene-card outline preview. The preview is not selected automatically.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "instruction": STRING,
                "include_hook": {"type": "boolean"},
                "settings": SETTINGS,
            },
            ["conversation_id", "instruction"],
        ),
    },
    {
        "name": "novel_save_outline",
        "description": "Save a JSON scene-card outline candidate and optionally select it.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "outline_id": STRING,
                "instruction": STRING,
                "content": STRING,
                "select": {"type": "boolean"},
                "settings": SETTINGS,
            },
            ["conversation_id", "instruction", "content"],
        ),
    },
    {
        "name": "novel_select_outline",
        "description": "Select a saved, valid JSON scene-card candidate for an outline group.",
        "inputSchema": object_schema(
            {"outline_id": STRING, "candidate_id": STRING},
            ["outline_id", "candidate_id"],
        ),
    },
    {
        "name": "novel_edit_outline_candidate",
        "description": "Edit the JSON content of a saved scene-card candidate after validation.",
        "inputSchema": object_schema(
            {"candidate_id": STRING, "content": STRING},
            ["candidate_id", "content"],
        ),
    },
    {
        "name": "novel_run_scene_workflow",
        "description": "Run selected JSON scene cards through drafting and checks, stopping at fragment review.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "instruction": STRING,
                "settings": SETTINGS,
            },
            ["conversation_id"],
        ),
    },
    {
        "name": "novel_regenerate_scene_fragment",
        "description": "Regenerate one scene fragment during review while preserving the other reviewed fragments.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "candidate_id": STRING,
                "outline_text": STRING,
                "scene_index": {"type": "integer", "minimum": 0},
                "scenes": {"type": "array", "items": SCENE_FRAGMENT, "minItems": 1},
                "instruction": STRING,
                "settings": SETTINGS,
            },
            ["conversation_id", "candidate_id", "scene_index", "scenes"],
        ),
    },
    {
        "name": "novel_accept_scene_workflow",
        "description": "Accept reviewed scene fragments as the final candidate without calling the model for polishing.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "candidate_id": STRING,
                "scenes": {"type": "array", "items": SCENE_FRAGMENT, "minItems": 1},
            },
            ["conversation_id", "candidate_id", "scenes"],
        ),
    },
    {
        "name": "novel_polish_scene_workflow",
        "description": "Finish reviewed scene fragments with continuity checking and final chapter polishing.",
        "inputSchema": object_schema(
            {
                "conversation_id": STRING,
                "candidate_id": STRING,
                "scenes": {"type": "array", "items": SCENE_FRAGMENT, "minItems": 1},
                "settings": SETTINGS,
            },
            ["conversation_id", "candidate_id", "scenes"],
        ),
    },
    {
        "name": "novel_create_character",
        "description": "Manually create a structured JSON character card in a document.",
        "inputSchema": object_schema(
            {"document_id": STRING, "card": {"type": "object"}},
            ["document_id", "card"],
        ),
    },
    {
        "name": "novel_extract_characters",
        "description": "Extract structured character cards from an explicit chapter-summary range.",
        "inputSchema": object_schema(
            {
                "document_id": STRING,
                "start_position": {"type": "integer", "minimum": 1},
                "end_position": {"type": "integer", "minimum": 1},
                "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 384000},
            },
            ["document_id", "start_position", "end_position"],
        ),
    },
    {
        "name": "novel_update_character",
        "description": "Update a structured JSON character card or its enabled state.",
        "inputSchema": object_schema(
            {"character_id": STRING, "changes": {"type": "object"}},
            ["character_id", "changes"],
        ),
    },
    {
        "name": "novel_update_character_event",
        "description": "Enable or disable one character experience for prompt injection.",
        "inputSchema": object_schema(
            {"event_id": STRING, "enabled": {"type": "boolean"}},
            ["event_id", "enabled"],
        ),
    },
    {
        "name": "novel_merge_characters",
        "description": "Merge two character cards while choosing the canonical name.",
        "inputSchema": object_schema(
            {
                "source_character_id": STRING,
                "target_character_id": STRING,
                "keep_name": STRING,
            },
            ["source_character_id", "target_character_id", "keep_name"],
        ),
    },
    {
        "name": "novel_delete_item",
        "description": "Delete a chapter, character, or document only after explicit user confirmation.",
        "inputSchema": object_schema(
            {
                "item_type": {
                    "type": "string",
                    "enum": [
                        "conversation",
                        "chapter",
                        "character",
                        "document",
                        "outline",
                        "outline_candidate",
                    ],
                },
                "item_id": STRING,
                "confirmed": {"type": "boolean", "const": True},
            },
            ["item_type", "item_id", "confirmed"],
        ),
    },
    {
        "name": "novel_stop_generation",
        "description": "Stop the currently running generation or analysis task.",
        "inputSchema": object_schema({}),
    },
]


MUTATING_TOOLS = {
    "novel_create_conversation",
    "novel_update_conversation",
    "novel_branch_candidate",
    "novel_generate",
    "novel_regenerate_candidate",
    "novel_select_candidate",
    "novel_update_project",
    "novel_update_document",
    "novel_import_text",
    "novel_create_document",
    "novel_update_chapter",
    "novel_preview_selection_rewrite",
    "novel_apply_selection_rewrite",
    "novel_summarize_chapters",
    "novel_append_chapter",
    "novel_generate_outline",
    "novel_save_outline",
    "novel_select_outline",
    "novel_edit_outline_candidate",
    "novel_run_scene_workflow",
    "novel_regenerate_scene_fragment",
    "novel_accept_scene_workflow",
    "novel_polish_scene_workflow",
    "novel_create_character",
    "novel_extract_characters",
    "novel_update_character",
    "novel_update_character_event",
    "novel_merge_characters",
    "novel_delete_item",
    "novel_stop_generation",
}


class NovelFactoryClient:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=None)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = self.client.request(
            method,
            path,
            json=json_body,
            content=content,
            headers=headers,
        )
        if response.status_code >= 400:
            raise RuntimeError(self._error_text(response))
        if response.status_code == 204:
            return None
        content_type = response.headers.get("content-type", "")
        return response.json() if "application/json" in content_type else response.text

    def stream(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        with self.client.stream("POST", path, json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise RuntimeError(self._error_text(response))
            event_name = "message"
            data_lines: list[str] = []
            for raw_line in response.iter_lines():
                line = raw_line.rstrip("\r")
                if not line:
                    if data_lines:
                        raw_data = "\n".join(data_lines)
                        try:
                            data = json.loads(raw_data)
                        except json.JSONDecodeError:
                            data = {"text": raw_data}
                        events.append({"event": event_name, "data": data})
                        if event_name == "error":
                            raise RuntimeError(
                                str(data.get("message") or "Novel-factory stream failed")
                            )
                    event_name = "message"
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        important = [
            event
            for event in events
            if event["event"]
            in {
                "done",
                "cancelled",
                "workflow_review_ready",
                "fragment_done",
                "outline_preview_done",
                "chapter_completed",
            }
        ]
        return {
            "result": (important[-1] if important else (events[-1] if events else None)),
            "event_count": len(events),
        }

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            payload = response.json()
            detail = payload.get("error") or payload.get("detail") or payload
            if isinstance(detail, dict):
                return str(detail.get("message") or detail.get("detail") or detail)
            return str(detail)
        except (ValueError, TypeError):
            return response.text[:1000] or f"HTTP {response.status_code}"

    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        token = ""
        try:
            started = self.request(
                "POST",
                "/api/agent/activity/start",
                json_body={"label": label},
            )
            token = str(started.get("token") or "")
            yield
        finally:
            if token:
                try:
                    self.request(
                        "POST",
                        f"/api/agent/activity/{quote(token, safe='')}/finish",
                    )
                except Exception:
                    pass


CLIENT = NovelFactoryClient(BASE_URL)


def clean_body(arguments: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: arguments[key] for key in keys if key in arguments and arguments[key] is not None}


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "novel_status":
        return {
            "health": CLIENT.request("GET", "/api/health"),
            "runtime": CLIENT.request("GET", "/api/runtime"),
        }
    if name == "novel_list_conversations":
        return CLIENT.request("GET", "/api/conversations")
    if name == "novel_get_conversation":
        return CLIENT.request("GET", f"/api/conversations/{arguments['conversation_id']}")
    if name == "novel_create_conversation":
        return CLIENT.request(
            "POST", "/api/conversations", json_body={"title": arguments.get("title", "新对话")}
        )
    if name == "novel_update_conversation":
        return CLIENT.request(
            "PATCH",
            f"/api/conversations/{arguments['conversation_id']}",
            json_body=arguments["changes"],
        )
    if name == "novel_branch_candidate":
        return CLIENT.request(
            "POST",
            f"/api/exchanges/{arguments['exchange_id']}/branch",
            json_body={"candidate_id": arguments["candidate_id"]},
        )
    if name == "novel_generate":
        body = {"content": arguments["instruction"]}
        if arguments.get("settings") is not None:
            body["settings"] = arguments["settings"]
        streamed = CLIENT.stream(
            f"/api/conversations/{arguments['conversation_id']}/generate", body
        )
        streamed["conversation"] = CLIENT.request(
            "GET", f"/api/conversations/{arguments['conversation_id']}"
        )
        return streamed
    if name == "novel_regenerate_candidate":
        return CLIENT.stream(
            f"/api/exchanges/{arguments['exchange_id']}/regenerate",
            clean_body(arguments, "settings"),
        )
    if name == "novel_select_candidate":
        return CLIENT.request(
            "PUT",
            f"/api/exchanges/{arguments['exchange_id']}/selection",
            json_body={"candidate_id": arguments["candidate_id"]},
        )
    if name == "novel_list_projects":
        return CLIENT.request("GET", "/api/projects")
    if name == "novel_get_project":
        return CLIENT.request("GET", f"/api/projects/{arguments['project_id']}")
    if name == "novel_update_project":
        return CLIENT.request(
            "PATCH",
            f"/api/projects/{arguments['project_id']}",
            json_body=arguments["changes"],
        )
    if name == "novel_get_document":
        return CLIENT.request("GET", f"/api/documents/{arguments['document_id']}/workspace")
    if name == "novel_update_document":
        return CLIENT.request(
            "PATCH",
            f"/api/documents/{arguments['document_id']}",
            json_body=arguments["changes"],
        )
    if name == "novel_import_text":
        return CLIENT.request(
            "POST",
            f"/api/projects/{arguments['project_id']}/import-txt",
            content=arguments["content"].encode("utf-8"),
            headers={"X-Filename": quote(arguments["filename"])},
        )
    if name == "novel_create_document":
        return CLIENT.request(
            "POST",
            f"/api/projects/{arguments['project_id']}/documents",
            json_body={"filename": arguments.get("filename", "未命名小说.txt")},
        )
    if name == "novel_get_chapter":
        return CLIENT.request("GET", f"/api/chapters/{arguments['chapter_id']}")
    if name == "novel_update_chapter":
        return CLIENT.request(
            "PATCH",
            f"/api/chapters/{arguments['chapter_id']}",
            json_body=arguments["changes"],
        )
    if name == "novel_preview_selection_rewrite":
        return CLIENT.stream(
            f"/api/chapters/{arguments['chapter_id']}/rewrite-selection",
            clean_body(arguments, "start", "end", "instruction", "settings"),
        )
    if name == "novel_apply_selection_rewrite":
        return CLIENT.request(
            "POST",
            f"/api/chapters/{arguments['chapter_id']}/rewrite-selection/apply",
            json_body=clean_body(
                arguments,
                "start",
                "end",
                "source_hash",
                "original_text",
                "replacement",
            ),
        )
    if name == "novel_summarize_chapters":
        return CLIENT.stream(
            f"/api/projects/{arguments['project_id']}/summarize",
            clean_body(
                arguments,
                "document_id",
                "chapter_ids",
                "start_position",
                "end_position",
                "regenerate",
                "max_tokens",
            ),
        )
    if name == "novel_append_chapter":
        return CLIENT.stream(
            f"/api/projects/{arguments['project_id']}/append",
            clean_body(
                arguments,
                "document_id",
                "chapter_id",
                "title",
                "content",
                "summarize_now",
                "max_tokens",
            ),
        )
    if name == "novel_get_prompt_preview":
        query = quote(str(arguments.get("query") or ""), safe="")
        return CLIENT.request(
            "GET",
            f"/api/conversations/{arguments['conversation_id']}/prompt-preview?query={query}",
        )
    if name == "novel_get_outline":
        return CLIENT.request(
            "GET", f"/api/conversations/{arguments['conversation_id']}/outline"
        )
    if name == "novel_generate_outline":
        return CLIENT.stream(
            f"/api/conversations/{arguments['conversation_id']}/outline/generate",
            clean_body(arguments, "instruction", "include_hook", "settings"),
        )
    if name == "novel_save_outline":
        return CLIENT.request(
            "POST",
            f"/api/conversations/{arguments['conversation_id']}/outline/candidates",
            json_body=clean_body(
                arguments,
                "outline_id",
                "instruction",
                "content",
                "select",
                "settings",
            ),
        )
    if name == "novel_select_outline":
        return CLIENT.request(
            "PUT",
            f"/api/outlines/{arguments['outline_id']}/selection",
            json_body={"candidate_id": arguments["candidate_id"]},
        )
    if name == "novel_edit_outline_candidate":
        return CLIENT.request(
            "PATCH",
            f"/api/outline-candidates/{arguments['candidate_id']}",
            json_body={"content": arguments["content"]},
        )
    if name == "novel_run_scene_workflow":
        return CLIENT.stream(
            f"/api/conversations/{arguments['conversation_id']}/scene-workflow",
            clean_body(arguments, "instruction", "settings"),
        )
    if name == "novel_regenerate_scene_fragment":
        return CLIENT.stream(
            f"/api/conversations/{arguments['conversation_id']}/scene-workflow/fragment",
            clean_body(
                arguments,
                "candidate_id",
                "outline_text",
                "scene_index",
                "scenes",
                "instruction",
                "settings",
            ),
        )
    if name == "novel_polish_scene_workflow":
        return CLIENT.stream(
            f"/api/conversations/{arguments['conversation_id']}/scene-workflow/polish",
            clean_body(arguments, "candidate_id", "scenes", "settings"),
        )
    if name == "novel_accept_scene_workflow":
        return CLIENT.request(
            "POST",
            f"/api/conversations/{arguments['conversation_id']}/scene-workflow/accept",
            json_body=clean_body(arguments, "candidate_id", "scenes"),
        )
    if name == "novel_create_character":
        return CLIENT.request(
            "POST",
            f"/api/documents/{arguments['document_id']}/characters",
            json_body={"card": arguments["card"]},
        )
    if name == "novel_extract_characters":
        return CLIENT.stream(
            f"/api/documents/{arguments['document_id']}/characters/extract",
            clean_body(arguments, "start_position", "end_position", "max_tokens"),
        )
    if name == "novel_update_character":
        return CLIENT.request(
            "PATCH",
            f"/api/characters/{arguments['character_id']}",
            json_body=arguments["changes"],
        )
    if name == "novel_update_character_event":
        return CLIENT.request(
            "PATCH",
            f"/api/character-events/{arguments['event_id']}",
            json_body={"enabled": arguments["enabled"]},
        )
    if name == "novel_merge_characters":
        return CLIENT.request(
            "POST",
            f"/api/characters/{arguments['source_character_id']}/merge",
            json_body={
                "target_character_id": arguments["target_character_id"],
                "keep_name": arguments["keep_name"],
            },
        )
    if name == "novel_delete_item":
        if arguments.get("confirmed") is not True:
            raise RuntimeError("Deletion requires confirmed=true after explicit user approval")
        path = {
            "conversation": "/api/conversations/{id}",
            "chapter": "/api/chapters/{id}",
            "character": "/api/characters/{id}",
            "document": "/api/documents/{id}",
            "outline": "/api/outlines/{id}",
            "outline_candidate": "/api/outline-candidates/{id}",
        }[arguments["item_type"]]
        return CLIENT.request(
            "DELETE",
            path.format(id=quote(arguments["item_id"], safe="")),
        )
    if name == "novel_stop_generation":
        return CLIENT.request("POST", "/api/generation/stop")
    raise RuntimeError(f"Unknown tool: {name}")


def tool_result(value: Any, *, is_error: bool = False) -> dict[str, Any]:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if not is_error and isinstance(value, dict):
        result["structuredContent"] = value
    return result


def handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        requested = str((message.get("params") or {}).get("protocolVersion") or "2025-06-18")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": requested,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": TOOLS},
        }
    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        try:
            if name in MUTATING_TOOLS:
                label = next(
                    (tool["description"] for tool in TOOLS if tool["name"] == name),
                    name,
                )
                with CLIENT.activity(label):
                    value = call_tool(name, arguments)
            else:
                value = call_tool(name, arguments)
            result = tool_result(value)
        except Exception as exc:
            result = tool_result({"error": str(exc)}, is_error=True)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            response = handle_request(message)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(exc)},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
