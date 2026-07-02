from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import backend.app as app_module
from backend.database import Database
from backend.novel_repository import NovelRepository


def test_stream_regenerate_select_and_continue(monkeypatch, tmp_path: Path) -> None:
    test_database = Database(tmp_path / "api.db")
    test_database.initialize()
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", NovelRepository(test_database))

    async def healthy() -> bool:
        return True

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)

    async def count_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages) // 2 + 8

    monkeypatch.setattr(app_module.llama_client, "count_chat_tokens", count_tokens)

    captured_messages: list[list[dict[str, str]]] = []
    outputs = iter(["雨落在旧站台上。", "她在末班车后回来。", "他终于抬起头。"])

    async def fake_stream(
        messages: list[dict[str, str]],
        _settings: dict[str, Any],
        _stop_event: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        captured_messages.append(messages)
        yield {"type": "content_delta", "text": next(outputs)}
        yield {"type": "timings", "value": {"prompt_tokens": 20, "completion_tokens": 2200}}
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "雨夜"}).json()
        first_response = client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"content": "写一个开场"},
        )
        assert first_response.status_code == 200
        assert "event: done" in first_response.text

        stored = client.get(f"/api/conversations/{conversation['id']}").json()
        exchange = stored["exchanges"][0]
        first_candidate_id = exchange["selected_candidate_id"]
        assert exchange["candidates"][0]["content"] == "雨落在旧站台上。"

        reroll_response = client.post(
            f"/api/exchanges/{exchange['id']}/regenerate",
            json={},
        )
        assert reroll_response.status_code == 200

        stored = client.get(f"/api/conversations/{conversation['id']}").json()
        exchange = stored["exchanges"][0]
        assert len(exchange["candidates"]) == 2
        assert exchange["selected_candidate_id"] == first_candidate_id
        second_candidate = exchange["candidates"][1]
        assert second_candidate["content"] == "她在末班车后回来。"

        selection = client.put(
            f"/api/exchanges/{exchange['id']}/selection",
            json={"candidate_id": second_candidate["id"]},
        )
        assert selection.status_code == 200

        final_response = client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"content": "继续"},
        )
        assert final_response.status_code == 200

    last_context = captured_messages[-1]
    assert {"role": "assistant", "content": "她在末班车后回来。"} in last_context
    assert "雨落在旧站台上。" not in str(last_context)


def test_generation_auto_continues_until_minimum_completion_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    test_database = Database(tmp_path / "auto-continue-api.db")
    test_database.initialize()
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", NovelRepository(test_database))

    async def healthy() -> bool:
        return True

    async def count_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages) // 2 + 8

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.llama_client, "count_chat_tokens", count_tokens)

    captured_calls: list[dict[str, Any]] = []
    chunks = ["第一段正文。", "第二段正文。"]
    completion_counts = [900, 1200]

    async def fake_stream(
        messages: list[dict[str, str]],
        settings: dict[str, Any],
        _stop_event: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        call_index = len(captured_calls)
        captured_calls.append({"messages": messages, "settings": dict(settings)})
        yield {"type": "content_delta", "text": chunks[call_index]}
        yield {
            "type": "timings",
            "value": {"prompt_tokens": 20 + call_index, "completion_tokens": completion_counts[call_index]},
        }
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "自动续写"}).json()
        client.patch(
            f"/api/conversations/{conversation['id']}",
            json={
                "style_guide": "表达直白，必要时不要改成委婉说法。",
                "style_lexicon": "暗星\n旧誓",
                "generation_settings": {"max_tokens": 1200},
            },
        )
        response = client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"content": "写下一章"},
        )
        assert response.status_code == 200
        assert "event: auto_continue_started" in response.text

        stored = client.get(f"/api/conversations/{conversation['id']}").json()

    assert len(captured_calls) == 2
    first_system = captured_calls[0]["messages"][0]["content"]
    assert "表达直白" in first_system
    assert "暗星" in first_system
    second_messages = captured_calls[1]["messages"]
    assert {"role": "assistant", "content": "第一段正文。"} in second_messages
    assert "隐藏续写指令" in second_messages[-1]["content"]
    assert captured_calls[1]["settings"]["max_tokens"] >= 1612
    candidate = stored["exchanges"][0]["candidates"][0]
    assert candidate["content"] == "第一段正文。第二段正文。"
    assert candidate["completion_tokens"] == 2100


def test_generation_respects_custom_minimum_completion_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    test_database = Database(tmp_path / "custom-auto-continue-api.db")
    test_database.initialize()
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", NovelRepository(test_database))

    async def healthy() -> bool:
        return True

    async def count_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages) // 2 + 8

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.llama_client, "count_chat_tokens", count_tokens)

    captured_calls: list[dict[str, Any]] = []

    async def fake_stream(
        messages: list[dict[str, str]],
        settings: dict[str, Any],
        _stop_event: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        captured_calls.append({"messages": messages, "settings": dict(settings)})
        yield {"type": "content_delta", "text": "短正文。"}
        yield {"type": "timings", "value": {"prompt_tokens": 20, "completion_tokens": 700}}
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "自定义续写阈值"}).json()
        client.patch(
            f"/api/conversations/{conversation['id']}",
            json={"generation_settings": {"max_tokens": 1200, "min_completion_tokens": 500}},
        )
        response = client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"content": "写下一章"},
        )
        stored = client.get(f"/api/conversations/{conversation['id']}").json()

    assert response.status_code == 200
    assert "event: auto_continue_started" not in response.text
    assert len(captured_calls) == 1
    assert "min_completion_tokens" not in captured_calls[0]["settings"]
    assert stored["generation_settings"]["min_completion_tokens"] == 500
    assert stored["exchanges"][0]["candidates"][0]["completion_tokens"] == 700


def test_import_summarize_character_and_outline_flow(monkeypatch, tmp_path: Path) -> None:
    test_database = Database(tmp_path / "novel-api.db")
    test_database.initialize()
    test_repository = NovelRepository(test_database)
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", test_repository)
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, experimental_material_system=True),
    )

    async def healthy() -> bool:
        return True

    async def count_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages) // 2 + 12

    async def analyze_chunk(
        title: str, content: str, _previous: str, _index: int,
        _total: int, _stop_event: Any, **_kwargs: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return (
            {"title": title, "summary": f"摘要：{content[:20]}"},
            [{"name": "林舟", "facts": ["正在查案"], "source_chapters": [title]}],
        )

    async def extract_facts(
        _title: str, content: str, _stop_event: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{
            "fact_key": f"clue-{content[:6]}",
            "fact_type": "foreshadowing",
            "subject": "钥匙",
            "predicate": "指向",
            "object": "旧车站",
            "state": "尚未回收",
            "status": "open",
            "evidence": content[:20],
        }]

    async def extract_unified_events(
        _title: str, _content: str, _stop_event: Any, **_kwargs: Any
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "plot_events": [{
                "title": "林舟发现钥匙线索",
                "description": "线索指向旧车站。",
                "event_type": "clue",
                "participants": ["林舟"],
                "confidence": 0.9,
            }],
            "character_events": [],
            "relationship_events": [],
            "location_events": [],
            "ability_events": [],
            "object_events": [],
            "unresolved_entities": [],
        }

    async def merge_chapter(
        title: str, parts: list[dict[str, Any]], _stop_event: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        return {"title": title, "summary": "；".join(item["summary"] for item in parts)}

    async def project_summary(
        _summaries: list[dict[str, Any]], _stop_event: Any, **_kwargs: Any
    ) -> str:
        callback = _kwargs.get("on_progress")
        if callback:
            callback("batch_started", 1, 1)
            callback("batch_completed", 1, 1)
        return "林舟追查苏晚失踪案，线索指向旧车站。"

    async def character_cards(
        _summaries: list[dict[str, Any]], _stop_event: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        callback = _kwargs.get("on_progress")
        if callback:
            callback("batch_started", 1, 1)
            callback("batch_completed", 1, 1)
        return [{"name": "林舟", "identity": "记者", "facts": ["正在查案"]}]

    async def summarize_increment(
        title: str,
        _previous_summary: str,
        new_content: str,
        _stop_event: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        callback = _kwargs.get("on_progress")
        if callback:
            callback("summary_chunk_started", 1, 1)
            callback("summary_chunk_completed", 1, 1)
            callback("character_chunk_started", 1, 1)
            callback("character_chunk_completed", 1, 1)
        return {
            "title": title,
            "summary": f"新增：{new_content}",
            "_character_observations": [
                {"name": "林舟", "facts": ["进入旧车站"]}
            ],
            "_chunk_summaries": [{"summary": new_content}],
        }

    async def fake_stream(
        _messages: list[dict[str, str]],
        _settings: dict[str, Any],
        _stop_event: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "content_delta", "text": "# 下一章\n\n林舟潜入旧车站档案室。"}
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.llama_client, "count_chat_tokens", count_tokens)
    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)
    monkeypatch.setattr(app_module.analysis_service, "analyze_chunk", analyze_chunk)
    monkeypatch.setattr(app_module.analysis_service, "extract_story_facts", extract_facts)
    monkeypatch.setattr(app_module.analysis_service, "extract_unified_events", extract_unified_events)
    monkeypatch.setattr(app_module.analysis_service, "merge_chapter_summaries", merge_chapter)
    monkeypatch.setattr(app_module.analysis_service, "build_project_summary", project_summary)
    monkeypatch.setattr(app_module.analysis_service, "extract_character_cards", character_cards)
    monkeypatch.setattr(app_module.analysis_service, "summarize_increment", summarize_increment)

    with TestClient(app_module.app) as client:
        imported = client.post(
            "/api/projects/default/import-txt",
            content="第一章 雨夜\n林舟遇见苏晚。\n\n第二章 失踪\n苏晚留下钥匙后失踪。".encode(),
            headers={"X-Filename": "%E6%97%A7%E7%A8%BF.txt"},
        )
        assert imported.status_code == 201
        assert len(imported.json()["chapters"]) == 2

        document_id = imported.json()["document"]["id"]
        summarized = client.post(
            "/api/projects/default/summarize",
            json={"document_id": document_id, "start_position": 1, "end_position": 2},
        )
        assert summarized.status_code == 200
        assert "event: characters_completed" in summarized.text
        assert "event: analysis_progress" in summarized.text

        project = client.get("/api/projects/default").json()
        workspace = client.get(f"/api/documents/{document_id}/workspace").json()
        assert all(chapter["status"] == "completed" for chapter in workspace["chapters"])
        assert workspace["global_summary"].startswith("林舟追查")
        assert workspace["characters"][0]["name"] == "林舟"
        assert workspace["facts"][0]["first_chapter"]
        assert workspace["chapters"][0]["character_observations"][0]["name"] == "林舟"
        material_overview = client.get(
            f"/api/experimental/material-system/documents/{document_id}/overview"
        ).json()
        assert any(
            event["title"] == "林舟发现钥匙线索"
            for event in material_overview["timeline"]["events"]
        )

        conversation = client.post("/api/conversations", json={"title": "续写"}).json()
        generated = client.post(
            f"/api/conversations/{conversation['id']}/outline/generate",
            json={"instruction": "加入一次潜入行动", "settings": {"max_tokens": 7000}},
        )
        assert generated.status_code == 200
        assert "event: done" in generated.text
        assert "outline_preview_created" in generated.text
        assert '"max_tokens": 7000' in generated.text

        assert client.get(f"/api/conversations/{conversation['id']}/outline").json() is None
        outline = client.post(
            f"/api/conversations/{conversation['id']}/outline/candidates",
            json={
                "instruction": "加入一次潜入行动",
                "content": "# 下一章\n\n林舟潜入旧车站档案室。",
                "select": True,
            },
        ).json()
        candidate = outline["candidates"][0]
        assert outline["selected_candidate_id"] == candidate["id"]
        edited = client.patch(
            f"/api/outline-candidates/{candidate['id']}",
            json={"content": "# 手调大纲\n\n林舟从地下通道进入。"},
        ).json()
        assert edited["candidates"][0]["edited_content"].startswith("# 手调大纲")
        enabled = client.patch(
            f"/api/outlines/{outline['id']}", json={"enabled": True}
        ).json()
        assert enabled["enabled"] is True

        context = client.post(
            f"/api/conversations/{conversation['id']}/context-count", json={"content": "写正文"}
        ).json()
        assert context["context_size"] in {32768, 65536}
        assert context["input_tokens"] > 0

        target_chapter = workspace["chapters"][-1]
        appended = client.post(
            "/api/projects/default/append",
            json={
                "chapter_id": target_chapter["id"],
                "document_id": document_id,
                "content": "林舟从地下通道进入旧车站。",
                "max_tokens": 5000,
                "summarize_now": False,
            },
        )
        assert appended.status_code == 200
        assert appended.json()["summarized"] is False
        updated_chapter = client.get(f"/api/chapters/{target_chapter['id']}").json()
        assert updated_chapter["content"].endswith("林舟从地下通道进入旧车站。")
        assert updated_chapter["status"] == "pending"
        exported = client.get(f"/api/documents/{document_id}/export.txt")
        assert exported.status_code == 200
        assert "地下通道" in exported.text
        exported_conversation = client.get(
            f"/api/conversations/{conversation['id']}/export",
            params={"format": "markdown", "include_all": True},
        )
        assert exported_conversation.status_code == 200
        assert "## 下一章大纲" in exported_conversation.text
        assert "状态：已启用" in exported_conversation.text
        assert "# 手调大纲" in exported_conversation.text
        assert "林舟从地下通道进入。" in exported_conversation.text


def test_append_immediate_summary_updates_only_relevant_character_cards(monkeypatch, tmp_path: Path) -> None:
    test_database = Database(tmp_path / "incremental-character-api.db")
    test_database.initialize()
    test_repository = NovelRepository(test_database)
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", test_repository)

    async def healthy() -> bool:
        return True

    async def summarize_increment(
        title: str,
        _previous_summary: str,
        new_content: str,
        _stop_event: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        callback = _kwargs.get("on_progress")
        if callback:
            callback("summary_chunk_started", 1, 1)
            callback("summary_chunk_completed", 1, 1)
            callback("character_chunk_started", 1, 1)
            callback("character_chunk_completed", 1, 1)
        return {
            "title": title,
            "summary": f"新增：{new_content}",
            "_chunk_summaries": [{"title": title, "summary": new_content}],
            "_character_observations": [
                {
                    "name": "林记者",
                    "aliases": ["林舟"],
                    "current_state": "潜入旧车站",
                    "source_chapters": [title],
                }
            ],
        }

    async def extract_facts(
        _title: str, _content: str, _stop_event: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        return []

    async def project_summary(
        _summaries: list[dict[str, Any]], _stop_event: Any, **_kwargs: Any
    ) -> str:
        return "林舟继续调查旧车站。"

    merge_calls: list[dict[str, Any]] = []

    async def merge_character_updates(
        existing_cards: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        _stop_event: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        merge_calls.append({"existing": existing_cards, "observations": observations})
        callback = _kwargs.get("on_progress")
        if callback:
            callback("batch_started", 1, 1)
            callback("batch_completed", 1, 1)
        return [{
            "id": existing_cards[0]["id"],
            "name": "林舟",
            "aliases": ["林记者"],
            "identity": "调查记者",
            "current_state": "潜入旧车站",
        }]

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.analysis_service, "summarize_increment", summarize_increment)
    monkeypatch.setattr(app_module.analysis_service, "extract_story_facts", extract_facts)
    monkeypatch.setattr(app_module.analysis_service, "build_project_summary", project_summary)
    monkeypatch.setattr(app_module.analysis_service, "merge_character_updates", merge_character_updates)

    imported = test_repository.import_document(
        "default", "旧稿.txt", "utf-8", "第一章 雨夜\n林舟被称作林记者。"
    )
    document_id = imported["document"]["id"]
    chapter_id = imported["chapters"][0]["id"]
    test_repository.save_chapter_summary(
        chapter_id,
        {"title": "第一章 雨夜", "summary": "林舟开始调查。"},
        [{"name": "林舟", "aliases": ["林记者"], "facts": ["开始调查"]}],
    )
    existing = test_repository.replace_characters(
        document_id,
        [{"name": "林舟", "aliases": ["林记者"], "identity": "调查记者"}],
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/projects/default/append",
            json={
                "chapter_id": chapter_id,
                "document_id": document_id,
                "content": "林记者从地下通道潜入旧车站。",
                "summarize_now": True,
            },
        )

    assert response.status_code == 200
    assert "event: characters_completed" in response.text
    assert merge_calls
    assert [card["id"] for card in merge_calls[0]["existing"]] == [existing[0]["id"]]
    assert [item["name"] for item in merge_calls[0]["observations"]] == ["林记者"]

    workspace = test_repository.get_document_workspace(document_id)
    assert len(workspace["characters"]) == 1
    assert workspace["characters"][0]["id"] == existing[0]["id"]
    assert workspace["characters"][0]["card"]["current_state"] == "潜入旧车站"


def test_analysis_can_pause_and_resume_from_saved_chunk(monkeypatch, tmp_path: Path) -> None:
    test_database = Database(tmp_path / "resume-api.db")
    test_database.initialize()
    test_repository = NovelRepository(test_database)
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", test_repository)

    async def healthy() -> bool:
        return True

    analyze_calls: list[str] = []

    async def analyze_chunk(
        title: str, _content: str, _previous: str, _index: int,
        _total: int, stop_event: Any, **_kwargs: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        analyze_calls.append(title)
        if len(analyze_calls) == 1:
            stop_event.set()
        return {"title": title, "summary": f"{title}摘要"}, []

    async def extract_facts(
        title: str, _content: str, stop_event: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        if stop_event.is_set():
            raise app_module.GenerationCancelled("测试暂停")
        return [{
            "fact_key": title,
            "fact_type": "timeline",
            "subject": title,
            "predicate": "发生",
            "object": "事件",
            "state": "已发生",
        }]

    async def merge_chapter(
        title: str, _parts: list[dict[str, Any]], _stop_event: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        return {"title": title, "summary": f"{title}合并摘要"}

    async def project_summary(
        _summaries: list[dict[str, Any]], _stop_event: Any, **_kwargs: Any
    ) -> str:
        return "全书总览"

    async def character_cards(
        _observations: list[dict[str, Any]], _stop_event: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.analysis_service, "analyze_chunk", analyze_chunk)
    monkeypatch.setattr(app_module.analysis_service, "extract_story_facts", extract_facts)
    monkeypatch.setattr(app_module.analysis_service, "merge_chapter_summaries", merge_chapter)
    monkeypatch.setattr(app_module.analysis_service, "build_project_summary", project_summary)
    monkeypatch.setattr(app_module.analysis_service, "extract_character_cards", character_cards)

    with TestClient(app_module.app) as client:
        imported = client.post(
            "/api/projects/default/import-txt",
            content="第一章 开端\n甲。\n\n第二章 继续\n乙。".encode(),
            headers={"X-Filename": "resume.txt"},
        ).json()
        document_id = imported["document"]["id"]
        paused = client.post(
            "/api/projects/default/summarize",
            json={"document_id": document_id, "start_position": 1, "end_position": 2},
        )
        assert "event: cancelled" in paused.text
        workspace = client.get(f"/api/documents/{document_id}/workspace").json()
        assert workspace["latest_job"]["status"] == "paused"
        first_chunk = client.get(
            f"/api/chapters/{workspace['chapters'][0]['id']}"
        ).json()["chunks"][0]
        assert first_chunk["summary"]["summary"].endswith("摘要")
        assert first_chunk["facts_status"] == "pending"

        resumed = client.post(
            "/api/projects/default/summarize",
            json={"resume_job_id": workspace["latest_job"]["id"]},
        )
        assert "event: done" in resumed.text
        final_workspace = client.get(f"/api/documents/{document_id}/workspace").json()
        assert final_workspace["latest_job"]["status"] == "completed"
        assert all(chapter["status"] == "completed" for chapter in final_workspace["chapters"])
        assert len(analyze_calls) == 2
