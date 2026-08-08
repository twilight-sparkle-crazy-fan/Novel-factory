from collections.abc import AsyncIterator
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import backend.app as app_module
from backend.database import Database
from backend.novel_repository import NovelRepository


def scene_outline_json(*scenes: tuple[str, str, str]) -> str:
    return json.dumps(
        {
            "chapter_goal": "推进下一章",
            "scenes": [
                {
                    "id": scene_id,
                    "title": title,
                    "purpose": "推进情节",
                    "entry": "夜晚，林舟来到旧车站，线索仍不完整",
                    "beats": [goal, "林舟确认钥匙仍在身上"],
                    "exit": "林舟获得新线索并准备继续深入",
                    "constraints": ["钥匙位置保持一致", "不能直接揭开全部真相"],
                    "budget": {"target_tokens": 800, "max_tokens": 1100},
                }
                for scene_id, title, goal in scenes
            ],
            "chapter_ending": "留下下一章钩子",
            "polish_checklist": ["场景衔接", "时间连续", "称呼一致"],
        },
        ensure_ascii=False,
    )


def test_dedupe_scene_continuation_removes_replayed_text() -> None:
    existing = (
        "林舟站在旧车站的月台边，听见铁轨深处传来细碎声响。"
        "他握紧剑柄，指节微微发白，却没有拔剑的意思。"
    )
    tail_replay = f"{existing[-24:]}门后的脚步声终于停下。"
    full_replay = f"{existing}门后的脚步声终于停下。"

    assert app_module.dedupe_scene_continuation(existing, tail_replay) == "门后的脚步声终于停下。"
    assert app_module.dedupe_scene_continuation(existing, full_replay) == "门后的脚步声终于停下。"
    punctuation_replay = "续写正文：他握紧剑柄，指节微微发白；却没有拔剑的意思。门后的脚步声终于停下。"
    assert app_module.dedupe_scene_continuation(existing, punctuation_replay) == "门后的脚步声终于停下。"


def test_outline_hook_instruction_can_be_disabled() -> None:
    with_hook = app_module.outline_instruction("推进调查", True)
    without_hook = app_module.outline_instruction("推进调查", False)

    assert "必须留下明确的新问题、危险、发现或行动驱动力" in with_hook
    assert "不要强行制造悬念、突发危险或未完句" in without_hook


def test_scene_cards_need_no_budget_and_legacy_budget_does_not_cap_output(monkeypatch) -> None:
    legacy_outline = json.loads(scene_outline_json(("S1", "潜入旧车站", "取得线索")))
    outline = json.loads(json.dumps(legacy_outline, ensure_ascii=False))
    outline["scenes"][0].pop("budget")

    scenes = app_module.parse_scene_cards(json.dumps(outline, ensure_ascii=False))
    legacy_scenes = app_module.parse_scene_cards(json.dumps(legacy_outline, ensure_ascii=False))

    monkeypatch.setattr(app_module, "active_max_output_tokens", lambda: 16_384)
    assert scenes[0]["label"] == "S1"
    assert legacy_scenes[0]["label"] == "S1"
    assert app_module.scene_output_token_limit() == 16_384
    prompt = app_module.outline_instruction("推进调查")
    assert '"budget"' not in prompt
    assert "target_tokens" in prompt
    assert "不要设置" in prompt


def test_scene_check_treats_rephrased_previous_state_as_deviation() -> None:
    prompt = app_module.scene_check_prompt(
        {"label": "S02", "title": "继续行动", "card": "第二场景"},
        "齿轮架的缺口泛着幽蓝光晕，林舟开始固定逆转轴。",
        "十二号齿轮之后留着一道缺口，边缘泛着幽蓝光晕。",
    )

    assert "用新措辞重新解释同一状态，仍算重复" in prompt
    assert "十二号齿轮之后留着一道缺口" in prompt


def test_chapter_selection_rewrite_uses_only_local_context_and_relevant_characters() -> None:
    selected = "林舟握住门把，没有立刻推门。"
    content = f"{'甲' * 7000}{selected}{'乙' * 7000}"
    start = content.index(selected)
    messages = app_module.chapter_selection_rewrite_messages(
        {
            "title": "第三章 暗门",
            "content": content,
        },
        {
            "short_summary": "林舟正在调查旧车站。",
            "characters": [
                {"name": "林舟", "aliases": [], "prompt_text": "林舟：调查记者，行动克制。"},
                {"name": "周岚", "aliases": [], "prompt_text": "周岚：机械师。"},
            ],
        },
        start,
        start + len(selected),
        "增强犹豫感，但不要开门。",
    )

    prompt = messages[-1]["content"]
    assert selected in prompt
    assert "增强犹豫感" in prompt
    assert "林舟：调查记者" in prompt
    assert "周岚：机械师" not in prompt
    assert "甲" * 6000 in prompt
    assert "甲" * 6001 not in prompt
    assert "乙" * 6000 in prompt
    assert "乙" * 6001 not in prompt


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
        **_kwargs: Any,
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


def test_generation_runs_single_pass_without_auto_continue(
    monkeypatch, tmp_path: Path
) -> None:
    test_database = Database(tmp_path / "single-pass-api.db")
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
        **_kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        captured_calls.append({"messages": messages, "settings": dict(settings)})
        yield {"type": "content_delta", "text": "第一段正文。"}
        yield {
            "type": "timings",
            "value": {"prompt_tokens": 20, "completion_tokens": 900},
        }
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "单次生成"}).json()
        client.patch(
            f"/api/conversations/{conversation['id']}",
            json={
                "style_guide": "表达直白，必要时不要改成委婉说法。",
                "style_lexicon": "暗星\n旧誓",
                "generation_settings": {"max_tokens": 1200, "min_completion_tokens": 3000},
            },
        )
        response = client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"content": "写下一章"},
        )
        assert response.status_code == 200
        assert "event: auto_continue_started" not in response.text

        stored = client.get(f"/api/conversations/{conversation['id']}").json()

    assert len(captured_calls) == 1
    first_system = captured_calls[0]["messages"][0]["content"]
    assert "表达直白" in first_system
    assert "暗星" in first_system
    assert "min_completion_tokens" not in captured_calls[0]["settings"]
    candidate = stored["exchanges"][0]["candidates"][0]
    assert candidate["content"] == "第一段正文。"
    assert candidate["completion_tokens"] == 900


def test_generation_ignores_legacy_minimum_completion_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    test_database = Database(tmp_path / "legacy-auto-continue-api.db")
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
        **_kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        captured_calls.append({"messages": messages, "settings": dict(settings)})
        yield {"type": "content_delta", "text": "短正文。"}
        yield {"type": "timings", "value": {"prompt_tokens": 20, "completion_tokens": 700}}
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "旧续写阈值兼容"}).json()
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
    assert "min_completion_tokens" not in stored["generation_settings"]
    assert stored["exchanges"][0]["candidates"][0]["completion_tokens"] == 700


def test_scene_workflow_pauses_for_fragment_review_then_polishes(monkeypatch, tmp_path: Path) -> None:
    test_database = Database(tmp_path / "scene-workflow-api.db")
    test_database.initialize()
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", NovelRepository(test_database))

    async def healthy() -> bool:
        return True

    async def count_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages) // 2 + 8

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.llama_client, "count_chat_tokens", count_tokens)

    calls: list[dict[str, Any]] = []
    prompts: list[str] = []

    async def fake_stream(
        messages: list[dict[str, str]],
        settings: dict[str, Any],
        _stop_event: Any,
        **_kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        prompt = messages[-1]["content"]
        calls.append({"messages": messages, "settings": dict(settings)})
        prompts.append(prompt)
        if "只返回 JSON" in prompt:
            yield {"type": "content_delta", "text": '{"status":"complete","reason":"完成","fix_instruction":""}'}
        elif "连续性检查结果" in prompt:
            yield {"type": "content_delta", "text": "最终章节正文。"}
        elif "请检查下面整章草稿的连续性" in prompt:
            yield {"type": "content_delta", "text": "衔接自然。"}
        else:
            yield {"type": "content_delta", "text": "当前场景正文。"}
        yield {"type": "timings", "value": {"prompt_tokens": 20, "completion_tokens": 30}}
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "场景流程"}).json()
        client.patch(
            f"/api/conversations/{conversation['id']}",
            json={
                "system_prompt": "不应进入独立润色系统提示词",
                "pinned_context": "不应进入独立润色固定资料",
                "style_guide": "不应进入独立润色风格要求",
                "style_lexicon": "不应进入独立润色词表",
            },
        )
        outline = client.post(
            f"/api/conversations/{conversation['id']}/outline/candidates",
            json={
                "instruction": "拆分下一章",
                "content": scene_outline_json(
                    ("S1", "潜入旧车站", "潜入旧车站"),
                    ("S2", "找到钥匙对应的门", "找到钥匙对应的门"),
                ),
                "select": True,
            },
        ).json()
        client.patch(f"/api/outlines/{outline['id']}", json={"enabled": True})
        response = client.post(
            f"/api/conversations/{conversation['id']}/scene-workflow",
            json={"instruction": "保持悬疑节奏", "settings": {"max_tokens": 1200}},
        )
        stored = client.get(f"/api/conversations/{conversation['id']}").json()
        candidate = stored["exchanges"][0]["candidates"][0]
        fragment_call_start = len(calls)
        fragment_response = client.post(
            f"/api/conversations/{conversation['id']}/scene-workflow/fragment",
            json={
                "candidate_id": candidate["id"],
                "instruction": "改写当前场景",
                "outline_text": outline["candidates"][0]["content"],
                "scene_index": 0,
                "scenes": [
                    {
                        "label": "S1",
                        "title": "潜入旧车站",
                        "card": json.dumps({"id": "S1", "title": "潜入旧车站"}, ensure_ascii=False),
                        "content": "当前场景正文。",
                    },
                    {
                        "label": "S2",
                        "title": "找到钥匙对应的门",
                        "card": json.dumps({"id": "S2", "title": "找到钥匙对应的门"}, ensure_ascii=False),
                        "content": "当前场景正文。",
                    },
                ],
                "settings": {"max_tokens": 1200},
            },
        )
        fragment_calls = calls[fragment_call_start:]
        calls_before_accept = len(calls)
        accept_response = client.post(
            f"/api/conversations/{conversation['id']}/scene-workflow/accept",
            json={
                "candidate_id": candidate["id"],
                "scenes": [
                    {
                        "label": "S1",
                        "title": "潜入旧车站",
                        "card": json.dumps({"id": "S1", "title": "潜入旧车站"}, ensure_ascii=False),
                        "content": "当前场景正文。",
                    },
                    {
                        "label": "S2",
                        "title": "找到钥匙对应的门",
                        "card": json.dumps({"id": "S2", "title": "找到钥匙对应的门"}, ensure_ascii=False),
                        "content": "当前场景正文。",
                    },
                ],
            },
        )
        calls_after_accept = len(calls)
        polish_response = client.post(
            f"/api/conversations/{conversation['id']}/scene-workflow/polish",
            json={
                "candidate_id": candidate["id"],
                "settings": {"max_tokens": 1200},
                "scenes": [
                    {
                        "label": "S1",
                        "title": "潜入旧车站",
                        "card": json.dumps({"id": "S1", "title": "潜入旧车站"}, ensure_ascii=False),
                        "content": "当前场景正文。",
                    },
                    {
                        "label": "S2",
                        "title": "找到钥匙对应的门",
                        "card": json.dumps({"id": "S2", "title": "找到钥匙对应的门"}, ensure_ascii=False),
                        "content": "当前场景正文。",
                    },
                ],
            },
        )
        polished = client.get(f"/api/conversations/{conversation['id']}").json()

    assert response.status_code == 200
    assert "event: workflow_step" in response.text
    assert "event: workflow_review_ready" in response.text
    assert fragment_response.status_code == 200
    assert "event: fragment_done" in fragment_response.text
    assert fragment_calls
    assert all(len(call["messages"]) == 2 for call in fragment_calls)
    assert all(
        not any(
            message["role"] == "assistant" and "当前场景正文" in message["content"]
            for message in call["messages"][:-1]
        )
        for call in fragment_calls
    )
    assert "scene_workflow_review_ready" in response.text
    draft_prompts = [prompt for prompt in prompts if "【当前场景：" in prompt]
    assert draft_prompts
    assert any("【当前场景：潜入旧车站】" in prompt for prompt in draft_prompts)
    assert all("场景目标：" in prompt for prompt in draft_prompts)
    assert all('"beats"' not in prompt and '"purpose"' not in prompt and '"id"' not in prompt for prompt in draft_prompts)
    draft_calls = [
        call for call in calls
        if call["messages"][-1]["content"].startswith("你正在按场景卡逐场景写作一章小说。")
    ]
    assert draft_calls
    assert all(
        call["settings"]["max_tokens"] == app_module.active_max_output_tokens()
        for call in draft_calls
    )
    assert "### S1" not in candidate["content"]
    assert stored["exchanges"][0]["user_content"].startswith("一键启动编排流程")
    assert accept_response.status_code == 200
    assert accept_response.json()["finish_reason"] == "scene_workflow_accepted"
    assert calls_after_accept == calls_before_accept
    assert "### S1" not in accept_response.json()["exchange"]["candidates"][0]["content"]
    assert polish_response.status_code == 200
    assert "event: content_replace" in polish_response.text
    assert "最终章节正文。" in polish_response.text
    assert polished["exchanges"][0]["candidates"][0]["content"] == "最终章节正文。"
    polish_calls = [
        call for call in calls
        if "整章草稿" in call["messages"][-1]["content"]
    ]
    assert len(polish_calls) == 2
    assert all(len(call["messages"]) == 2 for call in polish_calls)
    assert all(
        "不应进入独立润色" not in "\n".join(message["content"] for message in call["messages"])
        for call in polish_calls
    )
    assert polish_calls[-1]["settings"]["max_tokens"] > 3000
    assert polish_calls[-1]["settings"]["max_tokens"] <= app_module.POLISH_CONTEXT_TOKEN_LIMIT


def test_scene_workflow_dedupes_incomplete_continuation(monkeypatch, tmp_path: Path) -> None:
    test_database = Database(tmp_path / "scene-workflow-dedupe.db")
    test_database.initialize()
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", NovelRepository(test_database))

    async def healthy() -> bool:
        return True

    async def count_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages) // 2 + 8

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.llama_client, "count_chat_tokens", count_tokens)

    scene_text = (
        "林舟站在旧车站的月台边，听见铁轨深处传来细碎声响。"
        "他握紧剑柄，指节微微发白，却没有拔剑的意思。"
    )
    continuation_text = f"{scene_text[-28:]}门后的脚步声终于停下。"

    async def fake_stream(
        messages: list[dict[str, str]],
        _settings: dict[str, Any],
        _stop_event: Any,
        **_kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        prompt = messages[-1]["content"]
        if "只返回 JSON" in prompt:
            yield {"type": "content_delta", "text": '{"status":"incomplete","reason":"还缺少门后反应","fix_instruction":"补门后反应"}'}
        elif "当前场景还没有完成" in prompt:
            yield {"type": "content_delta", "text": continuation_text}
        else:
            yield {"type": "content_delta", "text": scene_text}
        yield {"type": "timings", "value": {"prompt_tokens": 20, "completion_tokens": 30}}
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "续写去重"}).json()
        outline = client.post(
            f"/api/conversations/{conversation['id']}/outline/candidates",
            json={
                "instruction": "拆分下一章",
                "content": scene_outline_json(("S1", "旧车站门后", "听见门后反应")),
                "select": True,
            },
        ).json()
        client.patch(f"/api/outlines/{outline['id']}", json={"enabled": True})
        response = client.post(
            f"/api/conversations/{conversation['id']}/scene-workflow",
            json={"instruction": "保持悬疑节奏", "settings": {"max_tokens": 1200}},
        )
        stored = client.get(f"/api/conversations/{conversation['id']}").json()

    assert response.status_code == 200
    content = stored["exchanges"][0]["candidates"][0]["content"]
    assert "门后的脚步声终于停下。" in content
    assert content.count(scene_text[-28:]) == 1


def test_outline_candidate_requires_json(monkeypatch, tmp_path: Path) -> None:
    test_database = Database(tmp_path / "outline-json-api.db")
    test_database.initialize()
    monkeypatch.setattr(app_module, "database", test_database)
    monkeypatch.setattr(app_module, "novels", NovelRepository(test_database))

    with TestClient(app_module.app) as client:
        conversation = client.post("/api/conversations", json={"title": "JSON 场景卡"}).json()
        response = client.post(
            f"/api/conversations/{conversation['id']}/outline/candidates",
            json={
                "instruction": "拆分下一章",
                "content": "# 场景卡\n\nS1：这不再允许。",
                "select": True,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "OUTLINE_JSON_INVALID"


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

    async def summarize_chapter(
        title: str, content: str, _stop_event: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        callback = _kwargs.get("on_progress")
        if callback:
            callback("summary_started", 1, 1)
            callback("summary_completed", 1, 1)
        return {"title": title, "summary": f"摘要：{content[:20]}"}

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

    async def new_character_cards(
        _title: str,
        _content: str,
        existing_cards: list[dict[str, Any]],
        _stop_event: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        callback = _kwargs.get("on_progress")
        if callback:
            callback("batch_started", 1, 1)
            callback("batch_completed", 1, 1)
        if existing_cards:
            return []
        return [{"name": "林舟", "identity": "记者", "facts": ["正在查案"]}]

    async def extract_cards_from_summaries(
        summaries: list[dict[str, str]],
        _existing: list[dict[str, Any]],
        _stop_event: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        assert len(summaries) == 2
        return [{
            "name": "苏晚",
            "card": {
                "character_name": "苏晚",
                "character_title": "线索持有人",
                "full_name": "苏晚",
                "aliases": [],
                "basic_info": {"identity": "失踪者"},
                "core_personality": {},
                "behavior_habits": [],
                "world_setting": "",
            },
            "source_chapters": [item["title"] for item in summaries],
        }]

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
        **_kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "content_delta", "text": scene_outline_json(("S1", "潜入档案室", "林舟潜入旧车站档案室"))}
        yield {"type": "done"}

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.llama_client, "count_chat_tokens", count_tokens)
    monkeypatch.setattr(app_module.llama_client, "stream_chat", fake_stream)
    monkeypatch.setattr(app_module.analysis_service, "analyze_chunk", analyze_chunk)
    monkeypatch.setattr(app_module.analysis_service, "summarize_chapter", summarize_chapter)
    monkeypatch.setattr(app_module.analysis_service, "extract_story_facts", extract_facts)
    monkeypatch.setattr(app_module.analysis_service, "extract_unified_events", extract_unified_events)
    monkeypatch.setattr(app_module.analysis_service, "merge_chapter_summaries", merge_chapter)
    monkeypatch.setattr(app_module.analysis_service, "build_project_summary", project_summary)
    monkeypatch.setattr(app_module.analysis_service, "extract_new_character_cards", new_character_cards)
    monkeypatch.setattr(
        app_module.analysis_service,
        "extract_character_cards_from_summaries",
        extract_cards_from_summaries,
    )
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
        assert "event: characters_completed" not in summarized.text
        assert "event: analysis_progress" in summarized.text

        project = client.get("/api/projects/default").json()
        workspace = client.get(f"/api/documents/{document_id}/workspace").json()
        assert all(chapter["status"] == "completed" for chapter in workspace["chapters"])
        assert workspace["global_summary"].startswith("林舟追查")
        assert workspace["characters"] == []
        assert workspace["facts"] == []
        assert workspace["chapters"][0]["character_observations"] == []
        created_character = client.post(
            f"/api/documents/{document_id}/characters",
            json={"card": {"character_name": "林舟", "character_title": "记者"}},
        )
        assert created_character.status_code == 201
        assert created_character.json()["card"]["character_name"] == "林舟"
        extracted = client.post(
            f"/api/documents/{document_id}/characters/extract",
            json={"start_position": 1, "end_position": 2},
        )
        assert extracted.status_code == 200
        assert "event: done" in extracted.text
        character_names = {
            item["name"]
            for item in client.get(f"/api/documents/{document_id}/workspace").json()["characters"]
        }
        assert character_names == {"林舟", "苏晚"}

        conversation = client.post("/api/conversations", json={"title": "续写"}).json()
        generated = client.post(
            f"/api/conversations/{conversation['id']}/outline/generate",
            json={"instruction": "加入一次潜入行动", "settings": {"max_tokens": 7000}},
        )
        assert generated.status_code == 200
        assert "event: done" in generated.text
        assert "outline_preview_created" in generated.text
        assert '"max_tokens": 10500' in generated.text

        assert client.get(f"/api/conversations/{conversation['id']}/outline").json() is None
        outline = client.post(
            f"/api/conversations/{conversation['id']}/outline/candidates",
            json={
                "instruction": "加入一次潜入行动",
                "content": scene_outline_json(("S1", "潜入档案室", "林舟潜入旧车站档案室")),
                "select": True,
            },
        ).json()
        candidate = outline["candidates"][0]
        assert outline["selected_candidate_id"] == candidate["id"]
        edited = client.patch(
            f"/api/outline-candidates/{candidate['id']}",
            json={"content": scene_outline_json(("S1", "地下通道", "林舟从地下通道进入"))},
        ).json()
        assert '"title": "地下通道"' in edited["candidates"][0]["edited_content"]
        enabled = client.patch(
            f"/api/outlines/{outline['id']}", json={"enabled": True}
        ).json()
        assert enabled["enabled"] is True

        context = client.post(
            f"/api/conversations/{conversation['id']}/context-count", json={"content": "写正文"}
        ).json()
        assert context["context_size"] in {40960, 81920}
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
        assert "## 下一章场景卡" in exported_conversation.text
        assert "状态：已启用" in exported_conversation.text
        assert '"title": "地下通道"' in exported_conversation.text
        assert "林舟从地下通道进入" in exported_conversation.text
        exported_backup = client.get(
            f"/api/conversations/{conversation['id']}/export",
            params={"format": "json"},
        )
        assert exported_backup.status_code == 200
        backup = exported_backup.json()
        assert backup["outline"]["enabled"] is True
        assert backup["outline"]["selected_candidate_id"] == candidate["id"]
        assert '"title": "地下通道"' in backup["outline"]["candidates"][0]["edited_content"]
        restored = client.post("/api/conversations/import", json=backup)
        assert restored.status_code == 201
        restored_conversation = restored.json()
        assert restored_conversation["id"] != conversation["id"]
        assert restored_conversation["title"] == "续写 · 备份恢复"
        restored_outline = client.get(
            f"/api/conversations/{restored_conversation['id']}/outline"
        ).json()
        assert restored_outline["enabled"] is True
        assert restored_outline["selected_candidate_id"] != candidate["id"]
        assert '"title": "地下通道"' in restored_outline["candidates"][0]["edited_content"]


def test_append_immediate_summary_does_not_auto_update_character_cards(monkeypatch, tmp_path: Path) -> None:
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
                    "name": "林舟",
                    "aliases": ["林记者"],
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

    first_appearance_calls: list[list[str]] = []

    async def extract_new_character_cards(
        _title: str,
        _content: str,
        existing_cards: list[dict[str, Any]],
        _stop_event: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        first_appearance_calls.append([card["id"] for card in existing_cards])
        callback = _kwargs.get("on_progress")
        if callback:
            callback("batch_started", 1, 1)
            callback("batch_completed", 1, 1)
        return []

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.analysis_service, "summarize_increment", summarize_increment)
    monkeypatch.setattr(app_module.analysis_service, "extract_story_facts", extract_facts)
    monkeypatch.setattr(app_module.analysis_service, "build_project_summary", project_summary)
    monkeypatch.setattr(app_module.analysis_service, "extract_new_character_cards", extract_new_character_cards)

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
    assert "event: characters_completed" not in response.text
    assert first_appearance_calls == []

    workspace = test_repository.get_document_workspace(document_id)
    assert len(workspace["characters"]) == 1
    assert workspace["characters"][0]["id"] == existing[0]["id"]
    assert "current_state" not in workspace["characters"][0]["card"]
    assert workspace["characters"][0]["events"] == []


def test_analysis_can_pause_and_resume_from_saved_chapter(monkeypatch, tmp_path: Path) -> None:
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

    async def summarize_chapter(
        title: str, _content: str, stop_event: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        analyze_calls.append(title)
        if len(analyze_calls) == 1:
            stop_event.set()
        callback = _kwargs.get("on_progress")
        if callback:
            callback("summary_started", 1, 1)
            callback("summary_completed", 1, 1)
        return {"title": title, "summary": f"{title}摘要"}

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

    async def new_character_cards(
        _title: str,
        _content: str,
        _existing: list[dict[str, Any]],
        _stop_event: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(app_module.llama_process, "is_healthy", healthy)
    monkeypatch.setattr(app_module.analysis_service, "analyze_chunk", analyze_chunk)
    monkeypatch.setattr(app_module.analysis_service, "summarize_chapter", summarize_chapter)
    monkeypatch.setattr(app_module.analysis_service, "extract_story_facts", extract_facts)
    monkeypatch.setattr(app_module.analysis_service, "merge_chapter_summaries", merge_chapter)
    monkeypatch.setattr(app_module.analysis_service, "build_project_summary", project_summary)
    monkeypatch.setattr(app_module.analysis_service, "extract_new_character_cards", new_character_cards)

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
        first_chapter = client.get(
            f"/api/chapters/{workspace['chapters'][0]['id']}"
        ).json()
        assert first_chapter["summary"]["summary"].endswith("摘要")

        resumed = client.post(
            "/api/projects/default/summarize",
            json={"resume_job_id": workspace["latest_job"]["id"]},
        )
        assert "event: done" in resumed.text
        final_workspace = client.get(f"/api/documents/{document_id}/workspace").json()
        assert final_workspace["latest_job"]["status"] == "completed"
        assert all(chapter["status"] == "completed" for chapter in final_workspace["chapters"])
        assert len(analyze_calls) == 2
