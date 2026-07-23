from pathlib import Path

from backend.database import Database
from backend.novel_repository import NovelRepository
from backend.text_import import decode_text, split_chapters


SETTINGS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 1200,
    "repeat_penalty": 1.08,
    "seed": 11,
}


def make_repository(tmp_path: Path) -> tuple[Database, NovelRepository]:
    database = Database(tmp_path / "novel.db")
    database.initialize()
    return database, NovelRepository(database)


def test_txt_decode_and_chapter_split() -> None:
    source = "书名\r\n\r\n第一章 雨夜\r\n她回到车站。\r\n\r\n第二章 来客\r\n门响了。"
    imported = decode_text(source.encode("gb18030"))
    chapters = split_chapters(imported.text)

    assert imported.encoding == "gb18030"
    assert [chapter.title for chapter in chapters] == ["序章", "第一章 雨夜", "第二章 来客"]
    assert chapters[-1].content == "门响了。"


def test_chapter_split_accepts_leading_and_inner_spaces() -> None:
    source = "　　第一章 雨夜\n她回到车站。\n\n  第 2 章 来客\n门响了。"
    chapters = split_chapters(source)

    assert [chapter.title for chapter in chapters] == ["第一章 雨夜", "第 2 章 来客"]
    assert chapters[0].content == "她回到车站。"
    assert chapters[1].content == "门响了。"


def test_project_context_contains_summary_character_and_selected_outline(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    conversation = database.create_conversation()
    imported = repository.import_document(
        "default",
        "旧稿.txt",
        "utf-8",
        "第一章 相遇\n林舟在雨夜遇见苏晚。\n\n第二章 失踪\n苏晚留下钥匙后失踪。",
    )
    first, second = imported["chapters"]
    document_id = imported["document"]["id"]
    repository.save_chapter_summary(first["id"], {"title": first["title"], "summary": "林舟遇见苏晚。"})
    repository.save_chapter_summary(second["id"], {"title": second["title"], "summary": "苏晚失踪并留下钥匙。"})
    repository.save_document_summary(document_id, "林舟正在调查苏晚失踪案。")
    repository.replace_characters(
        document_id,
        [{"name": "林舟", "identity": "调查记者", "facts": ["持有苏晚的钥匙"]}],
    )

    outline = repository.get_or_create_outline(conversation["id"])
    outline, candidate = repository.create_outline_candidate(outline["id"], SETTINGS, 11)
    outline = repository.finalize_outline_candidate(candidate["id"], "completed", "林舟用钥匙打开旧档案室。")
    repository.update_outline(outline["id"], enabled=True)

    context = repository.get_prompt_context(conversation["id"])
    assert "调查苏晚失踪案" in context["project_summary"]
    assert context["short_summary"] == ""
    assert "林舟遇见苏晚。" in context["recent_chapters"]
    assert "苏晚失踪并留下钥匙" in context["recent_chapters"]
    assert '"summary"' not in context["recent_chapters"]
    assert "调查记者" in context["characters"]
    assert "旧档案室" in context["outline"]


def test_character_cards_update_only_when_standard_name_matches(tmp_path: Path) -> None:
    _database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default", "人物.txt", "utf-8", "第一章 雨夜\n林舟被人称作林记者。"
    )
    document_id = imported["document"]["id"]

    first = repository.replace_characters(
        document_id,
        [{"name": "林舟", "aliases": ["林记者"], "identity": "调查记者"}],
    )
    character_id = first[0]["id"]

    second = repository.replace_characters(
        document_id,
        [{
            "name": "林舟",
            "aliases": ["林记者", "舟哥"],
            "facts": ["拿到旧钥匙"],
            "current_state": "进入旧车站",
        }],
    )

    assert len(second) == 1
    assert second[0]["id"] == character_id
    assert second[0]["name"] == "林舟"
    assert "林记者" in second[0]["aliases"]
    assert "舟哥" in second[0]["aliases"]
    assert second[0]["card"]["identity"] == "调查记者"
    assert second[0]["card"]["current_state"] == "进入旧车站"

    relevant = repository.get_relevant_character_cards(
        document_id, [{"name": "林舟", "aliases": []}]
    )
    assert [item["id"] for item in relevant] == [character_id]


def test_character_cards_do_not_merge_by_alias_only(tmp_path: Path) -> None:
    _database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "神雕.txt",
        "utf-8",
        "第一章 重逢\n杨过望见小龙女。众人有时称小龙女为龙姑娘。",
    )
    document_id = imported["document"]["id"]

    repository.replace_characters(
        document_id,
        [{"name": "杨过", "aliases": ["过儿"], "identity": "少年侠客"}],
    )

    characters = repository.replace_characters(
        document_id,
        [{
            "name": "小龙女",
            "aliases": ["杨过", "姑姑", "龙姑娘"],
            "identity": "古墓派人物",
            "appearance": "白衣清冷",
        }],
    )
    by_name = {item["name"]: item for item in characters}

    assert set(by_name) == {"杨过", "小龙女"}
    assert by_name["杨过"]["card"]["identity"] == "少年侠客"
    assert by_name["小龙女"]["card"]["identity"] == "古墓派人物"


def test_character_event_records_are_visible_and_opt_in(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default", "事件.txt", "utf-8", "第一章 雨夜\n林舟拿到旧钥匙。"
    )
    document_id = imported["document"]["id"]

    repository.replace_characters(
        document_id,
        [{
            "name": "林舟",
            "identity": "调查记者",
            "event_records": [{
                "chapter": "第一章 雨夜",
                "event": "林舟拿到旧钥匙",
                "impact": "获得进入旧车站的线索",
                "importance": "high",
                "tags": ["线索", "物品"],
                "consequences": {"Abstract": "林舟拿到旧钥匙，获得进入旧车站的线索。"},
            }],
        }],
    )

    workspace = repository.get_document_workspace(document_id)
    character = workspace["characters"][0]
    assert character["events"][0]["abstract"] == "林舟拿到旧钥匙，获得进入旧车站的线索。"
    assert character["events"][0]["tags"] == ["线索", "物品"]
    assert character["events"][0]["enabled"] is False
    assert "事件摘要：林舟拿到旧钥匙" not in character["prompt_text"]

    conversation = database.create_conversation(document_id=document_id)
    context = repository.get_prompt_context(conversation["id"])
    assert "林舟拿到旧钥匙，获得进入旧车站的线索。" not in context["characters"]

    repository.update_character_event(character["events"][0]["id"], {"enabled": True})
    context = repository.get_prompt_context(conversation["id"])
    assert "注入事件" in context["characters"]
    assert "林舟拿到旧钥匙，获得进入旧车站的线索。" in context["characters"]


def test_chapter_summary_becomes_idempotent_experience_for_mentioned_characters(tmp_path: Path) -> None:
    _database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default", "经历.txt", "utf-8", "第一章 雨夜\n林记者进入旧车站，苏晚没有出现。"
    )
    document_id = imported["document"]["id"]
    repository.replace_characters(
        document_id,
        [
            {"name": "林舟", "aliases": ["林记者"], "identity": "调查记者"},
            {"name": "周岚", "identity": "法医"},
        ],
    )

    first = repository.upsert_character_chapter_experiences(
        document_id,
        "第一章 雨夜",
        "林舟从地下通道进入旧车站。",
        "林记者进入旧车站，苏晚没有出现。",
    )
    by_name = {item["name"]: item for item in first}
    assert len(by_name["林舟"]["events"]) == 1
    assert by_name["林舟"]["events"][0]["abstract"] == "林舟从地下通道进入旧车站。"
    assert by_name["周岚"]["events"] == []

    repository.update_character_event(by_name["林舟"]["events"][0]["id"], {"enabled": True})
    second = repository.upsert_character_chapter_experiences(
        document_id,
        "第一章 雨夜",
        "林舟进入旧车站，并发现封闭档案室。",
        "林记者进入旧车站，并发现封闭档案室。",
    )
    updated = {item["name"]: item for item in second}["林舟"]
    assert len(updated["events"]) == 1
    assert updated["events"][0]["enabled"] is True
    assert updated["events"][0]["abstract"] == "林舟进入旧车站，并发现封闭档案室。"


def test_replace_chapter_selection_checks_hash_and_exact_original(tmp_path: Path) -> None:
    _database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default", "重写.txt", "utf-8", "第一章 雨夜\n开头。旧句。结尾。"
    )
    chapter = repository.get_chapter(imported["chapters"][0]["id"])
    start = chapter["content"].index("旧句")
    end = start + len("旧句")

    updated = repository.replace_chapter_selection(
        chapter["id"],
        start=start,
        end=end,
        source_hash=chapter["content_hash"],
        original_text="旧句",
        replacement="林舟压低声音说出新句",
    )
    assert "林舟压低声音说出新句" in updated["content"]
    assert updated["status"] == "pending"
    assert updated["summary"] == {}

    try:
        repository.replace_chapter_selection(
            chapter["id"],
            start=start,
            end=end,
            source_hash=chapter["content_hash"],
            original_text="旧句",
            replacement="过期覆盖",
        )
    except ValueError as exc:
        assert "已发生变化" in str(exc)
    else:
        raise AssertionError("过期 hash 不应覆盖最新章节")


def test_manual_character_merge_preserves_fields_events_and_chosen_name(tmp_path: Path) -> None:
    _database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default", "神雕.txt", "utf-8", "第一章 重逢\n杨过与小龙女重逢。"
    )
    document_id = imported["document"]["id"]
    repository.replace_characters(
        document_id,
        [{
            "name": "杨过",
            "aliases": ["过儿"],
            "identity": "少年侠客",
            "event_records": [{
                "chapter": "第一章 重逢",
                "event": "杨过等候小龙女",
                "impact": "推动重逢",
                "consequences": {"Abstract": "杨过在谷口等候小龙女。"},
            }],
        }],
    )
    repository.replace_characters(
        document_id,
        [{
            "name": "小龙女",
            "aliases": ["龙姑娘"],
            "identity": "古墓派传人",
            "appearance": "白衣清冷",
            "event_records": [{
                "chapter": "第一章 重逢",
                "event": "小龙女现身",
                "impact": "回应杨过等待",
                "consequences": {"Abstract": "小龙女在月下现身。"},
            }],
        }],
    )

    workspace = repository.get_document_workspace(document_id)
    by_name = {item["name"]: item for item in workspace["characters"]}
    repository.update_character_event(by_name["小龙女"]["events"][0]["id"], {"enabled": True})

    workspace = repository.merge_characters(
        by_name["小龙女"]["id"],
        by_name["杨过"]["id"],
        "小龙女",
    )

    assert len(workspace["characters"]) == 1
    character = workspace["characters"][0]
    assert character["name"] == "小龙女"
    assert "杨过" in character["aliases"]
    assert "过儿" in character["aliases"]
    assert "龙姑娘" in character["aliases"]
    assert "少年侠客" in str(character["card"]["identity"])
    assert "古墓派传人" in str(character["card"]["identity"])
    assert character["card"]["appearance"] == "白衣清冷"
    assert {event["abstract"] for event in character["events"]} == {
        "杨过在谷口等候小龙女。",
        "小龙女在月下现身。",
    }
    enabled_events = [event for event in character["events"] if event["enabled"]]
    assert [event["abstract"] for event in enabled_events] == ["小龙女在月下现身。"]


def test_new_outline_group_disables_old_prompt_outline(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    conversation = database.create_conversation()
    first = repository.get_or_create_outline(conversation["id"])
    first, candidate = repository.create_outline_candidate(first["id"], SETTINGS, 11)
    repository.finalize_outline_candidate(candidate["id"], "completed", "旧大纲")
    repository.update_outline(first["id"], enabled=True)

    second = repository.get_or_create_outline(conversation["id"], force_new=True)
    assert second["enabled"] is False
    assert repository.get_outline(first["id"])["enabled"] is False
    assert repository.get_prompt_context(conversation["id"])["outline"] == ""


def test_long_chapter_is_chunked_and_interrupted_status_recovers(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    long_text = "第一章 长夜\n" + "\n\n".join(
        f"第{index}段。" + "雨声" * 900 for index in range(18)
    )
    imported = repository.import_document("default", "长篇.txt", "utf-8", long_text)
    chapter = repository.get_chapter(imported["chapters"][0]["id"])

    assert chapter["chunk_count"] >= 3
    assert all(len(chunk["content"]) <= 12_000 for chunk in chapter["chunks"])

    repository.set_chapter_status(chapter["id"], "processing")
    database.initialize()
    recovered = repository.get_chapter(chapter["id"])
    assert recovered["status"] == "pending"
    assert all(chunk["status"] == "pending" for chunk in recovered["chunks"])


def test_append_delete_and_export_project_content(tmp_path: Path) -> None:
    _database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default", "正文.txt", "utf-8", "第一章 开端\n旧正文。"
    )
    document_id = imported["document"]["id"]
    chapter_id = imported["chapters"][0]["id"]
    repository.save_chapter_summary(
        chapter_id,
        {"title": "第一章 开端", "summary": "旧摘要"},
        [{"name": "林舟", "facts": ["是记者"]}],
    )
    repository.save_document_summary(document_id, "旧总览")
    repository.replace_characters(document_id, [{"name": "林舟", "identity": "记者"}])

    appended = repository.append_content(
        "default", "新增正文。", chapter_id=chapter_id
    )
    assert appended["previous_summary"]
    assert appended["chapter"]["character_observations"][0]["name"] == "林舟"
    assert appended["chapter"]["content"].endswith("新增正文。")
    assert appended["chapter"]["chunks"][-1]["status"] == "pending"

    name, exported = repository.export_document_text(document_id)
    assert name == "正文"
    assert "第一章 开端" in exported and "新增正文。" in exported

    returned_document_id = repository.delete_chapter(chapter_id)
    workspace = repository.get_document_workspace(returned_document_id)
    assert workspace["chapters"] == []
    assert workspace["characters"] == []
    assert workspace["global_summary"] == ""


def test_documents_are_isolated_and_prompt_switches_do_not_leak(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    first = repository.import_document(
        "default", "甲.txt", "utf-8", "第一章 甲章\n甲世界的青灯。"
    )
    second = repository.import_document(
        "default", "乙.txt", "utf-8", "第一章 乙章\n乙世界的红伞。"
    )
    first_document = first["document"]["id"]
    second_document = second["document"]["id"]
    assert first["chapters"][0]["position"] == 1
    assert second["chapters"][0]["position"] == 1

    repository.save_chapter_summary(
        first["chapters"][0]["id"],
        {"title": "第一章 甲章", "summary": "甲世界的青灯仍亮着。"},
    )
    repository.save_chapter_summary(
        second["chapters"][0]["id"],
        {"title": "第一章 乙章", "summary": "乙世界的红伞已经遗失。"},
    )
    repository.save_document_summary(first_document, "甲世界总览")
    repository.save_document_summary(second_document, "乙世界总览")
    repository.replace_characters(first_document, [{"name": "甲主角", "identity": "守灯人"}])
    repository.replace_characters(second_document, [{"name": "乙主角", "identity": "寻伞人"}])

    second_chunk = repository.get_chapter(second["chapters"][0]["id"])["chunks"][0]
    repository.save_story_facts(second_document, second["chapters"][0]["id"], second_chunk["id"], [{
        "fact_key": "red-umbrella",
        "fact_type": "item",
        "subject": "红伞",
        "predicate": "持有人",
        "object": "未知",
        "state": "遗失",
        "status": "active",
        "evidence": "红伞在雨夜遗失",
    }])

    conversation = database.create_conversation(document_id=second_document)
    context = repository.get_prompt_context(conversation["id"], query_text="红伞在哪里")
    assert "乙世界" in context["project_summary"]
    assert "红伞" in context["recent_chapters"]
    assert "寻伞人" in context["characters"]
    assert context["facts"] == ""
    assert "甲世界" not in str(context)
    assert "守灯人" not in str(context)

    repository.update_document(second_document, {
        "summary_enabled": False,
        "recent_chapters_enabled": False,
        "characters_enabled": False,
        "facts_enabled": False,
    })
    disabled = repository.get_prompt_context(conversation["id"], query_text="红伞在哪里")
    assert disabled == {
        "project_summary": "",
        "short_summary": "",
        "recent_chapters": "",
        "characters": "",
        "facts": "",
        "outline": "",
    }


def test_outline_candidate_can_be_deleted_or_cleared(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    conversation = database.create_conversation()
    outline = repository.save_outline_candidate(
        conversation["id"],
        outline_id=None,
        instruction="推进主线",
        content="第一版大纲",
        settings=SETTINGS,
        seed=11,
        select=True,
    )
    candidate_id = outline["selected_candidate_id"]
    outline = repository.delete_outline_candidate(candidate_id)
    assert outline["candidates"] == []
    assert outline["selected_candidate_id"] is None
    assert outline["enabled"] is False

    repository.delete_outline(outline["id"])
    assert repository.find_latest_outline(conversation["id"]) is None
