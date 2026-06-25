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
    assert "苏晚失踪并留下钥匙" in context["recent_chapters"]
    assert "调查记者" in context["characters"]
    assert "旧档案室" in context["outline"]


def test_character_cards_are_matched_by_alias_and_keep_one_id(tmp_path: Path) -> None:
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
            "name": "林记者",
            "aliases": ["林舟", "舟哥"],
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
        document_id, [{"name": "舟哥", "aliases": []}]
    )
    assert [item["id"] for item in relevant] == [character_id]


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
    assert "红伞" in context["facts"]
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
