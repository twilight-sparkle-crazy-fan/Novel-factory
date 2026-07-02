from pathlib import Path

import pytest

from backend.database import Database


SETTINGS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 100,
    "repeat_penalty": 1.08,
    "seed": 7,
}


def make_database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database


def complete(database: Database, candidate_id: str, text: str) -> None:
    database.finalize_candidate(
        candidate_id,
        status="completed",
        content=text,
        reasoning="",
    )


def test_regeneration_preserves_candidates_and_selection(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    conversation = database.create_conversation()
    exchange, first = database.create_exchange_with_candidate(
        conversation["id"], "续写", SETTINGS, 7
    )
    complete(database, first["id"], "第一版")
    exchange, second = database.create_candidate(exchange["id"], SETTINGS, 8)
    complete(database, second["id"], "第二版")

    stored = database.get_exchange(exchange["id"])
    assert [item["content"] for item in stored["candidates"]] == ["第一版", "第二版"]
    assert stored["selected_candidate_id"] == first["id"]

    stored = database.select_candidate(exchange["id"], second["id"])
    assert stored["selected_candidate_id"] == second["id"]


def test_context_uses_only_selected_candidate(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    conversation = database.create_conversation()
    first_exchange, first = database.create_exchange_with_candidate(
        conversation["id"], "第一问", SETTINGS, 7
    )
    complete(database, first["id"], "没有选中的版本")
    _, second = database.create_candidate(first_exchange["id"], SETTINGS, 8)
    complete(database, second["id"], "最终选中的版本")
    database.select_candidate(first_exchange["id"], second["id"])

    second_exchange, _ = database.create_exchange_with_candidate(
        conversation["id"], "第二问", SETTINGS, 9
    )
    _, history, current = database.get_context_source(second_exchange["id"])

    assert current == "第二问"
    assert {"role": "assistant", "content": "最终选中的版本"} in history
    assert "没有选中的版本" not in str(history)


def test_changing_historical_selection_requires_branch(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    conversation = database.create_conversation()
    first_exchange, first = database.create_exchange_with_candidate(
        conversation["id"], "第一问", SETTINGS, 7
    )
    complete(database, first["id"], "第一版")
    _, alternative = database.create_candidate(first_exchange["id"], SETTINGS, 8)
    complete(database, alternative["id"], "另一版")
    later_exchange, later = database.create_exchange_with_candidate(
        conversation["id"], "后续", SETTINGS, 9
    )
    complete(database, later["id"], "后续内容")

    with pytest.raises(RuntimeError, match="branch_required"):
        database.select_candidate(first_exchange["id"], alternative["id"])

    branch = database.create_branch(first_exchange["id"], alternative["id"])
    assert branch["title"].endswith("· 分支")
    assert len(branch["exchanges"]) == 1
    selected = branch["exchanges"][0]["selected_candidate_id"]
    candidate = next(item for item in branch["exchanges"][0]["candidates"] if item["id"] == selected)
    assert candidate["content"] == "另一版"


def test_import_conversation_backup_remaps_candidates(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    restored = database.import_conversation_backup({
        "title": "旧备份",
        "system_prompt": "写小说",
        "pinned_context": "固定资料",
        "style_guide": "冷峻",
        "style_lexicon": "暗星",
        "generation_settings": {**SETTINGS, "min_completion_tokens": 1200},
        "exchanges": [{
            "id": "old-exchange",
            "user_content": "第一问",
            "selected_candidate_id": "old-candidate-2",
            "candidates": [
                {
                    "id": "old-candidate-1",
                    "candidate_index": 1,
                    "content": "第一版",
                    "status": "completed",
                    "settings_snapshot": SETTINGS,
                    "seed": 7,
                },
                {
                    "id": "old-candidate-2",
                    "candidate_index": 2,
                    "content": "第二版",
                    "status": "completed",
                    "settings_snapshot": SETTINGS,
                    "seed": 8,
                },
                {
                    "id": "old-candidate-3",
                    "candidate_index": 3,
                    "content": "半截",
                    "status": "streaming",
                    "settings_snapshot": SETTINGS,
                    "seed": 9,
                },
            ],
        }],
    })

    assert restored["title"] == "旧备份 · 备份恢复"
    assert restored["system_prompt"] == "写小说"
    assert restored["pinned_context"] == "固定资料"
    assert restored["style_guide"] == "冷峻"
    assert restored["style_lexicon"] == "暗星"
    assert restored["generation_settings"]["min_completion_tokens"] == 1200

    exchange = restored["exchanges"][0]
    assert exchange["id"] != "old-exchange"
    assert exchange["selected_candidate_id"] not in {"old-candidate-1", "old-candidate-2"}
    selected = next(item for item in exchange["candidates"] if item["id"] == exchange["selected_candidate_id"])
    assert selected["content"] == "第二版"
    interrupted = next(item for item in exchange["candidates"] if item["content"] == "半截")
    assert interrupted["status"] == "cancelled"
    assert interrupted["error_message"] == "备份恢复时生成尚未完成"


def test_initialize_recovers_interrupted_candidate(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    conversation = database.create_conversation()
    exchange, candidate = database.create_exchange_with_candidate(
        conversation["id"], "续写", SETTINGS, 7
    )
    database.update_candidate_draft(candidate["id"], "部分正文", "")

    database.initialize()
    recovered = database.get_exchange(exchange["id"])["candidates"][0]
    assert recovered["status"] == "cancelled"
    assert recovered["content"] == "部分正文"
