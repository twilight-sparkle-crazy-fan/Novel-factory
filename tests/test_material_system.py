from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

import backend.app as app_module
from backend.database import Database
from backend.material_system import MATERIAL_SCHEMA_VERSION, PACKAGE_FORMAT
from backend.material_utils import stable_text_hash
from backend.novel_repository import NovelRepository


def make_repository(tmp_path: Path, name: str = "materials.db") -> tuple[Database, NovelRepository]:
    database = Database(tmp_path / name)
    database.initialize()
    return database, NovelRepository(database)


def test_material_package_export_validate_and_pure_new_import(tmp_path: Path) -> None:
    source_database, source_repository = make_repository(tmp_path, "source.db")
    imported = source_repository.import_document(
        "default",
        "星港.txt",
        "utf-8",
        "第一章 星港\n林舟抵达星港。\n\n第二章 旧誓\n苏晚留下旧誓。",
    )
    document_id = imported["document"]["id"]
    package = app_module.MaterialPackageService(source_database).export_document_package(document_id)

    with zipfile.ZipFile(BytesIO(package)) as archive:
        assert set(archive.namelist()) >= {
            "manifest.json",
            "documents.json",
            "chapters.jsonl",
            "chunks.jsonl",
            "provenance.jsonl",
        }
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert manifest["format"] == PACKAGE_FORMAT
        assert manifest["generator"]["schema_version"] == MATERIAL_SCHEMA_VERSION
        assert manifest["chapter_count"] == 2

    report = app_module.MaterialPackageService(source_database).validate_package(package)
    assert report["target"]["mode"] == "pure_new_file"
    assert report["can_create_new_document"] is True
    assert report["checks"]["package_source_document_hash"] == "match"
    assert report["checks"]["chapter_count"] == "match"
    assert set(report["package"]["material_layer_counts"]) == {
        "observations",
        "timeline",
        "characters",
        "reviews",
        "budget",
    }
    assert report["package"]["material_layer_counts"]["timeline"] == 0

    target_database, target_repository = make_repository(tmp_path, "target.db")
    result = app_module.MaterialPackageService(target_database).import_package(
        package,
        project_id="default",
        mode="create_document",
    )
    workspace = target_repository.get_document_workspace(result["document_id"])
    assert workspace["filename"] == "星港.txt"
    assert [chapter["title"] for chapter in workspace["chapters"]] == ["第一章 星港", "第二章 旧誓"]
    assert workspace["raw_text_hash"] == stable_text_hash(
        "第一章 星港\n林舟抵达星港。\n\n第二章 旧誓\n苏晚留下旧誓。"
    )
    with target_database.connect() as connection:
        provenance_count = connection.execute(
            "SELECT COUNT(*) FROM material_provenance WHERE document_id = ?",
            (result["document_id"],),
        ).fetchone()[0]
    assert provenance_count >= 1 + len(workspace["chapters"])


def test_material_package_validation_rejects_hash_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    original = repository.import_document("default", "原文.txt", "utf-8", "第一章 原文\n林舟出发。")
    target = repository.import_document("default", "改写.txt", "utf-8", "第一章 改写\n林舟没有出发。")
    package = app_module.MaterialPackageService(database).export_document_package(original["document"]["id"])

    report = app_module.MaterialPackageService(database).validate_package(
        package,
        target_document_id=target["document"]["id"],
    )

    assert report["checks"]["source_document_hash"] == "mismatch"
    assert report["can_import"] is False
    assert "纯新文件导入" in report["actions"][0]


def test_material_rebuild_projects_existing_library_into_experimental_views(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "旧城.txt",
        "utf-8",
        "第一章 相逢\n林舟见到苏晚。\n\n第二章 同行\n林舟和苏晚进入旧城。",
    )
    document_id = imported["document"]["id"]
    first_chapter = imported["chapters"][0]
    second_chapter = imported["chapters"][1]
    repository.save_chapter_summary(
        first_chapter["id"],
        {"summary": "林舟和苏晚相逢。", "key_events": ["两人交换线索"]},
        [],
    )
    repository.save_chapter_summary(
        second_chapter["id"],
        {"summary": "两人进入旧城。", "ending_state": "开始并肩调查"},
        [],
    )
    repository.replace_characters(
        document_id,
        [
            {"name": "林舟", "aliases": ["林记者"], "identity": "调查记者", "current_state": "追查旧城"},
            {"name": "苏晚", "aliases": [], "identity": "线索持有人", "current_state": "与林舟同行"},
        ],
    )
    chunk_id = repository.get_chapter(first_chapter["id"])["chunks"][0]["id"]
    repository.save_story_facts(
        document_id,
        first_chapter["id"],
        chunk_id,
        [
            {
                "fact_key": "timeline-meet",
                "fact_type": "timeline",
                "subject": "林舟",
                "predicate": "见到",
                "object": "苏晚",
                "state": "两人在旧城入口交换线索",
            },
            {
                "fact_key": "rel-trust",
                "fact_type": "relationship",
                "subject": "林舟",
                "predicate": "同盟",
                "object": "苏晚",
                "state": "暂时结成同盟",
            },
        ],
    )

    service = app_module.MaterialPackageService(database)
    overview = service.rebuild_document_material(document_id)
    prompt_plan = service.build_prompt_plan(document_id, query_text="继续写旧城调查", max_tokens=8000)

    assert any(node["node_type"] == "chapter_group" for node in overview["timeline"]["nodes"])
    assert overview["timeline"]["events"][0]["title"] == "林舟 见到 苏晚"
    assert [item["canonical_name"] for item in overview["characters"]] == ["林舟", "苏晚"]
    assert overview["characters"][0]["profiles"]
    assert overview["relationships"][0]["source_name"] == "林舟"
    assert overview["relationships"][0]["target_name"] == "苏晚"
    assert any(section["key"] == "character_snapshots" and section["included"] for section in prompt_plan["sections"])

    package = service.export_document_package(document_id)
    with zipfile.ZipFile(BytesIO(package)) as archive:
        names = set(archive.namelist())
        assert {
            "semantic_observations.jsonl",
            "timeline_nodes.jsonl",
            "timeline_events.jsonl",
            "character_entities.jsonl",
            "character_profiles.jsonl",
            "relationship_events.jsonl",
            "character_relationships.jsonl",
            "review_items.jsonl",
            "prompt_budget_profiles.jsonl",
        } <= names
        assert archive.read("character_entities.jsonl").strip()
        assert archive.read("timeline_nodes.jsonl").strip()
    report = service.validate_package(package)
    assert report["package"]["material_layer_counts"]["timeline"] > 0
    assert report["package"]["material_layer_counts"]["characters"] > 0

    target_database, _target_repository = make_repository(tmp_path, "material-target.db")
    imported_package = app_module.MaterialPackageService(target_database).import_package(
        package,
        project_id="default",
        mode="create_document",
    )
    imported_overview = app_module.MaterialPackageService(target_database).get_material_overview(
        imported_package["document_id"]
    )
    assert [item["canonical_name"] for item in imported_overview["characters"]] == ["林舟", "苏晚"]
    assert imported_overview["timeline"]["nodes"]
    assert imported_overview["relationships"][0]["relation_type"] == "同盟"


def test_material_package_merge_and_replace_existing_material_layer(tmp_path: Path) -> None:
    source_database, source_repository = make_repository(tmp_path, "merge-source.db")
    source = source_repository.import_document(
        "default",
        "同文.txt",
        "utf-8",
        "第一章 起点\n林舟遇见苏晚。",
    )
    document_id = source["document"]["id"]
    chapter_id = source["chapters"][0]["id"]
    source_repository.replace_characters(
        document_id,
        [{"name": "林舟", "identity": "调查者"}],
    )
    source_repository.save_chapter_summary(chapter_id, {"summary": "林舟遇见苏晚。"}, [])
    source_service = app_module.MaterialPackageService(source_database)
    source_service.rebuild_document_material(document_id)
    source_service.update_prompt_budget_profile(document_id, config={"project_summary": 321})
    package = source_service.export_document_package(document_id)

    target_database, target_repository = make_repository(tmp_path, "merge-target.db")
    target = target_repository.import_document(
        "default",
        "同文.txt",
        "utf-8",
        "第一章 起点\n林舟遇见苏晚。",
    )
    target_id = target["document"]["id"]
    target_service = app_module.MaterialPackageService(target_database)
    merged = target_service.import_package(
        package,
        project_id="default",
        mode="merge",
        target_document_id=target_id,
    )
    assert merged["overview"]["characters"][0]["canonical_name"] == "林舟"

    with target_database.connect() as connection:
        connection.execute(
            """
            INSERT INTO material_review_items
                (id, document_id, review_type, title, payload_json, status, created_at, updated_at)
            VALUES ('local-review', ?, 'local', '本地确认项', '{}', 'pending', 'now', 'now')
            """,
            (target_id,),
        )
    replaced = target_service.import_package(
        package,
        project_id="default",
        mode="replace_material",
        target_document_id=target_id,
    )
    assert all(item["id"] != "local-review" for item in replaced["overview"]["review_items"])
    assert replaced["overview"]["characters"][0]["canonical_name"] == "林舟"

    target_service.update_prompt_budget_profile(target_id, config={"project_summary": 999})
    with target_database.connect() as connection:
        connection.execute(
            """
            INSERT INTO material_review_items
                (id, document_id, review_type, title, payload_json, status, created_at, updated_at)
            VALUES ('local-review-2', ?, 'local', '本地确认项 2', '{}', 'pending', 'now', 'now')
            """,
            (target_id,),
        )
    budget_only = target_service.import_package(
        package,
        project_id="default",
        mode="replace_material",
        target_document_id=target_id,
        material_layers=["budget"],
    )
    assert budget_only["material_layers"] == ["budget"]
    assert target_service.ensure_prompt_budget_profile(target_id)["config"]["project_summary"] == 321
    assert any(item["id"] == "local-review-2" for item in budget_only["overview"]["review_items"])
    assert budget_only["overview"]["characters"][0]["canonical_name"] == "林舟"


def test_unified_events_project_to_observations_timeline_relationships_and_review_items(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "统一事件.txt",
        "utf-8",
        "第一章 会合\n林舟和苏晚在旧站台会合。",
    )
    document_id = imported["document"]["id"]
    chapter_id = imported["chapters"][0]["id"]
    chunk_id = repository.get_chapter(chapter_id)["chunks"][0]["id"]
    repository.replace_characters(
        document_id,
        [
            {"name": "林舟", "aliases": ["林记者"], "identity": "调查者"},
            {"name": "苏晚", "identity": "线索持有人"},
        ],
    )
    service = app_module.MaterialPackageService(database)
    service.seed_character_entities(document_id)

    projected = service.save_unified_events(
        document_id,
        chapter_id,
        chunk_id,
        {
            "plot_events": [
                {
                    "title": "林舟与苏晚会合",
                    "description": "两人在旧站台交换线索。",
                    "event_type": "meeting",
                    "participants": ["林舟", "苏晚"],
                    "confidence": 0.9,
                }
            ],
            "character_events": [
                {
                    "character": "林记者",
                    "event_type": "goal_update",
                    "value": "继续调查旧站台",
                    "confidence": 0.8,
                }
            ],
            "relationship_events": [
                {
                    "source": "林舟",
                    "target": "苏晚",
                    "relation_type": "同盟",
                    "event_type": "set",
                    "description": "二人暂时合作。",
                    "confidence": 0.85,
                },
                {
                    "source": "林舟",
                    "target": "未知人",
                    "relation_type": "怀疑",
                    "event_type": "set",
                    "description": "需要人工确认。",
                },
            ],
        },
    )

    assert any(item["observation_type"] == "plot_event" for item in projected["observations"])
    assert projected["timeline"]["events"][0]["title"] == "林舟与苏晚会合"
    assert projected["relationships"][0]["relation_type"] == "同盟"
    assert projected["review_items"][0]["review_type"] == "relationship_entity_missing"

    resolved = service.resolve_review_item(
        projected["review_items"][0]["id"],
        {"apply": "create_missing_entities", "names": ["未知人"]},
    )
    relationships = service.list_relationships(document_id)
    characters = service.list_character_entities(document_id)

    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["applied"]["projected"] == "relationship_event"
    assert any(item["canonical_name"] == "未知人" for item in characters)
    assert any(
        item["source_name"] == "林舟"
        and item["target_name"] == "未知人"
        and item["relation_type"] == "怀疑"
        for item in relationships
    )


def test_experimental_material_system_health_requires_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, experimental_material_system=False),
    )
    with TestClient(app_module.app) as client:
        disabled = client.get("/api/experimental/material-system/health")
    assert disabled.status_code == 404

    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, experimental_material_system=True),
    )
    with TestClient(app_module.app) as client:
        enabled = client.get("/api/experimental/material-system/health")
    assert enabled.status_code == 200
    assert enabled.json()["schema_version"] == MATERIAL_SCHEMA_VERSION


def test_experimental_material_system_api_rebuild_and_prompt_plan(monkeypatch, tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "实验.txt",
        "utf-8",
        "第一章 起点\n林舟遇见苏晚。",
    )
    document_id = imported["document"]["id"]
    chapter_id = imported["chapters"][0]["id"]
    repository.save_chapter_summary(
        chapter_id,
        {"summary": "林舟遇见苏晚。"},
        [],
    )
    repository.update_document(document_id, {"global_summary": "林舟和苏晚在起点相遇后决定同行调查。" * 20})
    repository.replace_characters(
        document_id,
        [
            {"name": "林舟", "identity": "调查者"},
            {"name": "苏晚", "identity": "协助者"},
        ],
    )
    chunk_id = repository.get_chapter(chapter_id)["chunks"][0]["id"]
    repository.save_story_facts(
        document_id,
        chapter_id,
        chunk_id,
        [
            {
                "fact_key": "api-timeline",
                "fact_type": "timeline",
                "subject": "林舟",
                "predicate": "遇见",
                "object": "苏晚",
                "state": "两人在起点相遇",
            },
            {
                "fact_key": "api-relationship",
                "fact_type": "relationship",
                "subject": "林舟",
                "predicate": "同伴",
                "object": "苏晚",
                "state": "开始同行",
            },
        ],
    )
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO material_review_items
                (id, document_id, review_type, title, payload_json, status, created_at, updated_at)
            VALUES
                ('api-review-resolve', ?, 'relationship_entity_missing', '待确认关系', '{"source":"林舟"}', 'pending', 'now', 'now'),
                ('api-review-reject', ?, 'character_entity_missing', '待确认人物', '{"character":"陌生人"}', 'pending', 'now', 'now')
            """,
            (document_id, document_id),
        )

    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "novels", repository)
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, experimental_material_system=True),
    )

    with TestClient(app_module.app) as client:
        rebuilt = client.post(f"/api/experimental/material-system/documents/{document_id}/rebuild")
        rebuilt_data = rebuilt.json()
        timeline_update = client.patch(
            f"/api/experimental/material-system/timeline-events/{rebuilt_data['timeline']['events'][0]['id']}",
            json={"title": "改写后的相遇", "status": "resolved"},
        )
        character_update = client.patch(
            f"/api/experimental/material-system/characters/entities/{rebuilt_data['characters'][0]['id']}",
            json={"canonical_name": "林舟改", "enabled": False, "profile": {"identity": "改名后的调查者"}},
        )
        relationship_update = client.patch(
            f"/api/experimental/material-system/relationships/{rebuilt_data['relationships'][0]['id']}",
            json={"relation_type": "伙伴", "strength": 0.8},
        )
        budget_profile = client.get(
            f"/api/experimental/material-system/documents/{document_id}/prompt-budget-profile"
        )
        updated_budget = client.patch(
            f"/api/experimental/material-system/documents/{document_id}/prompt-budget-profile",
            json={"config": {"project_summary": 4, "character_snapshots": 12}},
        )
        plan = client.post(
            f"/api/experimental/material-system/documents/{document_id}/prompt-plan",
            json={"query_text": "继续", "max_tokens": 8000},
        )
        entities = client.get(
            f"/api/experimental/material-system/documents/{document_id}/characters/entities"
        )
        conversation = client.post("/api/conversations", json={"title": "实验提示词"}).json()
        preview = client.get(
            f"/api/conversations/{conversation['id']}/prompt-preview?query=继续"
        )
        review_items = client.get(
            f"/api/experimental/material-system/documents/{document_id}/review-items"
        )
        resolved = client.post(
            "/api/experimental/material-system/review-items/api-review-resolve/resolve",
            json={"note": "确认"},
        )
        rejected = client.post(
            "/api/experimental/material-system/review-items/api-review-reject/reject",
            json={"note": "忽略"},
        )

    assert rebuilt.status_code == 200
    assert rebuilt.json()["timeline"]["nodes"]
    assert timeline_update.json()["title"] == "改写后的相遇"
    assert timeline_update.json()["status"] == "resolved"
    assert character_update.json()["canonical_name"] == "林舟改"
    assert character_update.json()["enabled"] is False
    assert character_update.json()["profiles"][0]["identity"] == "改名后的调查者"
    assert relationship_update.json()["relation_type"] == "伙伴"
    assert relationship_update.json()["strength"] == 0.8
    assert budget_profile.json()["config"]["project_summary"] > 4
    assert updated_budget.json()["config"]["project_summary"] == 4
    assert plan.status_code == 200
    plan_sections = {section["key"]: section for section in plan.json()["sections"]}
    assert plan_sections["project_summary"]["budget"] == 4
    assert plan_sections["project_summary"]["tokens"] <= 4
    assert any(item["key"] == "project_summary" and item["reason"] == "分段预算裁剪" for item in plan.json()["trimmed"])
    assert "character_snapshots" in plan_sections
    assert [item["canonical_name"] for item in entities.json()] == ["林舟改", "苏晚"]
    assert preview.json()["sources"]["recent_chapters"].startswith("当前时间线节点")
    assert "人物当前快照" in preview.json()["sources"]["characters"]
    assert {item["id"] for item in review_items.json()} >= {"api-review-resolve", "api-review-reject"}
    assert resolved.json()["status"] == "resolved"
    assert resolved.json()["resolution"]["note"] == "确认"
    assert rejected.json()["status"] == "rejected"
    assert rejected.json()["resolution"]["note"] == "忽略"


def test_material_character_alias_and_merge_api(monkeypatch, tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "别名合并.txt",
        "utf-8",
        "第一章 起点\n林记者和苏晚同行。",
    )
    document_id = imported["document"]["id"]
    chapter_id = imported["chapters"][0]["id"]
    chunk_id = repository.get_chapter(chapter_id)["chunks"][0]["id"]
    repository.replace_characters(
        document_id,
        [
            {"name": "林舟", "identity": "调查者"},
            {"name": "林记者", "identity": "调查者称谓"},
            {"name": "苏晚", "identity": "协助者"},
        ],
    )
    repository.save_story_facts(
        document_id,
        chapter_id,
        chunk_id,
        [
            {
                "fact_key": "merge-relationship",
                "fact_type": "relationship",
                "subject": "林记者",
                "predicate": "同伴",
                "object": "苏晚",
                "state": "以记者身份同行",
            }
        ],
    )

    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "novels", repository)
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, experimental_material_system=True),
    )

    with TestClient(app_module.app) as client:
        rebuilt = client.post(f"/api/experimental/material-system/documents/{document_id}/rebuild").json()
        by_name = {item["canonical_name"]: item for item in rebuilt["characters"]}
        alias_added = client.post(
            f"/api/experimental/material-system/characters/entities/{by_name['林舟']['id']}/aliases",
            json={"alias": "舟哥"},
        )
        merged = client.post(
            f"/api/experimental/material-system/characters/entities/{by_name['林记者']['id']}/merge",
            json={"target_character_id": by_name["林舟"]["id"]},
        )

    assert alias_added.status_code == 200
    assert "舟哥" in [item["alias"] for item in alias_added.json()["aliases"]]
    merged_characters = {item["canonical_name"]: item for item in merged.json()["characters"]}
    assert "林记者" not in merged_characters
    assert "林记者" in [item["alias"] for item in merged_characters["林舟"]["aliases"]]
    assert any(
        item["source_name"] == "林舟"
        and item["target_name"] == "苏晚"
        and item["relation_type"] == "同伴"
        for item in merged.json()["relationships"]
    )
