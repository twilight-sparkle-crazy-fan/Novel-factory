from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
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

    node_id = next(node["id"] for node in overview["timeline"]["nodes"] if node["node_type"] == "chapter_group")
    updated_node = service.update_timeline_node(
        node_id,
        {"title": "人工阶段", "summary": "人工锁定的阶段摘要。", "enabled": False},
    )
    rebuilt_after_manual_node = service.rebuild_timeline(document_id)
    preserved_node = next(node for node in rebuilt_after_manual_node["nodes"] if node["id"] == node_id)
    assert updated_node["manually_edited"] == 1
    assert preserved_node["title"] == "人工阶段"
    assert preserved_node["summary"] == "人工锁定的阶段摘要。"
    assert preserved_node["enabled"] == 0
    manual_node = service.create_timeline_node(
        document_id,
        {"title": "第二卷", "node_type": "volume", "summary": "人工卷节点"},
    )
    manual_child = service.create_timeline_node(
        document_id,
        {"title": "暗线阶段", "node_type": "stage", "parent_id": manual_node["id"]},
    )
    deleted_node = service.delete_timeline_node(manual_node["id"])
    child_after_delete = next(
        node for node in service.get_timeline(document_id)["nodes"]
        if node["id"] == manual_child["id"]
    )
    assert manual_node["manually_edited"] == 1
    assert manual_node["node_type"] == "volume"
    assert deleted_node["deleted"] is True
    assert child_after_delete["parent_id"] is None
    manual_event = service.create_timeline_event(
        document_id,
        {
            "title": "手工线索",
            "description": "林舟留下一个需要回收的暗示。",
            "event_type": "foreshadowing",
            "participants": ["林舟"],
            "chapter_id": first_chapter["id"],
            "chunk_id": chunk_id,
        },
    )
    deleted_event = service.delete_timeline_event(manual_event["id"])
    event_ids_after_delete = {
        event["id"] for event in service.get_timeline(document_id)["events"]
    }
    assert manual_event["manually_edited"] == 1
    assert manual_event["participants"] == ["林舟"]
    assert manual_event["chapter_id"] == first_chapter["id"]
    assert manual_event["chunk_id"] == chunk_id
    assert deleted_event["deleted"] is True
    assert manual_event["id"] not in event_ids_after_delete

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
        [
            {"name": "林舟", "identity": "调查者"},
            {"name": "苏晚", "identity": "线索持有人"},
        ],
    )
    source_repository.save_chapter_summary(chapter_id, {"summary": "林舟遇见苏晚。"}, [])
    chunk_id = source_repository.get_chapter(chapter_id)["chunks"][0]["id"]
    source_repository.save_story_facts(
        document_id,
        chapter_id,
        chunk_id,
        [
            {
                "fact_key": "timeline-meet",
                "fact_type": "timeline",
                "subject": "林舟",
                "predicate": "遇见",
                "object": "苏晚",
                "state": "林舟遇见苏晚。",
            },
            {
                "fact_key": "rel-ally",
                "fact_type": "relationship",
                "subject": "林舟",
                "predicate": "同盟",
                "object": "苏晚",
                "state": "两人开始交换线索。",
            },
        ],
    )
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
    preview = target_service.validate_package(package, target_document_id=target_id)
    assert preview["can_import"] is True
    assert preview["diff_preview"]["layers"]["characters"]["added"] > 0
    assert preview["diff_preview"]["layers"]["budget"]["added"] == 1

    merged = target_service.import_package(
        package,
        project_id="default",
        mode="merge",
        target_document_id=target_id,
    )
    assert merged["overview"]["characters"][0]["canonical_name"] == "林舟"
    merged_preview = target_service.validate_package(package, target_document_id=target_id)
    assert merged_preview["diff_preview"]["layers"]["characters"]["added"] == 0
    assert merged_preview["diff_preview"]["layers"]["characters"]["unchanged"] > 0
    character_id = merged["overview"]["characters"][0]["id"]
    target_service.update_character_entity(
        character_id,
        {
            "canonical_name": "林舟（人工确认）",
            "enabled": False,
            "profile": {"identity": "人工修订身份"},
        },
    )
    manual_merged = target_service.import_package(
        package,
        project_id="default",
        mode="merge",
        target_document_id=target_id,
    )
    manual_character = next(
        item for item in manual_merged["overview"]["characters"] if item["id"] == character_id
    )
    assert manual_character["canonical_name"] == "林舟（人工确认）"
    assert manual_character["enabled"] is False
    assert manual_character["profiles"][0]["identity"] == "人工修订身份"
    timeline_event_id = manual_merged["overview"]["timeline"]["events"][0]["id"]
    relationship_id = manual_merged["overview"]["relationships"][0]["id"]
    target_service.update_timeline_event(
        timeline_event_id,
        {"title": "人工修订事件", "description": "人工保留的事件描述", "status": "manual"},
    )
    target_service.update_relationship(
        relationship_id,
        {"relation_type": "人工关系", "status": "manual", "strength": 0.9},
    )
    manual_event_merged = target_service.import_package(
        package,
        project_id="default",
        mode="merge",
        target_document_id=target_id,
    )
    manual_event = next(
        item for item in manual_event_merged["overview"]["timeline"]["events"]
        if item["id"] == timeline_event_id
    )
    manual_relationship = next(
        item for item in manual_event_merged["overview"]["relationships"]
        if item["id"] == relationship_id
    )
    assert manual_event["title"] == "人工修订事件"
    assert manual_event["description"] == "人工保留的事件描述"
    assert manual_relationship["relation_type"] == "人工关系"
    assert manual_relationship["status"] == "manual"
    assert manual_relationship["strength"] == 0.9
    import_conflicts = [
        item for item in target_service.list_review_items(target_id)
        if item["review_type"] == "material_import_conflict"
    ]
    conflict_fields = {
        (item["payload"]["table"], field["field"])
        for item in import_conflicts
        for field in item["payload"].get("fields", [])
    }
    assert ("character_entities", "canonical_name") in conflict_fields
    assert ("character_profiles", "identity") in conflict_fields
    assert ("timeline_events", "title") in conflict_fields
    assert ("character_relationships", "relation_type") in conflict_fields
    timeline_conflict = next(
        item for item in import_conflicts
        if item["payload"]["table"] == "timeline_events"
    )
    incoming_title = next(
        field["incoming"] for field in timeline_conflict["payload"]["fields"]
        if field["field"] == "title"
    )
    resolved_conflict = target_service.resolve_review_item(
        timeline_conflict["id"],
        {"apply": "apply_import_conflict_incoming"},
    )
    applied_event = next(
        item for item in target_service.get_timeline(target_id)["events"]
        if item["id"] == timeline_event_id
    )
    assert resolved_conflict["status"] == "resolved"
    assert "title" in resolved_conflict["resolution"]["applied"]["fields"]
    assert applied_event["title"] == incoming_title

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


def test_material_package_merge_filters_by_chapter_scope(tmp_path: Path) -> None:
    source_database, source_repository = make_repository(tmp_path, "scope-source.db")
    source = source_repository.import_document(
        "default",
        "范围.txt",
        "utf-8",
        "第一章 起点\n林舟抵达。\n\n第二章 线索\n苏晚交出线索。",
    )
    document_id = source["document"]["id"]
    first_chapter, second_chapter = source["chapters"]
    first_chunk_id = source_repository.get_chapter(first_chapter["id"])["chunks"][0]["id"]
    second_chunk_id = source_repository.get_chapter(second_chapter["id"])["chunks"][0]["id"]
    source_repository.save_story_facts(
        document_id,
        first_chapter["id"],
        first_chunk_id,
        [
            {
                "fact_key": "scope-first",
                "fact_type": "timeline",
                "subject": "林舟",
                "predicate": "抵达",
                "object": "旧城",
                "state": "林舟抵达旧城。",
            }
        ],
    )
    source_repository.save_story_facts(
        document_id,
        second_chapter["id"],
        second_chunk_id,
        [
            {
                "fact_key": "scope-second",
                "fact_type": "timeline",
                "subject": "苏晚",
                "predicate": "交出",
                "object": "线索",
                "state": "苏晚交出关键线索。",
            }
        ],
    )
    source_service = app_module.MaterialPackageService(source_database)
    source_service.rebuild_document_material(document_id)
    package = source_service.export_document_package(document_id)

    target_database, target_repository = make_repository(tmp_path, "scope-target.db")
    target = target_repository.import_document(
        "default",
        "范围.txt",
        "utf-8",
        "第一章 起点\n林舟抵达。\n\n第二章 线索\n苏晚交出线索。",
    )
    target_id = target["document"]["id"]
    target_service = app_module.MaterialPackageService(target_database)
    report = target_service.validate_package(
        package,
        target_document_id=target_id,
        chapter_start=2,
        chapter_end=2,
    )

    assert report["scope"]["matched_chapter_count"] == 1
    assert report["package"]["scoped_material_layer_counts"]["timeline"] < report["package"]["material_layer_counts"]["timeline"]

    imported = target_service.import_package(
        package,
        project_id="default",
        mode="merge",
        target_document_id=target_id,
        material_layers=["timeline"],
        chapter_start=2,
        chapter_end=2,
    )
    event_titles = [event["title"] for event in imported["overview"]["timeline"]["events"]]
    assert event_titles == ["苏晚 交出 线索"]

    with pytest.raises(app_module.MaterialPackageError, match="章节范围过滤暂只支持合并导入"):
        target_service.import_package(
            package,
            project_id="default",
            mode="replace_material",
            target_document_id=target_id,
            chapter_start=2,
            chapter_end=2,
        )


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
            "location_events": [
                {
                    "location": "旧站台",
                    "description": "旧站台出现新的暗门。",
                    "confidence": 0.8,
                }
            ],
            "ability_events": [
                {
                    "character": "林舟",
                    "ability": "线索推理",
                    "state": "能从旧票据推断路线。",
                }
            ],
            "object_events": [
                {
                    "object": "铜钥匙",
                    "state": "钥匙能打开站台暗门。",
                }
            ],
            "unresolved_entities": [
                {
                    "title": "暗门后的脚印",
                    "description": "脚印主人尚未揭晓。",
                }
            ],
        },
    )

    assert any(item["observation_type"] == "plot_event" for item in projected["observations"])
    assert projected["timeline"]["events"][0]["title"] == "林舟与苏晚会合"
    assert projected["relationships"][0]["relation_type"] == "同盟"
    review_types = {item["review_type"] for item in projected["review_items"]}
    assert "relationship_entity_missing" in review_types
    assert {
        "location_observation",
        "ability_observation",
        "object_observation",
        "unresolved_observation",
    } <= review_types

    relationship_missing = next(
        item for item in projected["review_items"]
        if item["review_type"] == "relationship_entity_missing"
    )
    resolved = service.resolve_review_item(
        relationship_missing["id"],
        {"apply": "create_missing_entities", "names": ["未知人"]},
    )
    auxiliary_item = next(
        item for item in projected["review_items"]
        if item["review_type"] == "location_observation"
    )
    ability_item = next(
        item for item in projected["review_items"]
        if item["review_type"] == "ability_observation"
    )
    resolved_auxiliary = service.resolve_review_item(
        auxiliary_item["id"],
        {"apply": "apply_auxiliary_observation"},
    )
    resolved_ability = service.resolve_review_item(
        ability_item["id"],
        {"apply": "apply_auxiliary_observation"},
    )
    relationships = service.list_relationships(document_id)
    characters = service.list_character_entities(document_id)
    timeline = service.get_timeline(document_id)

    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["applied"]["projected"] == "relationship_event"
    assert resolved_auxiliary["status"] == "resolved"
    assert resolved_auxiliary["resolution"]["applied"]["projected"] == "location_observation"
    assert resolved_ability["resolution"]["applied"]["projected"] == "character_event"
    assert any(event["title"] == "地点：旧站台" and event["location_id"] for event in timeline["events"])
    with database.connect() as connection:
        ability_events = connection.execute(
            """
            SELECT ce.canonical_name, ev.event_type, ev.value
            FROM character_events ev
            JOIN character_entities ce ON ce.id = ev.character_id
            WHERE ce.document_id = ? AND ev.event_type = 'ability_update'
            """,
            (document_id,),
        ).fetchall()
    assert any(
        row["canonical_name"] == "林舟" and "线索推理" in row["value"]
        for row in ability_events
    )
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
        node_create = client.post(
            f"/api/experimental/material-system/documents/{document_id}/timeline/nodes",
            json={"title": "API 新阶段", "node_type": "stage", "summary": "接口创建"},
        )
        node_update = client.patch(
            f"/api/experimental/material-system/timeline-nodes/{rebuilt_data['timeline']['nodes'][0]['id']}",
            json={"title": "人工节点", "summary": "人工节点摘要", "enabled": False},
        )
        node_delete = client.delete(
            f"/api/experimental/material-system/timeline-nodes/{node_create.json()['id']}"
        )
        event_create = client.post(
            f"/api/experimental/material-system/documents/{document_id}/timeline/events",
            json={
                "title": "API 新事件",
                "description": "接口创建事件",
                "event_type": "manual",
                "participants": ["林舟"],
            },
        )
        event_delete = client.delete(
            f"/api/experimental/material-system/timeline-events/{event_create.json()['id']}"
        )
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
    assert node_create.status_code == 201
    assert node_create.json()["title"] == "API 新阶段"
    assert node_create.json()["node_type"] == "stage"
    assert node_delete.json()["deleted"] is True
    assert event_create.status_code == 201
    assert event_create.json()["title"] == "API 新事件"
    assert event_create.json()["participants"] == ["林舟"]
    assert event_delete.json()["deleted"] is True
    assert node_update.json()["title"] == "人工节点"
    assert node_update.json()["summary"] == "人工节点摘要"
    assert node_update.json()["enabled"] == 0
    assert node_update.json()["manually_edited"] == 1
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
