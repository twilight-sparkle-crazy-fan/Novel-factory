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
        assert manifest["file_hashes"]["documents.json"].startswith("sha256:")

    report = app_module.MaterialPackageService(source_database).validate_package(package)
    export_report = app_module.MaterialPackageService(source_database).export_document_package_report(document_id)
    assert report["target"]["mode"] == "pure_new_file"
    assert report["can_create_new_document"] is True
    assert report["checks"]["package_source_document_hash"] == "match"
    assert report["checks"]["package_file_hashes"] == "match"
    assert report["checks"]["chapter_count"] == "match"
    assert report["checks"]["provenance_source_hash"] == "match"
    assert report["checks"]["material_references"] == "match"
    assert set(report["package"]["material_layer_counts"]) == {
        "observations",
        "timeline",
        "characters",
        "reviews",
        "auxiliary",
        "budget",
    }
    assert report["package"]["material_layer_counts"]["timeline"] == 0
    assert export_report["target"]["mode"] == "existing_document"
    assert export_report["checks"]["source_document_hash"] == "match"
    assert export_report["export"]["package_bytes"] > 0
    assert export_report["package"]["chapter_count"] == 2

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


def test_material_package_validation_rejects_material_count_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "计数.txt", "utf-8", "第一章 原文\n林舟出发。")
    service = app_module.MaterialPackageService(database)
    package = service.export_document_package(imported["document"]["id"])
    buffer = BytesIO()
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
                manifest["material_counts"]["timeline_nodes.jsonl"] = (
                    int(manifest["material_counts"].get("timeline_nodes.jsonl", 0)) + 1
                )
                data = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            target.writestr(item, data)

    report = service.validate_package(buffer.getvalue(), target_document_id=imported["document"]["id"])

    assert report["checks"]["material_counts"] == "mismatch"
    assert report["package"]["actual_material_counts"]["timeline_nodes.jsonl"] == 0
    assert report["can_import"] is False
    assert "资料记录数" in report["actions"][0]


def test_material_package_validation_rejects_source_count_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "chunk计数.txt", "utf-8", "第一章 原文\n林舟出发。")
    service = app_module.MaterialPackageService(database)
    package = service.export_document_package(imported["document"]["id"])
    buffer = BytesIO()
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
                manifest["chunk_count"] = int(manifest.get("chunk_count", 0)) + 1
                data = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            target.writestr(item, data)

    report = service.validate_package(buffer.getvalue(), target_document_id=imported["document"]["id"])

    assert report["checks"]["chunk_count"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "chunk 数量" in report["actions"][0]


def test_material_package_validation_rejects_package_file_hash_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "文件hash.txt", "utf-8", "第一章 原文\n林舟出发。")
    service = app_module.MaterialPackageService(database)
    service.ensure_prompt_budget_profile(imported["document"]["id"])
    package = service.export_document_package(imported["document"]["id"])
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "prompt_budget_profiles.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["name"] = "被篡改的预算"
                changed = True
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=imported["document"]["id"])

    assert report["checks"]["package_file_hashes"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "文件 hash" in report["actions"][0]


def test_material_package_validation_rejects_material_unknown_fields(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "未知字段.txt", "utf-8", "第一章 原文\n林舟出发。")
    document_id = imported["document"]["id"]
    service = app_module.MaterialPackageService(database)
    character = service.create_character_entity(document_id, {"canonical_name": "林舟"})
    service.create_character_fact(character["id"], {"field": "身份", "value": "调查者"})
    package = service.export_document_package(document_id)
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
                manifest.pop("file_hashes", None)
                data = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            elif item.filename == "character_facts.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["unexpected_field"] = "不属于当前 schema"
                changed = True
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=document_id)

    assert report["checks"]["package_file_hashes"] == "missing"
    assert report["checks"]["material_unknown_fields"] == 1
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "不认识的字段" in report["actions"][0]


def test_material_package_validation_rejects_material_missing_required_fields(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "缺字段.txt", "utf-8", "第一章 原文\n林舟出发。")
    document_id = imported["document"]["id"]
    service = app_module.MaterialPackageService(database)
    character = service.create_character_entity(document_id, {"canonical_name": "林舟"})
    service.create_character_fact(character["id"], {"field": "身份", "value": "调查者"})
    package = service.export_document_package(document_id)
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
                manifest.pop("file_hashes", None)
                data = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            elif item.filename == "character_facts.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0].pop("value", None)
                changed = True
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=document_id)

    assert report["checks"]["package_file_hashes"] == "missing"
    assert report["checks"]["material_required_fields"] == "missing"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "必填字段" in report["actions"][0]


def test_material_package_validation_rejects_source_unknown_fields(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "章节未知字段.txt", "utf-8", "第一章 原文\n林舟出发。")
    service = app_module.MaterialPackageService(database)
    package = service.export_document_package(imported["document"]["id"])
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
                manifest.pop("file_hashes", None)
                data = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
            elif item.filename == "chapters.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["unexpected_source_field"] = "不属于当前 schema"
                changed = True
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=imported["document"]["id"])

    assert report["checks"]["package_file_hashes"] == "missing"
    assert report["checks"]["unknown_fields"] == 1
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "章节或 chunk JSONL" in report["actions"][0]


def test_material_package_validation_rejects_source_content_hash_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "章节hash.txt", "utf-8", "第一章 原文\n林舟出发。")
    service = app_module.MaterialPackageService(database)
    package = service.export_document_package(imported["document"]["id"])
    buffer = BytesIO()
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename in {"chapters.jsonl", "chunks.jsonl"}:
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["content_hash"] = "sha256:broken"
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)

    report = service.validate_package(buffer.getvalue(), target_document_id=imported["document"]["id"])

    assert report["checks"]["chapter_content_hash"] == "mismatch"
    assert report["checks"]["chunk_content_hash"] == "mismatch"
    assert report["checks"]["rejected_records"] == 2
    assert report["can_import"] is False
    assert "内容 hash" in report["actions"][0]


def test_material_package_validation_rejects_source_document_id_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "来源id.txt", "utf-8", "第一章 原文\n林舟出发。")
    service = app_module.MaterialPackageService(database)
    package = service.export_document_package(imported["document"]["id"])
    buffer = BytesIO()
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "chapters.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["document_id"] = "doc_other"
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)

    report = service.validate_package(buffer.getvalue(), target_document_id=imported["document"]["id"])

    assert report["checks"]["source_document_id"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "document_id" in report["actions"][0]


def test_material_package_validation_rejects_material_document_id_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "资料id.txt", "utf-8", "第一章 原文\n林舟出发。")
    document_id = imported["document"]["id"]
    chapter_id = imported["chapters"][0]["id"]
    chunk_id = repository.get_chapter(chapter_id)["chunks"][0]["id"]
    service = app_module.MaterialPackageService(database)
    service.save_unified_events(
        document_id,
        chapter_id,
        chunk_id,
        {
            "plot_events": [
                {
                    "title": "林舟出发",
                    "description": "林舟离开旧站。",
                    "participants": ["林舟"],
                }
            ]
        },
    )
    package = service.export_document_package(document_id)
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "semantic_observations.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["document_id"] = "doc_other"
                changed = True
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=document_id)

    assert report["checks"]["material_document_id"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "资料记录 document_id" in report["actions"][0]


def test_material_package_validation_rejects_provenance_hash_mismatch(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "来源hash.txt", "utf-8", "第一章 原文\n林舟出发。")
    service = app_module.MaterialPackageService(database)
    package = service.export_document_package(imported["document"]["id"])
    buffer = BytesIO()
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "provenance.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["source_hash"] = "sha256:broken"
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)

    report = service.validate_package(buffer.getvalue(), target_document_id=imported["document"]["id"])

    assert report["checks"]["provenance_source_hash"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "provenance" in report["actions"][0]


def test_material_package_validation_rejects_missing_material_provenance_reference(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "来源引用.txt", "utf-8", "第一章 原文\n林舟出发。")
    document_id = imported["document"]["id"]
    chapter_id = imported["chapters"][0]["id"]
    chunk_id = repository.get_chapter(chapter_id)["chunks"][0]["id"]
    service = app_module.MaterialPackageService(database)
    service.save_unified_events(
        document_id,
        chapter_id,
        chunk_id,
        {
            "plot_events": [
                {
                    "title": "林舟出发",
                    "description": "林舟离开旧站。",
                    "participants": ["林舟"],
                    "confidence": 0.8,
                }
            ]
        },
    )
    package = service.export_document_package(document_id)
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename in {"semantic_observations.jsonl", "timeline_events.jsonl"} and not changed:
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                for record in records:
                    if str(record.get("provenance_id") or "").strip():
                        record["provenance_id"] = "prov:missing"
                        changed = True
                        break
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=document_id)

    assert report["checks"]["material_provenance_refs"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "provenance_id" in report["actions"][0]


def test_material_package_validation_rejects_missing_material_internal_reference(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document("default", "内部引用.txt", "utf-8", "第一章 原文\n林舟出发。")
    document_id = imported["document"]["id"]
    service = app_module.MaterialPackageService(database)
    character = service.create_character_entity(
        document_id,
        {
            "canonical_name": "林舟",
            "profile": {"identity": "调查者"},
        },
    )
    service.create_character_fact(
        character["id"],
        {"field": "身份", "value": "调查者"},
    )
    package = service.export_document_package(document_id)
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "character_facts.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["character_id"] = "char_missing"
                changed = True
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=document_id)

    assert report["checks"]["material_references"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "资料记录引用" in report["actions"][0]


def test_material_package_validation_rejects_timeline_parent_cycle(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "时间线环.txt",
        "utf-8",
        "第一章 原文\n林舟出发。\n\n第二章 归来\n林舟归来。",
    )
    document_id = imported["document"]["id"]
    service = app_module.MaterialPackageService(database)
    service.rebuild_timeline(document_id)
    package = service.export_document_package(document_id)
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(BytesIO(package)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "timeline_nodes.jsonl":
                records = [
                    json.loads(line)
                    for line in data.decode("utf-8").splitlines()
                    if line.strip()
                ]
                records[0]["parent_id"] = records[0]["id"]
                changed = True
                data = "".join(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    for record in records
                ).encode("utf-8")
            target.writestr(item, data)
    assert changed

    report = service.validate_package(buffer.getvalue(), target_document_id=document_id)

    assert report["checks"]["material_references"] == "mismatch"
    assert report["checks"]["rejected_records"] == 1
    assert report["can_import"] is False
    assert "资料记录引用" in report["actions"][0]


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
    snapshot = service.current_material_snapshot(document_id, max_tokens=8000)

    assert any(node["node_type"] == "chapter_group" for node in overview["timeline"]["nodes"])
    assert overview["observation_ledger"]["observation_count"] >= 2
    assert overview["observation_ledger"]["type_counts"]["plot_event"] == 1
    assert overview["observation_ledger"]["type_counts"]["relationship_event"] == 1
    assert overview["timeline"]["events"][0]["title"] == "林舟 见到 苏晚"
    assert overview["timeline"]["tree"][0]["node_type"] == "project"
    assert overview["timeline"]["tree"][0]["children"][0]["node_type"] == "chapter_group"
    assert {
        child["node_type"]
        for child in overview["timeline"]["tree"][0]["children"][0]["children"]
    } == {"chapter"}
    assert [item["canonical_name"] for item in overview["characters"]] == ["林舟", "苏晚"]
    assert overview["characters"][0]["profiles"]
    assert overview["relationships"][0]["source_name"] == "林舟"
    assert overview["relationships"][0]["target_name"] == "苏晚"
    assert overview["relationship_network"]["node_count"] == 2
    assert overview["relationship_network"]["edge_count"] == 1
    assert overview["relationship_network"]["central_characters"][0]["degree"] == 1
    assert any(section["key"] == "character_snapshots" and section["included"] for section in prompt_plan["sections"])
    relationship_history = next(section for section in prompt_plan["sections"] if section["key"] == "relationship_history")
    assert relationship_history["included"] is True
    assert "林舟 -> 苏晚" in relationship_history["content"]
    assert "同盟" in relationship_history["content"]
    assert snapshot["sections"]
    assert "人物当前快照" in snapshot["content"]
    existing_fact = service.create_character_fact(
        overview["characters"][0]["id"],
        {"field": "位置", "value": "旧城入口", "certainty": 0.8},
    )
    updated_existing_fact = service.update_character_fact(
        existing_fact["id"],
        {"value": "旧城深处", "certainty": 0.9},
    )
    fact_plan = service.build_prompt_plan(document_id, query_text="继续写旧城调查", max_tokens=8000)
    character_snapshot = next(section for section in fact_plan["sections"] if section["key"] == "character_snapshots")
    assert updated_existing_fact["value"] == "旧城深处"
    assert updated_existing_fact["certainty"] == 0.9
    assert "人物事实：位置：旧城深处" in character_snapshot["content"]
    later_profile = service.create_character_profile(
        overview["characters"][0]["id"],
        {
            "title": "旧城后期",
            "start_chapter_id": second_chapter["id"],
            "identity": "深入旧城后的调查者",
        },
    )
    staged_profile_plan = service.build_prompt_plan(document_id, query_text="继续写旧城调查", max_tokens=8000)
    staged_snapshot = next(
        section for section in staged_profile_plan["sections"]
        if section["key"] == "character_snapshots"
    )
    assert later_profile["start_chapter_id"] == second_chapter["id"]
    assert "身份：深入旧城后的调查者" in staged_snapshot["content"]
    first_character_snapshot = service.character_snapshot(
        overview["characters"][0]["id"],
        chapter_id=first_chapter["id"],
    )
    second_character_snapshot = service.character_snapshot(
        overview["characters"][0]["id"],
        chapter_id=second_chapter["id"],
    )
    assert first_character_snapshot["chapter"]["id"] == first_chapter["id"]
    assert first_character_snapshot["selected_profile"]["id"] != later_profile["id"]
    assert second_character_snapshot["chapter"]["id"] == second_chapter["id"]
    assert second_character_snapshot["selected_profile"]["id"] == later_profile["id"]
    assert "身份：深入旧城后的调查者" in second_character_snapshot["text"]

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
    updated_child = service.update_timeline_node(
        manual_child["id"],
        {
            "start_chapter_id": first_chapter["id"],
            "end_chapter_id": second_chapter["id"],
            "position": 7,
        },
    )
    assert updated_child["start_chapter_id"] == first_chapter["id"]
    assert updated_child["end_chapter_id"] == second_chapter["id"]
    assert updated_child["position"] == 7
    with pytest.raises(ValueError, match="起始章节"):
        service.update_timeline_node(
            manual_child["id"],
            {"start_chapter_id": second_chapter["id"], "end_chapter_id": first_chapter["id"]},
        )
    with pytest.raises(ValueError, match="子节点"):
        service.update_timeline_node(manual_node["id"], {"parent_id": manual_child["id"]})
    manual_tree = service.get_timeline(document_id)["tree"]
    manual_volume = next(node for node in manual_tree if node["id"] == manual_node["id"])
    assert manual_volume["children"][0]["id"] == manual_child["id"]
    assert manual_volume["children"][0]["depth"] == manual_volume["depth"] + 1
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
    updated_manual_event = service.update_timeline_event(
        manual_event["id"],
        {
            "event_type": "turning_point",
            "chapter_id": second_chapter["id"],
            "participants": ["苏晚"],
            "sequence": 12,
        },
    )
    cleared_manual_event = service.update_timeline_event(
        manual_event["id"],
        {"chapter_id": None, "participants": []},
    )
    deleted_event = service.delete_timeline_event(manual_event["id"])
    event_ids_after_delete = {
        event["id"] for event in service.get_timeline(document_id)["events"]
    }
    assert manual_event["manually_edited"] == 1
    assert manual_event["participants"] == ["林舟"]
    assert manual_event["chapter_id"] == first_chapter["id"]
    assert manual_event["chunk_id"] == chunk_id
    assert updated_manual_event["event_type"] == "turning_point"
    assert updated_manual_event["chapter_id"] == second_chapter["id"]
    assert updated_manual_event["chunk_id"] is None
    assert updated_manual_event["participants"] == ["苏晚"]
    assert updated_manual_event["sequence"] == 12
    assert cleared_manual_event["chapter_id"] is None
    assert cleared_manual_event["participants"] == []
    assert deleted_event["deleted"] is True
    assert manual_event["id"] not in event_ids_after_delete
    manual_character = service.create_character_entity(
        document_id,
        {
            "canonical_name": "周岚",
            "aliases": ["周医师"],
            "profile": {"identity": "旧城诊所医师"},
        },
    )
    manual_profile = service.create_character_profile(
        manual_character["id"],
        {
            "title": "第二阶段",
            "identity": "旧城暗线协助者",
            "start_chapter_id": first_chapter["id"],
            "end_chapter_id": second_chapter["id"],
        },
    )
    updated_profile = service.update_character_profile(
        manual_profile["id"],
        {
            "title": "第二阶段修订",
            "identity": "旧城暗线核心协助者",
            "start_chapter_id": second_chapter["id"],
            "end_chapter_id": second_chapter["id"],
            "enabled": False,
        },
    )
    with pytest.raises(ValueError, match="阶段档案起始章节"):
        service.update_character_profile(
            manual_profile["id"],
            {"start_chapter_id": second_chapter["id"], "end_chapter_id": first_chapter["id"]},
        )
    deleted_profile = service.delete_character_profile(manual_profile["id"])
    manual_character_event = service.create_character_event(
        manual_character["id"],
        {"event_type": "ability", "value": "开始掌握旧城暗号。", "chapter_id": first_chapter["id"]},
    )
    updated_character_event = service.update_character_event(
        manual_character_event["id"],
        {
            "event_type": "decision",
            "value": "决定协助林舟进入旧城。",
            "chapter_id": second_chapter["id"],
            "sequence": 7,
        },
    )
    cleared_character_event = service.update_character_event(
        manual_character_event["id"],
        {"chapter_id": None},
    )
    deleted_character_event = service.delete_character_event(manual_character_event["id"])
    manual_character_fact = service.create_character_fact(
        manual_character["id"],
        {"field": "能力", "value": "识别旧城暗号", "certainty": 0.85},
    )
    updated_character_fact = service.update_character_fact(
        manual_character_fact["id"],
        {
            "field": "能力阶段",
            "value": "能独立解读旧城暗号",
            "valid_from_chapter_id": first_chapter["id"],
            "valid_to_chapter_id": second_chapter["id"],
            "certainty": 0.95,
        },
    )
    cleared_character_fact = service.update_character_fact(
        manual_character_fact["id"],
        {"valid_from_chapter_id": None, "valid_to_chapter_id": None},
    )
    deleted_character_fact = service.delete_character_fact(manual_character_fact["id"])
    deleted_character = service.delete_character_entity(manual_character["id"])
    character_ids_after_delete = {
        character["id"] for character in service.list_character_entities(document_id)
    }
    assert manual_character["manually_confirmed"] is True
    assert manual_character["aliases"][0]["alias"] == "周医师"
    assert manual_character["profiles"][0]["identity"] == "旧城诊所医师"
    assert manual_profile["title"] == "第二阶段"
    assert updated_profile["title"] == "第二阶段修订"
    assert updated_profile["identity"] == "旧城暗线核心协助者"
    assert updated_profile["start_chapter_id"] == second_chapter["id"]
    assert updated_profile["end_chapter_id"] == second_chapter["id"]
    assert updated_profile["enabled"] == 0
    assert deleted_profile["deleted"] is True
    assert manual_character_event["event_type"] == "ability"
    assert manual_character_event["chapter_id"] == first_chapter["id"]
    assert updated_character_event["event_type"] == "decision"
    assert updated_character_event["value"] == "决定协助林舟进入旧城。"
    assert updated_character_event["chapter_id"] == second_chapter["id"]
    assert updated_character_event["sequence"] == 7
    assert cleared_character_event["chapter_id"] is None
    assert deleted_character_event["deleted"] is True
    assert manual_character_fact["field"] == "能力"
    assert updated_character_fact["field"] == "能力阶段"
    assert updated_character_fact["value"] == "能独立解读旧城暗号"
    assert updated_character_fact["valid_from_chapter_id"] == first_chapter["id"]
    assert updated_character_fact["valid_to_chapter_id"] == second_chapter["id"]
    assert updated_character_fact["certainty"] == 0.95
    assert cleared_character_fact["valid_from_chapter_id"] is None
    assert cleared_character_fact["valid_to_chapter_id"] is None
    assert deleted_character_fact["deleted"] is True
    assert deleted_character["deleted"] is True
    assert manual_character["id"] not in character_ids_after_delete
    split_source = service.create_character_entity(
        document_id,
        {
            "canonical_name": "双生",
            "aliases": ["影子"],
            "profile": {"identity": "被误合并的人物"},
        },
    )
    split_fact = service.create_character_fact(
        split_source["id"],
        {"field": "身份", "value": "真正的影子"},
    )
    split_event = service.create_character_event(
        split_source["id"],
        {"event_type": "reveal", "value": "影子身份被识破。"},
    )
    split_relationship = service.create_relationship(
        document_id,
        {
            "source_character_id": split_source["id"],
            "target_character_id": overview["characters"][0]["id"],
            "relation_type": "误认同伴",
            "description": "影子曾被误认为双生。",
        },
    )
    service.create_relationship_event(
        split_relationship["id"],
        {"event_type": "truth_reveal", "description": "影子本体和林舟重新确认关系。"},
    )
    split_result = service.split_character_entity(
        split_source["id"],
        {
            "canonical_name": "影子本体",
            "aliases": ["影子"],
            "fact_ids": [split_fact["id"]],
            "event_ids": [split_event["id"]],
            "relationship_ids": [split_relationship["id"]],
        },
    )
    split_source_after = next(
        character for character in split_result["characters"]
        if character["id"] == split_source["id"]
    )
    split_target = next(
        character for character in split_result["characters"]
        if character["id"] == split_result["split"]["new_character_id"]
    )
    assert split_result["split"]["moved_aliases"] == ["影子"]
    assert split_result["split"]["moved_facts"] == 1
    assert split_result["split"]["moved_events"] == 1
    assert split_result["split"]["moved_relationships"] == 1
    assert split_result["split"]["moved_relationship_events"] == 2
    assert [alias["alias"] for alias in split_source_after["aliases"]] == []
    assert [alias["alias"] for alias in split_target["aliases"]] == ["影子"]
    assert split_target["facts"][0]["value"] == "真正的影子"
    assert split_target["events"][0]["value"] == "影子身份被识破。"
    assert split_target["profiles"]
    moved_relationship = next(
        relationship for relationship in split_result["relationships"]
        if relationship["id"] == split_relationship["id"]
    )
    split_target_dependencies = service.character_entity_dependencies(split_target["id"])
    assert moved_relationship["source_character_id"] == split_target["id"]
    assert moved_relationship["source_name"] == "影子本体"
    assert len(moved_relationship["events"]) == 2
    assert split_target_dependencies["can_delete"] is False
    assert split_target_dependencies["relationship_count"] == 1
    assert split_target_dependencies["relationship_event_count"] == 2
    assert split_target_dependencies["relationships"][0]["relation_type"] == "误认同伴"
    service.delete_relationship(split_relationship["id"])
    service.delete_character_entity(split_source_after["id"])
    service.delete_character_entity(split_target["id"])
    manual_relationship = service.create_relationship(
        document_id,
        {
            "source_character_id": overview["characters"][0]["id"],
            "target_character_id": overview["characters"][1]["id"],
            "relation_type": "师徒",
            "status": "active",
            "strength": 0.75,
            "description": "人工补充关系",
        },
    )
    manual_relationship_event = service.create_relationship_event(
        manual_relationship["id"],
        {"event_type": "trust_shift", "description": "苏晚开始信任林舟。", "strength_delta": 0.2},
    )
    first_chapter_relationship_event = service.create_relationship_event(
        manual_relationship["id"],
        {
            "event_type": "chapter_trust",
            "description": "第一章两人开始建立信任。",
            "chapter_id": first_chapter["id"],
        },
    )
    second_chapter_relationship_event = service.create_relationship_event(
        manual_relationship["id"],
        {
            "event_type": "later_split",
            "description": "第二章两人因暗线短暂分歧。",
            "chapter_id": second_chapter["id"],
        },
    )
    updated_relationship_event = service.update_relationship_event(
        manual_relationship_event["id"],
        {
            "event_type": "trust_confirmed",
            "description": "两人确认短期同盟。",
            "chapter_id": second_chapter["id"],
            "strength_delta": 0.3,
        },
    )
    cleared_relationship_event = service.update_relationship_event(
        manual_relationship_event["id"],
        {"chapter_id": None},
    )
    relationship_with_events = next(
        relationship for relationship in service.list_relationships(document_id)
        if relationship["id"] == manual_relationship["id"]
    )
    character_relationships = service.character_relationships(overview["characters"][0]["id"])
    active_character_relationships = service.character_relationships(
        overview["characters"][0]["id"],
        status="active",
    )
    relationship_history = service.relationship_history(
        overview["characters"][0]["id"],
        overview["characters"][1]["id"],
        relation_type="师徒",
    )
    first_chapter_snapshot = service.relationship_snapshot(
        document_id,
        chapter_id=first_chapter["id"],
    )
    snapshot_relationship = next(
        item for item in first_chapter_snapshot["relationships"]
        if item["id"] == manual_relationship["id"]
    )
    deleted_relationship_event = service.delete_relationship_event(manual_relationship_event["id"])
    deleted_relationship = service.delete_relationship(manual_relationship["id"])
    relationship_ids_after_delete = {
        relationship["id"] for relationship in service.list_relationships(document_id)
    }
    assert manual_relationship["manually_edited"] == 1
    assert manual_relationship["relation_type"] == "师徒"
    assert manual_relationship["strength"] == 0.75
    assert manual_relationship_event["event_type"] == "trust_shift"
    assert updated_relationship_event["event_type"] == "trust_confirmed"
    assert updated_relationship_event["chapter_id"] == second_chapter["id"]
    assert updated_relationship_event["strength_delta"] == 0.3
    assert cleared_relationship_event["chapter_id"] is None
    assert relationship_with_events["events"]
    assert character_relationships["relationship_count"] >= 1
    assert character_relationships["event_count"] >= 2
    assert active_character_relationships["filters"]["status"] == "active"
    assert all(item["status"] == "active" for item in active_character_relationships["relationships"])
    assert relationship_history["filters"]["relation_type"] == "师徒"
    assert relationship_history["relationship_count"] == 1
    assert "trust_confirmed" in [event["event_type"] for event in relationship_history["events"]]
    assert first_chapter_relationship_event["chapter_id"] == first_chapter["id"]
    assert second_chapter_relationship_event["chapter_id"] == second_chapter["id"]
    assert first_chapter_snapshot["chapter"]["id"] == first_chapter["id"]
    assert "chapter_trust" in [event["event_type"] for event in snapshot_relationship["events"]]
    assert "later_split" not in [event["event_type"] for event in snapshot_relationship["events"]]
    assert deleted_relationship_event["deleted"] is True
    assert deleted_relationship["deleted"] is True
    assert manual_relationship["id"] not in relationship_ids_after_delete
    manual_auxiliary = service.create_auxiliary_record(
        document_id,
        {
            "record_type": "location",
            "name": "旧城入口",
            "summary": "入口处出现可疑暗门。",
            "chapter_id": first_chapter["id"],
            "chunk_id": chunk_id,
        },
    )
    updated_auxiliary = service.update_auxiliary_record(
        manual_auxiliary["id"],
        {
            "summary": "入口暗门已确认存在。",
            "status": "resolved",
            "chapter_id": second_chapter["id"],
            "sequence": 9,
        },
    )
    cleared_auxiliary = service.update_auxiliary_record(
        manual_auxiliary["id"],
        {"chapter_id": None},
    )
    auxiliary_plan = service.build_prompt_plan(document_id, query_text="继续写旧城调查", max_tokens=8000)
    auxiliary_section = next(
        section for section in auxiliary_plan["sections"]
        if section["key"] == "auxiliary_records"
    )
    assert manual_auxiliary["record_type"] == "location"
    assert manual_auxiliary["manually_edited"] == 1
    assert updated_auxiliary["summary"] == "入口暗门已确认存在。"
    assert updated_auxiliary["status"] == "resolved"
    assert updated_auxiliary["chapter_id"] == second_chapter["id"]
    assert updated_auxiliary["chunk_id"] is None
    assert updated_auxiliary["sequence"] == 9
    assert cleared_auxiliary["chapter_id"] is None
    assert "旧城入口" in auxiliary_section["content"]

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
            "auxiliary_records.jsonl",
            "prompt_budget_profiles.jsonl",
        } <= names
        assert archive.read("character_entities.jsonl").strip()
        assert archive.read("timeline_nodes.jsonl").strip()
        assert archive.read("auxiliary_records.jsonl").strip()
    report = service.validate_package(package)
    assert report["package"]["material_layer_counts"]["timeline"] > 0
    assert report["package"]["material_layer_counts"]["characters"] > 0
    assert report["package"]["material_layer_counts"]["auxiliary"] > 0

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
    assert imported_overview["auxiliary_records"][0]["name"] == "旧城入口"


def test_character_fact_conflicts_enter_review_queue(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "人物事实冲突.txt",
        "utf-8",
        "第一章 入口\n林舟抵达旧城入口。\n\n第二章 深处\n林舟进入旧城深处。",
    )
    document_id = imported["document"]["id"]
    first_chapter = imported["chapters"][0]
    second_chapter = imported["chapters"][1]
    service = app_module.MaterialPackageService(database)
    character = service.create_character_entity(
        document_id,
        {
            "canonical_name": "林舟",
            "profile": {"identity": "调查者"},
        },
    )
    entrance_fact = service.create_character_fact(
        character["id"],
        {
            "field": "位置",
            "value": "旧城入口",
            "valid_from_chapter_id": first_chapter["id"],
            "valid_to_chapter_id": first_chapter["id"],
        },
    )
    deep_fact = service.create_character_fact(
        character["id"],
        {
            "field": "位置",
            "value": "旧城深处",
            "valid_from_chapter_id": second_chapter["id"],
            "valid_to_chapter_id": second_chapter["id"],
        },
    )
    assert [
        item for item in service.list_review_items(document_id)
        if item["review_type"] == "character_fact_conflict"
    ] == []

    updated = service.update_character_fact(
        entrance_fact["id"],
        {"valid_to_chapter_id": second_chapter["id"]},
    )
    conflicts = [
        item for item in service.list_review_items(document_id)
        if item["review_type"] == "character_fact_conflict"
    ]
    resolved = service.resolve_review_item(
        conflicts[0]["id"],
        {"apply": "apply_character_fact_conflict", "note": "用新事实覆盖"},
    )
    with database.connect() as connection:
        remaining_facts = connection.execute(
            "SELECT id, value FROM character_facts WHERE character_id = ? ORDER BY created_at",
            (character["id"],),
        ).fetchall()

    assert updated["valid_to_chapter_id"] == second_chapter["id"]
    assert len(conflicts) == 1
    assert conflicts[0]["title"] == "人物事实冲突：林舟 / 位置"
    assert conflicts[0]["payload"]["incoming_fact_id"] == entrance_fact["id"]
    assert conflicts[0]["payload"]["incoming_value"] == "旧城入口"
    assert conflicts[0]["payload"]["conflicts"][0]["fact_id"] == deep_fact["id"]
    assert conflicts[0]["payload"]["conflicts"][0]["value"] == "旧城深处"
    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["applied"]["projected"] == "character_fact"
    assert resolved["resolution"]["applied"]["deleted_conflict_count"] == 1
    assert [row["id"] for row in remaining_facts] == [entrance_fact["id"]]
    assert remaining_facts[0]["value"] == "旧城入口"


def test_relationship_overlap_conflicts_enter_review_queue(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "关系覆盖.txt",
        "utf-8",
        "第一章 关系\n林舟和苏晚暂时合作。",
    )
    document_id = imported["document"]["id"]
    service = app_module.MaterialPackageService(database)
    source = service.create_character_entity(
        document_id,
        {"canonical_name": "林舟", "profile": {"identity": "调查者"}},
    )
    target = service.create_character_entity(
        document_id,
        {"canonical_name": "苏晚", "profile": {"identity": "线索持有人"}},
    )
    alliance = service.create_relationship(
        document_id,
        {
            "source_character_id": source["id"],
            "target_character_id": target["id"],
            "relation_type": "同盟",
            "status": "active",
        },
    )
    assert [
        item for item in service.list_review_items(document_id)
        if item["review_type"] == "relationship_overlap_conflict"
    ] == []

    rivalry = service.create_relationship(
        document_id,
        {
            "source_character_id": source["id"],
            "target_character_id": target["id"],
            "relation_type": "敌对",
            "status": "active",
        },
    )
    conflicts = [
        item for item in service.list_review_items(document_id)
        if item["review_type"] == "relationship_overlap_conflict"
    ]
    resolved = service.resolve_review_item(
        conflicts[0]["id"],
        {"apply": "apply_relationship_overlap_conflict", "note": "接受新关系"},
    )
    relationships = {item["id"]: item for item in service.list_relationships(document_id)}

    assert len(conflicts) == 1
    assert conflicts[0]["title"] == "关系覆盖待确认：林舟 -> 苏晚"
    assert conflicts[0]["payload"]["incoming_relationship_id"] == rivalry["id"]
    assert conflicts[0]["payload"]["incoming_relation_type"] == "敌对"
    assert conflicts[0]["payload"]["conflicts"][0]["relationship_id"] == alliance["id"]
    assert conflicts[0]["payload"]["conflicts"][0]["relation_type"] == "同盟"
    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["applied"]["projected"] == "relationship_overlap"
    assert resolved["resolution"]["applied"]["updated_conflict_count"] == 1
    assert relationships[rivalry["id"]]["status"] == "active"
    assert relationships[alliance["id"]]["status"] == "superseded"


def test_weak_character_aliases_enter_review_queue(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "弱别名.txt",
        "utf-8",
        "第一章 称谓\n林舟被人称作林记者，也有人只叫他队长。",
    )
    document_id = imported["document"]["id"]
    repository.replace_characters(
        document_id,
        [
            {
                "name": "林舟",
                "aliases": ["林记者", "他", "队长"],
                "identity": "调查者",
            }
        ],
    )
    service = app_module.MaterialPackageService(database)
    characters = service.seed_character_entities(document_id)
    character = characters[0]
    aliases = {item["alias"] for item in character["aliases"]}
    pending_aliases = [
        item for item in service.list_review_items(document_id)
        if item["review_type"] == "character_alias_pending"
    ]
    resolved = service.resolve_review_item(
        pending_aliases[0]["id"],
        {"apply": "apply_character_alias"},
    )
    updated_character = service.list_character_entities(document_id)[0]
    updated_aliases = {item["alias"] for item in updated_character["aliases"]}

    assert aliases == {"林记者"}
    assert {item["payload"]["alias"] for item in pending_aliases} == {"他", "队长"}
    assert pending_aliases[0]["title"].startswith("别名待确认：林舟 / ")
    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["applied"]["projected"] == "character_alias"
    assert resolved["resolution"]["applied"]["alias"] in {"他", "队长"}
    assert resolved["resolution"]["applied"]["alias"] in updated_aliases


def test_character_merge_candidates_enter_review_queue(tmp_path: Path) -> None:
    database, repository = make_repository(tmp_path)
    imported = repository.import_document(
        "default",
        "人物合并候选.txt",
        "utf-8",
        "第一章 称谓\n林舟以林记者的身份调查。",
    )
    document_id = imported["document"]["id"]
    with database.connect() as connection:
        connection.executemany(
            """
            INSERT INTO document_characters
                (id, document_id, name, aliases_json, card_json, prompt_text,
                 source_chapters_json, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, '', '[]', 1, 'now', 'now')
            """,
            [
                (
                    "legacy-linzhou",
                    document_id,
                    "林舟",
                    json.dumps(["林记者"], ensure_ascii=False),
                    json.dumps({"identity": "调查者"}, ensure_ascii=False),
                ),
                (
                    "legacy-linjizhe",
                    document_id,
                    "林记者",
                    "[]",
                    json.dumps({"identity": "临时称谓"}, ensure_ascii=False),
                ),
            ],
        )
    service = app_module.MaterialPackageService(database)
    characters = service.seed_character_entities(document_id)
    candidates = [
        item for item in service.list_review_items(document_id)
        if item["review_type"] == "character_merge_candidate"
    ]
    resolved = service.resolve_review_item(
        candidates[0]["id"],
        {"apply": "merge_character_candidate"},
    )
    merged_characters = service.list_character_entities(document_id)
    merged_by_name = {item["canonical_name"]: item for item in merged_characters}

    assert {item["canonical_name"] for item in characters} == {"林舟", "林记者"}
    assert len(candidates) == 1
    assert candidates[0]["title"] == "人物合并待确认：林记者 -> 林舟"
    assert candidates[0]["payload"]["source"] == "林记者"
    assert candidates[0]["payload"]["target"] == "林舟"
    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["applied"]["projected"] == "character_merge"
    assert set(merged_by_name) == {"林舟"}
    assert "林记者" in [item["alias"] for item in merged_by_name["林舟"]["aliases"]]


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

    plot_observation = next(
        item for item in projected["observations"]
        if item["observation_type"] == "plot_event"
    )
    updated_observation = service.update_semantic_observation(
        plot_observation["id"],
        {"status": "disabled"},
    )
    observation_ledger = service.semantic_observation_ledger(document_id)
    disabled_observations = service.semantic_observation_ledger(document_id, status="disabled")
    plot_observations = service.semantic_observation_ledger(document_id, observation_type="plot_event")
    assert updated_observation["status"] == "disabled"
    assert updated_observation["manually_edited"] == 1
    assert observation_ledger["by_type_status"]["plot_event"]["disabled"] == 1
    assert disabled_observations["filters"]["status"] == "disabled"
    assert disabled_observations["filtered_count"] == 1
    assert disabled_observations["recent_observations"][0]["status"] == "disabled"
    assert plot_observations["filters"]["observation_type"] == "plot_event"
    assert all(item["observation_type"] == "plot_event" for item in plot_observations["recent_observations"])
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
    object_item = next(
        item for item in projected["review_items"]
        if item["review_type"] == "object_observation"
    )
    unresolved_item = next(
        item for item in projected["review_items"]
        if item["review_type"] == "unresolved_observation"
    )
    ability_item = next(
        item for item in projected["review_items"]
        if item["review_type"] == "ability_observation"
    )
    resolved_auxiliary = service.resolve_review_item(
        auxiliary_item["id"],
        {"apply": "apply_auxiliary_observation"},
    )
    resolved_object = service.resolve_review_item(
        object_item["id"],
        {"apply": "apply_auxiliary_observation"},
    )
    resolved_unresolved = service.resolve_review_item(
        unresolved_item["id"],
        {"apply": "apply_auxiliary_observation"},
    )
    resolved_ability = service.resolve_review_item(
        ability_item["id"],
        {"apply": "apply_auxiliary_observation"},
    )
    relationships = service.list_relationships(document_id)
    characters = service.list_character_entities(document_id)
    timeline = service.get_timeline(document_id)
    auxiliary_records = service.list_auxiliary_records(document_id)

    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["applied"]["projected"] == "relationship_event"
    assert resolved_auxiliary["status"] == "resolved"
    assert resolved_auxiliary["resolution"]["applied"]["projected"] == "location_observation"
    assert resolved_auxiliary["resolution"]["applied"]["auxiliary_record_id"]
    assert resolved_object["resolution"]["applied"]["projected"] == "object_observation"
    assert resolved_unresolved["resolution"]["applied"]["projected"] == "unresolved_observation"
    assert resolved_ability["resolution"]["applied"]["projected"] == "character_event"
    assert {item["record_type"] for item in auxiliary_records} == {
        "location",
        "object",
        "unresolved",
    }
    assert any(item["name"] == "旧站台" for item in auxiliary_records)
    assert any(item["name"] == "铜钥匙" for item in auxiliary_records)
    assert any(item["name"] == "暗门后的脚印" for item in auxiliary_records)
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
                ('api-review-reject', ?, 'character_entity_missing', '待确认人物', '{"character":"陌生人"}', 'pending', 'now', 'now'),
                ('api-review-batch-resolve', ?, 'local', '批量确认项', '{"note":"safe"}', 'pending', 'now', 'now'),
                ('api-review-batch-reject', ?, 'local', '批量忽略项', '{"note":"skip"}', 'pending', 'now', 'now')
            """,
            (document_id, document_id, document_id, document_id),
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
        package_report = client.get(f"/api/experimental/material-system/documents/{document_id}/package/report")
        observations = client.get(
            f"/api/experimental/material-system/documents/{document_id}/observations?limit=5"
        )
        observation_update = client.patch(
            f"/api/experimental/material-system/observations/{observations.json()['recent_observations'][0]['id']}",
            json={"status": "resolved"},
        )
        filtered_observations = client.get(
            f"/api/experimental/material-system/documents/{document_id}/observations?status=resolved&limit=5"
        )
        node_create = client.post(
            f"/api/experimental/material-system/documents/{document_id}/timeline/nodes",
            json={"title": "API 新阶段", "node_type": "stage", "summary": "接口创建"},
        )
        node_update = client.patch(
            f"/api/experimental/material-system/timeline-nodes/{rebuilt_data['timeline']['nodes'][0]['id']}",
            json={
                "title": "人工节点",
                "summary": "人工节点摘要",
                "parent_id": None,
                "start_chapter_id": chapter_id,
                "end_chapter_id": chapter_id,
                "position": 9,
                "enabled": False,
            },
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
        character_create = client.post(
            f"/api/experimental/material-system/documents/{document_id}/characters/entities",
            json={
                "canonical_name": "API 新人物",
                "aliases": ["新人物别名"],
                "profile": {"identity": "接口创建人物"},
            },
        )
        profile_create = client.post(
            f"/api/experimental/material-system/characters/entities/{character_create.json()['id']}/profiles",
            json={
                "title": "API 第二阶段",
                "identity": "接口创建阶段",
                "start_chapter_id": chapter_id,
                "end_chapter_id": chapter_id,
            },
        )
        profile_update = client.patch(
            f"/api/experimental/material-system/characters/profiles/{profile_create.json()['id']}",
            json={
                "title": "API 第二阶段修订",
                "identity": "接口修订阶段",
                "start_chapter_id": None,
                "end_chapter_id": None,
                "enabled": False,
            },
        )
        profile_delete = client.delete(
            f"/api/experimental/material-system/characters/profiles/{profile_create.json()['id']}"
        )
        character_event_create = client.post(
            f"/api/experimental/material-system/characters/entities/{character_create.json()['id']}/events",
            json={"event_type": "api_event", "value": "接口创建人物经历"},
        )
        character_event_update = client.patch(
            f"/api/experimental/material-system/characters/events/{character_event_create.json()['id']}",
            json={
                "event_type": "api_decision",
                "value": "接口修订人物经历",
                "chapter_id": chapter_id,
                "sequence": 3,
            },
        )
        character_event_clear_chapter = client.patch(
            f"/api/experimental/material-system/characters/events/{character_event_create.json()['id']}",
            json={"chapter_id": None},
        )
        character_event_delete = client.delete(
            f"/api/experimental/material-system/characters/events/{character_event_create.json()['id']}"
        )
        character_fact_create = client.post(
            f"/api/experimental/material-system/characters/entities/{character_create.json()['id']}/facts",
            json={"field": "位置", "value": "接口创建位置", "certainty": 0.7},
        )
        character_fact_update = client.patch(
            f"/api/experimental/material-system/characters/facts/{character_fact_create.json()['id']}",
            json={
                "field": "状态",
                "value": "接口修订状态",
                "valid_from_chapter_id": chapter_id,
                "valid_to_chapter_id": chapter_id,
                "certainty": 0.88,
            },
        )
        character_fact_clear_range = client.patch(
            f"/api/experimental/material-system/characters/facts/{character_fact_create.json()['id']}",
            json={"valid_from_chapter_id": None, "valid_to_chapter_id": None},
        )
        character_fact_delete = client.delete(
            f"/api/experimental/material-system/characters/facts/{character_fact_create.json()['id']}"
        )
        character_split_relationship = client.post(
            f"/api/experimental/material-system/documents/{document_id}/relationships",
            json={
                "source_character_id": character_create.json()["id"],
                "target_character_id": rebuilt_data["characters"][0]["id"],
                "relation_type": "API 拆分关系",
            },
        )
        character_dependencies = client.get(
            f"/api/experimental/material-system/characters/entities/{character_create.json()['id']}/dependencies"
        )
        character_split = client.post(
            f"/api/experimental/material-system/characters/entities/{character_create.json()['id']}/split",
            json={
                "canonical_name": "API 拆分人物",
                "aliases": ["新人物别名"],
                "relationship_ids": [character_split_relationship.json()["id"]],
            },
        )
        character_split_relationship_delete = client.delete(
            f"/api/experimental/material-system/relationships/{character_split_relationship.json()['id']}"
        )
        split_target_delete = client.delete(
            f"/api/experimental/material-system/characters/entities/{character_split.json()['split']['new_character_id']}"
        )
        character_delete = client.delete(
            f"/api/experimental/material-system/characters/entities/{character_create.json()['id']}"
        )
        relationship_create = client.post(
            f"/api/experimental/material-system/documents/{document_id}/relationships",
            json={
                "source_character_id": rebuilt_data["characters"][0]["id"],
                "target_character_id": rebuilt_data["characters"][1]["id"],
                "relation_type": "API 关系",
                "strength": 0.67,
            },
        )
        relationship_event_create = client.post(
            f"/api/experimental/material-system/relationships/{relationship_create.json()['id']}/events",
            json={
                "event_type": "api_shift",
                "description": "接口创建关系事件",
                "chapter_id": chapter_id,
                "strength_delta": 0.12,
            },
        )
        relationship_event_update = client.patch(
            f"/api/experimental/material-system/relationships/events/{relationship_event_create.json()['id']}",
            json={
                "event_type": "api_resolved",
                "description": "接口修订关系事件",
                "chapter_id": chapter_id,
                "strength_delta": 0.22,
            },
        )
        relationship_event_clear_chapter = client.patch(
            f"/api/experimental/material-system/relationships/events/{relationship_event_create.json()['id']}",
            json={"chapter_id": None},
        )
        relationship_snapshot_view = client.get(
            f"/api/experimental/material-system/documents/{document_id}/relationships/snapshot",
            params={"chapter_id": chapter_id},
        )
        character_relationships_view = client.get(
            f"/api/experimental/material-system/characters/entities/{rebuilt_data['characters'][0]['id']}/relationships",
            params={"status": "active"},
        )
        relationship_history_view = client.get(
            (
                "/api/experimental/material-system/characters/entities/"
                f"{rebuilt_data['characters'][0]['id']}/relationships/"
                f"{rebuilt_data['characters'][1]['id']}/history"
            ),
            params={"relation_type": "API 关系"},
        )
        relationship_event_delete = client.delete(
            f"/api/experimental/material-system/relationships/events/{relationship_event_create.json()['id']}"
        )
        relationship_delete = client.delete(
            f"/api/experimental/material-system/relationships/{relationship_create.json()['id']}"
        )
        auxiliary_create = client.post(
            f"/api/experimental/material-system/documents/{document_id}/auxiliary-records",
            json={
                "record_type": "object",
                "name": "API 铜钥匙",
                "summary": "接口创建物件账本",
                "status": "active",
                "chapter_id": chapter_id,
            },
        )
        auxiliary_update = client.patch(
            f"/api/experimental/material-system/auxiliary-records/{auxiliary_create.json()['id']}",
            json={"summary": "接口修订物件账本", "status": "resolved", "sequence": 5},
        )
        auxiliary_clear = client.patch(
            f"/api/experimental/material-system/auxiliary-records/{auxiliary_create.json()['id']}",
            json={"chapter_id": None},
        )
        auxiliary_delete = client.delete(
            f"/api/experimental/material-system/auxiliary-records/{auxiliary_create.json()['id']}"
        )
        timeline_update = client.patch(
            f"/api/experimental/material-system/timeline-events/{rebuilt_data['timeline']['events'][0]['id']}",
            json={
                "title": "改写后的相遇",
                "event_type": "api_rewrite",
                "status": "resolved",
                "chapter_id": chapter_id,
                "participants": ["林舟", "苏晚"],
                "sequence": 11,
            },
        )
        character_update = client.patch(
            f"/api/experimental/material-system/characters/entities/{rebuilt_data['characters'][0]['id']}",
            json={"canonical_name": "林舟改", "enabled": False, "profile": {"identity": "改名后的调查者"}},
        )
        relationship_update = client.patch(
            f"/api/experimental/material-system/relationships/{rebuilt_data['relationships'][0]['id']}",
            json={"relation_type": "伙伴", "strength": 0.8},
        )
        relationship_network = client.get(
            f"/api/experimental/material-system/documents/{document_id}/relationships/network"
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
        snapshot = client.get(
            f"/api/experimental/material-system/documents/{document_id}/snapshot?max_tokens=8000"
        )
        entities = client.get(
            f"/api/experimental/material-system/documents/{document_id}/characters/entities"
        )
        character_snapshot = client.get(
            f"/api/experimental/material-system/characters/entities/{rebuilt_data['characters'][0]['id']}/snapshot",
            params={"chapter_id": chapter_id},
        )
        conversation = client.post("/api/conversations", json={"title": "实验提示词"}).json()
        preview = client.get(
            f"/api/conversations/{conversation['id']}/prompt-preview?query=继续"
        )
        review_items = client.get(
            f"/api/experimental/material-system/documents/{document_id}/review-items"
        )
        filtered_review_items = client.get(
            f"/api/experimental/material-system/documents/{document_id}/review-items",
            params={"status": "pending", "review_type": "local"},
        )
        invalid_review_filter = client.get(
            f"/api/experimental/material-system/documents/{document_id}/review-items",
            params={"status": "done"},
        )
        batch_resolved = client.post(
            f"/api/experimental/material-system/documents/{document_id}/review-items/batch/resolve",
            json={
                "item_ids": ["api-review-batch-resolve", "api-review-resolve"],
                "resolution": {"note": "批量确认"},
            },
        )
        batch_rejected = client.post(
            f"/api/experimental/material-system/documents/{document_id}/review-items/batch/reject",
            json={
                "item_ids": ["api-review-batch-reject"],
                "resolution": {"note": "批量忽略"},
            },
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
    assert rebuilt.json()["timeline"]["tree"][0]["node_type"] == "project"
    assert rebuilt.json()["timeline"]["tree"][0]["children"]
    assert package_report.status_code == 200
    assert package_report.json()["target"]["mode"] == "existing_document"
    assert package_report.json()["checks"]["source_document_hash"] == "match"
    assert package_report.json()["package"]["material_layer_counts"]["timeline"] > 0
    assert observations.status_code == 200
    assert observations.json()["observation_count"] >= 2
    assert len(observations.json()["recent_observations"]) <= 5
    assert observation_update.status_code == 200
    assert observation_update.json()["status"] == "resolved"
    assert observation_update.json()["manually_edited"] == 1
    assert filtered_observations.status_code == 200
    assert filtered_observations.json()["filters"]["status"] == "resolved"
    assert filtered_observations.json()["filtered_count"] >= 1
    assert all(item["status"] == "resolved" for item in filtered_observations.json()["recent_observations"])
    assert node_create.status_code == 201
    assert node_create.json()["title"] == "API 新阶段"
    assert node_create.json()["node_type"] == "stage"
    assert node_delete.json()["deleted"] is True
    assert event_create.status_code == 201
    assert event_create.json()["title"] == "API 新事件"
    assert event_create.json()["participants"] == ["林舟"]
    assert event_delete.json()["deleted"] is True
    assert character_create.status_code == 201
    assert character_create.json()["canonical_name"] == "API 新人物"
    assert character_create.json()["aliases"][0]["alias"] == "新人物别名"
    assert character_create.json()["profiles"][0]["identity"] == "接口创建人物"
    assert profile_create.status_code == 201
    assert profile_create.json()["title"] == "API 第二阶段"
    assert profile_create.json()["start_chapter_id"] == chapter_id
    assert profile_create.json()["end_chapter_id"] == chapter_id
    assert profile_update.json()["title"] == "API 第二阶段修订"
    assert profile_update.json()["identity"] == "接口修订阶段"
    assert profile_update.json()["start_chapter_id"] is None
    assert profile_update.json()["end_chapter_id"] is None
    assert profile_update.json()["enabled"] == 0
    assert profile_delete.json()["deleted"] is True
    assert character_event_create.status_code == 201
    assert character_event_create.json()["event_type"] == "api_event"
    assert character_event_update.json()["event_type"] == "api_decision"
    assert character_event_update.json()["value"] == "接口修订人物经历"
    assert character_event_update.json()["chapter_id"] == chapter_id
    assert character_event_update.json()["sequence"] == 3
    assert character_event_clear_chapter.json()["chapter_id"] is None
    assert character_event_delete.json()["deleted"] is True
    assert character_fact_create.status_code == 201
    assert character_fact_create.json()["field"] == "位置"
    assert character_fact_update.json()["field"] == "状态"
    assert character_fact_update.json()["value"] == "接口修订状态"
    assert character_fact_update.json()["valid_from_chapter_id"] == chapter_id
    assert character_fact_update.json()["valid_to_chapter_id"] == chapter_id
    assert character_fact_update.json()["certainty"] == 0.88
    assert character_fact_clear_range.json()["valid_from_chapter_id"] is None
    assert character_fact_clear_range.json()["valid_to_chapter_id"] is None
    assert character_fact_delete.json()["deleted"] is True
    assert character_split_relationship.status_code == 201
    assert character_dependencies.status_code == 200
    assert character_dependencies.json()["can_delete"] is False
    assert character_dependencies.json()["relationship_count"] == 1
    assert character_split.status_code == 200
    assert character_split.json()["split"]["moved_aliases"] == ["新人物别名"]
    assert character_split.json()["split"]["moved_relationships"] == 1
    split_target = next(
        character for character in character_split.json()["characters"]
        if character["id"] == character_split.json()["split"]["new_character_id"]
    )
    split_relationship = next(
        relationship for relationship in character_split.json()["relationships"]
        if relationship["id"] == character_split_relationship.json()["id"]
    )
    assert split_target["canonical_name"] == "API 拆分人物"
    assert split_target["aliases"][0]["alias"] == "新人物别名"
    assert split_relationship["source_character_id"] == split_target["id"]
    assert split_relationship["source_name"] == "API 拆分人物"
    assert character_split_relationship_delete.json()["deleted"] is True
    assert split_target_delete.json()["deleted"] is True
    assert character_delete.json()["deleted"] is True
    assert relationship_create.status_code == 201
    assert relationship_create.json()["relation_type"] == "API 关系"
    assert relationship_create.json()["strength"] == 0.67
    assert relationship_event_create.status_code == 201
    assert relationship_event_create.json()["event_type"] == "api_shift"
    assert relationship_event_update.json()["event_type"] == "api_resolved"
    assert relationship_event_update.json()["description"] == "接口修订关系事件"
    assert relationship_event_update.json()["chapter_id"] == chapter_id
    assert relationship_event_update.json()["strength_delta"] == 0.22
    assert relationship_event_clear_chapter.json()["chapter_id"] is None
    assert relationship_snapshot_view.status_code == 200
    assert relationship_snapshot_view.json()["chapter"]["id"] == chapter_id
    assert relationship_snapshot_view.json()["relationship_count"] >= 1
    assert any(
        item["id"] == relationship_create.json()["id"]
        for item in relationship_snapshot_view.json()["relationships"]
    )
    assert character_relationships_view.status_code == 200
    assert character_relationships_view.json()["filters"]["status"] == "active"
    assert any(
        item["id"] == relationship_create.json()["id"]
        for item in character_relationships_view.json()["relationships"]
    )
    assert relationship_history_view.status_code == 200
    assert relationship_history_view.json()["filters"]["relation_type"] == "API 关系"
    assert relationship_history_view.json()["event_count"] >= 2
    assert relationship_history_view.json()["events"][-1]["event_type"] == "api_resolved"
    assert relationship_event_delete.json()["deleted"] is True
    assert relationship_delete.json()["deleted"] is True
    assert auxiliary_create.status_code == 201
    assert auxiliary_create.json()["record_type"] == "object"
    assert auxiliary_create.json()["name"] == "API 铜钥匙"
    assert auxiliary_update.json()["summary"] == "接口修订物件账本"
    assert auxiliary_update.json()["status"] == "resolved"
    assert auxiliary_update.json()["chapter_id"] == chapter_id
    assert auxiliary_update.json()["sequence"] == 5
    assert auxiliary_update.json()["manually_edited"] == 1
    assert auxiliary_clear.json()["chapter_id"] is None
    assert auxiliary_delete.json()["deleted"] is True
    assert node_update.json()["title"] == "人工节点"
    assert node_update.json()["summary"] == "人工节点摘要"
    assert node_update.json()["parent_id"] is None
    assert node_update.json()["start_chapter_id"] == chapter_id
    assert node_update.json()["end_chapter_id"] == chapter_id
    assert node_update.json()["position"] == 9
    assert node_update.json()["enabled"] == 0
    assert node_update.json()["manually_edited"] == 1
    assert timeline_update.json()["title"] == "改写后的相遇"
    assert timeline_update.json()["event_type"] == "api_rewrite"
    assert timeline_update.json()["status"] == "resolved"
    assert timeline_update.json()["chapter_id"] == chapter_id
    assert timeline_update.json()["participants"] == ["林舟", "苏晚"]
    assert timeline_update.json()["sequence"] == 11
    assert character_update.json()["canonical_name"] == "林舟改"
    assert character_update.json()["enabled"] is False
    assert character_update.json()["profiles"][0]["identity"] == "改名后的调查者"
    assert relationship_update.json()["relation_type"] == "伙伴"
    assert relationship_update.json()["strength"] == 0.8
    assert relationship_network.status_code == 200
    assert relationship_network.json()["node_count"] >= 2
    assert relationship_network.json()["edge_count"] >= 1
    assert relationship_network.json()["central_characters"][0]["degree"] >= 1
    assert budget_profile.json()["config"]["project_summary"] > 4
    assert updated_budget.json()["config"]["project_summary"] == 4
    assert plan.status_code == 200
    plan_sections = {section["key"]: section for section in plan.json()["sections"]}
    assert plan_sections["project_summary"]["budget"] == 4
    assert plan_sections["project_summary"]["tokens"] <= 4
    project_summary_trim = next(
        item for item in plan.json()["trimmed"]
        if item["key"] == "project_summary" and item["reason"] == "分段预算裁剪"
    )
    assert plan_sections["project_summary"]["original_tokens"] >= plan_sections["project_summary"]["tokens"]
    assert project_summary_trim["label"] == "前文总览"
    assert project_summary_trim["budget"] == 4
    assert project_summary_trim["original_tokens"] >= project_summary_trim["tokens"]
    assert "character_snapshots" in plan_sections
    assert "relationship_history" in plan_sections
    assert plan_sections["relationship_history"]["included"] is True
    assert snapshot.status_code == 200
    assert snapshot.json()["sections"]
    assert "人物当前快照" in snapshot.json()["content"]
    assert "人物关系历史" in snapshot.json()["content"]
    assert [item["canonical_name"] for item in entities.json()] == ["林舟改", "苏晚"]
    assert character_snapshot.status_code == 200
    assert character_snapshot.json()["chapter"]["id"] == chapter_id
    assert character_snapshot.json()["selected_profile"]
    assert "text" in character_snapshot.json()
    assert preview.json()["sources"]["recent_chapters"].startswith("当前时间线节点")
    assert "人物当前快照" in preview.json()["sources"]["characters"]
    assert "人物关系历史" in preview.json()["sources"]["characters"]
    assert {item["id"] for item in review_items.json()} >= {"api-review-resolve", "api-review-reject"}
    assert filtered_review_items.status_code == 200
    assert {item["id"] for item in filtered_review_items.json()} == {
        "api-review-batch-resolve",
        "api-review-batch-reject",
    }
    assert all(item["status"] == "pending" for item in filtered_review_items.json())
    assert all(item["review_type"] == "local" for item in filtered_review_items.json())
    assert invalid_review_filter.status_code == 400
    assert batch_resolved.status_code == 200
    assert batch_resolved.json()["updated_count"] == 1
    assert batch_resolved.json()["skipped_count"] == 1
    assert batch_resolved.json()["skipped"][0]["reason"] == "requires_manual_payload"
    assert next(
        item for item in batch_resolved.json()["review_items"]
        if item["id"] == "api-review-batch-resolve"
    )["status"] == "resolved"
    assert batch_rejected.status_code == 200
    assert batch_rejected.json()["updated_count"] == 1
    assert next(
        item for item in batch_rejected.json()["review_items"]
        if item["id"] == "api-review-batch-reject"
    )["status"] == "rejected"
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
