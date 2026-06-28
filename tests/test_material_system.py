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
