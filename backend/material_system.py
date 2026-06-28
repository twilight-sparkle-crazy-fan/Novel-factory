from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Iterable

from .database import Database, new_id, utc_now
from .material_utils import stable_text_hash
from .text_import import split_long_text


PACKAGE_FORMAT = "novel-factory-analysis-package"
PACKAGE_FORMAT_VERSION = "1.0"
MATERIAL_SCHEMA_VERSION = "material-schema-v1"
GENERATOR_VERSION = "experimental-material-system-0.1"


class MaterialPackageError(ValueError):
    pass


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"


class MaterialPackageService:
    def __init__(self, database: Database):
        self.database = database

    def export_document_package(self, document_id: str) -> bytes:
        now = utc_now()
        with self.database.connect() as connection:
            document = connection.execute(
                "SELECT * FROM source_documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None:
                raise KeyError("document_not_found")
            chapters = connection.execute(
                "SELECT * FROM chapters WHERE document_id = ? ORDER BY position",
                (document_id,),
            ).fetchall()
            chunks = connection.execute(
                """
                SELECT cc.*, c.document_id
                FROM chapter_chunks cc
                JOIN chapters c ON c.id = cc.chapter_id
                WHERE c.document_id = ?
                ORDER BY c.position, cc.position
                """,
                (document_id,),
            ).fetchall()
            self._ensure_source_provenance(connection, document, chapters, chunks, now)
            provenance_rows = connection.execute(
                "SELECT * FROM material_provenance WHERE document_id = ? ORDER BY source_type, source_id",
                (document_id,),
            ).fetchall()

        document_hash = document["raw_text_hash"] or stable_text_hash(document["raw_text"])
        manifest = {
            "format": PACKAGE_FORMAT,
            "format_version": PACKAGE_FORMAT_VERSION,
            "project_id": document["project_id"],
            "document_id": document["id"],
            "source_document_hash": document_hash,
            "chapter_count": len(chapters),
            "chunk_count": len(chunks),
            "created_at": now,
            "generator": {
                "application": "Novel-factory",
                "application_version": GENERATOR_VERSION,
                "model": "local-or-remote-model-name",
                "prompt_version": "source-provenance-v1",
                "schema_version": MATERIAL_SCHEMA_VERSION,
                "analysis_profile": "source-package",
            },
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as package:
            package.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            )
            package.writestr(
                "documents.json",
                json.dumps(
                    [self._document_record(document)],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )
            package.writestr(
                "chapters.jsonl",
                "".join(_json_line(self._chapter_record(row)) for row in chapters),
            )
            package.writestr(
                "chunks.jsonl",
                "".join(_json_line(self._chunk_record(row)) for row in chunks),
            )
            package.writestr(
                "provenance.jsonl",
                "".join(_json_line(dict(row)) for row in provenance_rows),
            )
        return buffer.getvalue()

    def validate_package(
        self, package_bytes: bytes, *, target_document_id: str | None = None
    ) -> dict[str, Any]:
        package = self._open_package(package_bytes)
        with package:
            manifest = self._read_manifest(package)
            documents = self._read_documents(package)
            package_document = documents[0] if documents else {}
            chapter_stats = self._jsonl_stats(
                package,
                "chapters.jsonl",
                self._chapter_record_keys(),
                {"id", "document_id"},
            )
            chunk_stats = self._jsonl_stats(
                package,
                "chunks.jsonl",
                self._chunk_record_keys(),
                {"id", "chapter_id"},
            )

        format_state = (
            "compatible"
            if manifest.get("format") == PACKAGE_FORMAT
            and manifest.get("format_version") == PACKAGE_FORMAT_VERSION
            else "incompatible"
        )
        schema_version = manifest.get("generator", {}).get("schema_version", "")
        schema_state = "compatible" if schema_version == MATERIAL_SCHEMA_VERSION else "incompatible"
        source_hash = manifest.get("source_document_hash", "")
        report: dict[str, Any] = {
            "ok": format_state == "compatible" and schema_state == "compatible",
            "package": {
                "format": manifest.get("format", ""),
                "format_version": manifest.get("format_version", ""),
                "schema_version": schema_version,
                "project_id": manifest.get("project_id", ""),
                "document_id": manifest.get("document_id", ""),
                "filename": package_document.get("filename", ""),
                "source_document_hash": source_hash,
                "chapter_count": manifest.get("chapter_count", 0),
                "chunk_count": manifest.get("chunk_count", 0),
            },
            "target": {
                "document_id": target_document_id,
                "mode": "existing_document" if target_document_id else "pure_new_file",
            },
            "checks": {
                "format": format_state,
                "schema": schema_state,
                "package_source_document_hash": "not_checked",
                "source_document_hash": "no_target",
                "chapter_count": "not_checked",
                "unknown_fields": chapter_stats["unknown_fields"] + chunk_stats["unknown_fields"],
                "source_chapter_missing": 0,
                "safe_records": chapter_stats["count"] + chunk_stats["count"] + len(documents),
                "review_records": 0,
                "rejected_records": 0,
            },
            "actions": [],
            "can_import": False,
            "can_create_new_document": False,
        }
        if format_state != "compatible":
            report["actions"].append("拒绝导入：分析包格式或版本不兼容。")
            return report
        if schema_state != "compatible":
            report["actions"].append("拒绝导入：schema 版本不兼容。")
            return report

        missing_source_ids = (
            chapter_stats["missing_required"]
            + chunk_stats["missing_required"]
            + len(chunk_stats["chapter_refs"] - chapter_stats["ids"])
        )
        if missing_source_ids:
            report["ok"] = False
            report["checks"]["source_chapter_missing"] = missing_source_ids
            report["checks"]["rejected_records"] = missing_source_ids
            report["actions"].append("拒绝导入：章节或 chunk 的来源 ID 不完整。")
            return report

        package_raw_text = str(package_document.get("raw_text") or "")
        recorded_document_hash = str(package_document.get("raw_text_hash") or "")
        if package_raw_text:
            computed_document_hash = stable_text_hash(package_raw_text)
            if recorded_document_hash and recorded_document_hash != computed_document_hash:
                report["ok"] = False
                report["checks"]["package_source_document_hash"] = "mismatch"
                report["checks"]["rejected_records"] = report["checks"]["safe_records"]
                report["actions"].append("拒绝导入：documents.json 原文 hash 与 raw_text 不一致。")
                return report
            if source_hash != computed_document_hash:
                report["ok"] = False
                report["checks"]["package_source_document_hash"] = "mismatch"
                report["checks"]["rejected_records"] = report["checks"]["safe_records"]
                report["actions"].append("拒绝导入：manifest 原文 hash 与 raw_text 不一致。")
                return report
            report["checks"]["package_source_document_hash"] = "match"
        else:
            report["checks"]["package_source_document_hash"] = "missing"

        manifest_chapter_count = int(manifest.get("chapter_count") or 0)
        if manifest_chapter_count == chapter_stats["count"]:
            report["checks"]["chapter_count"] = "match"
        else:
            report["checks"]["chapter_count"] = "mismatch"
            report["checks"]["review_records"] += abs(manifest_chapter_count - chapter_stats["count"])

        with self.database.connect() as connection:
            if target_document_id:
                target = connection.execute(
                    """
                    SELECT d.*,
                           (SELECT COUNT(*) FROM chapters c WHERE c.document_id = d.id) AS chapter_count
                    FROM source_documents d WHERE d.id = ?
                    """,
                    (target_document_id,),
                ).fetchone()
                if target is None:
                    report["checks"]["source_document_hash"] = "missing_target"
                    report["checks"]["rejected_records"] = report["checks"]["safe_records"]
                    report["actions"].append("拒绝导入：目标文档不存在。")
                    report["ok"] = False
                    return report
                target_hash = target["raw_text_hash"] or stable_text_hash(target["raw_text"])
                report["target"]["source_document_hash"] = target_hash
                report["target"]["filename"] = target["filename"]
                report["target"]["chapter_count"] = target["chapter_count"]
                report["checks"]["source_document_hash"] = (
                    "match" if target_hash == source_hash else "mismatch"
                )
                report["checks"]["chapter_count"] = (
                    "match"
                    if int(target["chapter_count"]) == manifest_chapter_count
                    else "mismatch"
                )
                if report["checks"]["source_document_hash"] == "match":
                    report["can_import"] = True
                    report["actions"].append("可以导入：目标原文哈希一致。")
                else:
                    report["can_import"] = False
                    report["actions"].append(
                        "默认拒绝导入：目标原文哈希不一致。可改用纯新文件导入创建新文档。"
                    )
            else:
                matches = connection.execute(
                    "SELECT id, filename FROM source_documents WHERE raw_text_hash = ?",
                    (source_hash,),
                ).fetchall()
                report["target"]["matching_documents"] = [dict(row) for row in matches]
                report["can_create_new_document"] = bool(package_document.get("raw_text"))
                report["can_import"] = bool(package_document.get("raw_text"))
                if matches:
                    report["actions"].append(
                        "检测到已有相同原文；可选择匹配现有文档，也可作为新文档导入。"
                    )
                else:
                    report["actions"].append(
                        "纯新文件导入：包内带有原文，可创建新文档并导入章节、chunk 与来源记录。"
                    )
        return report

    def import_package(
        self,
        package_bytes: bytes,
        *,
        project_id: str,
        mode: str,
    ) -> dict[str, Any]:
        if mode != "create_document":
            raise MaterialPackageError("当前阶段只支持 mode=create_document 的纯新文件导入")
        report = self.validate_package(package_bytes)
        if not report["can_create_new_document"]:
            raise MaterialPackageError("分析包缺少原文，不能创建新文档")
        package = self._open_package(package_bytes)
        with package:
            manifest = self._read_manifest(package)
            document_record = self._read_documents(package)[0]
            chapters = list(self._iter_jsonl(package, "chapters.jsonl"))
            chunks = list(self._iter_jsonl(package, "chunks.jsonl"))
            provenance = list(self._iter_jsonl(package, "provenance.jsonl"))

        now = utc_now()
        document_id = document_record.get("id") or manifest.get("document_id") or new_id()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
                connection.rollback()
                raise KeyError("project_not_found")
            if connection.execute("SELECT 1 FROM source_documents WHERE id = ?", (document_id,)).fetchone():
                connection.rollback()
                raise MaterialPackageError("分析包中的 document_id 已存在，不能作为纯新文件导入")
            raw_text = str(document_record.get("raw_text") or "")
            if not raw_text.strip():
                connection.rollback()
                raise MaterialPackageError("分析包缺少原文，不能创建新文档")
            source_hash = document_record.get("raw_text_hash") or stable_text_hash(raw_text)
            connection.execute(
                """
                INSERT INTO source_documents
                    (id, project_id, filename, encoding, raw_text, raw_text_hash,
                     global_summary, library_enabled, summary_enabled,
                     recent_chapters_enabled, characters_enabled, facts_enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    project_id,
                    document_record.get("filename") or "导入分析包.txt",
                    document_record.get("encoding") or "utf-8",
                    raw_text,
                    source_hash,
                    document_record.get("global_summary") or "",
                    int(bool(document_record.get("library_enabled", True))),
                    int(bool(document_record.get("summary_enabled", True))),
                    int(bool(document_record.get("recent_chapters_enabled", True))),
                    int(bool(document_record.get("characters_enabled", True))),
                    int(bool(document_record.get("facts_enabled", True))),
                    document_record.get("created_at") or now,
                ),
            )
            for chapter in chapters:
                chapter_id = chapter.get("id") or new_id()
                content = str(chapter.get("content") or "")
                connection.execute(
                    """
                    INSERT INTO chapters
                        (id, document_id, project_id, position, title, content, content_hash,
                         summary_json, character_observations_json, edited_summary,
                         status, error_message, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chapter_id,
                        document_id,
                        project_id,
                        int(chapter.get("position") or 0),
                        chapter.get("title") or "未命名章节",
                        content,
                        chapter.get("content_hash") or stable_text_hash(content),
                        json.dumps(chapter.get("summary") or {}, ensure_ascii=False),
                        json.dumps(chapter.get("character_observations") or [], ensure_ascii=False),
                        chapter.get("edited_summary") or "",
                        chapter.get("status") or "pending",
                        chapter.get("error_message"),
                        chapter.get("created_at") or now,
                        chapter.get("updated_at") or now,
                    ),
                )
            if chunks:
                for chunk in chunks:
                    content = str(chunk.get("content") or "")
                    connection.execute(
                        """
                        INSERT INTO chapter_chunks
                            (id, chapter_id, position, content, content_hash, summary_json,
                             character_observations_json, facts_status, status, error_message,
                             created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.get("id") or new_id(),
                            chunk.get("chapter_id"),
                            int(chunk.get("position") or 0),
                            content,
                            chunk.get("content_hash") or stable_text_hash(content),
                            json.dumps(chunk.get("summary") or {}, ensure_ascii=False),
                            json.dumps(chunk.get("character_observations") or [], ensure_ascii=False),
                            chunk.get("facts_status") or "pending",
                            chunk.get("status") or "pending",
                            chunk.get("error_message"),
                            chunk.get("created_at") or now,
                            chunk.get("updated_at") or now,
                        ),
                    )
            else:
                for chapter in chapters:
                    for position, content in enumerate(split_long_text(str(chapter.get("content") or "")), start=1):
                        connection.execute(
                            """
                            INSERT INTO chapter_chunks
                                (id, chapter_id, position, content, content_hash, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_id(),
                                chapter.get("id"),
                                position,
                                content,
                                stable_text_hash(content),
                                now,
                                now,
                            ),
                        )
            for item in provenance:
                if item.get("document_id") != document_id:
                    item = {**item, "document_id": document_id}
                connection.execute(
                    """
                    INSERT OR REPLACE INTO material_provenance
                        (id, document_id, source_type, source_id, source_hash,
                         analysis_version, prompt_version, model_id, generated_at, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.get("id") or new_id(),
                        item.get("document_id"),
                        item.get("source_type") or "",
                        item.get("source_id") or "",
                        item.get("source_hash") or "",
                        item.get("analysis_version") or "",
                        item.get("prompt_version") or "",
                        item.get("model_id") or "",
                        item.get("generated_at") or now,
                        float(item.get("confidence") or 0.7),
                    ),
                )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
            connection.execute(
                "UPDATE conversations SET document_id = ? WHERE project_id = ? AND document_id IS NULL",
                (document_id, project_id),
            )
            connection.commit()
        return {"document_id": document_id, "report": self.validate_package(package_bytes)}

    def _ensure_source_provenance(
        self,
        connection: Any,
        document: Any,
        chapters: Iterable[Any],
        chunks: Iterable[Any],
        generated_at: str,
    ) -> None:
        records: list[tuple[str, str, str, str]] = [
            (
                f"prov:{document['id']}:source_document:{document['id']}",
                "source_document",
                document["id"],
                document["raw_text_hash"] or stable_text_hash(document["raw_text"]),
            )
        ]
        records.extend(
            (
                f"prov:{document['id']}:chapter:{row['id']}",
                "chapter",
                row["id"],
                row["content_hash"] or stable_text_hash(row["content"]),
            )
            for row in chapters
        )
        records.extend(
            (
                f"prov:{document['id']}:chunk:{row['id']}",
                "chunk",
                row["id"],
                row["content_hash"] or stable_text_hash(row["content"]),
            )
            for row in chunks
        )
        for provenance_id, source_type, source_id, source_hash in records:
            connection.execute(
                """
                INSERT OR REPLACE INTO material_provenance
                    (id, document_id, source_type, source_id, source_hash,
                     analysis_version, prompt_version, model_id, generated_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provenance_id,
                    document["id"],
                    source_type,
                    source_id,
                    source_hash,
                    GENERATOR_VERSION,
                    "source-provenance-v1",
                    "",
                    generated_at,
                    1.0,
                ),
            )

    def _document_record(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "filename": row["filename"],
            "encoding": row["encoding"],
            "raw_text": row["raw_text"],
            "raw_text_hash": row["raw_text_hash"] or stable_text_hash(row["raw_text"]),
            "global_summary": row["global_summary"],
            "library_enabled": bool(row["library_enabled"]),
            "summary_enabled": bool(row["summary_enabled"]),
            "recent_chapters_enabled": bool(row["recent_chapters_enabled"]),
            "characters_enabled": bool(row["characters_enabled"]),
            "facts_enabled": bool(row["facts_enabled"]),
            "created_at": row["created_at"],
        }

    def _chapter_record(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "project_id": row["project_id"],
            "position": row["position"],
            "title": row["title"],
            "content": row["content"],
            "content_hash": row["content_hash"] or stable_text_hash(row["content"]),
            "summary": _json_load(row["summary_json"], {}),
            "character_observations": _json_load(row["character_observations_json"], []),
            "edited_summary": row["edited_summary"],
            "status": row["status"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _chunk_record(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "chapter_id": row["chapter_id"],
            "position": row["position"],
            "content": row["content"],
            "content_hash": row["content_hash"] or stable_text_hash(row["content"]),
            "summary": _json_load(row["summary_json"], {}),
            "character_observations": _json_load(row["character_observations_json"], []),
            "facts_status": row["facts_status"],
            "status": row["status"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _open_package(self, package_bytes: bytes) -> zipfile.ZipFile:
        try:
            return zipfile.ZipFile(io.BytesIO(package_bytes))
        except zipfile.BadZipFile as exc:
            raise MaterialPackageError("不是有效的 .llm4pkg/ZIP 分析包") from exc

    def _read_manifest(self, package: zipfile.ZipFile) -> dict[str, Any]:
        try:
            with package.open("manifest.json") as handle:
                return json.loads(handle.read().decode("utf-8"))
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MaterialPackageError("分析包缺少有效 manifest.json") from exc

    def _read_documents(self, package: zipfile.ZipFile) -> list[dict[str, Any]]:
        try:
            with package.open("documents.json") as handle:
                value = json.loads(handle.read().decode("utf-8"))
        except KeyError:
            return []
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MaterialPackageError("documents.json 不是有效 JSON") from exc
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    def _iter_jsonl(self, package: zipfile.ZipFile, name: str) -> Iterable[dict[str, Any]]:
        try:
            handle = package.open(name)
        except KeyError:
            return
        with handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value

    def _jsonl_stats(
        self,
        package: zipfile.ZipFile,
        name: str,
        known_fields: set[str],
        required_fields: set[str],
    ) -> dict[str, Any]:
        count = 0
        unknown_fields = 0
        missing_required = 0
        ids: set[str] = set()
        chapter_refs: set[str] = set()
        for record in self._iter_jsonl(package, name):
            count += 1
            unknown_fields += len(set(record) - known_fields)
            if any(not record.get(field) for field in required_fields):
                missing_required += 1
            if record.get("id"):
                ids.add(str(record["id"]))
            if record.get("chapter_id"):
                chapter_refs.add(str(record["chapter_id"]))
        return {
            "count": count,
            "unknown_fields": unknown_fields,
            "missing_required": missing_required,
            "ids": ids,
            "chapter_refs": chapter_refs,
        }

    def _chapter_record_keys(self) -> set[str]:
        return {
            "id", "document_id", "project_id", "position", "title", "content",
            "content_hash", "summary", "character_observations", "edited_summary",
            "status", "error_message", "created_at", "updated_at",
        }

    def _chunk_record_keys(self) -> set[str]:
        return {
            "id", "document_id", "chapter_id", "position", "content",
            "content_hash", "summary", "character_observations", "facts_status",
            "status", "error_message", "created_at", "updated_at",
        }
