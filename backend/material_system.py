from __future__ import annotations

import io
import json
import hashlib
import zipfile
from typing import Any, Iterable

from .database import Database, new_id, utc_now
from .material_utils import stable_json_hash, stable_text_hash
from .text_import import split_long_text


PACKAGE_FORMAT = "novel-factory-analysis-package"
PACKAGE_FORMAT_VERSION = "1.0"
MATERIAL_SCHEMA_VERSION = "material-schema-v1"
GENERATOR_VERSION = "experimental-material-system-0.1"


class MaterialPackageError(ValueError):
    pass


DEFAULT_PROMPT_BUDGET = {
    "project_summary": 800,
    "current_timeline_node": 1200,
    "recent_chapter_summaries": 1800,
    "timeline_events": 1200,
    "character_snapshots": 2600,
    "relationships": 1000,
    "relationship_history": 900,
    "auxiliary_records": 900,
    "facts": 1000,
    "outline": 1600,
}

TIMELINE_NODE_TYPE_ORDER = {
    "project": 0,
    "volume": 1,
    "arc": 2,
    "stage": 3,
    "chapter_group": 4,
    "chapter": 5,
    "scene": 6,
}

MATERIAL_JSONL_TABLES = {
    "semantic_observations.jsonl": {
        "table": "semantic_observations",
        "columns": [
            "id", "document_id", "chapter_id", "chunk_id", "observation_type",
            "payload_json", "normalized_key", "status", "confidence",
            "provenance_id", "created_at", "updated_at",
        ],
    },
    "timeline_nodes.jsonl": {
        "table": "timeline_nodes",
        "columns": [
            "id", "document_id", "parent_id", "node_type", "title",
            "start_chapter_id", "end_chapter_id", "position", "summary",
            "summary_version", "enabled", "manually_edited", "created_at", "updated_at",
        ],
    },
    "timeline_events.jsonl": {
        "table": "timeline_events",
        "columns": [
            "id", "document_id", "event_type", "title", "description",
            "chapter_id", "chunk_id", "sequence", "participants_json",
            "location_id", "causes_json", "consequences_json", "status",
            "confidence", "manually_edited", "provenance_id", "created_at", "updated_at",
        ],
    },
    "character_entities.jsonl": {
        "table": "character_entities",
        "columns": [
            "id", "document_id", "canonical_name", "entity_type", "enabled",
            "manually_confirmed", "created_at", "updated_at",
        ],
    },
    "character_aliases.jsonl": {
        "table": "character_aliases",
        "columns": [
            "id", "character_id", "alias", "alias_type", "first_chapter_id",
            "last_chapter_id", "confidence", "manually_confirmed", "created_at", "updated_at",
        ],
    },
    "character_profiles.jsonl": {
        "table": "character_profiles",
        "columns": [
            "id", "character_id", "title", "start_chapter_id", "end_chapter_id",
            "identity", "personality", "goals", "behavior_pattern", "ability_stage",
            "social_status", "enabled", "manually_edited", "created_at", "updated_at",
        ],
    },
    "character_facts.jsonl": {
        "table": "character_facts",
        "columns": [
            "id", "character_id", "field", "value", "valid_from_chapter_id",
            "valid_to_chapter_id", "certainty", "provenance_id", "created_at", "updated_at",
        ],
    },
    "character_events.jsonl": {
        "table": "character_events",
        "columns": [
            "id", "character_id", "event_type", "value", "chapter_id",
            "chunk_id", "sequence", "provenance_id", "created_at", "updated_at",
        ],
    },
    "relationship_events.jsonl": {
        "table": "relationship_events",
        "columns": [
            "id", "document_id", "source_character_id", "target_character_id",
            "relation_type", "event_type", "description", "chapter_id", "chunk_id",
            "sequence", "strength_delta", "confidence", "provenance_id",
            "created_at", "updated_at",
        ],
    },
    "character_relationships.jsonl": {
        "table": "character_relationships",
        "columns": [
            "id", "document_id", "source_character_id", "target_character_id",
            "relation_type", "direction", "status", "strength", "start_chapter_id",
            "end_chapter_id", "confidence", "manually_edited", "provenance_id",
            "created_at", "updated_at",
        ],
    },
    "review_items.jsonl": {
        "table": "material_review_items",
        "columns": [
            "id", "document_id", "review_type", "title", "payload_json",
            "status", "resolution_json", "created_at", "updated_at",
        ],
    },
    "auxiliary_records.jsonl": {
        "table": "auxiliary_records",
        "columns": [
            "id", "document_id", "record_type", "name", "summary", "status",
            "chapter_id", "chunk_id", "sequence", "payload_json", "confidence",
            "manually_edited", "provenance_id", "created_at", "updated_at",
        ],
    },
    "prompt_budget_profiles.jsonl": {
        "table": "prompt_budget_profiles",
        "columns": [
            "id", "document_id", "name", "config_json", "is_default",
            "created_at", "updated_at",
        ],
    },
}

MATERIAL_DOCUMENT_TABLES = [
    "semantic_observations",
    "timeline_nodes",
    "timeline_events",
    "character_relationships",
    "relationship_events",
    "material_review_items",
    "auxiliary_records",
    "prompt_budget_profiles",
]

MATERIAL_CHARACTER_TABLES = [
    "character_aliases",
    "character_profiles",
    "character_facts",
    "character_events",
]

WEAK_CHARACTER_ALIASES = {
    "他", "她", "它", "ta", "TA", "男人", "女人", "少年", "少女",
    "老人", "老者", "青年", "中年人", "孩子", "小孩", "老师", "师父",
    "师傅", "队长", "老板", "小姐", "先生", "夫人", "姑娘", "那人",
    "此人", "对方", "那个人", "这个人",
}

MATERIAL_IMPORT_LAYERS = {
    "observations": {"semantic_observations.jsonl"},
    "timeline": {"timeline_nodes.jsonl", "timeline_events.jsonl"},
    "characters": {
        "character_entities.jsonl",
        "character_aliases.jsonl",
        "character_profiles.jsonl",
        "character_facts.jsonl",
        "character_events.jsonl",
        "relationship_events.jsonl",
        "character_relationships.jsonl",
    },
    "reviews": {"review_items.jsonl"},
    "auxiliary": {"auxiliary_records.jsonl"},
    "budget": {"prompt_budget_profiles.jsonl"},
}

MATERIAL_MANUAL_FIELD_RULES = {
    "timeline_nodes": {
        "marker": "manually_edited",
        "fields": {
            "title", "summary", "enabled", "manually_edited", "updated_at",
        },
    },
    "timeline_events": {
        "marker": "manually_edited",
        "fields": {
            "event_type", "title", "description", "status", "confidence",
            "manually_edited", "updated_at",
        },
    },
    "character_entities": {
        "marker": "manually_confirmed",
        "fields": {
            "canonical_name", "entity_type", "enabled", "manually_confirmed", "updated_at",
        },
    },
    "character_aliases": {
        "marker": "manually_confirmed",
        "fields": {
            "alias", "alias_type", "confidence", "manually_confirmed", "updated_at",
        },
    },
    "character_profiles": {
        "marker": "manually_edited",
        "fields": {
            "title", "identity", "personality", "goals", "behavior_pattern",
            "ability_stage", "social_status", "enabled", "manually_edited", "updated_at",
        },
    },
    "character_relationships": {
        "marker": "manually_edited",
        "fields": {
            "relation_type", "direction", "status", "strength", "confidence",
            "manually_edited", "updated_at",
        },
    },
    "auxiliary_records": {
        "marker": "manually_edited",
        "fields": {
            "record_type", "name", "summary", "status", "confidence",
            "manually_edited", "updated_at",
        },
    },
}


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2) if text.strip() else 0


def _safe_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalized_fact_value(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_weak_character_alias(value: Any) -> bool:
    alias = str(value or "").strip()
    return bool(alias) and alias in WEAK_CHARACTER_ALIASES


def _name_list(value: Any) -> list[str]:
    if isinstance(value, list):
        names = [str(item).strip() for item in value]
    else:
        text = str(value or "")
        for separator in ("、", "，", ",", "\n", ";", "；", "|"):
            text = text.replace(separator, "\n")
        names = [item.strip() for item in text.split("\n")]
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _summary_text(summary: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("summary", "key_events", "ending_state", "character_changes"):
        value = summary.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return "；".join(values)


def _character_card_text(name: str, aliases: list[str], card: dict[str, Any]) -> str:
    lines = [name]
    if aliases:
        lines.append("别名：" + "、".join(aliases))
    for key, label in (
        ("identity", "身份"),
        ("personality", "性格"),
        ("goals", "目标"),
        ("abilities", "能力"),
        ("relationships", "关系"),
        ("current_state", "当前状态"),
    ):
        value = card.get(key)
        if isinstance(value, list):
            rendered = "；".join(str(item) for item in value if item)
        elif isinstance(value, dict):
            rendered = "；".join(f"{item_key}：{item_value}" for item_key, item_value in value.items())
        else:
            rendered = str(value or "")
        if rendered:
            lines.append(f"{label}：{rendered}")
    return "\n".join(lines)


def _chapter_position(chapter_positions: dict[str, int], chapter_id: Any) -> int | None:
    if not chapter_id:
        return None
    return chapter_positions.get(str(chapter_id))


def _select_character_profile(
    profiles: list[dict[str, Any]],
    chapter_positions: dict[str, int] | None = None,
    current_position: int | None = None,
) -> dict[str, Any]:
    enabled_profiles = [profile for profile in profiles if profile.get("enabled", 1)]
    candidates = enabled_profiles or profiles
    if not candidates:
        return {}
    if not chapter_positions or not current_position:
        return candidates[0]

    def profile_score(profile: dict[str, Any]) -> tuple[int, int, int, str]:
        start_position = _chapter_position(chapter_positions, profile.get("start_chapter_id"))
        end_position = _chapter_position(chapter_positions, profile.get("end_chapter_id"))
        starts_before = start_position is None or start_position <= current_position
        ends_after = end_position is None or end_position >= current_position
        covers_current = starts_before and ends_after
        specificity = int(start_position is not None) + int(end_position is not None)
        recency = start_position or 0
        return (
            int(covers_current),
            specificity,
            recency,
            str(profile.get("created_at") or ""),
        )

    return max(candidates, key=profile_score)


def _character_snapshot_text(
    character: dict[str, Any],
    chapter_positions: dict[str, int] | None = None,
    current_position: int | None = None,
) -> str:
    profiles = character.get("profiles", [])
    profile = _select_character_profile(profiles, chapter_positions, current_position)
    text = _character_card_text(
        character["canonical_name"],
        [alias["alias"] for alias in character["aliases"]],
        profile,
    )
    lines = [text]
    facts = [
        f"{fact['field']}：{fact['value']}"
        for fact in character.get("facts", [])
        if fact.get("field") and fact.get("value")
    ]
    if facts:
        lines.append("人物事实：" + "；".join(facts[:12]))
    events = [
        f"{event['event_type']}：{event['value']}"
        for event in character.get("events", [])
        if event.get("event_type") and event.get("value")
    ]
    if events:
        lines.append("近期经历：" + "；".join(events[-8:]))
    return "\n".join(line for line in lines if line)


def _auxiliary_record_text(record: dict[str, Any]) -> str:
    label = {
        "location": "地点",
        "object": "物件",
        "unresolved": "悬念",
    }.get(str(record.get("record_type") or ""), "辅助")
    name = str(record.get("name") or "未命名").strip()
    summary = str(record.get("summary") or "").strip()
    status = str(record.get("status") or "active").strip()
    line = f"- [{label}/{status}] {name}"
    if summary:
        line += f"：{summary}"
    return line


def _relationship_history_text(event: dict[str, Any]) -> str:
    chapter = (
        f"第 {int(event['chapter_position'])} 章"
        if event.get("chapter_position")
        else "无章节"
    )
    delta = float(event.get("strength_delta") or 0)
    delta_text = f"，强度{delta:+.2f}" if delta else ""
    description = str(event.get("description") or "").strip()
    detail = f"：{description}" if description else ""
    return (
        f"- {chapter}，{event.get('source_name') or '?'} -> {event.get('target_name') or '?'}"
        f"：{event.get('relation_type') or 'related'} / {event.get('event_type') or 'event'}"
        f"{delta_text}{detail}"
    )


def _timeline_node_sort_key(node: dict[str, Any]) -> tuple[int, int, str]:
    return (
        TIMELINE_NODE_TYPE_ORDER.get(str(node.get("node_type") or ""), 99),
        int(node.get("position") or 0),
        str(node.get("title") or ""),
    )


def _timeline_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_map = {
        node["id"]: {**node, "children": [], "depth": 0}
        for node in nodes
    }
    roots: list[dict[str, Any]] = []
    for node in node_map.values():
        parent_id = node.get("parent_id")
        parent = node_map.get(parent_id)
        if parent and parent_id != node["id"]:
            parent["children"].append(node)
        else:
            roots.append(node)

    def assign_depth(items: list[dict[str, Any]], depth: int, path: set[str]) -> None:
        items.sort(key=_timeline_node_sort_key)
        for item in items:
            item["depth"] = depth
            item_id = str(item.get("id") or "")
            if item_id in path:
                item["children"] = []
                continue
            assign_depth(item["children"], depth + 1, {*path, item_id})

    if not roots and node_map:
        roots = list(node_map.values())
    assign_depth(roots, 0, set())
    return roots


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
            material_records = self._material_package_records(connection, document_id)

        document_hash = document["raw_text_hash"] or stable_text_hash(document["raw_text"])
        manifest = {
            "format": PACKAGE_FORMAT,
            "format_version": PACKAGE_FORMAT_VERSION,
            "project_id": document["project_id"],
            "document_id": document["id"],
            "source_document_hash": document_hash,
            "chapter_count": len(chapters),
            "chunk_count": len(chunks),
            "material_counts": {
                name: len(records) for name, records in material_records.items()
            },
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
            for name, records in material_records.items():
                package.writestr(
                    name,
                    "".join(_json_line(record) for record in records),
                )
        return buffer.getvalue()

    def export_document_package_report(self, document_id: str) -> dict[str, Any]:
        package_bytes = self.export_document_package(document_id)
        report = self.validate_package(package_bytes, target_document_id=document_id)
        report["export"] = {
            "document_id": document_id,
            "package_bytes": len(package_bytes),
        }
        return report

    def validate_package(
        self,
        package_bytes: bytes,
        *,
        target_document_id: str | None = None,
        chapter_start: int | None = None,
        chapter_end: int | None = None,
    ) -> dict[str, Any]:
        chapter_scope = self._normalise_chapter_scope(chapter_start, chapter_end)
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
            package_chapters = list(self._iter_jsonl(package, "chapters.jsonl"))
            package_chunks = list(self._iter_jsonl(package, "chunks.jsonl"))
            package_provenance = (
                list(self._iter_jsonl(package, "provenance.jsonl")) if chapter_scope else []
            )
            package_material_records = (
                self._read_material_records(package) if target_document_id else {}
            )
            if chapter_scope and package_material_records:
                package_material_records = self._filter_material_records_by_chapter_scope(
                    package_chapters,
                    package_chunks,
                    package_provenance,
                    package_material_records,
                    chapter_scope,
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
        raw_material_counts = manifest.get("material_counts")
        if not isinstance(raw_material_counts, dict):
            raw_material_counts = {}
        material_counts = {
            name: _safe_count(count)
            for name, count in raw_material_counts.items()
        }
        material_layer_counts = {
            layer: sum(material_counts.get(name, 0) for name in files)
            for layer, files in MATERIAL_IMPORT_LAYERS.items()
        }
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
                "material_counts": material_counts,
                "material_layer_counts": material_layer_counts,
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
        if chapter_scope:
            scoped_chapters = self._chapters_in_scope(package_chapters, chapter_scope)
            scoped_material_counts = {
                name: len(records) for name, records in package_material_records.items()
            }
            report["scope"] = {
                "enabled": True,
                "chapter_start": chapter_scope["start"],
                "chapter_end": chapter_scope["end"],
                "matched_chapter_count": len(scoped_chapters),
            }
            report["package"]["scoped_material_counts"] = scoped_material_counts
            report["package"]["scoped_material_layer_counts"] = {
                layer: sum(scoped_material_counts.get(name, 0) for name in files)
                for layer, files in MATERIAL_IMPORT_LAYERS.items()
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
                report["diff_preview"] = self._material_diff_preview(
                    connection,
                    target_document_id,
                    package_chapters,
                    package_chunks,
                    package_material_records,
                )
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
        target_document_id: str | None = None,
        material_layers: list[str] | None = None,
        chapter_start: int | None = None,
        chapter_end: int | None = None,
    ) -> dict[str, Any]:
        chapter_scope = self._normalise_chapter_scope(chapter_start, chapter_end)
        if mode not in {"create_document", "merge", "replace_material"}:
            raise MaterialPackageError("导入模式必须是 create_document、merge 或 replace_material")
        if chapter_scope and mode != "merge":
            raise MaterialPackageError("章节范围过滤暂只支持合并导入")
        report = self.validate_package(
            package_bytes,
            target_document_id=target_document_id,
            chapter_start=chapter_start,
            chapter_end=chapter_end,
        )
        selected_files = self._material_files_for_layers(material_layers)
        if mode in {"merge", "replace_material"}:
            if not target_document_id:
                raise MaterialPackageError("合并或替换导入必须提供目标 document_id")
            if not report["can_import"]:
                raise MaterialPackageError("目标文档与分析包原文不匹配，不能默认导入")
            return self._import_material_records_into_existing(
                package_bytes,
                target_document_id=target_document_id,
                mode=mode,
                report=report,
                selected_files=selected_files,
                chapter_scope=chapter_scope,
            )
        if not report["can_create_new_document"]:
            raise MaterialPackageError("分析包缺少原文，不能创建新文档")
        package = self._open_package(package_bytes)
        with package:
            manifest = self._read_manifest(package)
            document_record = self._read_documents(package)[0]
            chapters = list(self._iter_jsonl(package, "chapters.jsonl"))
            chunks = list(self._iter_jsonl(package, "chunks.jsonl"))
            provenance = list(self._iter_jsonl(package, "provenance.jsonl"))
            material_records = self._read_material_records(package, selected_files=selected_files)

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
            self._import_material_records(connection, document_id, material_records, selected_files=selected_files)
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
            connection.execute(
                "UPDATE conversations SET document_id = ? WHERE project_id = ? AND document_id IS NULL",
                (document_id, project_id),
            )
            connection.commit()
        return {
            "document_id": document_id,
            "material_layers": sorted(
                layer for layer, files in MATERIAL_IMPORT_LAYERS.items()
                if files & selected_files
            ),
            "report": self.validate_package(package_bytes),
        }

    def rebuild_document_material(self, document_id: str) -> dict[str, Any]:
        self.rebuild_semantic_observations(document_id)
        timeline = self.rebuild_timeline(document_id)
        characters = self.seed_character_entities(document_id)
        relationships = self.rebuild_relationships(document_id)
        profile = self.ensure_prompt_budget_profile(document_id)
        return {
            "document_id": document_id,
            "observation_ledger": self.semantic_observation_ledger(document_id),
            "timeline": timeline,
            "characters": characters,
            "relationships": relationships,
            "relationship_network": self.relationship_network(document_id),
            "auxiliary_records": self.list_auxiliary_records(document_id),
            "prompt_budget_profile": profile,
            "review_items": self.list_review_items(document_id),
        }

    def get_material_overview(self, document_id: str) -> dict[str, Any]:
        self._require_document(document_id)
        return {
            "document_id": document_id,
            "observation_ledger": self.semantic_observation_ledger(document_id),
            "timeline": self.get_timeline(document_id),
            "characters": self.list_character_entities(document_id),
            "relationships": self.list_relationships(document_id),
            "relationship_network": self.relationship_network(document_id),
            "auxiliary_records": self.list_auxiliary_records(document_id),
            "review_items": self.list_review_items(document_id),
            "prompt_budget_profile": self.ensure_prompt_budget_profile(document_id),
        }

    def rebuild_semantic_observations(self, document_id: str) -> list[dict[str, Any]]:
        now = utc_now()
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            facts = connection.execute(
                """
                SELECT sf.*, fs.chapter_id AS source_chapter_id, fs.chunk_id AS source_chunk_id
                FROM story_facts sf
                LEFT JOIN fact_sources fs ON fs.fact_id = sf.id
                WHERE sf.document_id = ?
                GROUP BY sf.id
                ORDER BY sf.updated_at
                """,
                (document_id,),
            ).fetchall()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM semantic_observations WHERE document_id = ? AND normalized_key LIKE 'story_fact:%'",
                (document_id,),
            )
            for fact in facts:
                payload = dict(fact)
                observation_type = {
                    "timeline": "plot_event",
                    "relationship": "relationship_event",
                    "location": "location_event",
                    "item": "object_event",
                    "foreshadowing": "unresolved_reference",
                }.get(fact["fact_type"], "plot_event")
                connection.execute(
                    """
                    INSERT INTO semantic_observations
                        (id, document_id, chapter_id, chunk_id, observation_type,
                         payload_json, normalized_key, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id("obs", document_id, fact["id"]),
                        document_id,
                        fact["source_chapter_id"] or fact["first_chapter_id"],
                        fact["source_chunk_id"],
                        observation_type,
                        json.dumps(payload, ensure_ascii=False),
                        f"story_fact:{fact['id']}",
                        float(fact["confidence"] or 0.7),
                        now,
                        now,
                    ),
                )
            connection.commit()
        return self.list_semantic_observations(document_id)

    def save_unified_events(
        self,
        document_id: str,
        chapter_id: str,
        chunk_id: str,
        events: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        now = utc_now()
        type_map = {
            "plot_events": "plot_event",
            "character_events": "character_event",
            "relationship_events": "relationship_event",
            "location_events": "location_event",
            "ability_events": "ability_event",
            "object_events": "object_event",
            "unresolved_entities": "unresolved_reference",
        }
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_document(document_id, connection)
            provenance_id = _stable_id("prov", document_id, "chunk", chunk_id)
            chunk = connection.execute(
                "SELECT content_hash FROM chapter_chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            connection.execute(
                """
                INSERT OR REPLACE INTO material_provenance
                    (id, document_id, source_type, source_id, source_hash,
                     analysis_version, prompt_version, model_id, generated_at, confidence)
                VALUES (?, ?, 'chunk', ?, ?, ?, 'unified-event-extractor-v1', '', ?, 1.0)
                """,
                (
                    provenance_id,
                    document_id,
                    chunk_id,
                    chunk["content_hash"] if chunk else "",
                    GENERATOR_VERSION,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM semantic_observations WHERE document_id = ? AND chunk_id = ? AND normalized_key LIKE 'unified:%'",
                (document_id, chunk_id),
            )
            for group_key, observation_type in type_map.items():
                for index, payload in enumerate(events.get(group_key, []), start=1):
                    observation_id = _stable_id("obs", document_id, chunk_id, group_key, index, stable_json_hash(payload))
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO semantic_observations
                            (id, document_id, chapter_id, chunk_id, observation_type,
                             payload_json, normalized_key, confidence, provenance_id,
                             created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            observation_id,
                            document_id,
                            chapter_id,
                            chunk_id,
                            observation_type,
                            json.dumps(payload, ensure_ascii=False),
                            f"unified:{group_key}:{chunk_id}:{index}",
                            float(payload.get("confidence") or 0.7),
                            provenance_id,
                            now,
                            now,
                        ),
                    )
                    self._project_unified_event(
                        connection,
                        document_id,
                        chapter_id,
                        chunk_id,
                        observation_id,
                        provenance_id,
                        observation_type,
                        payload,
                        index,
                        now,
                    )
            connection.commit()
        return {
            "observations": self.list_semantic_observations(document_id),
            "timeline": self.get_timeline(document_id),
            "relationships": self.list_relationships(document_id),
            "auxiliary_records": self.list_auxiliary_records(document_id),
            "review_items": self.list_review_items(document_id),
        }

    def list_semantic_observations(self, document_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            rows = connection.execute(
                "SELECT * FROM semantic_observations WHERE document_id = ? ORDER BY created_at",
                (document_id,),
            ).fetchall()
        return [
            {**dict(row), "payload": _json_load(row["payload_json"], {})}
            for row in rows
        ]

    def semantic_observation_ledger(
        self,
        document_id: str,
        *,
        limit: int = 40,
        observation_type: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        requested_observation_type = str(observation_type or "").strip()
        requested_status = str(status or "").strip()
        bounded_limit = max(1, min(200, int(limit or 40)))
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            type_rows = connection.execute(
                """
                SELECT observation_type, status, COUNT(*) AS count
                FROM semantic_observations
                WHERE document_id = ?
                GROUP BY observation_type, status
                ORDER BY observation_type, status
                """,
                (document_id,),
            ).fetchall()
            clauses = ["so.document_id = ?"]
            params: list[Any] = [document_id]
            if requested_observation_type:
                clauses.append("so.observation_type = ?")
                params.append(requested_observation_type)
            if requested_status:
                clauses.append("so.status = ?")
                params.append(requested_status)
            where_sql = " AND ".join(clauses)
            rows = connection.execute(
                f"""
                SELECT so.*, c.title AS chapter_title, c.position AS chapter_position
                FROM semantic_observations so
                LEFT JOIN chapters c ON c.id = so.chapter_id
                WHERE {where_sql}
                ORDER BY so.updated_at DESC, so.created_at DESC
                LIMIT ?
                """,
                (*params, bounded_limit),
            ).fetchall()
        type_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        by_type_status: dict[str, dict[str, int]] = {}
        for row in type_rows:
            row_observation_type = row["observation_type"]
            row_status = row["status"]
            count = int(row["count"] or 0)
            type_counts[row_observation_type] = type_counts.get(row_observation_type, 0) + count
            status_counts[row_status] = status_counts.get(row_status, 0) + count
            by_type_status.setdefault(row_observation_type, {})[row_status] = count
        observations = [
            {
                **dict(row),
                "payload": _json_load(row["payload_json"], {}),
            }
            for row in rows
        ]
        return {
            "document_id": document_id,
            "filters": {
                "observation_type": requested_observation_type,
                "status": requested_status,
                "limit": bounded_limit,
            },
            "observation_count": sum(type_counts.values()),
            "filtered_count": len(observations),
            "type_counts": type_counts,
            "status_counts": status_counts,
            "by_type_status": by_type_status,
            "recent_observations": observations,
        }

    def update_semantic_observation(
        self,
        observation_id: str,
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_statuses = {"active", "resolved", "disabled"}
        assignments: list[str] = []
        values: list[Any] = []
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM semantic_observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("semantic_observation_not_found")
            if "status" in changes:
                status = str(changes.get("status") or "").strip() or "active"
                if status not in allowed_statuses:
                    connection.rollback()
                    raise ValueError("语义观察状态只支持 active / resolved / disabled")
                assignments.append("status = ?")
                values.append(status)
            if not assignments:
                connection.rollback()
                raise ValueError("no_semantic_observation_changes")
            assignments.extend(["manually_edited = 1", "updated_at = ?"])
            values.extend([now, observation_id])
            connection.execute(
                f"UPDATE semantic_observations SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            updated = connection.execute(
                "SELECT * FROM semantic_observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
            connection.commit()
        return {**dict(updated), "payload": _json_load(updated["payload_json"], {})}

    def rebuild_timeline(self, document_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            document = self._require_document(document_id, connection)
            chapters = connection.execute(
                "SELECT * FROM chapters WHERE document_id = ? ORDER BY position",
                (document_id,),
            ).fetchall()
            facts = connection.execute(
                """
                SELECT sf.*, fs.chapter_id AS source_chapter_id, fs.chunk_id AS source_chunk_id
                FROM story_facts sf
                LEFT JOIN fact_sources fs ON fs.fact_id = sf.id
                WHERE sf.document_id = ? AND sf.fact_type = 'timeline'
                GROUP BY sf.id
                ORDER BY COALESCE(sf.first_chapter_id, ''), sf.updated_at
                """,
                (document_id,),
            ).fetchall()
            group_size = self._timeline_group_size(len(chapters))
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM timeline_nodes WHERE document_id = ? AND manually_edited = 0",
                (document_id,),
            )
            connection.execute("DELETE FROM timeline_events WHERE document_id = ?", (document_id,))
            root_id = _stable_id("tl", document_id, "project")
            self._upsert_generated_timeline_node(
                connection,
                id=root_id,
                document_id=document_id,
                parent_id=None,
                node_type="project",
                title=document["filename"],
                start_chapter_id=chapters[0]["id"] if chapters else None,
                end_chapter_id=chapters[-1]["id"] if chapters else None,
                position=0,
                summary=document["global_summary"],
                now=now,
            )
            for group_index, start in enumerate(range(0, len(chapters), group_size), start=1):
                group = chapters[start : start + group_size]
                if not group:
                    continue
                group_id = _stable_id("tl", document_id, "group", group[0]["position"], group[-1]["position"])
                group_summary = " / ".join(
                    text for text in (
                        _summary_text(_json_load(chapter["summary_json"], {}))
                        for chapter in group
                    ) if text
                )[:1200]
                self._upsert_generated_timeline_node(
                    connection,
                    id=group_id,
                    document_id=document_id,
                    parent_id=root_id,
                    node_type="chapter_group",
                    title=f"章节组 {group[0]['position']}-{group[-1]['position']}",
                    start_chapter_id=group[0]["id"],
                    end_chapter_id=group[-1]["id"],
                    position=group_index,
                    summary=group_summary,
                    now=now,
                )
                for chapter in group:
                    summary = _summary_text(_json_load(chapter["summary_json"], {}))
                    self._upsert_generated_timeline_node(
                        connection,
                        id=_stable_id("tl", document_id, "chapter", chapter["id"]),
                        document_id=document_id,
                        parent_id=group_id,
                        node_type="chapter",
                        title=chapter["title"],
                        start_chapter_id=chapter["id"],
                        end_chapter_id=chapter["id"],
                        position=chapter["position"],
                        summary=summary,
                        now=now,
                    )
            for sequence, fact in enumerate(facts, start=1):
                title = " ".join(
                    part for part in [fact["subject"], fact["predicate"], fact["object"]]
                    if part
                ) or fact["state"] or "时间线事件"
                connection.execute(
                    """
                    INSERT OR REPLACE INTO timeline_events
                        (id, document_id, event_type, title, description, chapter_id,
                         chunk_id, sequence, participants_json, status, confidence,
                         manually_edited, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        _stable_id("tle", document_id, fact["id"]),
                        document_id,
                        fact["predicate"] or "event",
                        title,
                        fact["state"] or fact["object"] or "",
                        fact["source_chapter_id"] or fact["first_chapter_id"],
                        fact["source_chunk_id"],
                        sequence,
                        json.dumps([fact["subject"]] if fact["subject"] else [], ensure_ascii=False),
                        fact["status"],
                        float(fact["confidence"] or 0.7),
                        now,
                        now,
                    ),
                )
            connection.commit()
        return self.get_timeline(document_id)

    def get_timeline(self, document_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            nodes = connection.execute(
                """
                SELECT * FROM timeline_nodes
                WHERE document_id = ?
                ORDER BY
                    CASE node_type
                        WHEN 'project' THEN 0
                        WHEN 'volume' THEN 1
                        WHEN 'arc' THEN 2
                        WHEN 'stage' THEN 3
                        WHEN 'chapter_group' THEN 4
                        WHEN 'chapter' THEN 5
                        WHEN 'scene' THEN 6
                        ELSE 99
                    END,
                    position,
                    title
                """,
                (document_id,),
            ).fetchall()
            events = connection.execute(
                "SELECT * FROM timeline_events WHERE document_id = ? ORDER BY sequence",
                (document_id,),
            ).fetchall()
        node_records = [dict(row) for row in nodes]
        return {
            "nodes": node_records,
            "tree": _timeline_tree(node_records),
            "events": [
                {
                    **dict(row),
                    "participants": _json_load(row["participants_json"], []),
                    "causes": _json_load(row["causes_json"], []),
                    "consequences": _json_load(row["consequences_json"], []),
                }
                for row in events
            ],
        }

    def _upsert_generated_timeline_node(
        self,
        connection: Any,
        *,
        id: str,
        document_id: str,
        parent_id: str | None,
        node_type: str,
        title: str,
        start_chapter_id: str | None,
        end_chapter_id: str | None,
        position: int,
        summary: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO timeline_nodes
                (id, document_id, parent_id, node_type, title, start_chapter_id,
                 end_chapter_id, position, summary, summary_version, enabled,
                 manually_edited, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document_id = excluded.document_id,
                parent_id = excluded.parent_id,
                node_type = excluded.node_type,
                title = excluded.title,
                start_chapter_id = excluded.start_chapter_id,
                end_chapter_id = excluded.end_chapter_id,
                position = excluded.position,
                summary = excluded.summary,
                summary_version = excluded.summary_version,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            WHERE timeline_nodes.manually_edited = 0
            """,
            (
                id,
                document_id,
                parent_id,
                node_type,
                title,
                start_chapter_id,
                end_chapter_id,
                int(position or 0),
                summary,
                GENERATOR_VERSION,
                now,
                now,
            ),
        )

    def update_timeline_node(self, node_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "title", "summary", "enabled", "parent_id",
            "start_chapter_id", "end_chapter_id", "position",
        }
        assignments: list[str] = []
        values: list[Any] = []
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM timeline_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("timeline_node_not_found")
            document_id = row["document_id"]
            start_chapter_id = row["start_chapter_id"]
            end_chapter_id = row["end_chapter_id"]
            for key, value in changes.items():
                if key not in allowed:
                    continue
                if key == "enabled":
                    value = int(bool(value))
                elif key == "parent_id":
                    value = self._timeline_parent_id_for_document(connection, document_id, value)
                    if value == node_id:
                        connection.rollback()
                        raise ValueError("父时间线节点不能是自己")
                    if value and self._timeline_parent_would_cycle(connection, node_id, value):
                        connection.rollback()
                        raise ValueError("父时间线节点不能是当前节点的子节点")
                elif key == "start_chapter_id":
                    value = self._timeline_chapter_id_for_document(connection, document_id, value)
                    start_chapter_id = value
                elif key == "end_chapter_id":
                    value = self._timeline_chapter_id_for_document(connection, document_id, value)
                    end_chapter_id = value
                elif key == "position":
                    value = max(0, int(value or 0))
                else:
                    value = str(value or "").strip()
                    if key == "title" and not value:
                        connection.rollback()
                        raise ValueError("时间线节点标题不能为空")
                assignments.append(f"{key} = ?")
                values.append(value)
            if not assignments:
                connection.rollback()
                raise ValueError("no_timeline_node_changes")
            self._validate_timeline_chapter_range(connection, document_id, start_chapter_id, end_chapter_id)
            assignments.extend(["manually_edited = 1", "updated_at = ?"])
            values.extend([now, node_id])
            connection.execute(
                f"UPDATE timeline_nodes SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            connection.commit()
        return next(
            item for item in self.get_timeline(document_id)["nodes"]
            if item["id"] == node_id
        )

    def create_timeline_node(self, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title") or "").strip()
        if not title:
            raise ValueError("时间线节点标题不能为空")
        node_type = str(payload.get("node_type") or "stage").strip() or "stage"
        allowed_types = {"volume", "arc", "stage", "chapter_group", "scene"}
        if node_type not in allowed_types:
            raise ValueError("时间线节点类型必须是 volume、arc、stage、chapter_group 或 scene")
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_document(document_id, connection)
            parent_id = self._timeline_parent_id_for_document(
                connection,
                document_id,
                payload.get("parent_id"),
            )
            start_chapter_id = self._timeline_chapter_id_for_document(
                connection,
                document_id,
                payload.get("start_chapter_id"),
            )
            end_chapter_id = self._timeline_chapter_id_for_document(
                connection,
                document_id,
                payload.get("end_chapter_id"),
            )
            self._validate_timeline_chapter_range(connection, document_id, start_chapter_id, end_chapter_id)
            if payload.get("position") is None:
                position = connection.execute(
                    "SELECT COALESCE(MAX(position), 0) + 1 FROM timeline_nodes WHERE document_id = ?",
                    (document_id,),
                ).fetchone()[0]
            else:
                position = max(0, int(payload.get("position") or 0))
            node_id = new_id()
            connection.execute(
                """
                INSERT INTO timeline_nodes
                    (id, document_id, parent_id, node_type, title, start_chapter_id,
                     end_chapter_id, position, summary, summary_version, enabled,
                     manually_edited, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    node_id,
                    document_id,
                    parent_id,
                    node_type,
                    title,
                    start_chapter_id,
                    end_chapter_id,
                    position,
                    str(payload.get("summary") or "").strip(),
                    "manual",
                    int(bool(payload.get("enabled", True))),
                    now,
                    now,
                ),
            )
            connection.commit()
        return next(
            item for item in self.get_timeline(document_id)["nodes"]
            if item["id"] == node_id
        )

    def delete_timeline_node(self, node_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM timeline_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("timeline_node_not_found")
            connection.execute(
                "UPDATE timeline_nodes SET parent_id = ?, updated_at = ? WHERE parent_id = ?",
                (row["parent_id"], now, node_id),
            )
            connection.execute("DELETE FROM timeline_nodes WHERE id = ?", (node_id,))
            connection.commit()
        return {
            "id": node_id,
            "document_id": row["document_id"],
            "deleted": True,
        }

    def _timeline_parent_id_for_document(
        self,
        connection: Any,
        document_id: str,
        parent_id: Any,
    ) -> str | None:
        if not parent_id:
            return None
        row = connection.execute(
            "SELECT id FROM timeline_nodes WHERE id = ? AND document_id = ?",
            (str(parent_id), document_id),
        ).fetchone()
        if row is None:
            raise ValueError("父时间线节点不存在")
        return row["id"]

    def _timeline_parent_would_cycle(
        self,
        connection: Any,
        node_id: str,
        parent_id: str,
    ) -> bool:
        seen: set[str] = set()
        current_id: str | None = parent_id
        while current_id:
            if current_id == node_id:
                return True
            if current_id in seen:
                return True
            seen.add(current_id)
            row = connection.execute(
                "SELECT parent_id FROM timeline_nodes WHERE id = ?",
                (current_id,),
            ).fetchone()
            current_id = row["parent_id"] if row else None
        return False

    def _timeline_chapter_id_for_document(
        self,
        connection: Any,
        document_id: str,
        chapter_id: Any,
    ) -> str | None:
        if not chapter_id:
            return None
        row = connection.execute(
            "SELECT id FROM chapters WHERE id = ? AND document_id = ?",
            (str(chapter_id), document_id),
        ).fetchone()
        if row is None:
            raise ValueError("时间线节点章节范围不属于当前 TXT")
        return row["id"]

    def _validate_timeline_chapter_range(
        self,
        connection: Any,
        document_id: str,
        start_chapter_id: str | None,
        end_chapter_id: str | None,
    ) -> None:
        if not start_chapter_id or not end_chapter_id:
            return
        rows = connection.execute(
            """
            SELECT id, position FROM chapters
            WHERE document_id = ? AND id IN (?, ?)
            """,
            (document_id, start_chapter_id, end_chapter_id),
        ).fetchall()
        positions = {row["id"]: int(row["position"] or 0) for row in rows}
        if positions.get(start_chapter_id, 0) > positions.get(end_chapter_id, 0):
            raise ValueError("时间线节点起始章节不能晚于结束章节")

    def _timeline_chunk_for_document(
        self,
        connection: Any,
        document_id: str,
        chunk_id: Any,
    ) -> tuple[str, str] | None:
        if not chunk_id:
            return None
        row = connection.execute(
            """
            SELECT chapter_chunks.id, chapter_chunks.chapter_id
            FROM chapter_chunks
            JOIN chapters ON chapters.id = chapter_chunks.chapter_id
            WHERE chapter_chunks.id = ? AND chapters.document_id = ?
            """,
            (str(chunk_id), document_id),
        ).fetchone()
        if row is None:
            raise ValueError("时间线事件分片不属于当前 TXT")
        return row["id"], row["chapter_id"]

    def create_timeline_event(self, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title") or "").strip()
        if not title:
            raise ValueError("时间线事件标题不能为空")
        event_type = str(payload.get("event_type") or "event").strip() or "event"
        status = str(payload.get("status") or "active").strip() or "active"
        confidence = max(0, min(1, float(payload.get("confidence", 0.7))))
        participants = [
            str(item).strip()
            for item in payload.get("participants") or []
            if str(item).strip()
        ]
        causes = [
            str(item).strip()
            for item in payload.get("causes") or []
            if str(item).strip()
        ]
        consequences = [
            str(item).strip()
            for item in payload.get("consequences") or []
            if str(item).strip()
        ]
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_document(document_id, connection)
            chapter_id = self._timeline_chapter_id_for_document(
                connection,
                document_id,
                payload.get("chapter_id"),
            )
            chunk = self._timeline_chunk_for_document(
                connection,
                document_id,
                payload.get("chunk_id"),
            )
            chunk_id = None
            if chunk is not None:
                chunk_id, chunk_chapter_id = chunk
                if chapter_id and chapter_id != chunk_chapter_id:
                    raise ValueError("时间线事件分片不属于指定章节")
                chapter_id = chapter_id or chunk_chapter_id
            if payload.get("sequence") is None:
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM timeline_events WHERE document_id = ?",
                    (document_id,),
                ).fetchone()[0]
            else:
                sequence = max(0, int(payload.get("sequence") or 0))
            event_id = new_id()
            connection.execute(
                """
                INSERT INTO timeline_events
                    (id, document_id, event_type, title, description, chapter_id,
                     chunk_id, sequence, participants_json, causes_json,
                     consequences_json, status, confidence, manually_edited,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    event_id,
                    document_id,
                    event_type,
                    title,
                    str(payload.get("description") or "").strip(),
                    chapter_id,
                    chunk_id,
                    sequence,
                    json.dumps(participants, ensure_ascii=False),
                    json.dumps(causes, ensure_ascii=False),
                    json.dumps(consequences, ensure_ascii=False),
                    status,
                    confidence,
                    now,
                    now,
                ),
            )
            connection.commit()
        return next(
            item for item in self.get_timeline(document_id)["events"]
            if item["id"] == event_id
        )

    def delete_timeline_event(self, event_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM timeline_events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("timeline_event_not_found")
            connection.execute("DELETE FROM timeline_events WHERE id = ?", (event_id,))
            connection.commit()
        return {
            "id": event_id,
            "document_id": row["document_id"],
            "deleted": True,
        }

    def update_timeline_event(self, event_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM timeline_events WHERE id = ?", (event_id,)
            ).fetchone()
            if row is None:
                raise KeyError("timeline_event_not_found")
            document_id = row["document_id"]
            chapter_id = row["chapter_id"]
            chunk_id = row["chunk_id"]
            if "chapter_id" in changes:
                next_chapter_id = self._timeline_chapter_id_for_document(
                    connection,
                    document_id,
                    changes.get("chapter_id"),
                )
                if next_chapter_id != chapter_id and "chunk_id" not in changes:
                    chunk_id = None
                chapter_id = next_chapter_id
            if "chunk_id" in changes:
                chunk = self._timeline_chunk_for_document(
                    connection,
                    document_id,
                    changes.get("chunk_id"),
                )
                chunk_id = None
                if chunk is not None:
                    chunk_id, chunk_chapter_id = chunk
                    if chapter_id and chapter_id != chunk_chapter_id:
                        raise ValueError("时间线事件分片不属于指定章节")
                    chapter_id = chapter_id or chunk_chapter_id
            assignments: list[str] = []
            values: list[Any] = []
            for key, value in changes.items():
                if key in {"title", "description", "event_type", "status"}:
                    assignments.append(f"{key} = ?")
                    values.append(str(value or "").strip())
                elif key == "confidence":
                    assignments.append("confidence = ?")
                    values.append(max(0, min(1, float(value))))
                elif key == "sequence":
                    assignments.append("sequence = ?")
                    values.append(max(0, int(value or 0)))
                elif key == "participants":
                    participants = [
                        str(item).strip()
                        for item in value or []
                        if str(item).strip()
                    ]
                    assignments.append("participants_json = ?")
                    values.append(json.dumps(participants, ensure_ascii=False))
                elif key == "causes":
                    causes = [
                        str(item).strip()
                        for item in value or []
                        if str(item).strip()
                    ]
                    assignments.append("causes_json = ?")
                    values.append(json.dumps(causes, ensure_ascii=False))
                elif key == "consequences":
                    consequences = [
                        str(item).strip()
                        for item in value or []
                        if str(item).strip()
                    ]
                    assignments.append("consequences_json = ?")
                    values.append(json.dumps(consequences, ensure_ascii=False))
            if "chapter_id" in changes:
                assignments.append("chapter_id = ?")
                values.append(chapter_id)
            if "chunk_id" in changes or ("chapter_id" in changes and chunk_id is None):
                assignments.append("chunk_id = ?")
                values.append(chunk_id)
            if not assignments:
                raise ValueError("no_timeline_event_changes")
            assignments.extend(["manually_edited = 1", "updated_at = ?"])
            values.extend([utc_now(), event_id])
            cursor = connection.execute(
                f"UPDATE timeline_events SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        if cursor.rowcount == 0:
            raise KeyError("timeline_event_not_found")
        return next(
            item for item in self.get_timeline(document_id)["events"]
            if item["id"] == event_id
        )

    def seed_character_entities(self, document_id: str) -> list[dict[str, Any]]:
        now = utc_now()
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            cards = connection.execute(
                "SELECT * FROM document_characters WHERE document_id = ? ORDER BY name",
                (document_id,),
            ).fetchall()
            card_names = {row["name"].strip() for row in cards if row["name"].strip()}
            connection.execute("BEGIN IMMEDIATE")
            for card_row in cards:
                name = card_row["name"].strip()
                if not name:
                    continue
                aliases = [
                    str(alias).strip()
                    for alias in _json_load(card_row["aliases_json"], [])
                    if str(alias).strip()
                ]
                card = _json_load(card_row["card_json"], {})
                entity_id = _stable_id("char", document_id, name)
                connection.execute(
                    """
                    INSERT INTO character_entities
                        (id, document_id, canonical_name, entity_type, enabled,
                         manually_confirmed, created_at, updated_at)
                    VALUES (?, ?, ?, 'person', ?, 1, ?, ?)
                    ON CONFLICT(document_id, canonical_name) DO UPDATE SET
                        enabled = excluded.enabled,
                        updated_at = excluded.updated_at
                    """,
                    (entity_id, document_id, name, int(card_row["enabled"]), now, now),
                )
                for alias in aliases:
                    if alias == name:
                        continue
                    if alias in card_names:
                        self._write_review_item(
                            connection,
                            document_id,
                            "character_merge_candidate",
                            f"人物合并待确认：{alias} -> {name}",
                            {
                                "source_character_id": _stable_id("char", document_id, alias),
                                "target_character_id": entity_id,
                                "source": alias,
                                "target": name,
                                "alias": alias,
                                "reason": "人物名同时作为另一人物别名出现",
                            },
                            now,
                        )
                        continue
                    if _is_weak_character_alias(alias):
                        self._write_review_item(
                            connection,
                            document_id,
                            "character_alias_pending",
                            f"别名待确认：{name} / {alias}",
                            {
                                "character_id": entity_id,
                                "character": name,
                                "alias": alias,
                                "alias_type": "weak_candidate",
                                "reason": "弱称谓不自动作为强别名入库",
                            },
                            now,
                        )
                        continue
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO character_aliases
                            (id, character_id, alias, alias_type, confidence,
                             manually_confirmed, created_at, updated_at)
                        VALUES (?, ?, ?, 'name', 0.8, 1, ?, ?)
                        """,
                        (_stable_id("alias", entity_id, alias), entity_id, alias, now, now),
                    )
                profile_id = _stable_id("profile", entity_id, "current")
                connection.execute(
                    """
                    INSERT INTO character_profiles
                        (id, character_id, title, identity, personality, goals,
                         behavior_pattern, ability_stage, social_status, enabled,
                         manually_edited, created_at, updated_at)
                    VALUES (?, ?, '当前档案', ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        identity = excluded.identity,
                        personality = excluded.personality,
                        goals = excluded.goals,
                        behavior_pattern = excluded.behavior_pattern,
                        ability_stage = excluded.ability_stage,
                        social_status = excluded.social_status,
                        enabled = excluded.enabled,
                        updated_at = excluded.updated_at
                    """,
                    (
                        profile_id,
                        entity_id,
                        str(card.get("identity") or ""),
                        str(card.get("personality") or ""),
                        str(card.get("goals") or ""),
                        str(card.get("current_state") or ""),
                        str(card.get("abilities") or ""),
                        str(card.get("social_status") or ""),
                        int(card_row["enabled"]),
                        now,
                        now,
                    ),
                )
            connection.commit()
        return self.list_character_entities(document_id)

    def list_character_entities(self, document_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            rows = connection.execute(
                "SELECT * FROM character_entities WHERE document_id = ? ORDER BY canonical_name",
                (document_id,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                aliases = connection.execute(
                    "SELECT * FROM character_aliases WHERE character_id = ? ORDER BY alias",
                    (row["id"],),
                ).fetchall()
                profiles = connection.execute(
                    "SELECT * FROM character_profiles WHERE character_id = ? ORDER BY created_at",
                    (row["id"],),
                ).fetchall()
                facts = connection.execute(
                    "SELECT * FROM character_facts WHERE character_id = ? ORDER BY field, created_at",
                    (row["id"],),
                ).fetchall()
                events = connection.execute(
                    "SELECT * FROM character_events WHERE character_id = ? ORDER BY sequence, created_at",
                    (row["id"],),
                ).fetchall()
                result.append({
                    **dict(row),
                    "enabled": bool(row["enabled"]),
                    "manually_confirmed": bool(row["manually_confirmed"]),
                    "aliases": [dict(alias) for alias in aliases],
                    "profiles": [dict(profile) for profile in profiles],
                    "facts": [dict(fact) for fact in facts],
                    "events": [dict(event) for event in events],
                })
        return result

    def create_character_entity(self, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        canonical_name = str(payload.get("canonical_name") or "").strip()
        if not canonical_name:
            raise ValueError("人物名称不能为空")
        entity_type = str(payload.get("entity_type") or "person").strip()[:50] or "person"
        aliases = [
            alias for alias in _name_list(payload.get("aliases") or [])
            if alias != canonical_name
        ]
        profile_payload = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
        allowed_profile = {
            "title", "identity", "personality", "goals", "behavior_pattern",
            "ability_stage", "social_status",
        }
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_document(document_id, connection)
            duplicate = connection.execute(
                "SELECT id FROM character_entities WHERE document_id = ? AND canonical_name = ?",
                (document_id, canonical_name),
            ).fetchone()
            if duplicate:
                connection.rollback()
                raise ValueError("人物名称已存在")
            character_id = new_id()
            connection.execute(
                """
                INSERT INTO character_entities
                    (id, document_id, canonical_name, entity_type, enabled,
                     manually_confirmed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    character_id,
                    document_id,
                    canonical_name,
                    entity_type,
                    int(bool(payload.get("enabled", True))),
                    now,
                    now,
                ),
            )
            for alias in aliases:
                owner = connection.execute(
                    """
                    SELECT ca.character_id
                    FROM character_aliases ca
                    JOIN character_entities ce ON ce.id = ca.character_id
                    WHERE ce.document_id = ? AND ca.alias = ?
                    """,
                    (document_id, alias),
                ).fetchone()
                if owner:
                    connection.rollback()
                    raise ValueError("别名已属于另一个人物；请使用人物合并")
                connection.execute(
                    """
                    INSERT INTO character_aliases
                        (id, character_id, alias, alias_type, confidence,
                         manually_confirmed, created_at, updated_at)
                    VALUES (?, ?, ?, 'name', 1.0, 1, ?, ?)
                    """,
                    (_stable_id("alias", character_id, alias), character_id, alias, now, now),
                )
            profile_id = _stable_id("charprofile", character_id, "manual")
            profile_values = {
                key: str(profile_payload.get(key) or "").strip()
                for key in allowed_profile
            }
            connection.execute(
                """
                INSERT INTO character_profiles
                    (id, character_id, title, identity, personality, goals,
                     behavior_pattern, ability_stage, social_status, enabled,
                     manually_edited, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                """,
                (
                    profile_id,
                    character_id,
                    profile_values["title"] or "人工档案",
                    profile_values["identity"],
                    profile_values["personality"],
                    profile_values["goals"],
                    profile_values["behavior_pattern"],
                    profile_values["ability_stage"],
                    profile_values["social_status"],
                    now,
                    now,
                ),
            )
            connection.commit()
        return next(
            item for item in self.list_character_entities(document_id)
            if item["id"] == character_id
        )

    def delete_character_entity(self, character_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (character_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("character_entity_not_found")
            document_id = row["document_id"]
            relationship_count = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM character_relationships
                     WHERE source_character_id = ? OR target_character_id = ?)
                    +
                    (SELECT COUNT(*) FROM relationship_events
                     WHERE source_character_id = ? OR target_character_id = ?)
                """,
                (character_id, character_id, character_id, character_id),
            ).fetchone()[0]
            if relationship_count:
                connection.rollback()
                raise ValueError("人物仍被关系引用；请先合并人物或清理关系")
            connection.execute("DELETE FROM character_entities WHERE id = ?", (character_id,))
            connection.commit()
        return {
            "id": character_id,
            "document_id": document_id,
            "deleted": True,
        }

    def character_entity_dependencies(self, character_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (character_id,),
            ).fetchone()
            if row is None:
                raise KeyError("character_entity_not_found")
            document_id = row["document_id"]
            relationships = connection.execute(
                """
                SELECT cr.*, source.canonical_name AS source_name,
                       target.canonical_name AS target_name
                FROM character_relationships cr
                JOIN character_entities source ON source.id = cr.source_character_id
                JOIN character_entities target ON target.id = cr.target_character_id
                WHERE cr.document_id = ?
                  AND (cr.source_character_id = ? OR cr.target_character_id = ?)
                ORDER BY source.canonical_name, target.canonical_name, cr.relation_type
                """,
                (document_id, character_id, character_id),
            ).fetchall()
            relationship_events = connection.execute(
                """
                SELECT re.*, source.canonical_name AS source_name,
                       target.canonical_name AS target_name
                FROM relationship_events re
                LEFT JOIN character_entities source ON source.id = re.source_character_id
                LEFT JOIN character_entities target ON target.id = re.target_character_id
                WHERE re.document_id = ?
                  AND (re.source_character_id = ? OR re.target_character_id = ?)
                ORDER BY re.sequence, re.created_at
                """,
                (document_id, character_id, character_id),
            ).fetchall()
        return {
            "character_id": character_id,
            "document_id": document_id,
            "canonical_name": row["canonical_name"],
            "relationship_count": len(relationships),
            "relationship_event_count": len(relationship_events),
            "can_delete": not relationships and not relationship_events,
            "relationships": [dict(item) for item in relationships],
            "relationship_events": [dict(item) for item in relationship_events],
        }

    def _chapter_context_for_document(
        self,
        connection: Any,
        document_id: str,
        *,
        chapter_id: str | None = None,
        chapter_position: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, int], int]:
        chapter_rows = connection.execute(
            "SELECT id, title, position FROM chapters WHERE document_id = ? ORDER BY position",
            (document_id,),
        ).fetchall()
        chapter_positions = {
            row["id"]: int(row["position"] or 0)
            for row in chapter_rows
        }
        if chapter_id:
            chapter = next((dict(row) for row in chapter_rows if row["id"] == chapter_id), None)
            if chapter is None:
                raise ValueError("快照章节不属于当前 TXT")
            return chapter, chapter_positions, int(chapter["position"] or 0)
        if chapter_position is not None:
            chapter = next(
                (dict(row) for row in chapter_rows if int(row["position"] or 0) == int(chapter_position)),
                None,
            )
            if chapter is None:
                raise ValueError("快照章节不存在")
            return chapter, chapter_positions, int(chapter["position"] or 0)
        return None, chapter_positions, max(chapter_positions.values() or [0])

    def character_snapshot(
        self,
        character_id: str,
        *,
        chapter_id: str | None = None,
        chapter_position: int | None = None,
    ) -> dict[str, Any]:
        with self.database.connect() as connection:
            character_row = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (character_id,),
            ).fetchone()
            if character_row is None:
                raise KeyError("character_entity_not_found")
            document_id = character_row["document_id"]
            chapter, chapter_positions, current_position = self._chapter_context_for_document(
                connection,
                document_id,
                chapter_id=chapter_id,
                chapter_position=chapter_position,
            )
            aliases = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM character_aliases WHERE character_id = ? ORDER BY alias",
                    (character_id,),
                ).fetchall()
            ]
            profiles = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM character_profiles WHERE character_id = ? ORDER BY created_at",
                    (character_id,),
                ).fetchall()
            ]
            facts = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM character_facts WHERE character_id = ? ORDER BY field, created_at",
                    (character_id,),
                ).fetchall()
            ]
            events = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM character_events WHERE character_id = ? ORDER BY sequence, created_at",
                    (character_id,),
                ).fetchall()
            ]
            relationships = [
                dict(row) for row in connection.execute(
                    """
                    SELECT cr.*, source.canonical_name AS source_name,
                           target.canonical_name AS target_name
                    FROM character_relationships cr
                    JOIN character_entities source ON source.id = cr.source_character_id
                    JOIN character_entities target ON target.id = cr.target_character_id
                    WHERE cr.document_id = ?
                      AND (cr.source_character_id = ? OR cr.target_character_id = ?)
                    ORDER BY source.canonical_name, target.canonical_name, cr.relation_type
                    """,
                    (document_id, character_id, character_id),
                ).fetchall()
            ]
        selected_profile = _select_character_profile(profiles, chapter_positions, current_position)

        def fact_active(fact: dict[str, Any]) -> bool:
            start = _chapter_position(chapter_positions, fact.get("valid_from_chapter_id"))
            end = _chapter_position(chapter_positions, fact.get("valid_to_chapter_id"))
            return (start is None or start <= current_position) and (end is None or end >= current_position)

        def event_active(event: dict[str, Any]) -> bool:
            event_position = _chapter_position(chapter_positions, event.get("chapter_id"))
            return event_position is None or event_position <= current_position

        active_facts = [fact for fact in facts if fact_active(fact)]
        active_events = [event for event in events if event_active(event)]
        character = {
            **dict(character_row),
            "enabled": bool(character_row["enabled"]),
            "manually_confirmed": bool(character_row["manually_confirmed"]),
            "aliases": aliases,
            "profiles": [selected_profile] if selected_profile else [],
            "facts": active_facts,
            "events": active_events,
        }
        return {
            "document_id": document_id,
            "chapter": chapter,
            "current_position": current_position,
            "character": {
                **dict(character_row),
                "enabled": bool(character_row["enabled"]),
                "manually_confirmed": bool(character_row["manually_confirmed"]),
                "aliases": aliases,
            },
            "selected_profile": selected_profile,
            "facts": active_facts,
            "events": active_events[-8:],
            "relationships": relationships,
            "text": _character_snapshot_text(character, chapter_positions, current_position),
        }

    def _profile_chapter_id_for_document(
        self,
        connection: Any,
        document_id: str,
        chapter_id: Any,
    ) -> str | None:
        if not chapter_id:
            return None
        row = connection.execute(
            "SELECT id FROM chapters WHERE id = ? AND document_id = ?",
            (str(chapter_id), document_id),
        ).fetchone()
        if row is None:
            raise ValueError("阶段档案章节范围不属于当前 TXT")
        return row["id"]

    def _validate_profile_chapter_range(
        self,
        connection: Any,
        document_id: str,
        start_chapter_id: str | None,
        end_chapter_id: str | None,
    ) -> None:
        if not start_chapter_id or not end_chapter_id:
            return
        rows = connection.execute(
            """
            SELECT id, position FROM chapters
            WHERE document_id = ? AND id IN (?, ?)
            """,
            (document_id, start_chapter_id, end_chapter_id),
        ).fetchall()
        positions = {row["id"]: int(row["position"] or 0) for row in rows}
        if positions.get(start_chapter_id, 0) > positions.get(end_chapter_id, 0):
            raise ValueError("阶段档案起始章节不能晚于结束章节")

    def _profile_payload_values(
        self,
        connection: Any,
        document_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        text_fields = {
            "title", "identity", "personality", "goals", "behavior_pattern",
            "ability_stage", "social_status",
        }
        for field in text_fields:
            if field in payload:
                values[field] = str(payload.get(field) or "").strip()
        if "title" in values and not values["title"]:
            raise ValueError("阶段档案标题不能为空")
        if "enabled" in payload:
            values["enabled"] = int(bool(payload["enabled"]))
        if "start_chapter_id" in payload:
            values["start_chapter_id"] = self._profile_chapter_id_for_document(
                connection,
                document_id,
                payload.get("start_chapter_id"),
            )
        if "end_chapter_id" in payload:
            values["end_chapter_id"] = self._profile_chapter_id_for_document(
                connection,
                document_id,
                payload.get("end_chapter_id"),
            )
        return values

    def _get_character_profile(self, connection: Any, profile_id: str) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT cp.*, ce.document_id
            FROM character_profiles cp
            JOIN character_entities ce ON ce.id = cp.character_id
            WHERE cp.id = ?
            """,
            (profile_id,),
        ).fetchone()
        if row is None:
            raise KeyError("character_profile_not_found")
        return dict(row)

    def create_character_profile(self, character_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            character = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (character_id,),
            ).fetchone()
            if character is None:
                connection.rollback()
                raise KeyError("character_entity_not_found")
            document_id = character["document_id"]
            values = self._profile_payload_values(connection, document_id, payload)
            self._validate_profile_chapter_range(
                connection,
                document_id,
                values.get("start_chapter_id"),
                values.get("end_chapter_id"),
            )
            title = values.get("title") or "阶段档案"
            profile_id = new_id()
            connection.execute(
                """
                INSERT INTO character_profiles
                    (id, character_id, title, start_chapter_id, end_chapter_id,
                     identity, personality, goals, behavior_pattern, ability_stage,
                     social_status, enabled, manually_edited, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    profile_id,
                    character_id,
                    title,
                    values.get("start_chapter_id"),
                    values.get("end_chapter_id"),
                    values.get("identity", ""),
                    values.get("personality", ""),
                    values.get("goals", ""),
                    values.get("behavior_pattern", ""),
                    values.get("ability_stage", ""),
                    values.get("social_status", ""),
                    values.get("enabled", 1),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
                (now, character_id),
            )
            profile = self._get_character_profile(connection, profile_id)
            connection.commit()
        return profile

    def update_character_profile(self, profile_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            profile = self._get_character_profile(connection, profile_id)
            values = self._profile_payload_values(connection, profile["document_id"], changes)
            if not values:
                connection.rollback()
                raise ValueError("no_character_profile_changes")
            start_chapter_id = values.get("start_chapter_id", profile["start_chapter_id"])
            end_chapter_id = values.get("end_chapter_id", profile["end_chapter_id"])
            self._validate_profile_chapter_range(
                connection,
                profile["document_id"],
                start_chapter_id,
                end_chapter_id,
            )
            assignments = [f"{field} = ?" for field in values]
            parameters = list(values.values())
            assignments.extend(["manually_edited = 1", "updated_at = ?"])
            parameters.extend([now, profile_id])
            connection.execute(
                f"UPDATE character_profiles SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            updated = self._get_character_profile(connection, profile_id)
            connection.commit()
        return updated

    def delete_character_profile(self, profile_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            profile = self._get_character_profile(connection, profile_id)
            connection.execute("DELETE FROM character_profiles WHERE id = ?", (profile_id,))
            connection.commit()
        return {
            "id": profile_id,
            "character_id": profile["character_id"],
            "document_id": profile["document_id"],
            "deleted": True,
        }

    def _character_event_payload_values(
        self,
        connection: Any,
        document_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if "event_type" in payload:
            event_type = str(payload.get("event_type") or "").strip()
            if not event_type:
                raise ValueError("人物经历类型不能为空")
            values["event_type"] = event_type
        if "value" in payload:
            value = str(payload.get("value") or "").strip()
            if not value:
                raise ValueError("人物经历内容不能为空")
            values["value"] = value
        if "sequence" in payload:
            values["sequence"] = max(0, int(payload.get("sequence") or 0))
        if "chapter_id" in payload:
            values["chapter_id"] = self._profile_chapter_id_for_document(
                connection,
                document_id,
                payload.get("chapter_id"),
            )
        if "chunk_id" in payload:
            chunk = self._timeline_chunk_for_document(
                connection,
                document_id,
                payload.get("chunk_id"),
            )
            values["chunk_id"] = chunk[0] if chunk else None
            if chunk and values.get("chapter_id") and values["chapter_id"] != chunk[1]:
                raise ValueError("人物经历分片不属于指定章节")
            if chunk and not values.get("chapter_id"):
                values["chapter_id"] = chunk[1]
        return values

    def _get_character_event(self, connection: Any, event_id: str) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT ce.*, entity.document_id
            FROM character_events ce
            JOIN character_entities entity ON entity.id = ce.character_id
            WHERE ce.id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError("character_event_not_found")
        return dict(row)

    def create_character_event(self, character_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            character = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (character_id,),
            ).fetchone()
            if character is None:
                connection.rollback()
                raise KeyError("character_entity_not_found")
            document_id = character["document_id"]
            values = self._character_event_payload_values(connection, document_id, payload)
            event_type = values.get("event_type") or "event"
            value = values.get("value")
            if not value:
                connection.rollback()
                raise ValueError("人物经历内容不能为空")
            if "sequence" in values:
                sequence = values["sequence"]
            else:
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM character_events WHERE character_id = ?",
                    (character_id,),
                ).fetchone()[0]
            event_id = new_id()
            connection.execute(
                """
                INSERT INTO character_events
                    (id, character_id, event_type, value, chapter_id, chunk_id,
                     sequence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    character_id,
                    event_type,
                    value,
                    values.get("chapter_id"),
                    values.get("chunk_id"),
                    sequence,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
                (now, character_id),
            )
            event = self._get_character_event(connection, event_id)
            connection.commit()
        return event

    def update_character_event(self, event_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._get_character_event(connection, event_id)
            values = self._character_event_payload_values(
                connection,
                existing["document_id"],
                changes,
            )
            if (
                "chapter_id" in values
                and values["chapter_id"] != existing["chapter_id"]
                and "chunk_id" not in values
            ):
                values["chunk_id"] = None
            if not values:
                connection.rollback()
                raise ValueError("no_character_event_changes")
            assignments = [f"{field} = ?" for field in values]
            parameters = list(values.values())
            assignments.append("updated_at = ?")
            parameters.extend([now, event_id])
            connection.execute(
                f"UPDATE character_events SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            updated = self._get_character_event(connection, event_id)
            connection.commit()
        return updated

    def delete_character_event(self, event_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            event = self._get_character_event(connection, event_id)
            connection.execute("DELETE FROM character_events WHERE id = ?", (event_id,))
            connection.commit()
        return {
            "id": event_id,
            "character_id": event["character_id"],
            "document_id": event["document_id"],
            "deleted": True,
        }

    def _character_fact_payload_values(
        self,
        connection: Any,
        document_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if "field" in payload:
            field = str(payload.get("field") or "").strip()
            if not field:
                raise ValueError("人物事实字段不能为空")
            values["field"] = field
        if "value" in payload:
            value = str(payload.get("value") or "").strip()
            if not value:
                raise ValueError("人物事实内容不能为空")
            values["value"] = value
        if "certainty" in payload:
            values["certainty"] = max(0, min(1, float(payload.get("certainty", 1.0))))
        if "valid_from_chapter_id" in payload:
            values["valid_from_chapter_id"] = self._profile_chapter_id_for_document(
                connection,
                document_id,
                payload.get("valid_from_chapter_id"),
            )
        if "valid_to_chapter_id" in payload:
            values["valid_to_chapter_id"] = self._profile_chapter_id_for_document(
                connection,
                document_id,
                payload.get("valid_to_chapter_id"),
            )
        return values

    def _get_character_fact(self, connection: Any, fact_id: str) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT cf.*, entity.document_id
            FROM character_facts cf
            JOIN character_entities entity ON entity.id = cf.character_id
            WHERE cf.id = ?
            """,
            (fact_id,),
        ).fetchone()
        if row is None:
            raise KeyError("character_fact_not_found")
        return dict(row)

    def _character_fact_ranges_overlap(
        self,
        chapter_positions: dict[str, int],
        left_start: str | None,
        left_end: str | None,
        right_start: str | None,
        right_end: str | None,
    ) -> bool:
        left_start_position = chapter_positions.get(left_start or "", -1_000_000)
        left_end_position = chapter_positions.get(left_end or "", 1_000_000)
        right_start_position = chapter_positions.get(right_start or "", -1_000_000)
        right_end_position = chapter_positions.get(right_end or "", 1_000_000)
        return left_start_position <= right_end_position and right_start_position <= left_end_position

    def _write_character_fact_conflict_review_item(
        self,
        connection: Any,
        fact: dict[str, Any],
        now: str,
    ) -> None:
        normalized_value = _normalized_fact_value(fact.get("value"))
        if not fact.get("field") or not normalized_value:
            return
        chapter_rows = connection.execute(
            "SELECT id, position FROM chapters WHERE document_id = ?",
            (fact["document_id"],),
        ).fetchall()
        chapter_positions = {row["id"]: int(row["position"]) for row in chapter_rows}
        rows = connection.execute(
            """
            SELECT cf.*, ce.canonical_name
            FROM character_facts cf
            JOIN character_entities ce ON ce.id = cf.character_id
            WHERE cf.character_id = ? AND cf.id != ? AND cf.field = ?
            ORDER BY cf.created_at
            """,
            (fact["character_id"], fact["id"], fact["field"]),
        ).fetchall()
        conflicts: list[dict[str, Any]] = []
        character_name = ""
        for row in rows:
            character_name = row["canonical_name"] or character_name
            if _normalized_fact_value(row["value"]) == normalized_value:
                continue
            if not self._character_fact_ranges_overlap(
                chapter_positions,
                fact.get("valid_from_chapter_id"),
                fact.get("valid_to_chapter_id"),
                row["valid_from_chapter_id"],
                row["valid_to_chapter_id"],
            ):
                continue
            conflicts.append(
                {
                    "fact_id": row["id"],
                    "value": row["value"],
                    "valid_from_chapter_id": row["valid_from_chapter_id"],
                    "valid_to_chapter_id": row["valid_to_chapter_id"],
                    "certainty": row["certainty"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        if not conflicts:
            return
        if not character_name:
            character = connection.execute(
                "SELECT canonical_name FROM character_entities WHERE id = ?",
                (fact["character_id"],),
            ).fetchone()
            character_name = character["canonical_name"] if character else ""
        self._write_review_item(
            connection,
            fact["document_id"],
            "character_fact_conflict",
            f"人物事实冲突：{character_name or '未知人物'} / {fact['field']}",
            {
                "character_id": fact["character_id"],
                "character": character_name,
                "field": fact["field"],
                "incoming_fact_id": fact["id"],
                "incoming_value": fact["value"],
                "incoming_valid_from_chapter_id": fact.get("valid_from_chapter_id"),
                "incoming_valid_to_chapter_id": fact.get("valid_to_chapter_id"),
                "incoming_certainty": fact.get("certainty"),
                "conflicts": conflicts,
            },
            now,
        )

    def create_character_fact(self, character_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            character = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (character_id,),
            ).fetchone()
            if character is None:
                connection.rollback()
                raise KeyError("character_entity_not_found")
            document_id = character["document_id"]
            values = self._character_fact_payload_values(connection, document_id, payload)
            field = values.get("field")
            value = values.get("value")
            if not field:
                connection.rollback()
                raise ValueError("人物事实字段不能为空")
            if not value:
                connection.rollback()
                raise ValueError("人物事实内容不能为空")
            fact_id = new_id()
            connection.execute(
                """
                INSERT INTO character_facts
                    (id, character_id, field, value, valid_from_chapter_id,
                     valid_to_chapter_id, certainty, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    character_id,
                    field,
                    value,
                    values.get("valid_from_chapter_id"),
                    values.get("valid_to_chapter_id"),
                    values.get("certainty", 1.0),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
                (now, character_id),
            )
            fact = self._get_character_fact(connection, fact_id)
            self._write_character_fact_conflict_review_item(connection, fact, now)
            connection.commit()
        return fact

    def update_character_fact(self, fact_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._get_character_fact(connection, fact_id)
            values = self._character_fact_payload_values(
                connection,
                existing["document_id"],
                changes,
            )
            if not values:
                connection.rollback()
                raise ValueError("no_character_fact_changes")
            assignments = [f"{field} = ?" for field in values]
            parameters = list(values.values())
            assignments.append("updated_at = ?")
            parameters.extend([now, fact_id])
            connection.execute(
                f"UPDATE character_facts SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            updated = self._get_character_fact(connection, fact_id)
            self._write_character_fact_conflict_review_item(connection, updated, now)
            connection.commit()
        return updated

    def delete_character_fact(self, fact_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            fact = self._get_character_fact(connection, fact_id)
            connection.execute("DELETE FROM character_facts WHERE id = ?", (fact_id,))
            connection.commit()
        return {
            "id": fact_id,
            "character_id": fact["character_id"],
            "document_id": fact["document_id"],
            "deleted": True,
        }

    def update_character_entity(self, character_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        entity_assignments: list[str] = []
        entity_values: list[Any] = []
        profile_changes = changes.get("profile") if isinstance(changes.get("profile"), dict) else {}
        allowed_profile = {
            "title", "identity", "personality", "goals", "behavior_pattern",
            "ability_stage", "social_status",
        }
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?", (character_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("character_entity_not_found")
            document_id = row["document_id"]
            if "canonical_name" in changes:
                canonical_name = str(changes.get("canonical_name") or "").strip()
                if not canonical_name:
                    connection.rollback()
                    raise ValueError("人物名称不能为空")
                duplicate = connection.execute(
                    """
                    SELECT id FROM character_entities
                    WHERE document_id = ? AND canonical_name = ? AND id != ?
                    """,
                    (document_id, canonical_name, character_id),
                ).fetchone()
                if duplicate:
                    connection.rollback()
                    raise ValueError("人物名称已存在")
                entity_assignments.append("canonical_name = ?")
                entity_values.append(canonical_name)
            if "enabled" in changes:
                entity_assignments.append("enabled = ?")
                entity_values.append(int(bool(changes["enabled"])))
            if "manually_confirmed" in changes:
                entity_assignments.append("manually_confirmed = ?")
                entity_values.append(int(bool(changes["manually_confirmed"])))
            if entity_assignments:
                entity_assignments.append("updated_at = ?")
                entity_values.extend([now, character_id])
                connection.execute(
                    f"UPDATE character_entities SET {', '.join(entity_assignments)} WHERE id = ?",
                    entity_values,
                )
            if profile_changes:
                profile_row = connection.execute(
                    """
                    SELECT * FROM character_profiles
                    WHERE character_id = ? ORDER BY created_at LIMIT 1
                    """,
                    (character_id,),
                ).fetchone()
                profile_id = profile_row["id"] if profile_row else _stable_id("charprofile", character_id, "manual")
                if profile_row is None:
                    connection.execute(
                        """
                        INSERT INTO character_profiles
                            (id, character_id, title, enabled, manually_edited,
                             created_at, updated_at)
                        VALUES (?, ?, '人工编辑档案', 1, 1, ?, ?)
                        """,
                        (profile_id, character_id, now, now),
                    )
                profile_assignments: list[str] = []
                profile_values: list[Any] = []
                for key, value in profile_changes.items():
                    if key not in allowed_profile:
                        continue
                    profile_assignments.append(f"{key} = ?")
                    profile_values.append(str(value or "").strip())
                if profile_assignments:
                    profile_assignments.extend(["manually_edited = 1", "updated_at = ?"])
                    profile_values.extend([now, profile_id])
                    connection.execute(
                        f"UPDATE character_profiles SET {', '.join(profile_assignments)} WHERE id = ?",
                        profile_values,
                    )
            connection.commit()
        return next(
            item for item in self.list_character_entities(document_id)
            if item["id"] == character_id
        )

    def add_character_alias(
        self,
        character_id: str,
        alias: str,
        *,
        alias_type: str = "name",
    ) -> dict[str, Any]:
        alias = alias.strip()
        alias_type = (alias_type or "name").strip()[:50] or "name"
        if not alias:
            raise ValueError("别名不能为空")
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?", (character_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("character_entity_not_found")
            document_id = row["document_id"]
            if alias == row["canonical_name"]:
                connection.commit()
                return next(
                    item for item in self.list_character_entities(document_id)
                    if item["id"] == character_id
                )
            existing = connection.execute(
                """
                SELECT ca.character_id
                FROM character_aliases ca
                JOIN character_entities ce ON ce.id = ca.character_id
                WHERE ce.document_id = ? AND ca.alias = ?
                """,
                (document_id, alias),
            ).fetchone()
            if existing and existing["character_id"] != character_id:
                connection.rollback()
                raise ValueError("别名已属于另一个人物；请使用人物合并")
            connection.execute(
                """
                INSERT INTO character_aliases
                    (id, character_id, alias, alias_type, confidence,
                     manually_confirmed, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1.0, 1, ?, ?)
                ON CONFLICT(character_id, alias) DO UPDATE SET
                    alias_type = excluded.alias_type,
                    confidence = 1.0,
                    manually_confirmed = 1,
                    updated_at = excluded.updated_at
                """,
                (_stable_id("alias", character_id, alias), character_id, alias, alias_type, now, now),
            )
            connection.execute(
                "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
                (now, character_id),
            )
            connection.commit()
        return next(
            item for item in self.list_character_entities(document_id)
            if item["id"] == character_id
        )

    def _merge_character_entities_in_connection(
        self,
        connection: Any,
        source_character_id: str,
        target_character_id: str,
        *,
        keep_source_name_as_alias: bool,
        now: str,
    ) -> str:
        if source_character_id == target_character_id:
            raise ValueError("不能把人物合并到自身")
        source = connection.execute(
            "SELECT * FROM character_entities WHERE id = ?", (source_character_id,)
        ).fetchone()
        target = connection.execute(
            "SELECT * FROM character_entities WHERE id = ?", (target_character_id,)
        ).fetchone()
        if source is None or target is None:
            raise KeyError("character_entity_not_found")
        if source["document_id"] != target["document_id"]:
            raise ValueError("只能合并同一 TXT 内的人物")
        document_id = target["document_id"]
        alias_values = [
            row["alias"]
            for row in connection.execute(
                "SELECT alias FROM character_aliases WHERE character_id = ?",
                (source_character_id,),
            ).fetchall()
        ]
        if keep_source_name_as_alias:
            alias_values.insert(0, source["canonical_name"])
        for alias in _name_list(alias_values):
            if alias == target["canonical_name"]:
                continue
            owner = connection.execute(
                """
                SELECT ca.character_id
                FROM character_aliases ca
                JOIN character_entities ce ON ce.id = ca.character_id
                WHERE ce.document_id = ? AND ca.alias = ?
                """,
                (document_id, alias),
            ).fetchone()
            if owner and owner["character_id"] not in {source_character_id, target_character_id}:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO character_aliases
                    (id, character_id, alias, alias_type, confidence,
                     manually_confirmed, created_at, updated_at)
                VALUES (?, ?, ?, 'merged', 1.0, 1, ?, ?)
                """,
                (_stable_id("alias", target_character_id, alias), target_character_id, alias, now, now),
            )
        for table in ("character_profiles", "character_facts", "character_events"):
            connection.execute(
                f"UPDATE {table} SET character_id = ?, updated_at = ? WHERE character_id = ?",
                (target_character_id, now, source_character_id),
            )
        connection.execute(
            "UPDATE relationship_events SET source_character_id = ? WHERE source_character_id = ?",
            (target_character_id, source_character_id),
        )
        connection.execute(
            "UPDATE relationship_events SET target_character_id = ? WHERE target_character_id = ?",
            (target_character_id, source_character_id),
        )
        connection.execute(
            "DELETE FROM relationship_events WHERE source_character_id = target_character_id AND document_id = ?",
            (document_id,),
        )
        connection.execute(
            "UPDATE character_relationships SET source_character_id = ?, updated_at = ? WHERE source_character_id = ?",
            (target_character_id, now, source_character_id),
        )
        connection.execute(
            "UPDATE character_relationships SET target_character_id = ?, updated_at = ? WHERE target_character_id = ?",
            (target_character_id, now, source_character_id),
        )
        connection.execute(
            "DELETE FROM character_relationships WHERE source_character_id = target_character_id AND document_id = ?",
            (document_id,),
        )
        connection.execute(
            "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
            (now, target_character_id),
        )
        connection.execute("DELETE FROM character_entities WHERE id = ?", (source_character_id,))
        return document_id

    def merge_character_entities(
        self,
        source_character_id: str,
        target_character_id: str,
        *,
        keep_source_name_as_alias: bool = True,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                document_id = self._merge_character_entities_in_connection(
                    connection,
                    source_character_id,
                    target_character_id,
                    keep_source_name_as_alias=keep_source_name_as_alias,
                    now=now,
                )
            except Exception:
                connection.rollback()
                raise
            connection.commit()
        return {
            "merged": {"source_character_id": source_character_id, "target_character_id": target_character_id},
            "characters": self.list_character_entities(document_id),
            "relationships": self.list_relationships(document_id),
        }

    def _move_character_owned_rows(
        self,
        connection: Any,
        table: str,
        source_character_id: str,
        target_character_id: str,
        row_ids: list[str],
        *,
        label: str,
        now: str,
    ) -> int:
        ids = [str(row_id).strip() for row_id in row_ids if str(row_id).strip()]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT id FROM {table} WHERE character_id = ? AND id IN ({placeholders})",
            (source_character_id, *ids),
        ).fetchall()
        found = {row["id"] for row in rows}
        missing = [row_id for row_id in ids if row_id not in found]
        if missing:
            raise ValueError(f"{label}不属于当前人物")
        connection.execute(
            f"UPDATE {table} SET character_id = ?, updated_at = ? WHERE id IN ({placeholders})",
            (target_character_id, now, *ids),
        )
        return len(ids)

    def _move_character_relationships(
        self,
        connection: Any,
        source_character_id: str,
        target_character_id: str,
        relationship_ids: list[str],
        *,
        document_id: str,
        now: str,
    ) -> tuple[int, int]:
        ids = [str(row_id).strip() for row_id in relationship_ids if str(row_id).strip()]
        if not ids:
            return 0, 0
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"""
            SELECT * FROM character_relationships
            WHERE document_id = ? AND id IN ({placeholders})
            """,
            (document_id, *ids),
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        missing = [row_id for row_id in ids if row_id not in by_id]
        if missing:
            raise ValueError("要拆分的关系边不属于当前 TXT")
        moved_events = 0
        for relationship_id in ids:
            row = by_id[relationship_id]
            if (
                row["source_character_id"] != source_character_id
                and row["target_character_id"] != source_character_id
            ):
                raise ValueError("要拆分的关系边不属于当前人物")
            new_source_id = (
                target_character_id
                if row["source_character_id"] == source_character_id
                else row["source_character_id"]
            )
            new_target_id = (
                target_character_id
                if row["target_character_id"] == source_character_id
                else row["target_character_id"]
            )
            if new_source_id == new_target_id:
                raise ValueError("拆分后的关系不能指向同一人物")
            event_result = connection.execute(
                """
                UPDATE relationship_events
                SET source_character_id = ?, target_character_id = ?, updated_at = ?
                WHERE document_id = ? AND source_character_id = ?
                  AND target_character_id = ? AND relation_type = ?
                """,
                (
                    new_source_id,
                    new_target_id,
                    now,
                    document_id,
                    row["source_character_id"],
                    row["target_character_id"],
                    row["relation_type"],
                ),
            )
            moved_events += event_result.rowcount if event_result.rowcount > 0 else 0
            connection.execute(
                """
                UPDATE character_relationships
                SET source_character_id = ?, target_character_id = ?,
                    manually_edited = 1, updated_at = ?
                WHERE id = ?
                """,
                (new_source_id, new_target_id, now, relationship_id),
            )
        return len(ids), moved_events

    def split_character_entity(self, source_character_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        canonical_name = str(payload.get("canonical_name") or "").strip()
        if not canonical_name:
            raise ValueError("拆分后人物名称不能为空")
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (source_character_id,),
            ).fetchone()
            if source is None:
                connection.rollback()
                raise KeyError("character_entity_not_found")
            document_id = source["document_id"]
            if canonical_name == source["canonical_name"]:
                connection.rollback()
                raise ValueError("拆分后人物名称不能与原人物相同")
            duplicate = connection.execute(
                "SELECT id FROM character_entities WHERE document_id = ? AND canonical_name = ?",
                (document_id, canonical_name),
            ).fetchone()
            if duplicate:
                connection.rollback()
                raise ValueError("人物名称已存在")
            target_character_id = new_id()
            connection.execute(
                """
                INSERT INTO character_entities
                    (id, document_id, canonical_name, entity_type, enabled,
                     manually_confirmed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    target_character_id,
                    document_id,
                    canonical_name,
                    source["entity_type"],
                    int(source["enabled"]),
                    now,
                    now,
                ),
            )

            moved_aliases = []
            alias_values = [
                alias for alias in _name_list(payload.get("aliases") or [])
                if alias != canonical_name
            ]
            if alias_values:
                placeholders = ",".join("?" for _ in alias_values)
                alias_rows = connection.execute(
                    f"""
                    SELECT * FROM character_aliases
                    WHERE character_id = ? AND alias IN ({placeholders})
                    """,
                    (source_character_id, *alias_values),
                ).fetchall()
                by_alias = {row["alias"]: row for row in alias_rows}
                missing_aliases = [alias for alias in alias_values if alias not in by_alias]
                if missing_aliases:
                    connection.rollback()
                    raise ValueError("要拆分的别名不属于当前人物")
                for alias in alias_values:
                    row = by_alias[alias]
                    connection.execute(
                        "UPDATE character_aliases SET character_id = ?, updated_at = ? WHERE id = ?",
                        (target_character_id, now, row["id"]),
                    )
                    moved_aliases.append(alias)

            moved_profiles = self._move_character_owned_rows(
                connection,
                "character_profiles",
                source_character_id,
                target_character_id,
                payload.get("profile_ids") or [],
                label="阶段档案",
                now=now,
            )
            moved_facts = self._move_character_owned_rows(
                connection,
                "character_facts",
                source_character_id,
                target_character_id,
                payload.get("fact_ids") or [],
                label="人物事实",
                now=now,
            )
            moved_events = self._move_character_owned_rows(
                connection,
                "character_events",
                source_character_id,
                target_character_id,
                payload.get("event_ids") or [],
                label="人物经历",
                now=now,
            )
            moved_relationships, moved_relationship_events = self._move_character_relationships(
                connection,
                source_character_id,
                target_character_id,
                payload.get("relationship_ids") or [],
                document_id=document_id,
                now=now,
            )

            has_profile = connection.execute(
                "SELECT 1 FROM character_profiles WHERE character_id = ? LIMIT 1",
                (target_character_id,),
            ).fetchone()
            if not has_profile and payload.get("copy_current_profile", True):
                profile = connection.execute(
                    """
                    SELECT * FROM character_profiles
                    WHERE character_id = ? ORDER BY created_at LIMIT 1
                    """,
                    (source_character_id,),
                ).fetchone()
                if profile:
                    connection.execute(
                        """
                        INSERT INTO character_profiles
                            (id, character_id, title, start_chapter_id, end_chapter_id,
                             identity, personality, goals, behavior_pattern, ability_stage,
                             social_status, enabled, manually_edited, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (
                            new_id(),
                            target_character_id,
                            profile["title"],
                            profile["start_chapter_id"],
                            profile["end_chapter_id"],
                            profile["identity"],
                            profile["personality"],
                            profile["goals"],
                            profile["behavior_pattern"],
                            profile["ability_stage"],
                            profile["social_status"],
                            int(profile["enabled"]),
                            now,
                            now,
                        ),
                    )
            connection.execute(
                "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
                (now, source_character_id),
            )
            connection.commit()
        return {
            "split": {
                "source_character_id": source_character_id,
                "new_character_id": target_character_id,
                "moved_aliases": moved_aliases,
                "moved_profiles": moved_profiles,
                "moved_facts": moved_facts,
                "moved_events": moved_events,
                "moved_relationships": moved_relationships,
                "moved_relationship_events": moved_relationship_events,
            },
            "characters": self.list_character_entities(document_id),
            "relationships": self.list_relationships(document_id),
        }

    def rebuild_relationships(self, document_id: str) -> list[dict[str, Any]]:
        now = utc_now()
        characters = self.seed_character_entities(document_id)
        by_name = {item["canonical_name"]: item["id"] for item in characters}
        for character in characters:
            for alias in character["aliases"]:
                by_name[alias["alias"]] = character["id"]
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            facts = connection.execute(
                """
                SELECT sf.*, fs.chapter_id AS source_chapter_id, fs.chunk_id AS source_chunk_id
                FROM story_facts sf
                LEFT JOIN fact_sources fs ON fs.fact_id = sf.id
                WHERE sf.document_id = ? AND sf.fact_type = 'relationship'
                GROUP BY sf.id
                ORDER BY sf.updated_at
                """,
                (document_id,),
            ).fetchall()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM relationship_events WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM character_relationships WHERE document_id = ?", (document_id,))
            for fact in facts:
                source_id = by_name.get(fact["subject"])
                target_id = by_name.get(fact["object"])
                if not source_id or not target_id:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO material_review_items
                            (id, document_id, review_type, title, payload_json,
                             status, created_at, updated_at)
                        VALUES (?, ?, 'relationship_entity_missing', ?, ?, 'pending', ?, ?)
                        """,
                        (
                            _stable_id("review", document_id, fact["id"], "relationship_entity_missing"),
                            document_id,
                            "关系事实缺少可匹配人物实体",
                            json.dumps(dict(fact), ensure_ascii=False),
                            now,
                            now,
                        ),
                    )
                    continue
                relation_type = fact["predicate"] or "related"
                event_id = _stable_id("relev", document_id, fact["id"])
                relationship_id = _stable_id("rel", document_id, source_id, target_id, relation_type)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO relationship_events
                        (id, document_id, source_character_id, target_character_id,
                         relation_type, event_type, description, chapter_id, chunk_id,
                         sequence, strength_delta, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'set', ?, ?, ?, 0, 0, ?, ?, ?)
                    """,
                    (
                        event_id,
                        document_id,
                        source_id,
                        target_id,
                        relation_type,
                        fact["state"] or fact["object"] or "",
                        fact["source_chapter_id"] or fact["first_chapter_id"],
                        fact["source_chunk_id"],
                        float(fact["confidence"] or 0.7),
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO character_relationships
                        (id, document_id, source_character_id, target_character_id,
                         relation_type, direction, status, strength, start_chapter_id,
                         confidence, manually_edited, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'directed', ?, 0.5, ?, ?, 0, ?, ?)
                    """,
                    (
                        relationship_id,
                        document_id,
                        source_id,
                        target_id,
                        relation_type,
                        fact["status"],
                        fact["first_chapter_id"],
                        float(fact["confidence"] or 0.7),
                        now,
                        now,
                    ),
                )
            connection.commit()
        return self.list_relationships(document_id)

    def list_relationships(self, document_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            rows = connection.execute(
                """
                SELECT cr.*, source.canonical_name AS source_name,
                       target.canonical_name AS target_name
                FROM character_relationships cr
                JOIN character_entities source ON source.id = cr.source_character_id
                JOIN character_entities target ON target.id = cr.target_character_id
                WHERE cr.document_id = ?
                ORDER BY source.canonical_name, target.canonical_name, cr.relation_type
                """,
                (document_id,),
            ).fetchall()
            result = []
            for row in rows:
                events = connection.execute(
                    """
                    SELECT * FROM relationship_events
                    WHERE document_id = ? AND source_character_id = ?
                      AND target_character_id = ? AND relation_type = ?
                    ORDER BY sequence, created_at
                    """,
                    (
                        document_id,
                        row["source_character_id"],
                        row["target_character_id"],
                        row["relation_type"],
                    ),
                ).fetchall()
                result.append({**dict(row), "events": [dict(event) for event in events]})
        return result

    def character_relationships(
        self,
        character_id: str,
        *,
        status: str | None = None,
    ) -> dict[str, Any]:
        requested_status = str(status or "").strip()
        with self.database.connect() as connection:
            character = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (character_id,),
            ).fetchone()
            if character is None:
                raise KeyError("character_not_found")
            clauses = [
                "cr.document_id = ?",
                "(cr.source_character_id = ? OR cr.target_character_id = ?)",
            ]
            params: list[Any] = [character["document_id"], character_id, character_id]
            if requested_status:
                clauses.append("cr.status = ?")
                params.append(requested_status)
            where_sql = " AND ".join(clauses)
            rows = connection.execute(
                f"""
                SELECT cr.*, source.canonical_name AS source_name,
                       target.canonical_name AS target_name
                FROM character_relationships cr
                JOIN character_entities source ON source.id = cr.source_character_id
                JOIN character_entities target ON target.id = cr.target_character_id
                WHERE {where_sql}
                ORDER BY cr.status, source.canonical_name, target.canonical_name, cr.relation_type
                """,
                params,
            ).fetchall()
            relationships = []
            for row in rows:
                events = connection.execute(
                    """
                    SELECT re.*, c.title AS chapter_title, c.position AS chapter_position
                    FROM relationship_events re
                    LEFT JOIN chapters c ON c.id = re.chapter_id
                    WHERE re.document_id = ? AND re.source_character_id = ?
                      AND re.target_character_id = ? AND re.relation_type = ?
                    ORDER BY re.sequence, re.created_at
                    """,
                    (
                        row["document_id"],
                        row["source_character_id"],
                        row["target_character_id"],
                        row["relation_type"],
                    ),
                ).fetchall()
                relationships.append({**dict(row), "events": [dict(event) for event in events]})
        return {
            "document_id": character["document_id"],
            "character": dict(character),
            "filters": {"status": requested_status},
            "relationship_count": len(relationships),
            "event_count": sum(len(item["events"]) for item in relationships),
            "relationships": relationships,
        }

    def relationship_history(
        self,
        source_character_id: str,
        target_character_id: str,
        *,
        relation_type: str | None = None,
    ) -> dict[str, Any]:
        requested_relation_type = str(relation_type or "").strip()
        with self.database.connect() as connection:
            source = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (source_character_id,),
            ).fetchone()
            target = connection.execute(
                "SELECT * FROM character_entities WHERE id = ?",
                (target_character_id,),
            ).fetchone()
            if source is None or target is None:
                raise KeyError("character_not_found")
            if source["document_id"] != target["document_id"]:
                raise ValueError("关系人物不属于同一 TXT")
            clauses = [
                "re.document_id = ?",
                "re.source_character_id = ?",
                "re.target_character_id = ?",
            ]
            params: list[Any] = [source["document_id"], source_character_id, target_character_id]
            if requested_relation_type:
                clauses.append("re.relation_type = ?")
                params.append(requested_relation_type)
            where_sql = " AND ".join(clauses)
            relationship_rows = connection.execute(
                """
                SELECT cr.*, source.canonical_name AS source_name,
                       target.canonical_name AS target_name
                FROM character_relationships cr
                JOIN character_entities source ON source.id = cr.source_character_id
                JOIN character_entities target ON target.id = cr.target_character_id
                WHERE cr.document_id = ? AND cr.source_character_id = ?
                  AND cr.target_character_id = ?
                ORDER BY cr.status, cr.relation_type
                """,
                (source["document_id"], source_character_id, target_character_id),
            ).fetchall()
            event_rows = connection.execute(
                f"""
                SELECT re.*, c.title AS chapter_title, c.position AS chapter_position
                FROM relationship_events re
                LEFT JOIN chapters c ON c.id = re.chapter_id
                WHERE {where_sql}
                ORDER BY re.sequence, re.created_at
                """,
                params,
            ).fetchall()
        relationships = [
            dict(row) for row in relationship_rows
            if not requested_relation_type or row["relation_type"] == requested_relation_type
        ]
        return {
            "document_id": source["document_id"],
            "source": dict(source),
            "target": dict(target),
            "filters": {"relation_type": requested_relation_type},
            "relationship_count": len(relationships),
            "event_count": len(event_rows),
            "relationships": relationships,
            "events": [dict(row) for row in event_rows],
        }

    def relationship_network(self, document_id: str) -> dict[str, Any]:
        characters = self.list_character_entities(document_id)
        relationships = self.list_relationships(document_id)
        nodes: dict[str, dict[str, Any]] = {
            character["id"]: {
                "id": character["id"],
                "name": character["canonical_name"],
                "enabled": bool(character["enabled"]),
                "alias_count": len(character.get("aliases") or []),
                "degree": 0,
                "in_degree": 0,
                "out_degree": 0,
                "relationship_count": 0,
                "event_count": 0,
            }
            for character in characters
        }
        edges: list[dict[str, Any]] = []
        for relationship in relationships:
            source_id = relationship["source_character_id"]
            target_id = relationship["target_character_id"]
            events = relationship.get("events") or []
            edge = {
                "id": relationship["id"],
                "source_character_id": source_id,
                "target_character_id": target_id,
                "source_name": relationship["source_name"],
                "target_name": relationship["target_name"],
                "relation_type": relationship["relation_type"],
                "status": relationship["status"],
                "strength": relationship["strength"],
                "confidence": relationship["confidence"],
                "event_count": len(events),
            }
            edges.append(edge)
            if source_id in nodes:
                nodes[source_id]["degree"] += 1
                nodes[source_id]["out_degree"] += 1
                nodes[source_id]["relationship_count"] += 1
                nodes[source_id]["event_count"] += len(events)
            if target_id in nodes:
                nodes[target_id]["degree"] += 1
                nodes[target_id]["in_degree"] += 1
                nodes[target_id]["relationship_count"] += 1
                nodes[target_id]["event_count"] += len(events)
        ordered_nodes = sorted(
            nodes.values(),
            key=lambda item: (-int(item["degree"]), str(item["name"])),
        )
        return {
            "document_id": document_id,
            "node_count": len(ordered_nodes),
            "edge_count": len(edges),
            "event_count": sum(edge["event_count"] for edge in edges),
            "central_characters": ordered_nodes[:5],
            "nodes": ordered_nodes,
            "edges": sorted(
                edges,
                key=lambda item: (
                    str(item["source_name"]),
                    str(item["target_name"]),
                    str(item["relation_type"]),
                ),
            ),
        }

    def relationship_snapshot(
        self,
        document_id: str,
        *,
        chapter_id: str | None = None,
        chapter_position: int | None = None,
    ) -> dict[str, Any]:
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            target_chapter = None
            if chapter_id:
                target_chapter = connection.execute(
                    "SELECT id, title, position FROM chapters WHERE id = ? AND document_id = ?",
                    (chapter_id, document_id),
                ).fetchone()
                if target_chapter is None:
                    raise ValueError("关系快照章节不属于当前 TXT")
            elif chapter_position is not None:
                target_chapter = connection.execute(
                    "SELECT id, title, position FROM chapters WHERE document_id = ? AND position = ?",
                    (document_id, int(chapter_position)),
                ).fetchone()
                if target_chapter is None:
                    raise ValueError("关系快照章节不存在")
            target_position = int(target_chapter["position"]) if target_chapter else None
            rows = connection.execute(
                """
                SELECT cr.*, source.canonical_name AS source_name,
                       target.canonical_name AS target_name
                FROM character_relationships cr
                JOIN character_entities source ON source.id = cr.source_character_id
                JOIN character_entities target ON target.id = cr.target_character_id
                WHERE cr.document_id = ?
                ORDER BY source.canonical_name, target.canonical_name, cr.relation_type
                """,
                (document_id,),
            ).fetchall()
            relationships = []
            for row in rows:
                event_rows = connection.execute(
                    """
                    SELECT re.*, c.title AS chapter_title, c.position AS chapter_position
                    FROM relationship_events re
                    LEFT JOIN chapters c ON c.id = re.chapter_id
                    WHERE re.document_id = ? AND re.source_character_id = ?
                      AND re.target_character_id = ? AND re.relation_type = ?
                    ORDER BY re.sequence, re.created_at
                    """,
                    (
                        document_id,
                        row["source_character_id"],
                        row["target_character_id"],
                        row["relation_type"],
                    ),
                ).fetchall()
                events = []
                for event in event_rows:
                    event_position = event["chapter_position"]
                    if (
                        target_position is None
                        or event_position is None
                        or int(event_position) <= target_position
                    ):
                        events.append(dict(event))
                if target_position is not None and event_rows and not events:
                    continue
                latest_event = events[-1] if events else None
                relationships.append({
                    **dict(row),
                    "events": events,
                    "event_count_to_chapter": len(events),
                    "latest_event": latest_event,
                })
        return {
            "document_id": document_id,
            "chapter": dict(target_chapter) if target_chapter else None,
            "relationship_count": len(relationships),
            "event_count": sum(item["event_count_to_chapter"] for item in relationships),
            "relationships": relationships,
        }

    def _relationship_character_id_for_document(
        self,
        connection: Any,
        document_id: str,
        character_id: Any,
    ) -> str:
        character_id = str(character_id or "").strip()
        if not character_id:
            raise ValueError("关系人物不能为空")
        row = connection.execute(
            "SELECT id FROM character_entities WHERE id = ? AND document_id = ?",
            (character_id, document_id),
        ).fetchone()
        if row is None:
            raise ValueError("关系人物不属于当前 TXT")
        return row["id"]

    def _write_relationship_overlap_review_item(
        self,
        connection: Any,
        relationship_id: str,
        now: str,
    ) -> None:
        relationship = connection.execute(
            """
            SELECT cr.*, source.canonical_name AS source_name,
                   target.canonical_name AS target_name
            FROM character_relationships cr
            JOIN character_entities source ON source.id = cr.source_character_id
            JOIN character_entities target ON target.id = cr.target_character_id
            WHERE cr.id = ?
            """,
            (relationship_id,),
        ).fetchone()
        if relationship is None or relationship["status"] != "active":
            return
        rows = connection.execute(
            """
            SELECT cr.*, source.canonical_name AS source_name,
                   target.canonical_name AS target_name
            FROM character_relationships cr
            JOIN character_entities source ON source.id = cr.source_character_id
            JOIN character_entities target ON target.id = cr.target_character_id
            WHERE cr.document_id = ? AND cr.source_character_id = ?
              AND cr.target_character_id = ? AND cr.id != ?
              AND cr.status = 'active' AND cr.relation_type != ?
            ORDER BY cr.updated_at DESC
            """,
            (
                relationship["document_id"],
                relationship["source_character_id"],
                relationship["target_character_id"],
                relationship["id"],
                relationship["relation_type"],
            ),
        ).fetchall()
        if not rows:
            return
        conflicts = [
            {
                "relationship_id": row["id"],
                "relation_type": row["relation_type"],
                "status": row["status"],
                "strength": row["strength"],
                "confidence": row["confidence"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
        self._write_review_item(
            connection,
            relationship["document_id"],
            "relationship_overlap_conflict",
            f"关系覆盖待确认：{relationship['source_name']} -> {relationship['target_name']}",
            {
                "source_character_id": relationship["source_character_id"],
                "target_character_id": relationship["target_character_id"],
                "source": relationship["source_name"],
                "target": relationship["target_name"],
                "incoming_relationship_id": relationship["id"],
                "incoming_relation_type": relationship["relation_type"],
                "incoming_status": relationship["status"],
                "incoming_strength": relationship["strength"],
                "conflicts": conflicts,
            },
            now,
        )

    def create_relationship(self, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        relation_type = str(payload.get("relation_type") or "related").strip() or "related"
        direction = str(payload.get("direction") or "directed").strip() or "directed"
        status = str(payload.get("status") or "active").strip() or "active"
        strength = max(0, min(1, float(payload.get("strength", 0.5))))
        confidence = max(0, min(1, float(payload.get("confidence", 1.0))))
        description = str(payload.get("description") or "").strip()
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_document(document_id, connection)
            source_id = self._relationship_character_id_for_document(
                connection,
                document_id,
                payload.get("source_character_id"),
            )
            target_id = self._relationship_character_id_for_document(
                connection,
                document_id,
                payload.get("target_character_id"),
            )
            if source_id == target_id:
                connection.rollback()
                raise ValueError("关系不能指向同一人物")
            duplicate = connection.execute(
                """
                SELECT id FROM character_relationships
                WHERE document_id = ? AND source_character_id = ?
                  AND target_character_id = ? AND relation_type = ?
                """,
                (document_id, source_id, target_id, relation_type),
            ).fetchone()
            if duplicate:
                connection.rollback()
                raise ValueError("关系边已存在")
            relationship_id = new_id()
            event_id = new_id()
            connection.execute(
                """
                INSERT INTO character_relationships
                    (id, document_id, source_character_id, target_character_id,
                     relation_type, direction, status, strength, confidence,
                     manually_edited, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    relationship_id,
                    document_id,
                    source_id,
                    target_id,
                    relation_type,
                    direction,
                    status,
                    strength,
                    confidence,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO relationship_events
                    (id, document_id, source_character_id, target_character_id,
                     relation_type, event_type, description, sequence,
                     strength_delta, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'manual_set', ?, 0, 0, ?, ?, ?)
                """,
                (
                    event_id,
                    document_id,
                    source_id,
                    target_id,
                    relation_type,
                    description,
                    confidence,
                    now,
                    now,
                ),
            )
            self._write_relationship_overlap_review_item(connection, relationship_id, now)
            connection.commit()
        return next(
            item for item in self.list_relationships(document_id)
            if item["id"] == relationship_id
        )

    def delete_relationship(self, relationship_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM character_relationships WHERE id = ?",
                (relationship_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("relationship_not_found")
            connection.execute(
                """
                DELETE FROM relationship_events
                WHERE document_id = ? AND source_character_id = ?
                  AND target_character_id = ? AND relation_type = ?
                """,
                (
                    row["document_id"],
                    row["source_character_id"],
                    row["target_character_id"],
                    row["relation_type"],
                ),
            )
            connection.execute(
                "DELETE FROM character_relationships WHERE id = ?",
                (relationship_id,),
            )
            connection.commit()
        return {
            "id": relationship_id,
            "document_id": row["document_id"],
            "deleted": True,
        }

    def _relationship_event_payload_values(
        self,
        connection: Any,
        document_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if "event_type" in payload:
            event_type = str(payload.get("event_type") or "").strip()
            if not event_type:
                raise ValueError("关系事件类型不能为空")
            values["event_type"] = event_type
        if "description" in payload:
            values["description"] = str(payload.get("description") or "").strip()
        if "sequence" in payload:
            values["sequence"] = max(0, int(payload.get("sequence") or 0))
        if "strength_delta" in payload:
            values["strength_delta"] = max(-1, min(1, float(payload.get("strength_delta", 0))))
        if "confidence" in payload:
            values["confidence"] = max(0, min(1, float(payload.get("confidence", 1.0))))
        if "chapter_id" in payload:
            values["chapter_id"] = self._profile_chapter_id_for_document(
                connection,
                document_id,
                payload.get("chapter_id"),
            )
        if "chunk_id" in payload:
            chunk = self._timeline_chunk_for_document(
                connection,
                document_id,
                payload.get("chunk_id"),
            )
            values["chunk_id"] = chunk[0] if chunk else None
            if chunk and values.get("chapter_id") and values["chapter_id"] != chunk[1]:
                raise ValueError("关系事件分片不属于指定章节")
            if chunk and not values.get("chapter_id"):
                values["chapter_id"] = chunk[1]
        return values

    def _get_relationship_event(self, connection: Any, event_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT * FROM relationship_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError("relationship_event_not_found")
        return dict(row)

    def create_relationship_event(self, relationship_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            relationship = connection.execute(
                "SELECT * FROM character_relationships WHERE id = ?",
                (relationship_id,),
            ).fetchone()
            if relationship is None:
                connection.rollback()
                raise KeyError("relationship_not_found")
            document_id = relationship["document_id"]
            values = self._relationship_event_payload_values(connection, document_id, payload)
            event_type = values.get("event_type") or "manual"
            if "sequence" in values:
                sequence = values["sequence"]
            else:
                sequence = connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1 FROM relationship_events
                    WHERE document_id = ? AND source_character_id = ?
                      AND target_character_id = ? AND relation_type = ?
                    """,
                    (
                        document_id,
                        relationship["source_character_id"],
                        relationship["target_character_id"],
                        relationship["relation_type"],
                    ),
                ).fetchone()[0]
            event_id = new_id()
            connection.execute(
                """
                INSERT INTO relationship_events
                    (id, document_id, source_character_id, target_character_id,
                     relation_type, event_type, description, chapter_id, chunk_id,
                     sequence, strength_delta, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    document_id,
                    relationship["source_character_id"],
                    relationship["target_character_id"],
                    relationship["relation_type"],
                    event_type,
                    values.get("description", ""),
                    values.get("chapter_id"),
                    values.get("chunk_id"),
                    sequence,
                    values.get("strength_delta", 0),
                    values.get("confidence", 1.0),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE character_relationships SET manually_edited = 1, updated_at = ? WHERE id = ?",
                (now, relationship_id),
            )
            event = self._get_relationship_event(connection, event_id)
            connection.commit()
        return event

    def update_relationship_event(self, event_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._get_relationship_event(connection, event_id)
            values = self._relationship_event_payload_values(
                connection,
                existing["document_id"],
                changes,
            )
            if (
                "chapter_id" in values
                and values["chapter_id"] != existing["chapter_id"]
                and "chunk_id" not in values
            ):
                values["chunk_id"] = None
            if not values:
                connection.rollback()
                raise ValueError("no_relationship_event_changes")
            assignments = [f"{field} = ?" for field in values]
            parameters = list(values.values())
            assignments.append("updated_at = ?")
            parameters.extend([now, event_id])
            connection.execute(
                f"UPDATE relationship_events SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            updated = self._get_relationship_event(connection, event_id)
            connection.commit()
        return updated

    def delete_relationship_event(self, event_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            event = self._get_relationship_event(connection, event_id)
            connection.execute("DELETE FROM relationship_events WHERE id = ?", (event_id,))
            connection.commit()
        return {
            "id": event_id,
            "document_id": event["document_id"],
            "deleted": True,
        }

    def update_relationship(self, relationship_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"relation_type", "direction", "status", "strength", "confidence"}
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key not in allowed:
                continue
            if key in {"strength", "confidence"}:
                value = max(0, min(1, float(value)))
            else:
                value = str(value or "").strip()
                if not value:
                    continue
            assignments.append(f"{key} = ?")
            values.append(value)
        if not assignments:
            raise ValueError("no_relationship_changes")
        now = utc_now()
        assignments.extend(["manually_edited = 1", "updated_at = ?"])
        values.extend([now, relationship_id])
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"UPDATE character_relationships SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            row = connection.execute(
                "SELECT document_id FROM character_relationships WHERE id = ?",
                (relationship_id,),
            ).fetchone()
            if cursor.rowcount == 0 or row is None:
                connection.rollback()
                raise KeyError("relationship_not_found")
            self._write_relationship_overlap_review_item(connection, relationship_id, now)
            connection.commit()
        return next(
            item for item in self.list_relationships(row["document_id"])
            if item["id"] == relationship_id
        )

    def list_auxiliary_records(
        self,
        document_id: str,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            params: list[Any] = [document_id]
            where = "document_id = ?"
            if record_type:
                where += " AND record_type = ?"
                params.append(self._auxiliary_record_type(record_type))
            rows = connection.execute(
                f"""
                SELECT * FROM auxiliary_records
                WHERE {where}
                ORDER BY record_type, sequence, updated_at DESC
                """,
                params,
            ).fetchall()
        return [self._format_auxiliary_record(row) for row in rows]

    def create_auxiliary_record(self, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record_type = self._auxiliary_record_type(payload.get("record_type"))
        name = self._auxiliary_record_name(record_type, payload)
        if not name:
            raise ValueError("辅助账本名称不能为空")
        summary = self._auxiliary_record_summary(payload)
        status = str(payload.get("status") or "active").strip() or "active"
        confidence = max(0, min(1, float(payload.get("confidence", 1.0))))
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_document(document_id, connection)
            chapter_id = self._timeline_chapter_id_for_document(
                connection, document_id, payload.get("chapter_id")
            )
            chunk = self._timeline_chunk_for_document(
                connection, document_id, payload.get("chunk_id")
            )
            chunk_id = chunk[0] if chunk else None
            if chunk and not chapter_id:
                chapter_id = chunk[1]
            record_id = new_id()
            stored_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
            connection.execute(
                """
                INSERT INTO auxiliary_records
                    (id, document_id, record_type, name, summary, status,
                     chapter_id, chunk_id, sequence, payload_json, confidence,
                     manually_edited, provenance_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    record_id,
                    document_id,
                    record_type,
                    name,
                    summary,
                    status,
                    chapter_id,
                    chunk_id,
                    int(payload.get("sequence") or 0),
                    json.dumps(stored_payload, ensure_ascii=False),
                    confidence,
                    payload.get("provenance_id"),
                    now,
                    now,
                ),
            )
            connection.commit()
        return next(
            item for item in self.list_auxiliary_records(document_id)
            if item["id"] == record_id
        )

    def update_auxiliary_record(self, record_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "record_type", "name", "summary", "status", "chapter_id",
            "chunk_id", "sequence", "payload", "confidence",
        }
        assignments: list[str] = []
        values: list[Any] = []
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM auxiliary_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("auxiliary_record_not_found")
            document_id = row["document_id"]
            for key, value in changes.items():
                if key not in allowed:
                    continue
                if key == "record_type":
                    assignments.append("record_type = ?")
                    values.append(self._auxiliary_record_type(value))
                elif key == "name":
                    name = str(value or "").strip()
                    if not name:
                        connection.rollback()
                        raise ValueError("辅助账本名称不能为空")
                    assignments.append("name = ?")
                    values.append(name)
                elif key == "summary":
                    assignments.append("summary = ?")
                    values.append(str(value or ""))
                elif key == "status":
                    assignments.append("status = ?")
                    values.append(str(value or "active").strip() or "active")
                elif key == "chapter_id":
                    next_chapter_id = self._timeline_chapter_id_for_document(connection, document_id, value)
                    assignments.append("chapter_id = ?")
                    values.append(next_chapter_id)
                    if next_chapter_id != row["chapter_id"] and "chunk_id" not in changes:
                        assignments.append("chunk_id = ?")
                        values.append(None)
                elif key == "chunk_id":
                    chunk = self._timeline_chunk_for_document(connection, document_id, value)
                    assignments.append("chunk_id = ?")
                    values.append(chunk[0] if chunk else None)
                elif key == "sequence":
                    assignments.append("sequence = ?")
                    values.append(int(value or 0))
                elif key == "payload":
                    stored_payload = value if isinstance(value, dict) else {}
                    assignments.append("payload_json = ?")
                    values.append(json.dumps(stored_payload, ensure_ascii=False))
                elif key == "confidence":
                    assignments.append("confidence = ?")
                    values.append(max(0, min(1, float(value if value is not None else 0.7))))
            if not assignments:
                connection.rollback()
                raise ValueError("no_auxiliary_record_changes")
            assignments.append("manually_edited = 1")
            assignments.append("updated_at = ?")
            values.append(now)
            values.append(record_id)
            connection.execute(
                f"UPDATE auxiliary_records SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            connection.commit()
        return next(
            item for item in self.list_auxiliary_records(document_id)
            if item["id"] == record_id
        )

    def delete_auxiliary_record(self, record_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM auxiliary_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("auxiliary_record_not_found")
            connection.execute("DELETE FROM auxiliary_records WHERE id = ?", (record_id,))
            connection.commit()
        return {
            "id": record_id,
            "document_id": row["document_id"],
            "deleted": True,
        }

    def _format_auxiliary_record(self, row: Any) -> dict[str, Any]:
        return {**dict(row), "payload": _json_load(row["payload_json"], {})}

    def _auxiliary_record_type(self, value: Any) -> str:
        record_type = str(value or "").strip() or "unresolved"
        aliases = {
            "location_event": "location",
            "location_observation": "location",
            "object_event": "object",
            "object_observation": "object",
            "item": "object",
            "unresolved_reference": "unresolved",
            "unresolved_observation": "unresolved",
            "foreshadowing": "unresolved",
        }
        record_type = aliases.get(record_type, record_type)
        if record_type not in {"location", "object", "unresolved"}:
            raise ValueError("辅助账本类型只支持 location / object / unresolved")
        return record_type

    def _auxiliary_record_name(self, record_type: str, payload: dict[str, Any]) -> str:
        fallback = {
            "location": "未命名地点",
            "object": "未命名物件",
            "unresolved": "未命名悬念",
        }[record_type]
        name = str(
            payload.get("name")
            or payload.get("location")
            or payload.get("object")
            or payload.get("item")
            or payload.get("title")
            or payload.get("subject")
            or fallback
        ).strip()
        return name[:200]

    def _auxiliary_record_summary(self, payload: dict[str, Any]) -> str:
        return str(
            payload.get("summary")
            or payload.get("description")
            or payload.get("state")
            or payload.get("value")
            or payload.get("evidence")
            or ""
        ).strip()

    def list_review_items(
        self,
        document_id: str,
        status: str | None = None,
        review_type: str | None = None,
    ) -> list[dict[str, Any]]:
        requested_status = str(status or "").strip()
        requested_type = str(review_type or "").strip()
        if requested_status and requested_status not in {"pending", "resolved", "rejected"}:
            raise ValueError("确认队列状态只支持 pending / resolved / rejected")
        where = ["document_id = ?"]
        params: list[Any] = [document_id]
        if requested_status:
            where.append("status = ?")
            params.append(requested_status)
        if requested_type:
            where.append("review_type = ?")
            params.append(requested_type)
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            rows = connection.execute(
                f"""
                SELECT * FROM material_review_items
                WHERE {" AND ".join(where)}
                ORDER BY created_at
                """,
                params,
            ).fetchall()
        return [
            {**dict(row), "payload": _json_load(row["payload_json"], {}),
             "resolution": _json_load(row["resolution_json"], {})}
            for row in rows
        ]

    def resolve_review_item(self, item_id: str, resolution: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._set_review_item_status(item_id, "resolved", resolution or {})

    def reject_review_item(self, item_id: str, resolution: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._set_review_item_status(item_id, "rejected", resolution or {})

    def batch_update_review_items(
        self,
        document_id: str,
        item_ids: list[str],
        status: str,
        resolution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status = str(status or "").strip()
        if status not in {"resolved", "rejected"}:
            raise ValueError("确认队列状态只支持 resolved / rejected")
        unique_ids = list(dict.fromkeys(str(item_id).strip() for item_id in item_ids if str(item_id).strip()))
        if not unique_ids:
            raise ValueError("请先选择确认项")
        if len(unique_ids) > 500:
            raise ValueError("一次最多批量处理 500 条确认项")
        skipped_review_types = {"relationship_entity_missing", "character_entity_missing"}
        base_resolution = dict(resolution or {})
        updated: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            placeholders = ",".join("?" for _ in unique_ids)
            rows = connection.execute(
                f"""
                SELECT id, document_id, review_type, status
                FROM material_review_items
                WHERE id IN ({placeholders})
                """,
                unique_ids,
            ).fetchall()
        rows_by_id = {row["id"]: dict(row) for row in rows}
        for item_id in unique_ids:
            row = rows_by_id.get(item_id)
            if row is None or row["document_id"] != document_id:
                skipped.append({"id": item_id, "reason": "not_found"})
                continue
            if row["status"] != "pending":
                skipped.append({"id": item_id, "reason": "not_pending"})
                continue
            if status == "resolved" and row["review_type"] in skipped_review_types:
                skipped.append({"id": item_id, "reason": "requires_manual_payload"})
                continue
            item_resolution = {
                "source": "workspace_ui_batch",
                "action": status,
                "handled_at": utc_now(),
                **base_resolution,
            }
            try:
                if status == "resolved":
                    updated.append(self.resolve_review_item(item_id, item_resolution))
                else:
                    updated.append(self.reject_review_item(item_id, item_resolution))
            except Exception as exc:  # pragma: no cover - defensive per-item isolation
                errors.append({"id": item_id, "reason": str(exc)})
        return {
            "document_id": document_id,
            "status": status,
            "requested_count": len(unique_ids),
            "updated_count": len(updated),
            "skipped_count": len(skipped),
            "error_count": len(errors),
            "updated_items": updated,
            "skipped": skipped,
            "errors": errors,
            "review_items": self.list_review_items(document_id),
        }

    def ensure_prompt_budget_profile(self, document_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            self._require_document(document_id, connection)
            row = connection.execute(
                """
                SELECT * FROM prompt_budget_profiles
                WHERE document_id = ? AND is_default = 1
                ORDER BY created_at LIMIT 1
                """,
                (document_id,),
            ).fetchone()
            if row is None:
                profile_id = _stable_id("budget", document_id, "default")
                connection.execute(
                    """
                    INSERT INTO prompt_budget_profiles
                        (id, document_id, name, config_json, is_default, created_at, updated_at)
                    VALUES (?, ?, '默认预算', ?, 1, ?, ?)
                    """,
                    (
                        profile_id,
                        document_id,
                        json.dumps(DEFAULT_PROMPT_BUDGET, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM prompt_budget_profiles WHERE id = ?", (profile_id,)
                ).fetchone()
        config = {**DEFAULT_PROMPT_BUDGET, **_json_load(row["config_json"], {})}
        return {**dict(row), "config": config}

    def update_prompt_budget_profile(
        self,
        document_id: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self.ensure_prompt_budget_profile(document_id)
        next_config = dict(profile["config"])
        for key, value in (config or {}).items():
            if key not in DEFAULT_PROMPT_BUDGET:
                continue
            next_config[key] = max(0, min(50000, int(value)))
        profile_name = (name or profile["name"] or "默认预算").strip()[:100]
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE prompt_budget_profiles
                SET name = ?, config_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    profile_name,
                    json.dumps(next_config, ensure_ascii=False),
                    now,
                    profile["id"],
                ),
            )
        return self.ensure_prompt_budget_profile(document_id)

    def _relationship_history_events_for_prompt(
        self,
        connection: Any,
        document_id: str,
        characters: list[dict[str, Any]],
        query_text: str,
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        query = str(query_text or "").strip()
        relevant_character_ids: list[str] = []
        if query:
            for character in characters:
                names = [
                    str(character.get("canonical_name") or "").strip(),
                    *[
                        str(alias.get("alias") or "").strip()
                        for alias in character.get("aliases", [])
                    ],
                ]
                if any(name and name in query for name in names):
                    relevant_character_ids.append(character["id"])

        clauses = ["re.document_id = ?"]
        params: list[Any] = [document_id]
        if relevant_character_ids:
            placeholders = ",".join("?" for _ in relevant_character_ids)
            clauses.append(
                f"(re.source_character_id IN ({placeholders}) OR "
                f"re.target_character_id IN ({placeholders}))"
            )
            params.extend(relevant_character_ids)
            params.extend(relevant_character_ids)
        where_sql = " AND ".join(clauses)
        rows = connection.execute(
            f"""
            SELECT re.*, source.canonical_name AS source_name,
                   target.canonical_name AS target_name,
                   c.title AS chapter_title, c.position AS chapter_position,
                   cr.status AS relationship_status,
                   cr.strength AS relationship_strength
            FROM relationship_events re
            JOIN character_entities source ON source.id = re.source_character_id
            JOIN character_entities target ON target.id = re.target_character_id
            LEFT JOIN chapters c ON c.id = re.chapter_id
            LEFT JOIN character_relationships cr
              ON cr.document_id = re.document_id
             AND cr.source_character_id = re.source_character_id
             AND cr.target_character_id = re.target_character_id
             AND cr.relation_type = re.relation_type
            WHERE {where_sql} AND COALESCE(cr.status, 'active') != 'disabled'
            ORDER BY re.sequence DESC, re.created_at DESC
            LIMIT ?
            """,
            (*params, max(1, min(40, int(limit)))),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def build_prompt_plan(
        self, document_id: str, *, query_text: str = "", max_tokens: int = 8000
    ) -> dict[str, Any]:
        profile = self.ensure_prompt_budget_profile(document_id)
        budget = {**DEFAULT_PROMPT_BUDGET, **profile["config"]}
        with self.database.connect() as connection:
            document = self._require_document(document_id, connection)
            chapters = connection.execute(
                "SELECT * FROM chapters WHERE document_id = ? ORDER BY position DESC LIMIT 6",
                (document_id,),
            ).fetchall()
            chapter_position_rows = connection.execute(
                "SELECT id, position FROM chapters WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            chapter_positions = {
                row["id"]: int(row["position"] or 0)
                for row in chapter_position_rows
            }
            current_position = max(chapter_positions.values() or [0])
            timeline_nodes = connection.execute(
                """
                SELECT * FROM timeline_nodes
                WHERE document_id = ? AND enabled = 1 AND summary != ''
                ORDER BY CASE node_type WHEN 'project' THEN 0 WHEN 'chapter_group' THEN 1 ELSE 2 END,
                         position DESC LIMIT 4
                """,
                (document_id,),
            ).fetchall()
            timeline_events = connection.execute(
                "SELECT * FROM timeline_events WHERE document_id = ? ORDER BY sequence DESC LIMIT 12",
                (document_id,),
            ).fetchall()
            characters = self.list_character_entities(document_id)
            relationships = self.list_relationships(document_id)
            relationship_history_events = self._relationship_history_events_for_prompt(
                connection,
                document_id,
                characters,
                query_text,
            )
            auxiliary_records = connection.execute(
                """
                SELECT * FROM auxiliary_records
                WHERE document_id = ? AND status != 'disabled'
                ORDER BY record_type, sequence DESC, updated_at DESC
                LIMIT 40
                """,
                (document_id,),
            ).fetchall()
            facts = connection.execute(
                "SELECT * FROM story_facts WHERE document_id = ? AND status != 'resolved' ORDER BY updated_at DESC LIMIT 30",
                (document_id,),
            ).fetchall()

        sections = [
            self._plan_section("project_summary", "前文总览", document["global_summary"], budget),
            self._plan_section(
                "current_timeline_node",
                "当前时间线节点",
                "\n".join(f"- {row['title']}：{row['summary']}" for row in timeline_nodes),
                budget,
                [row["id"] for row in timeline_nodes],
            ),
            self._plan_section(
                "recent_chapter_summaries",
                "最近章节摘要",
                "\n".join(
                    f"- {row['title']}：{_summary_text(_json_load(row['summary_json'], {}))}"
                    for row in reversed(chapters)
                ),
                budget,
                [row["id"] for row in chapters],
            ),
            self._plan_section(
                "timeline_events",
                "时间线事件",
                "\n".join(f"- {row['title']}：{row['description']}" for row in timeline_events),
                budget,
                [row["id"] for row in timeline_events],
            ),
            self._plan_section(
                "character_snapshots",
                "人物当前快照",
                "\n\n".join(
                    _character_snapshot_text(item, chapter_positions, current_position)
                    for item in characters
                    if item["enabled"]
                ),
                budget,
                [item["id"] for item in characters],
            ),
            self._plan_section(
                "relationships",
                "人物关系",
                "\n".join(
                    f"- {row['source_name']} -> {row['target_name']}：{row['relation_type']}（{row['status']}）"
                    for row in relationships
                ),
                budget,
                [row["id"] for row in relationships],
            ),
            self._plan_section(
                "relationship_history",
                "人物关系历史",
                "\n".join(_relationship_history_text(row) for row in relationship_history_events),
                budget,
                [row["id"] for row in relationship_history_events],
            ),
            self._plan_section(
                "auxiliary_records",
                "地点 / 物件 / 悬念",
                "\n".join(_auxiliary_record_text(dict(row)) for row in auxiliary_records),
                budget,
                [row["id"] for row in auxiliary_records],
            ),
            self._plan_section(
                "facts",
                "结构化事实",
                "\n".join(
                    f"- [{row['fact_type']}/{row['status']}] {row['subject']} {row['predicate']} {row['object']} {row['state']}"
                    for row in facts
                ),
                budget,
                [row["id"] for row in facts],
            ),
        ]
        used = 0
        trimmed: list[dict[str, str]] = []
        planned = []
        for section in sections:
            if not section["content"].strip():
                section["included"] = False
                section["reason"] = "empty"
            elif used + section["tokens"] <= max_tokens:
                section["included"] = True
                used += section["tokens"]
                if section.get("trimmed_to_budget"):
                    trimmed.append({"key": section["key"], "reason": "分段预算裁剪"})
            else:
                section["included"] = False
                section["reason"] = "预算不足"
                trimmed.append({"key": section["key"], "reason": "预算不足"})
            planned.append(section)
        return {
            "document_id": document_id,
            "query_text": query_text,
            "total_tokens": used,
            "max_tokens": max_tokens,
            "sections": planned,
            "trimmed": trimmed,
        }

    def current_material_snapshot(self, document_id: str, *, max_tokens: int = 8000) -> dict[str, Any]:
        plan = self.build_prompt_plan(document_id, query_text="", max_tokens=max_tokens)
        sections = [
            section for section in plan["sections"]
            if section.get("included") and str(section.get("content") or "").strip()
        ]
        rendered = "\n\n".join(
            f"{section['label']}：\n{str(section['content']).strip()}"
            for section in sections
        )
        return {
            "document_id": document_id,
            "max_tokens": plan["max_tokens"],
            "total_tokens": plan["total_tokens"],
            "sections": sections,
            "trimmed": plan["trimmed"],
            "content": rendered,
        }

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

    def _material_package_records(self, connection: Any, document_id: str) -> dict[str, list[dict[str, Any]]]:
        records: dict[str, list[dict[str, Any]]] = {}
        for name, spec in MATERIAL_JSONL_TABLES.items():
            table = spec["table"]
            columns = spec["columns"]
            if table in MATERIAL_DOCUMENT_TABLES or table == "character_entities":
                select_list = ", ".join(columns)
                rows = connection.execute(
                    f"SELECT {select_list} FROM {table} WHERE document_id = ? ORDER BY id",
                    (document_id,),
                ).fetchall()
            elif table in MATERIAL_CHARACTER_TABLES:
                select_list = ", ".join(f"{table}.{column} AS {column}" for column in columns)
                rows = connection.execute(
                    f"""
                    SELECT {select_list}
                    FROM {table}
                    JOIN character_entities ce ON ce.id = {table}.character_id
                    WHERE ce.document_id = ?
                    ORDER BY {table}.id
                    """,
                    (document_id,),
                ).fetchall()
            else:
                rows = []
            records[name] = [dict(row) for row in rows]
        return records

    def _material_files_for_layers(self, layers: list[str] | None) -> set[str]:
        if not layers:
            return set(MATERIAL_JSONL_TABLES)
        selected: set[str] = set()
        for layer in layers:
            key = str(layer or "").strip()
            if not key:
                continue
            if key == "all":
                return set(MATERIAL_JSONL_TABLES)
            if key not in MATERIAL_IMPORT_LAYERS:
                raise MaterialPackageError(f"未知资料层：{key}")
            selected.update(MATERIAL_IMPORT_LAYERS[key])
        return selected or set(MATERIAL_JSONL_TABLES)

    def _read_material_records(
        self,
        package: zipfile.ZipFile,
        *,
        selected_files: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        files = selected_files or set(MATERIAL_JSONL_TABLES)
        return {
            name: list(self._iter_jsonl(package, name))
            for name in MATERIAL_JSONL_TABLES
            if name in files
        }

    def _normalise_chapter_scope(
        self,
        chapter_start: int | None,
        chapter_end: int | None,
    ) -> dict[str, int | None] | None:
        if chapter_start is None and chapter_end is None:
            return None
        start = _safe_count(chapter_start) or 1
        end = _safe_count(chapter_end) if chapter_end is not None else None
        if end is not None and end < start:
            raise MaterialPackageError("章节范围结束位置不能小于开始位置")
        return {"start": start, "end": end}

    def _chapters_in_scope(
        self,
        chapters: list[dict[str, Any]],
        chapter_scope: dict[str, int | None],
    ) -> list[dict[str, Any]]:
        return [
            chapter for chapter in chapters
            if self._position_in_scope(_safe_count(chapter.get("position")), chapter_scope)
        ]

    def _filter_provenance_by_chapter_scope(
        self,
        chapters: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        provenance: list[dict[str, Any]],
        chapter_scope: dict[str, int | None],
    ) -> list[dict[str, Any]]:
        chapter_positions, chunk_chapter_positions = self._package_source_positions(chapters, chunks)
        return [
            item for item in provenance
            if self._provenance_in_scope(item, chapter_positions, chunk_chapter_positions, chapter_scope)
        ]

    def _filter_material_records_by_chapter_scope(
        self,
        chapters: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        provenance: list[dict[str, Any]],
        material_records: dict[str, list[dict[str, Any]]],
        chapter_scope: dict[str, int | None],
    ) -> dict[str, list[dict[str, Any]]]:
        chapter_positions, chunk_chapter_positions = self._package_source_positions(chapters, chunks)
        provenance_sources = {
            str(item.get("id")): item
            for item in provenance
            if item.get("id")
        }
        return {
            name: [
                record for record in records
                if self._material_record_in_scope(
                    record,
                    chapter_positions,
                    chunk_chapter_positions,
                    provenance_sources,
                    chapter_scope,
                )
            ]
            for name, records in material_records.items()
        }

    def _package_source_positions(
        self,
        chapters: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> tuple[dict[str, int], dict[str, int]]:
        chapter_positions = {
            str(chapter.get("id")): _safe_count(chapter.get("position"))
            for chapter in chapters
            if chapter.get("id")
        }
        chunk_chapter_positions = {
            str(chunk.get("id")): chapter_positions.get(str(chunk.get("chapter_id")), 0)
            for chunk in chunks
            if chunk.get("id")
        }
        return chapter_positions, chunk_chapter_positions

    def _material_record_in_scope(
        self,
        record: dict[str, Any],
        chapter_positions: dict[str, int],
        chunk_chapter_positions: dict[str, int],
        provenance_sources: dict[str, dict[str, Any]],
        chapter_scope: dict[str, int | None],
    ) -> bool:
        checks = [
            self._source_payload_in_scope(
                record,
                chapter_positions,
                chunk_chapter_positions,
                chapter_scope,
            )
        ]
        provenance_id = record.get("provenance_id")
        if provenance_id and str(provenance_id) in provenance_sources:
            checks.append(
                self._provenance_in_scope(
                    provenance_sources[str(provenance_id)],
                    chapter_positions,
                    chunk_chapter_positions,
                    chapter_scope,
                )
            )
        payload = _json_load(record.get("payload_json"), {})
        if isinstance(payload, dict):
            checks.append(
                self._source_payload_in_scope(
                    payload.get("__context") if isinstance(payload.get("__context"), dict) else payload,
                    chapter_positions,
                    chunk_chapter_positions,
                    chapter_scope,
                )
            )
        scoped_checks = [value for value in checks if value is not None]
        return any(scoped_checks) if scoped_checks else True

    def _source_payload_in_scope(
        self,
        payload: dict[str, Any],
        chapter_positions: dict[str, int],
        chunk_chapter_positions: dict[str, int],
        chapter_scope: dict[str, int | None],
    ) -> bool | None:
        if not isinstance(payload, dict):
            return None
        matched = False
        for key in ("chapter_id", "source_chapter_id", "first_chapter_id", "last_chapter_id"):
            value = payload.get(key)
            if value:
                matched = True
                if self._chapter_id_in_scope(str(value), chapter_positions, chapter_scope):
                    return True
        for start_key, end_key in (
            ("start_chapter_id", "end_chapter_id"),
            ("valid_from_chapter_id", "valid_to_chapter_id"),
        ):
            start_id = payload.get(start_key)
            end_id = payload.get(end_key)
            if start_id or end_id:
                matched = True
                if self._chapter_span_in_scope(
                    str(start_id or ""),
                    str(end_id or ""),
                    chapter_positions,
                    chapter_scope,
                ):
                    return True
        for key in ("chunk_id", "source_chunk_id"):
            value = payload.get(key)
            if value:
                matched = True
                if self._chunk_id_in_scope(str(value), chunk_chapter_positions, chapter_scope):
                    return True
        return False if matched else None

    def _provenance_in_scope(
        self,
        provenance: dict[str, Any],
        chapter_positions: dict[str, int],
        chunk_chapter_positions: dict[str, int],
        chapter_scope: dict[str, int | None],
    ) -> bool:
        source_type = str(provenance.get("source_type") or "")
        source_id = str(provenance.get("source_id") or "")
        if source_type == "chapter":
            return self._chapter_id_in_scope(source_id, chapter_positions, chapter_scope)
        if source_type == "chunk":
            return self._chunk_id_in_scope(source_id, chunk_chapter_positions, chapter_scope)
        return True

    def _chapter_id_in_scope(
        self,
        chapter_id: str,
        chapter_positions: dict[str, int],
        chapter_scope: dict[str, int | None],
    ) -> bool:
        return self._position_in_scope(chapter_positions.get(chapter_id, 0), chapter_scope)

    def _chunk_id_in_scope(
        self,
        chunk_id: str,
        chunk_chapter_positions: dict[str, int],
        chapter_scope: dict[str, int | None],
    ) -> bool:
        return self._position_in_scope(chunk_chapter_positions.get(chunk_id, 0), chapter_scope)

    def _chapter_span_in_scope(
        self,
        start_id: str,
        end_id: str,
        chapter_positions: dict[str, int],
        chapter_scope: dict[str, int | None],
    ) -> bool:
        start_position = chapter_positions.get(start_id, 0) if start_id else 0
        end_position = chapter_positions.get(end_id, 0) if end_id else start_position
        if not start_position and not end_position:
            return False
        if not start_position:
            start_position = end_position
        if not end_position:
            end_position = start_position
        scope_end = chapter_scope["end"] or max(chapter_positions.values() or [chapter_scope["start"]])
        return start_position <= scope_end and end_position >= int(chapter_scope["start"] or 1)

    def _position_in_scope(
        self,
        position: int,
        chapter_scope: dict[str, int | None],
    ) -> bool:
        if position <= 0:
            return False
        if position < int(chapter_scope["start"] or 1):
            return False
        return chapter_scope["end"] is None or position <= int(chapter_scope["end"])

    def _material_diff_preview(
        self,
        connection: Any,
        target_document_id: str,
        package_chapters: list[dict[str, Any]],
        package_chunks: list[dict[str, Any]],
        material_records: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        id_maps = self._source_id_maps(
            connection,
            target_document_id,
            package_chapters,
            package_chunks,
        )
        current_records = self._material_package_records(connection, target_document_id)
        files: dict[str, dict[str, Any]] = {}
        for name, spec in MATERIAL_JSONL_TABLES.items():
            columns = spec["columns"]
            existing = {
                str(record.get("id")): stable_json_hash(
                    {column: record.get(column) for column in columns}
                )
                for record in current_records.get(name, [])
                if record.get("id")
            }
            seen_ids: set[str] = set()
            file_preview: dict[str, Any] = {
                "incoming": 0,
                "added": 0,
                "updated": 0,
                "unchanged": 0,
                "local_only": 0,
                "samples": [],
            }
            for raw_record in material_records.get(name, []):
                record = self._remap_record(dict(raw_record), target_document_id, id_maps)
                normalized = {
                    column: self._material_record_value(record, column)
                    for column in columns
                }
                record_id = str(normalized.get("id") or "")
                file_preview["incoming"] += 1
                if record_id:
                    seen_ids.add(record_id)
                if not record_id or record_id not in existing:
                    status = "added"
                elif stable_json_hash(normalized) == existing[record_id]:
                    status = "unchanged"
                else:
                    status = "updated"
                file_preview[status] += 1
                if len(file_preview["samples"]) < 5 and status != "unchanged":
                    file_preview["samples"].append(
                        {
                            "status": status,
                            "file": name,
                            "id": record_id,
                            "label": self._material_record_label(name, normalized),
                        }
                    )
            file_preview["local_only"] = len(set(existing) - seen_ids)
            if not file_preview["samples"]:
                for raw_record in material_records.get(name, [])[:2]:
                    record = self._remap_record(dict(raw_record), target_document_id, id_maps)
                    normalized = {
                        column: self._material_record_value(record, column)
                        for column in columns
                    }
                    file_preview["samples"].append(
                        {
                            "status": "unchanged",
                            "file": name,
                            "id": str(normalized.get("id") or ""),
                            "label": self._material_record_label(name, normalized),
                        }
                    )
            files[name] = file_preview

        layers: dict[str, dict[str, Any]] = {}
        for layer, layer_files in MATERIAL_IMPORT_LAYERS.items():
            layer_preview: dict[str, Any] = {
                "incoming": 0,
                "added": 0,
                "updated": 0,
                "unchanged": 0,
                "local_only": 0,
                "samples": [],
            }
            for name in MATERIAL_JSONL_TABLES:
                if name not in layer_files:
                    continue
                file_preview = files[name]
                for key in ("incoming", "added", "updated", "unchanged", "local_only"):
                    layer_preview[key] += int(file_preview[key])
                for sample in file_preview["samples"]:
                    if len(layer_preview["samples"]) < 6:
                        layer_preview["samples"].append(sample)
            layers[layer] = layer_preview
        return {
            "target_document_id": target_document_id,
            "layers": layers,
            "files": files,
        }

    def _material_record_label(self, file_name: str, record: dict[str, Any]) -> str:
        if file_name == "character_facts.jsonl":
            value = record.get("value")
            text = f"{record.get('field') or '人物事实'}：{value}" if value else record.get("field")
            return self._short_preview_label(text, record.get("id"))
        for key in (
            "title",
            "canonical_name",
            "alias",
            "relation_type",
            "observation_type",
            "review_type",
            "field",
            "name",
            "normalized_key",
        ):
            if record.get(key):
                return self._short_preview_label(record.get(key), record.get("id"))
        return str(record.get("id") or file_name)

    def _short_preview_label(self, value: Any, fallback: Any = "") -> str:
        text = str(value or fallback or "").strip()
        if len(text) <= 48:
            return text
        return f"{text[:45]}..."

    def _import_material_records_into_existing(
        self,
        package_bytes: bytes,
        *,
        target_document_id: str,
        mode: str,
        report: dict[str, Any],
        selected_files: set[str],
        chapter_scope: dict[str, int | None] | None = None,
    ) -> dict[str, Any]:
        package = self._open_package(package_bytes)
        with package:
            chapters = list(self._iter_jsonl(package, "chapters.jsonl"))
            chunks = list(self._iter_jsonl(package, "chunks.jsonl"))
            provenance = list(self._iter_jsonl(package, "provenance.jsonl"))
            material_records = self._read_material_records(package, selected_files=selected_files)
            if chapter_scope:
                material_records = self._filter_material_records_by_chapter_scope(
                    chapters,
                    chunks,
                    provenance,
                    material_records,
                    chapter_scope,
                )
                provenance = self._filter_provenance_by_chapter_scope(
                    chapters,
                    chunks,
                    provenance,
                    chapter_scope,
                )
                if not self._chapters_in_scope(chapters, chapter_scope):
                    raise MaterialPackageError("章节范围内没有可导入章节")

        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_document(target_document_id, connection)
            id_maps = self._source_id_maps(connection, target_document_id, chapters, chunks)
            if mode == "replace_material":
                self._clear_material_layer(connection, target_document_id, selected_files=selected_files)
            for item in provenance:
                record = self._remap_record(dict(item), target_document_id, id_maps)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO material_provenance
                        (id, document_id, source_type, source_id, source_hash,
                         analysis_version, prompt_version, model_id, generated_at, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("id") or new_id(),
                        target_document_id,
                        record.get("source_type") or "",
                        record.get("source_id") or "",
                        record.get("source_hash") or "",
                        record.get("analysis_version") or "",
                        record.get("prompt_version") or "",
                        record.get("model_id") or "",
                        record.get("generated_at") or now,
                        float(record.get("confidence") or 0.7),
                    ),
                )
            self._import_material_records(
                connection,
                target_document_id,
                material_records,
                id_maps,
                selected_files=selected_files,
                preserve_manual_fields=mode == "merge",
            )
            connection.commit()
        return {
            "document_id": target_document_id,
            "mode": mode,
            "material_layers": sorted(
                layer for layer, files in MATERIAL_IMPORT_LAYERS.items()
                if files & selected_files
            ),
            "report": report,
            "overview": self.get_material_overview(target_document_id),
        }

    def _import_material_records(
        self,
        connection: Any,
        document_id: str,
        material_records: dict[str, list[dict[str, Any]]],
        id_maps: dict[str, dict[str, str]] | None = None,
        selected_files: set[str] | None = None,
        preserve_manual_fields: bool = False,
    ) -> None:
        maps = id_maps or {"chapters": {}, "chunks": {}, "documents": {}}
        files = selected_files or set(MATERIAL_JSONL_TABLES)
        for name, spec in MATERIAL_JSONL_TABLES.items():
            if name not in files:
                continue
            table = spec["table"]
            columns = spec["columns"]
            placeholders = ", ".join("?" for _ in columns)
            column_list = ", ".join(columns)
            update_list = ", ".join(
                f"{column} = excluded.{column}" for column in columns if column != "id"
            )
            for raw_record in material_records.get(name, []):
                record = self._remap_record(dict(raw_record), document_id, maps)
                if preserve_manual_fields:
                    record = self._preserve_manual_fields(connection, table, columns, record)
                values = [self._material_record_value(record, column) for column in columns]
                connection.execute(
                    f"""
                    INSERT INTO {table} ({column_list}) VALUES ({placeholders})
                    ON CONFLICT(id) DO UPDATE SET {update_list}
                    """,
                    values,
                )

    def _material_record_value(self, record: dict[str, Any], column: str) -> Any:
        value = record.get(column)
        if column == "manually_edited" and value is None:
            return 0
        return value

    def _preserve_manual_fields(
        self,
        connection: Any,
        table: str,
        columns: list[str],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        rule = MATERIAL_MANUAL_FIELD_RULES.get(table)
        record_id = record.get("id")
        if not rule or not record_id:
            return record
        existing = connection.execute(
            f"SELECT {', '.join(columns)} FROM {table} WHERE id = ?",
            (record_id,),
        ).fetchone()
        if existing is None or not existing[rule["marker"]]:
            return record
        merged = dict(record)
        conflict_fields = self._manual_conflict_fields(rule["fields"], existing, record)
        if conflict_fields:
            self._write_import_conflict_review_item(
                connection,
                table,
                record,
                existing,
                conflict_fields,
            )
        for field in rule["fields"]:
            if field in columns:
                merged[field] = existing[field]
        return merged

    def _manual_conflict_fields(
        self,
        fields: set[str],
        existing: Any,
        incoming: dict[str, Any],
    ) -> list[str]:
        ignored = {"manually_edited", "manually_confirmed", "updated_at"}
        return [
            field for field in sorted(fields - ignored)
            if field in incoming and incoming.get(field) != existing[field]
        ]

    def _write_import_conflict_review_item(
        self,
        connection: Any,
        table: str,
        incoming: dict[str, Any],
        existing: Any,
        conflict_fields: list[str],
    ) -> None:
        document_id = self._material_document_id_for_record(connection, table, incoming)
        if not document_id:
            return
        file_name = self._material_file_for_table(table)
        record_id = str(incoming.get("id") or "")
        payload = {
            "table": table,
            "file": file_name,
            "record_id": record_id,
            "label": self._material_record_label(file_name, incoming),
            "action": "preserve_local_manual_fields",
            "fields": [
                {
                    "field": field,
                    "local": existing[field],
                    "incoming": incoming.get(field),
                }
                for field in conflict_fields
            ],
        }
        now = utc_now()
        connection.execute(
            """
            INSERT OR IGNORE INTO material_review_items
                (id, document_id, review_type, title, payload_json,
                 status, created_at, updated_at)
            VALUES (?, ?, 'material_import_conflict', ?, ?, 'pending', ?, ?)
            """,
            (
                _stable_id(
                    "review",
                    document_id,
                    "material_import_conflict",
                    table,
                    record_id,
                    stable_json_hash(payload),
                ),
                document_id,
                f"导入冲突：{payload['label']}",
                json.dumps(payload, ensure_ascii=False),
                now,
                now,
            ),
        )

    def _material_document_id_for_record(
        self,
        connection: Any,
        table: str,
        record: dict[str, Any],
    ) -> str:
        if record.get("document_id"):
            return str(record["document_id"])
        if record.get("character_id"):
            row = connection.execute(
                "SELECT document_id FROM character_entities WHERE id = ?",
                (record["character_id"],),
            ).fetchone()
            return str(row["document_id"]) if row else ""
        return ""

    def _material_file_for_table(self, table: str) -> str:
        for name, spec in MATERIAL_JSONL_TABLES.items():
            if spec["table"] == table:
                return name
        return table

    def _clear_material_layer(
        self,
        connection: Any,
        document_id: str,
        *,
        selected_files: set[str] | None = None,
    ) -> None:
        files = selected_files or set(MATERIAL_JSONL_TABLES)
        if files == set(MATERIAL_JSONL_TABLES):
            self._clear_all_material_layer(connection, document_id)
            return
        if "semantic_observations.jsonl" in files:
            connection.execute("DELETE FROM semantic_observations WHERE document_id = ?", (document_id,))
        if files & {"timeline_nodes.jsonl", "timeline_events.jsonl"}:
            connection.execute("DELETE FROM timeline_events WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM timeline_nodes WHERE document_id = ?", (document_id,))
        if files & MATERIAL_IMPORT_LAYERS["characters"]:
            character_ids = [
                row["id"] for row in connection.execute(
                    "SELECT id FROM character_entities WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            if character_ids:
                placeholders = ", ".join("?" for _ in character_ids)
                for table in MATERIAL_CHARACTER_TABLES:
                    connection.execute(
                        f"DELETE FROM {table} WHERE character_id IN ({placeholders})",
                        character_ids,
                    )
            connection.execute("DELETE FROM relationship_events WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM character_relationships WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM character_entities WHERE document_id = ?", (document_id,))
        if "review_items.jsonl" in files:
            connection.execute("DELETE FROM material_review_items WHERE document_id = ?", (document_id,))
        if "auxiliary_records.jsonl" in files:
            connection.execute("DELETE FROM auxiliary_records WHERE document_id = ?", (document_id,))
        if "prompt_budget_profiles.jsonl" in files:
            connection.execute("DELETE FROM prompt_budget_profiles WHERE document_id = ?", (document_id,))

    def _clear_all_material_layer(self, connection: Any, document_id: str) -> None:
        character_ids = [
            row["id"] for row in connection.execute(
                "SELECT id FROM character_entities WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        ]
        if character_ids:
            placeholders = ", ".join("?" for _ in character_ids)
            for table in MATERIAL_CHARACTER_TABLES:
                connection.execute(
                    f"DELETE FROM {table} WHERE character_id IN ({placeholders})",
                    character_ids,
                )
        for table in (
            "relationship_events",
            "character_relationships",
            "character_entities",
            "timeline_events",
            "timeline_nodes",
            "semantic_observations",
            "material_review_items",
            "auxiliary_records",
            "prompt_budget_profiles",
            "material_provenance",
        ):
            connection.execute(f"DELETE FROM {table} WHERE document_id = ?", (document_id,))

    def _source_id_maps(
        self,
        connection: Any,
        target_document_id: str,
        package_chapters: list[dict[str, Any]],
        package_chunks: list[dict[str, Any]],
    ) -> dict[str, dict[str, str]]:
        target_chapters = connection.execute(
            "SELECT id, position, content_hash FROM chapters WHERE document_id = ?",
            (target_document_id,),
        ).fetchall()
        chapters_by_hash = {row["content_hash"]: row["id"] for row in target_chapters if row["content_hash"]}
        chapters_by_position = {int(row["position"]): row["id"] for row in target_chapters}
        chapter_map: dict[str, str] = {}
        for chapter in package_chapters:
            source_id = str(chapter.get("id") or "")
            target_id = chapters_by_hash.get(str(chapter.get("content_hash") or ""))
            if not target_id:
                target_id = chapters_by_position.get(int(chapter.get("position") or 0))
            if source_id and target_id:
                chapter_map[source_id] = target_id

        target_chunks = connection.execute(
            """
            SELECT cc.id, cc.position, cc.content_hash, cc.chapter_id, c.position AS chapter_position
            FROM chapter_chunks cc
            JOIN chapters c ON c.id = cc.chapter_id
            WHERE c.document_id = ?
            """,
            (target_document_id,),
        ).fetchall()
        chunks_by_hash = {row["content_hash"]: row["id"] for row in target_chunks if row["content_hash"]}
        chunks_by_position = {
            (int(row["chapter_position"]), int(row["position"])): row["id"]
            for row in target_chunks
        }
        package_chapter_positions = {
            str(chapter.get("id")): int(chapter.get("position") or 0)
            for chapter in package_chapters
        }
        chunk_map: dict[str, str] = {}
        for chunk in package_chunks:
            source_id = str(chunk.get("id") or "")
            target_id = chunks_by_hash.get(str(chunk.get("content_hash") or ""))
            if not target_id:
                chapter_position = package_chapter_positions.get(str(chunk.get("chapter_id")), 0)
                target_id = chunks_by_position.get((chapter_position, int(chunk.get("position") or 0)))
            if source_id and target_id:
                chunk_map[source_id] = target_id
        return {
            "documents": {},
            "chapters": chapter_map,
            "chunks": chunk_map,
        }

    def _remap_record(
        self,
        record: dict[str, Any],
        document_id: str,
        id_maps: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        if "document_id" in record:
            record["document_id"] = document_id
        if record.get("source_type") == "source_document":
            record["source_id"] = document_id
        elif record.get("source_type") == "chapter" and record.get("source_id") in id_maps["chapters"]:
            record["source_id"] = id_maps["chapters"][record["source_id"]]
        elif record.get("source_type") == "chunk" and record.get("source_id") in id_maps["chunks"]:
            record["source_id"] = id_maps["chunks"][record["source_id"]]
        for key in (
            "chapter_id", "start_chapter_id", "end_chapter_id",
            "first_chapter_id", "last_chapter_id", "valid_from_chapter_id",
            "valid_to_chapter_id",
        ):
            value = record.get(key)
            if value in id_maps["chapters"]:
                record[key] = id_maps["chapters"][value]
        if record.get("chunk_id") in id_maps["chunks"]:
            record["chunk_id"] = id_maps["chunks"][record["chunk_id"]]
        return record

    def _require_document(self, document_id: str, connection: Any | None = None) -> Any:
        if connection is not None:
            row = connection.execute(
                "SELECT * FROM source_documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise KeyError("document_not_found")
            return row
        with self.database.connect() as owned_connection:
            return self._require_document(document_id, owned_connection)

    def _timeline_group_size(self, chapter_count: int) -> int:
        if chapter_count <= 30:
            return 20
        if chapter_count <= 100:
            return 12
        return 8

    def _set_review_item_status(
        self, item_id: str, status: str, resolution: dict[str, Any]
    ) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM material_review_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("review_item_not_found")
            final_resolution = dict(resolution)
            if status == "resolved":
                applied = self._apply_review_resolution(connection, row, final_resolution, now)
                if applied:
                    final_resolution["applied"] = applied
            connection.execute(
                """
                UPDATE material_review_items
                SET status = ?, resolution_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    json.dumps(final_resolution, ensure_ascii=False),
                    now,
                    item_id,
                ),
            )
            document_id = row["document_id"]
            connection.commit()
        return next(
            item for item in self.list_review_items(document_id)
            if item["id"] == item_id
        )

    def _apply_review_resolution(
        self,
        connection: Any,
        row: Any,
        resolution: dict[str, Any],
        now: str,
    ) -> dict[str, Any]:
        apply_action = resolution.get("apply")
        if apply_action == "apply_import_conflict_incoming":
            return self._apply_import_conflict_resolution(connection, row, resolution, now)
        if apply_action == "apply_auxiliary_observation":
            return self._apply_auxiliary_observation_resolution(connection, row, now)
        if apply_action == "apply_character_alias":
            return self._apply_character_alias_resolution(connection, row, resolution, now)
        if apply_action == "merge_character_candidate":
            return self._apply_character_merge_candidate_resolution(connection, row, resolution, now)
        if apply_action != "create_missing_entities":
            return {}
        document_id = row["document_id"]
        review_type = row["review_type"]
        payload = _json_load(row["payload_json"], {})
        context = payload.get("__context") if isinstance(payload.get("__context"), dict) else {}
        names = _name_list(resolution.get("names"))
        if review_type == "character_entity_missing" and not names:
            names = _name_list(payload.get("character") or payload.get("name") or payload.get("subject"))
        if review_type == "relationship_entity_missing":
            source_name = str(resolution.get("source") or payload.get("source") or payload.get("subject") or "").strip()
            target_name = str(resolution.get("target") or payload.get("target") or payload.get("object") or "").strip()
            if not names:
                names = [name for name in (source_name, target_name) if name]
            if not source_name and names:
                source_name = names[0]
            if not target_name and len(names) > 1:
                target_name = names[1]

        if not names:
            return {}

        entities: list[dict[str, Any]] = []
        by_name: dict[str, str] = {}
        for name in names:
            entity_id, created = self._ensure_character_entity(
                connection, document_id, name, now, manually_confirmed=True
            )
            entities.append({"name": name, "id": entity_id, "created": created})
            by_name[name] = entity_id

        chapter_id = (
            context.get("chapter_id")
            or payload.get("source_chapter_id")
            or payload.get("first_chapter_id")
            or payload.get("chapter_id")
            or None
        )
        chunk_id = context.get("chunk_id") or payload.get("source_chunk_id") or payload.get("chunk_id") or None
        sequence = int(context.get("sequence") or payload.get("sequence") or 0)
        provenance_id = context.get("provenance_id") or payload.get("provenance_id") or None
        record_id = context.get("observation_id") or payload.get("id") or row["id"]

        projected = ""
        if review_type == "character_entity_missing":
            character_name = (
                str(resolution.get("character") or payload.get("character") or payload.get("name") or "").strip()
                or names[0]
            )
            character_id = by_name.get(character_name) or by_name[names[0]]
            self._insert_character_event_projection(
                connection,
                character_id,
                payload,
                chapter_id=chapter_id,
                chunk_id=chunk_id,
                sequence=sequence,
                provenance_id=provenance_id,
                record_id=record_id,
                now=now,
            )
            projected = "character_event"
        elif review_type == "relationship_entity_missing":
            source_name = str(resolution.get("source") or payload.get("source") or payload.get("subject") or "").strip()
            target_name = str(resolution.get("target") or payload.get("target") or payload.get("object") or "").strip()
            source_id = (
                by_name.get(source_name)
                or self._find_character_entity(connection, document_id, source_name)
                or (by_name.get(names[0]) if names else None)
            )
            target_id = (
                by_name.get(target_name)
                or self._find_character_entity(connection, document_id, target_name)
                or (by_name.get(names[1]) if len(names) > 1 else None)
            )
            if source_id and target_id:
                self._insert_relationship_projection(
                    connection,
                    document_id,
                    source_id,
                    target_id,
                    payload,
                    chapter_id=chapter_id,
                    chunk_id=chunk_id,
                    sequence=sequence,
                    provenance_id=provenance_id,
                    record_id=record_id,
                    now=now,
                )
                projected = "relationship_event"

        return {"entities": entities, "projected": projected}

    def _apply_character_alias_resolution(
        self,
        connection: Any,
        row: Any,
        resolution: dict[str, Any],
        now: str,
    ) -> dict[str, Any]:
        if row["review_type"] != "character_alias_pending":
            return {}
        payload = _json_load(row["payload_json"], {})
        document_id = row["document_id"]
        character_id = str(resolution.get("character_id") or payload.get("character_id") or "").strip()
        alias = str(resolution.get("alias") or payload.get("alias") or "").strip()
        if not character_id or not alias:
            return {}
        character = connection.execute(
            "SELECT * FROM character_entities WHERE id = ? AND document_id = ?",
            (character_id, document_id),
        ).fetchone()
        if character is None:
            return {}
        if alias == character["canonical_name"]:
            return {
                "projected": "character_alias",
                "character_id": character_id,
                "alias": alias,
                "skipped": "canonical_name",
            }
        existing = connection.execute(
            """
            SELECT ca.character_id
            FROM character_aliases ca
            JOIN character_entities ce ON ce.id = ca.character_id
            WHERE ce.document_id = ? AND ca.alias = ?
            """,
            (document_id, alias),
        ).fetchone()
        if existing and existing["character_id"] != character_id:
            raise ValueError("别名已属于另一个人物；请使用人物合并")
        alias_type = str(
            resolution.get("alias_type")
            or payload.get("alias_type")
            or ("weak_confirmed" if _is_weak_character_alias(alias) else "name")
        ).strip()[:50] or "name"
        connection.execute(
            """
            INSERT INTO character_aliases
                (id, character_id, alias, alias_type, confidence,
                 manually_confirmed, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1.0, 1, ?, ?)
            ON CONFLICT(character_id, alias) DO UPDATE SET
                alias_type = excluded.alias_type,
                confidence = 1.0,
                manually_confirmed = 1,
                updated_at = excluded.updated_at
            """,
            (_stable_id("alias", character_id, alias), character_id, alias, alias_type, now, now),
        )
        connection.execute(
            "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
            (now, character_id),
        )
        return {
            "projected": "character_alias",
            "character_id": character_id,
            "character": character["canonical_name"],
            "alias": alias,
            "alias_type": alias_type,
        }

    def _apply_character_merge_candidate_resolution(
        self,
        connection: Any,
        row: Any,
        resolution: dict[str, Any],
        now: str,
    ) -> dict[str, Any]:
        if row["review_type"] != "character_merge_candidate":
            return {}
        payload = _json_load(row["payload_json"], {})
        source_character_id = str(
            resolution.get("source_character_id") or payload.get("source_character_id") or ""
        ).strip()
        target_character_id = str(
            resolution.get("target_character_id") or payload.get("target_character_id") or ""
        ).strip()
        if not source_character_id or not target_character_id:
            return {}
        keep_source_name_as_alias = bool(resolution.get("keep_source_name_as_alias", True))
        document_id = self._merge_character_entities_in_connection(
            connection,
            source_character_id,
            target_character_id,
            keep_source_name_as_alias=keep_source_name_as_alias,
            now=now,
        )
        return {
            "projected": "character_merge",
            "document_id": document_id,
            "source_character_id": source_character_id,
            "target_character_id": target_character_id,
            "keep_source_name_as_alias": keep_source_name_as_alias,
        }

    def _apply_auxiliary_observation_resolution(
        self,
        connection: Any,
        row: Any,
        now: str,
    ) -> dict[str, Any]:
        if row["review_type"] not in {
            "location_observation",
            "ability_observation",
            "object_observation",
            "unresolved_observation",
        }:
            return {}
        document_id = row["document_id"]
        payload = _json_load(row["payload_json"], {})
        context = payload.get("__context") if isinstance(payload.get("__context"), dict) else {}
        chapter_id = context.get("chapter_id") or payload.get("chapter_id") or None
        chunk_id = context.get("chunk_id") or payload.get("chunk_id") or None
        sequence = int(context.get("sequence") or payload.get("sequence") or 0)
        provenance_id = context.get("provenance_id") or payload.get("provenance_id") or None
        observation_id = context.get("observation_id") or row["id"]
        if row["review_type"] == "ability_observation":
            character_name = str(
                payload.get("character")
                or payload.get("name")
                or payload.get("source")
                or payload.get("subject")
                or ""
            ).strip()
            if not character_name:
                return {}
            character_id, created = self._ensure_character_entity(
                connection,
                document_id,
                character_name,
                now,
                manually_confirmed=True,
            )
            event_id = self._insert_character_event_projection(
                connection,
                character_id,
                {
                    **payload,
                    "event_type": payload.get("event_type") or "ability_update",
                    "value": self._auxiliary_payload_description(payload),
                },
                chapter_id=chapter_id,
                chunk_id=chunk_id,
                sequence=sequence,
                provenance_id=provenance_id,
                record_id=observation_id,
                now=now,
            )
            return {
                "projected": "character_event",
                "event_id": event_id,
                "character_id": character_id,
                "character_created": created,
            }

        record_type = self._auxiliary_record_type(row["review_type"])
        auxiliary_record_id = self._insert_auxiliary_record_projection(
            connection,
            document_id,
            record_type,
            payload,
            chapter_id=chapter_id,
            chunk_id=chunk_id,
            sequence=sequence,
            provenance_id=provenance_id,
            record_id=observation_id,
            now=now,
        )
        event_id = self._insert_auxiliary_timeline_event(
            connection,
            document_id,
            row["review_type"],
            payload,
            chapter_id=chapter_id,
            chunk_id=chunk_id,
            sequence=sequence,
            provenance_id=provenance_id,
            record_id=observation_id,
            auxiliary_record_id=auxiliary_record_id,
            now=now,
        )
        return {
            "projected": row["review_type"],
            "timeline_event_id": event_id,
            "auxiliary_record_id": auxiliary_record_id,
        }

    def _apply_import_conflict_resolution(
        self,
        connection: Any,
        row: Any,
        resolution: dict[str, Any],
        now: str,
    ) -> dict[str, Any]:
        if row["review_type"] != "material_import_conflict":
            return {}
        payload = _json_load(row["payload_json"], {})
        table = str(payload.get("table") or "")
        record_id = str(payload.get("record_id") or "")
        rule = MATERIAL_MANUAL_FIELD_RULES.get(table)
        columns = self._material_columns_for_table(table)
        if not rule or not columns or not record_id:
            return {}
        requested_fields = {
            str(field) for field in resolution.get("fields", [])
            if str(field).strip()
        }
        allowed_fields = set(rule["fields"]) - {"manually_edited", "manually_confirmed", "updated_at"}
        assignments: list[str] = []
        values: list[Any] = []
        applied_fields: list[str] = []
        for field_item in payload.get("fields", []):
            if not isinstance(field_item, dict):
                continue
            field = str(field_item.get("field") or "")
            if requested_fields and field not in requested_fields:
                continue
            if field not in allowed_fields or field not in columns:
                continue
            assignments.append(f"{field} = ?")
            values.append(field_item.get("incoming"))
            applied_fields.append(field)
        if not assignments:
            return {}
        marker = str(rule["marker"])
        if marker in columns:
            assignments.append(f"{marker} = 1")
        if "updated_at" in columns:
            assignments.append("updated_at = ?")
            values.append(now)
        values.append(record_id)
        connection.execute(
            f"UPDATE {table} SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        return {
            "table": table,
            "record_id": record_id,
            "fields": applied_fields,
        }

    def _material_columns_for_table(self, table: str) -> list[str]:
        for spec in MATERIAL_JSONL_TABLES.values():
            if spec["table"] == table:
                return list(spec["columns"])
        return []

    def _ensure_character_entity(
        self,
        connection: Any,
        document_id: str,
        name: str,
        now: str,
        *,
        manually_confirmed: bool = False,
    ) -> tuple[str, bool]:
        existing_id = self._find_character_entity(connection, document_id, name)
        if existing_id:
            if manually_confirmed:
                connection.execute(
                    "UPDATE character_entities SET manually_confirmed = 1, updated_at = ? WHERE id = ?",
                    (now, existing_id),
                )
            return existing_id, False
        entity_id = _stable_id("charent", document_id, name)
        connection.execute(
            """
            INSERT INTO character_entities
                (id, document_id, canonical_name, entity_type, enabled,
                 manually_confirmed, created_at, updated_at)
            VALUES (?, ?, ?, 'person', 1, ?, ?, ?)
            ON CONFLICT(document_id, canonical_name) DO UPDATE SET
                manually_confirmed = CASE
                    WHEN excluded.manually_confirmed = 1 THEN 1
                    ELSE manually_confirmed
                END,
                updated_at = excluded.updated_at
            """,
            (
                entity_id,
                document_id,
                name,
                int(manually_confirmed),
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO character_profiles
                (id, character_id, title, enabled, manually_edited, created_at, updated_at)
            VALUES (?, ?, '人工确认档案', 1, 0, ?, ?)
            """,
            (_stable_id("charprofile", entity_id, "manual"), entity_id, now, now),
        )
        return entity_id, True

    def _project_unified_event(
        self,
        connection: Any,
        document_id: str,
        chapter_id: str,
        chunk_id: str,
        observation_id: str,
        provenance_id: str,
        observation_type: str,
        payload: dict[str, Any],
        sequence: int,
        now: str,
    ) -> None:
        if observation_type == "plot_event":
            title = str(payload.get("title") or payload.get("description") or "剧情事件")
            connection.execute(
                """
                INSERT OR REPLACE INTO timeline_events
                    (id, document_id, event_type, title, description, chapter_id,
                     chunk_id, sequence, participants_json, causes_json, consequences_json,
                     status, confidence, manually_edited, provenance_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0, ?, ?, ?)
                """,
                (
                    _stable_id("tle", observation_id),
                    document_id,
                    str(payload.get("event_type") or "event"),
                    title,
                    str(payload.get("description") or ""),
                    chapter_id,
                    chunk_id,
                    sequence,
                    json.dumps(payload.get("participants") or [], ensure_ascii=False),
                    json.dumps(payload.get("causes") or [], ensure_ascii=False),
                    json.dumps(payload.get("consequences") or [], ensure_ascii=False),
                    float(payload.get("confidence") or 0.7),
                    provenance_id,
                    now,
                    now,
                ),
            )
        elif observation_type == "character_event":
            character_name = str(payload.get("character") or payload.get("name") or "").strip()
            character_id = self._find_character_entity(connection, document_id, character_name)
            if not character_id:
                self._write_review_item(
                    connection, document_id, "character_entity_missing",
                    "人物事件缺少可匹配人物实体", payload, now,
                    context={
                        "observation_id": observation_id,
                        "provenance_id": provenance_id,
                        "chapter_id": chapter_id,
                        "chunk_id": chunk_id,
                        "sequence": sequence,
                    },
                )
                return
            self._insert_character_event_projection(
                connection,
                character_id,
                payload,
                chapter_id=chapter_id,
                chunk_id=chunk_id,
                sequence=sequence,
                provenance_id=provenance_id,
                record_id=observation_id,
                now=now,
            )
        elif observation_type == "relationship_event":
            source_name = str(payload.get("source") or "").strip()
            target_name = str(payload.get("target") or "").strip()
            source_id = self._find_character_entity(connection, document_id, source_name)
            target_id = self._find_character_entity(connection, document_id, target_name)
            if not source_id or not target_id:
                self._write_review_item(
                    connection, document_id, "relationship_entity_missing",
                    "关系事件缺少可匹配人物实体", payload, now,
                    context={
                        "observation_id": observation_id,
                        "provenance_id": provenance_id,
                        "chapter_id": chapter_id,
                        "chunk_id": chunk_id,
                        "sequence": sequence,
                    },
                )
                return
            self._insert_relationship_projection(
                connection,
                document_id,
                source_id,
                target_id,
                payload,
                chapter_id=chapter_id,
                chunk_id=chunk_id,
                sequence=sequence,
                provenance_id=provenance_id,
                record_id=observation_id,
                now=now,
            )
        elif observation_type in {
            "location_event",
            "ability_event",
            "object_event",
            "unresolved_reference",
        }:
            review_type, title = self._auxiliary_observation_review(observation_type, payload)
            self._write_review_item(
                connection,
                document_id,
                review_type,
                title,
                payload,
                now,
                context={
                    "observation_id": observation_id,
                    "provenance_id": provenance_id,
                    "chapter_id": chapter_id,
                    "chunk_id": chunk_id,
                    "sequence": sequence,
                    "observation_type": observation_type,
                },
            )

    def _auxiliary_observation_review(
        self,
        observation_type: str,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        label = str(
            payload.get("title")
            or payload.get("name")
            or payload.get("location")
            or payload.get("ability")
            or payload.get("object")
            or payload.get("description")
            or payload.get("state")
            or "待确认观察"
        ).strip()
        mapping = {
            "location_event": ("location_observation", "位置观察待确认"),
            "ability_event": ("ability_observation", "能力观察待确认"),
            "object_event": ("object_observation", "物件观察待确认"),
            "unresolved_reference": ("unresolved_observation", "悬念线索待确认"),
        }
        review_type, prefix = mapping[observation_type]
        return review_type, f"{prefix}：{label[:40]}"

    def _insert_character_event_projection(
        self,
        connection: Any,
        character_id: str,
        payload: dict[str, Any],
        *,
        chapter_id: str | None,
        chunk_id: str | None,
        sequence: int,
        provenance_id: str | None,
        record_id: str,
        now: str,
    ) -> str:
        event_id = _stable_id("chev", record_id)
        connection.execute(
            """
            INSERT OR REPLACE INTO character_events
                (id, character_id, event_type, value, chapter_id, chunk_id,
                 sequence, provenance_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                character_id,
                str(payload.get("event_type") or "event"),
                str(payload.get("value") or payload.get("description") or payload.get("state") or ""),
                chapter_id,
                chunk_id,
                sequence,
                provenance_id,
                now,
                now,
            ),
        )
        return event_id

    def _insert_auxiliary_record_projection(
        self,
        connection: Any,
        document_id: str,
        record_type: str,
        payload: dict[str, Any],
        *,
        chapter_id: str | None,
        chunk_id: str | None,
        sequence: int,
        provenance_id: str | None,
        record_id: str,
        now: str,
    ) -> str:
        auxiliary_record_id = _stable_id("aux", record_id, record_type)
        name = self._auxiliary_record_name(record_type, payload)
        summary = self._auxiliary_record_summary(payload)
        status = str(payload.get("status") or "active").strip() or "active"
        connection.execute(
            """
            INSERT INTO auxiliary_records
                (id, document_id, record_type, name, summary, status,
                 chapter_id, chunk_id, sequence, payload_json, confidence,
                 manually_edited, provenance_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                record_type = CASE
                    WHEN auxiliary_records.manually_edited = 1 THEN auxiliary_records.record_type
                    ELSE excluded.record_type
                END,
                name = CASE
                    WHEN auxiliary_records.manually_edited = 1 THEN auxiliary_records.name
                    ELSE excluded.name
                END,
                summary = CASE
                    WHEN auxiliary_records.manually_edited = 1 THEN auxiliary_records.summary
                    ELSE excluded.summary
                END,
                status = CASE
                    WHEN auxiliary_records.manually_edited = 1 THEN auxiliary_records.status
                    ELSE excluded.status
                END,
                chapter_id = excluded.chapter_id,
                chunk_id = excluded.chunk_id,
                sequence = excluded.sequence,
                payload_json = excluded.payload_json,
                confidence = CASE
                    WHEN auxiliary_records.manually_edited = 1 THEN auxiliary_records.confidence
                    ELSE excluded.confidence
                END,
                provenance_id = excluded.provenance_id,
                updated_at = excluded.updated_at
            """,
            (
                auxiliary_record_id,
                document_id,
                record_type,
                name,
                summary,
                status,
                chapter_id,
                chunk_id,
                sequence,
                json.dumps(payload, ensure_ascii=False),
                float(payload.get("confidence") or 0.7),
                provenance_id,
                now,
                now,
            ),
        )
        return auxiliary_record_id

    def _insert_auxiliary_timeline_event(
        self,
        connection: Any,
        document_id: str,
        review_type: str,
        payload: dict[str, Any],
        *,
        chapter_id: str | None,
        chunk_id: str | None,
        sequence: int,
        provenance_id: str | None,
        record_id: str,
        now: str,
        auxiliary_record_id: str | None = None,
    ) -> str:
        event_id = _stable_id("tle", record_id, review_type)
        labels = {
            "location_observation": ("location", "地点"),
            "object_observation": ("object", "物件"),
            "unresolved_observation": ("unresolved_reference", "悬念"),
        }
        event_type, label = labels.get(review_type, ("auxiliary", "观察"))
        name = str(
            payload.get("location")
            or payload.get("object")
            or payload.get("title")
            or payload.get("name")
            or ""
        ).strip()
        title = str(payload.get("title") or (f"{label}：{name}" if name else label)).strip()
        description = self._auxiliary_payload_description(payload)
        location_id = (
            auxiliary_record_id or _stable_id("loc", document_id, name)
            if review_type == "location_observation" and name
            else None
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO timeline_events
                (id, document_id, event_type, title, description, chapter_id,
                 chunk_id, sequence, participants_json, location_id, causes_json,
                 consequences_json, status, confidence, manually_edited,
                 provenance_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, '[]', '[]', 'active', ?, 0, ?, ?, ?)
            """,
            (
                event_id,
                document_id,
                event_type,
                title,
                description,
                chapter_id,
                chunk_id,
                sequence,
                location_id,
                float(payload.get("confidence") or 0.7),
                provenance_id,
                now,
                now,
            ),
        )
        return event_id

    def _auxiliary_payload_description(self, payload: dict[str, Any]) -> str:
        text = str(
            payload.get("description")
            or payload.get("state")
            or payload.get("value")
            or payload.get("evidence")
            or ""
        ).strip()
        ability = str(payload.get("ability") or "").strip()
        if ability and ability not in text:
            return f"{ability}：{text}" if text else ability
        return text

    def _insert_relationship_projection(
        self,
        connection: Any,
        document_id: str,
        source_id: str,
        target_id: str,
        payload: dict[str, Any],
        *,
        chapter_id: str | None,
        chunk_id: str | None,
        sequence: int,
        provenance_id: str | None,
        record_id: str,
        now: str,
    ) -> None:
        relation_type = str(payload.get("relation_type") or payload.get("predicate") or "related")
        relationship_id = _stable_id("rel", document_id, source_id, target_id, relation_type)
        connection.execute(
            """
            INSERT OR REPLACE INTO relationship_events
                (id, document_id, source_character_id, target_character_id,
                 relation_type, event_type, description, chapter_id, chunk_id,
                 sequence, strength_delta, confidence, provenance_id,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _stable_id("relev", record_id),
                document_id,
                source_id,
                target_id,
                relation_type,
                str(payload.get("event_type") or "set"),
                str(payload.get("description") or payload.get("state") or payload.get("object") or ""),
                chapter_id,
                chunk_id,
                sequence,
                float(payload.get("strength_delta") or 0),
                float(payload.get("confidence") or 0.7),
                provenance_id,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO character_relationships
                (id, document_id, source_character_id, target_character_id,
                 relation_type, direction, status, strength, start_chapter_id,
                 confidence, manually_edited, provenance_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'directed', 'active', 0.5, ?, ?, 0, ?, ?, ?)
            """,
            (
                relationship_id,
                document_id,
                source_id,
                target_id,
                relation_type,
                chapter_id,
                float(payload.get("confidence") or 0.7),
                provenance_id,
                now,
                now,
            ),
        )
        self._write_relationship_overlap_review_item(connection, relationship_id, now)

    def _find_character_entity(
        self, connection: Any, document_id: str, name: str
    ) -> str | None:
        if not name:
            return None
        row = connection.execute(
            """
            SELECT id FROM character_entities
            WHERE document_id = ? AND canonical_name = ?
            """,
            (document_id, name),
        ).fetchone()
        if row:
            return row["id"]
        row = connection.execute(
            """
            SELECT ce.id
            FROM character_aliases ca
            JOIN character_entities ce ON ce.id = ca.character_id
            WHERE ce.document_id = ? AND ca.alias = ?
            """,
            (document_id, name),
        ).fetchone()
        return row["id"] if row else None

    def _write_review_item(
        self,
        connection: Any,
        document_id: str,
        review_type: str,
        title: str,
        payload: dict[str, Any],
        now: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        stored_payload = dict(payload)
        if context:
            stored_payload["__context"] = context
        connection.execute(
            """
            INSERT OR IGNORE INTO material_review_items
                (id, document_id, review_type, title, payload_json,
                 status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                _stable_id("review", document_id, review_type, stable_json_hash(stored_payload)),
                document_id,
                review_type,
                title,
                json.dumps(stored_payload, ensure_ascii=False),
                now,
                now,
            ),
        )

    def _plan_section(
        self,
        key: str,
        label: str,
        content: str,
        budget: dict[str, Any],
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        text = content.strip()
        section_budget = int(budget.get(key, 0) or 0)
        if text and section_budget <= 0:
            text = ""
        trimmed_to_budget = False
        tokens = _estimate_tokens(text)
        if text and section_budget and tokens > section_budget:
            suffix = "\n..." if section_budget >= 3 else ""
            max_chars = max(1, section_budget * 2 - len(suffix))
            text = text[:max_chars].rstrip()
            if text and suffix:
                text += suffix
            tokens = _estimate_tokens(text)
            trimmed_to_budget = True
        return {
            "key": key,
            "label": label,
            "tokens": tokens,
            "budget": section_budget,
            "included": False,
            "source_ids": source_ids or [],
            "content": text,
            "trimmed_to_budget": trimmed_to_budget,
        }

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
