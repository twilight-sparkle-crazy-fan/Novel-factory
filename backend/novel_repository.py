from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from .database import Database, new_id, utc_now
from .material_utils import stable_text_hash
from .text_import import split_chapters, split_long_text


def json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def format_chapter_summary(summary: dict[str, Any]) -> str:
    lines = []
    title = summary.get("title") or "章节摘要"
    lines.append(f"### {title}")
    mapping = [
        ("summary", "概要"),
        ("time", "时间"),
        ("location", "地点"),
        ("pov", "视角"),
        ("key_events", "关键事件"),
        ("conflicts", "冲突变化"),
        ("worldbuilding", "新增设定"),
        ("clues", "伏笔与线索"),
        ("unresolved", "未解决问题"),
        ("ending_state", "结尾状态"),
        ("character_changes", "人物状态变化"),
    ]
    for key, label in mapping:
        value = summary.get(key)
        if not value:
            continue
        if isinstance(value, list):
            rendered = "；".join(str(item) for item in value if item)
        else:
            rendered = str(value)
        if rendered:
            lines.append(f"- {label}：{rendered}")
    return "\n".join(lines)


def format_character_card(name: str, card: dict[str, Any], aliases: list[str]) -> str:
    lines = [f"## {name}"]
    if aliases:
        lines.append(f"- 别名/称谓：{'、'.join(aliases)}")
    mapping = [
        ("identity", "身份"),
        ("age", "年龄"),
        ("appearance", "外貌"),
        ("personality", "性格"),
        ("goals", "目标"),
        ("fears", "恐惧/弱点"),
        ("secrets", "秘密"),
        ("speech_style", "语言习惯"),
        ("abilities", "能力与资源"),
        ("relationships", "人物关系"),
        ("arc", "人物弧光"),
        ("current_state", "当前状态"),
        ("facts", "已确认事实"),
        ("inferences", "模型推断"),
        ("uncertainties", "不确定信息"),
    ]
    for key, label in mapping:
        value = card.get(key)
        if not value:
            continue
        if isinstance(value, list):
            rendered = "；".join(str(item) for item in value if item)
        elif isinstance(value, dict):
            rendered = "；".join(f"{item_key}：{item_value}" for item_key, item_value in value.items())
        else:
            rendered = str(value)
        if rendered:
            lines.append(f"- {label}：{rendered}")
    return "\n".join(lines)


WEAK_CHARACTER_ALIASES = {
    "他", "她", "它", "他们", "她们", "自己", "我", "你", "男主", "女主",
    "男人", "女人", "男子", "女子", "少年", "少女", "老人", "孩子", "小孩",
    "父亲", "母亲", "哥哥", "姐姐", "弟弟", "妹妹", "老师", "老板", "医生",
    "警察", "记者", "队长", "先生", "小姐", "夫人", "大人", "前辈", "后辈",
}
CHARACTER_META_FIELDS = {
    "id", "project_id", "document_id", "name", "aliases", "source_chapters",
    "prompt_text", "enabled", "updated_at", "created_at", "card",
}


def normalize_character_key(value: str | None) -> str:
    """Return a conservative comparable key for a character name or alias."""
    if not value:
        return ""
    normalized = str(value).strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[·•・\-_—~～'\"“”‘’`´]", "", normalized)
    normalized = re.sub(r"[《》〈〉【】\[\]{}（）()]", "", normalized)
    normalized = re.sub(r"[，,。.!！?？:：;；、/\\|]", "", normalized)
    return normalized


def _name_fragments(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    fragments = [text]
    for outer, inner in re.findall(r"(.+?)[（(]([^）)]+)[）)]", text):
        fragments.extend([outer.strip(), inner.strip()])
    for part in re.split(r"[、,，/|；;]+", text):
        part = part.strip()
        if part:
            fragments.append(part)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in fragments:
        item = re.sub(r"^(别名|又名|称谓|本名)[:：]", "", item.strip())
        key = normalize_character_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def _character_tokens(
    name: str | None,
    aliases: list[Any] | None,
    *,
    include_weak_primary: bool = True,
) -> set[str]:
    tokens: set[str] = set()
    for fragment in _name_fragments(name):
        key = normalize_character_key(fragment)
        if key and (include_weak_primary or key not in WEAK_CHARACTER_ALIASES):
            tokens.add(key)
    for alias in aliases or []:
        for fragment in _name_fragments(str(alias)):
            key = normalize_character_key(fragment)
            if not key or key in WEAK_CHARACTER_ALIASES:
                continue
            # Single-character aliases are too collision-prone in Chinese prose.
            if len(key) < 2 and not re.search(r"[A-Za-z0-9]", key):
                continue
            tokens.add(key)
    return tokens


def _standard_name_tokens(name: str | None) -> set[str]:
    return _character_tokens(name, [], include_weak_primary=True)


def _standard_names_match(left: str | None, right: str | None) -> bool:
    left_tokens = _standard_name_tokens(left)
    right_tokens = _standard_name_tokens(right)
    return bool(left_tokens and right_tokens and left_tokens & right_tokens)


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = normalize_character_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _unique_strings(value)
    if isinstance(value, tuple):
        return _unique_strings(list(value))
    if isinstance(value, str):
        return _unique_strings(_name_fragments(value) or [value])
    return _unique_strings([value])


def _merge_json_value(old: Any, new: Any) -> Any:
    if new in (None, "", [], {}):
        return old
    if old in (None, "", [], {}):
        return new
    if isinstance(old, list) or isinstance(new, list):
        old_items = old if isinstance(old, list) else [old]
        new_items = new if isinstance(new, list) else [new]
        return _unique_strings([*old_items, *new_items])
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for key, value in new.items():
            merged[key] = _merge_json_value(merged.get(key), value)
        return merged
    if str(old).strip() == str(new).strip():
        return old
    # The newest model pass represents the freshest interpretation, while older
    # non-empty values are preserved when the new value is absent.
    return new


def _character_payload(item: dict[str, Any]) -> dict[str, Any]:
    if isinstance(item.get("card"), dict):
        card = dict(item["card"])
        for key, value in item.items():
            if key not in CHARACTER_META_FIELDS and value not in (None, "", [], {}):
                card[key] = value
        return card
    return {
        key: value
        for key, value in item.items()
        if key not in CHARACTER_META_FIELDS and value not in (None, "", [], {})
    }


def _merge_character_item(
    base: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    base_name = str(base.get("name") or "").strip()
    incoming_name = str(incoming.get("name") or "").strip()
    name = base_name or incoming_name
    aliases = _unique_strings([
        *_as_string_list(base.get("aliases")),
        *_as_string_list(incoming.get("aliases")),
        *([] if not incoming_name or normalize_character_key(incoming_name) == normalize_character_key(name) else [incoming_name]),
    ])
    sources = _unique_strings([
        *_as_string_list(base.get("source_chapters")),
        *_as_string_list(incoming.get("source_chapters")),
    ])
    old_card = _character_payload(base)
    new_card = _character_payload(incoming)
    card = dict(old_card)
    for key, value in new_card.items():
        card[key] = _merge_json_value(card.get(key), value)
    merged = {**base, **incoming}
    merged["name"] = name
    merged["aliases"] = aliases
    merged["source_chapters"] = sources
    merged["card"] = card
    return merged


class NovelRepository:
    def __init__(self, database: Database):
        self.database = database

    def list_projects(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM source_documents d WHERE d.project_id = p.id) AS document_count,
                       (SELECT COUNT(*) FROM chapters c WHERE c.project_id = p.id) AS chapter_count,
                       (SELECT COUNT(*) FROM characters ch WHERE ch.project_id = p.id) AS character_count
                FROM projects p ORDER BY p.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            project = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if project is None:
                raise KeyError("project_not_found")
            documents = connection.execute(
                """
                SELECT d.*,
                       (SELECT COUNT(*) FROM chapters c WHERE c.document_id = d.id) AS chapter_count,
                       (SELECT COUNT(*) FROM document_characters dc WHERE dc.document_id = d.id) AS character_count,
                       (SELECT COUNT(*) FROM story_facts sf WHERE sf.document_id = d.id) AS fact_count
                FROM source_documents d WHERE d.project_id = ? ORDER BY d.created_at
                """,
                (project_id,),
            ).fetchall()
        return {
            "id": project["id"],
            "name": project["name"],
            "global_summary": project["global_summary"],
            "summary_enabled": bool(project["summary_enabled"]),
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
            "documents": [self._document(row) for row in documents],
            "chapters": [],
            "characters": [],
        }

    def _document(self, row: Any) -> dict[str, Any]:
        keys = set(row.keys())
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "filename": row["filename"],
            "encoding": row["encoding"],
            "raw_text_hash": row["raw_text_hash"] if "raw_text_hash" in keys else "",
            "global_summary": row["global_summary"],
            "library_enabled": bool(row["library_enabled"]),
            "summary_enabled": bool(row["summary_enabled"]),
            "recent_chapters_enabled": bool(row["recent_chapters_enabled"]),
            "characters_enabled": bool(row["characters_enabled"]),
            "facts_enabled": bool(row["facts_enabled"]),
            "chapter_count": row["chapter_count"] if "chapter_count" in keys else 0,
            "character_count": row["character_count"] if "character_count" in keys else 0,
            "fact_count": row["fact_count"] if "fact_count" in keys else 0,
            "created_at": row["created_at"],
        }

    def get_document_workspace(self, document_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            document = connection.execute(
                """
                SELECT d.*,
                       (SELECT COUNT(*) FROM chapters c WHERE c.document_id = d.id) AS chapter_count,
                       (SELECT COUNT(*) FROM document_characters dc WHERE dc.document_id = d.id) AS character_count,
                       (SELECT COUNT(*) FROM story_facts sf WHERE sf.document_id = d.id) AS fact_count
                FROM source_documents d WHERE d.id = ?
                """,
                (document_id,),
            ).fetchone()
            if document is None:
                raise KeyError("document_not_found")
            chapters = connection.execute(
                """
                SELECT c.*,
                       (SELECT COUNT(*) FROM chapter_chunks cc WHERE cc.chapter_id = c.id) AS chunk_count,
                       (SELECT COUNT(*) FROM chapter_chunks cc WHERE cc.chapter_id = c.id
                        AND cc.facts_status != 'completed') AS pending_fact_count
                FROM chapters c WHERE c.document_id = ? ORDER BY c.position
                """,
                (document_id,),
            ).fetchall()
            characters = connection.execute(
                "SELECT * FROM document_characters WHERE document_id = ? ORDER BY name",
                (document_id,),
            ).fetchall()
            facts = connection.execute(
                """
                SELECT sf.*, first_chapter.title AS first_chapter,
                       last_chapter.title AS last_chapter
                FROM story_facts sf
                LEFT JOIN chapters first_chapter ON first_chapter.id = sf.first_chapter_id
                LEFT JOIN chapters last_chapter ON last_chapter.id = sf.last_chapter_id
                WHERE sf.document_id = ? ORDER BY sf.fact_type, sf.updated_at DESC
                """,
                (document_id,),
            ).fetchall()
            job = connection.execute(
                "SELECT * FROM analysis_jobs WHERE document_id = ? ORDER BY created_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
        return {
            **self._document(document),
            "chapters": [self._chapter(row) for row in chapters],
            "characters": [self._character(row) for row in characters],
            "facts": [dict(row) for row in facts],
            "latest_job": dict(job) if job else None,
        }

    def _chapter(self, row: Any, *, include_content: bool = False) -> dict[str, Any]:
        summary = json_load(row["summary_json"], {})
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "project_id": row["project_id"],
            "position": row["position"],
            "title": row["title"],
            "content": row["content"] if include_content else "",
            "content_hash": row["content_hash"] if "content_hash" in row.keys() else "",
            "content_preview": row["content"][:320],
            "character_count": len(row["content"]),
            "chunk_count": row["chunk_count"] if "chunk_count" in row.keys() else 0,
            "pending_fact_count": row["pending_fact_count"] if "pending_fact_count" in row.keys() else 0,
            "summary": summary,
            "character_observations": json_load(row["character_observations_json"], []),
            "summary_text": row["edited_summary"] or format_chapter_summary(summary),
            "edited_summary": row["edited_summary"],
            "status": row["status"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _character(self, row: Any) -> dict[str, Any]:
        aliases = _as_string_list(json_load(row["aliases_json"], []))
        card = json_load(row["card_json"], {})
        keys = set(row.keys())
        return {
            "id": row["id"],
            "project_id": row["project_id"] if "project_id" in keys else None,
            "document_id": row["document_id"] if "document_id" in keys else None,
            "name": row["name"],
            "aliases": aliases,
            "card": card,
            "prompt_text": row["prompt_text"] or format_character_card(row["name"], card, aliases),
            "source_chapters": _as_string_list(json_load(row["source_chapters_json"], [])),
            "enabled": bool(row["enabled"]),
            "updated_at": row["updated_at"],
        }

    def update_project(self, project_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "global_summary", "summary_enabled"}
        assignments = []
        values: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(int(value) if key == "summary_enabled" else value)
        if assignments:
            assignments.append("updated_at = ?")
            values.extend([utc_now(), project_id])
            with self.database.connect() as connection:
                cursor = connection.execute(
                    f"UPDATE projects SET {', '.join(assignments)} WHERE id = ?", values
                )
                if cursor.rowcount == 0:
                    raise KeyError("project_not_found")
        return self.get_project(project_id)

    def update_document(self, document_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "filename", "global_summary", "library_enabled", "summary_enabled",
            "recent_chapters_enabled", "characters_enabled", "facts_enabled",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key not in allowed:
                continue
            assignments.append(f"{key} = ?")
            values.append(int(value) if key.endswith("_enabled") else value)
        if assignments:
            values.append(document_id)
            with self.database.connect() as connection:
                cursor = connection.execute(
                    f"UPDATE source_documents SET {', '.join(assignments)} WHERE id = ?",
                    values,
                )
                if cursor.rowcount == 0:
                    raise KeyError("document_not_found")
        return self.get_document_workspace(document_id)

    def import_document(
        self, project_id: str, filename: str, encoding: str, text: str
    ) -> dict[str, Any]:
        document_id = new_id()
        now = utc_now()
        parts = split_chapters(text)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
                connection.rollback()
                raise KeyError("project_not_found")
            base_position = 0
            connection.execute(
                """
                INSERT INTO source_documents
                    (id, project_id, filename, encoding, raw_text, raw_text_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (document_id, project_id, filename, encoding, text, stable_text_hash(text), now),
            )
            for offset, part in enumerate(parts, start=1):
                chapter_id = new_id()
                connection.execute(
                    """
                    INSERT INTO chapters
                        (id, document_id, project_id, position, title, content,
                         content_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chapter_id, document_id, project_id, base_position + offset,
                        part.title, part.content, stable_text_hash(part.content), now, now,
                    ),
                )
                for chunk_position, chunk in enumerate(split_long_text(part.content), start=1):
                    connection.execute(
                        """
                        INSERT INTO chapter_chunks
                            (id, chapter_id, position, content, content_hash, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id(), chapter_id, chunk_position, chunk,
                            stable_text_hash(chunk), now, now,
                        ),
                    )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
            connection.execute(
                "UPDATE conversations SET document_id = ? WHERE project_id = ? AND document_id IS NULL",
                (document_id, project_id),
            )
            connection.commit()
        workspace = self.get_document_workspace(document_id)
        return {
            "document": {key: value for key, value in workspace.items() if key not in {"chapters", "characters", "facts", "latest_job"}},
            "chapters": workspace["chapters"],
        }

    def get_chapter(self, chapter_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT c.*,
                       (SELECT COUNT(*) FROM chapter_chunks cc WHERE cc.chapter_id = c.id) AS chunk_count,
                       (SELECT COUNT(*) FROM chapter_chunks cc WHERE cc.chapter_id = c.id
                        AND cc.facts_status != 'completed') AS pending_fact_count
                FROM chapters c WHERE c.id = ?
                """,
                (chapter_id,),
            ).fetchone()
            chunk_rows = connection.execute(
                "SELECT * FROM chapter_chunks WHERE chapter_id = ? ORDER BY position",
                (chapter_id,),
            ).fetchall()
        if row is None:
            raise KeyError("chapter_not_found")
        chapter = self._chapter(row, include_content=True)
        chapter["chunks"] = [
            {
                "id": chunk["id"],
                "position": chunk["position"],
                "content": chunk["content"],
                "content_hash": chunk["content_hash"],
                "summary": json_load(chunk["summary_json"], {}),
                "character_observations": json_load(chunk["character_observations_json"], []),
                "status": chunk["status"],
                "facts_status": chunk["facts_status"],
                "error_message": chunk["error_message"],
            }
            for chunk in chunk_rows
        ]
        return chapter

    def update_chapter(self, chapter_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"title", "content", "edited_summary"}
        assignments = []
        values: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(value)
                if key == "content":
                    assignments.append("content_hash = ?")
                    values.append(stable_text_hash(value))
        if assignments:
            assignments.append("updated_at = ?")
            values.extend([utc_now(), chapter_id])
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    f"UPDATE chapters SET {', '.join(assignments)} WHERE id = ?", values
                )
                if cursor.rowcount == 0:
                    connection.rollback()
                    raise KeyError("chapter_not_found")
                if "content" in changes:
                    connection.execute("DELETE FROM chapter_chunks WHERE chapter_id = ?", (chapter_id,))
                    now = utc_now()
                    for position, chunk in enumerate(split_long_text(changes["content"]), start=1):
                        connection.execute(
                            """
                            INSERT INTO chapter_chunks
                                (id, chapter_id, position, content, content_hash, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_id(), chapter_id, position, chunk,
                                stable_text_hash(chunk), now, now,
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE chapters SET summary_json = '', edited_summary = '',
                            character_observations_json = '[]',
                            status = 'pending', error_message = NULL, updated_at = ?
                        WHERE id = ?
                        """,
                        (now, chapter_id),
                    )
                connection.commit()
        return self.get_chapter(chapter_id)

    def set_chapter_status(self, chapter_id: str, status: str, error: str | None = None) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE chapters SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
                (status, error, utc_now(), chapter_id),
            )
            if status in {"pending", "failed"}:
                connection.execute(
                    """
                    UPDATE chapter_chunks SET status = ?, error_message = ?, updated_at = ?
                    WHERE chapter_id = ? AND status = 'processing'
                    """,
                    (status, error, utc_now(), chapter_id),
                )

    def save_chunk_summaries(
        self,
        chapter_id: str,
        summaries: list[dict[str, Any]],
        start_position: int = 1,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id FROM chapter_chunks WHERE chapter_id = ? AND position >= ? ORDER BY position",
                (chapter_id, start_position),
            ).fetchall()
            for row, summary in zip(rows, summaries, strict=False):
                connection.execute(
                    """
                    UPDATE chapter_chunks SET summary_json = ?, status = 'completed',
                        error_message = NULL, updated_at = ? WHERE id = ?
                    """,
                    (json.dumps(summary, ensure_ascii=False), utc_now(), row["id"]),
                )
            connection.commit()

    def append_content(
        self,
        project_id: str,
        content: str,
        *,
        chapter_id: str | None = None,
        document_id: str | None = None,
        title: str | None = None,
        source_candidate_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        content = content.strip()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM projects WHERE id = ?", (project_id,)
            ).fetchone() is None:
                connection.rollback()
                raise KeyError("project_not_found")
            if source_candidate_id and connection.execute(
                "SELECT 1 FROM library_increments WHERE source_candidate_id = ?",
                (source_candidate_id,),
            ).fetchone():
                connection.rollback()
                raise ValueError("candidate_already_appended")

            previous_summary = ""
            if chapter_id:
                chapter = connection.execute(
                    "SELECT * FROM chapters WHERE id = ? AND project_id = ?",
                    (chapter_id, project_id),
                ).fetchone()
                if chapter is None:
                    connection.rollback()
                    raise KeyError("chapter_not_found")
                previous_summary = chapter["edited_summary"] or format_chapter_summary(
                    json_load(chapter["summary_json"], {})
                )
                document_id = chapter["document_id"]
                separator = "\n\n" if chapter["content"].strip() else ""
                updated_content = f"{chapter['content']}{separator}{content}"
                connection.execute(
                    """
                    UPDATE chapters SET content = ?, content_hash = ?, status = 'pending',
                        error_message = NULL, updated_at = ? WHERE id = ?
                    """,
                    (updated_content, stable_text_hash(updated_content), now, chapter_id),
                )
                chunk_start = connection.execute(
                    "SELECT COALESCE(MAX(position), 0) + 1 FROM chapter_chunks WHERE chapter_id = ?",
                    (chapter_id,),
                ).fetchone()[0]
            else:
                if not document_id:
                    connection.rollback()
                    raise ValueError("document_required")
                document = connection.execute(
                    "SELECT 1 FROM source_documents WHERE id = ? AND project_id = ?",
                    (document_id, project_id),
                ).fetchone()
                if document is None:
                    connection.rollback()
                    raise KeyError("document_not_found")
                chapter_id = new_id()
                chapter_title = (title or "新增章节").strip() or "新增章节"
                chapter_position = connection.execute(
                    "SELECT COALESCE(MAX(position), 0) + 1 FROM chapters WHERE document_id = ?",
                    (document_id,),
                ).fetchone()[0]
                connection.execute(
                    """
                    INSERT INTO chapters
                        (id, document_id, project_id, position, title, content,
                         content_hash, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        chapter_id, document_id, project_id, chapter_position,
                        chapter_title, content, stable_text_hash(content), now, now,
                    ),
                )
                chunk_start = 1

            for offset, chunk in enumerate(split_long_text(content)):
                connection.execute(
                    """
                    INSERT INTO chapter_chunks
                        (id, chapter_id, position, content, content_hash, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        new_id(), chapter_id, chunk_start + offset, chunk,
                        stable_text_hash(chunk), now, now,
                    ),
                )
            connection.execute(
                """
                INSERT INTO library_increments
                    (id, project_id, chapter_id, source_candidate_id, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_id(), project_id, chapter_id, source_candidate_id, content, now),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id)
            )
            connection.commit()
        return {
            "chapter": self.get_chapter(chapter_id),
            "document_id": document_id,
            "previous_summary": previous_summary,
            "chunk_start_position": chunk_start,
            "new_content": content,
        }

    def mark_increment_failed(
        self, chapter_id: str, start_position: int, error: str
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE chapters SET status = 'failed', error_message = ?, updated_at = ? WHERE id = ?",
                (error, utc_now(), chapter_id),
            )
            connection.execute(
                """
                UPDATE chapter_chunks SET status = 'failed', error_message = ?, updated_at = ?
                    , facts_status = CASE WHEN facts_status = 'processing' THEN 'failed' ELSE facts_status END
                WHERE chapter_id = ? AND position >= ?
                """,
                (error, utc_now(), chapter_id, start_position),
            )

    def save_chapter_summary(
        self,
        chapter_id: str,
        summary: dict[str, Any],
        character_observations: list[dict[str, Any]] | None = None,
        *,
        append_observations: bool = False,
    ) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if character_observations is not None and append_observations:
                row = connection.execute(
                    "SELECT character_observations_json FROM chapters WHERE id = ?",
                    (chapter_id,),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    raise KeyError("chapter_not_found")
                character_observations = [
                    *json_load(row["character_observations_json"], []),
                    *character_observations,
                ]
            connection.execute(
                """
                UPDATE chapters SET summary_json = ?, edited_summary = '', status = 'completed',
                    character_observations_json = COALESCE(?, character_observations_json),
                    error_message = NULL, updated_at = ? WHERE id = ?
                """,
                (
                    json.dumps(summary, ensure_ascii=False),
                    json.dumps(character_observations, ensure_ascii=False)
                    if character_observations is not None
                    else None,
                    utc_now(),
                    chapter_id,
                ),
            )
            connection.execute(
                """
                UPDATE chapter_chunks SET status = 'completed', error_message = NULL, updated_at = ?
                WHERE chapter_id = ? AND status IN ('processing', 'pending')
                """,
                (utc_now(), chapter_id),
            )
            connection.commit()
        return self.get_chapter(chapter_id)

    def _character_item_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "document_id": row["document_id"] if "document_id" in row.keys() else None,
            "name": row["name"],
            "aliases": _as_string_list(json_load(row["aliases_json"], [])),
            "card": json_load(row["card_json"], {}),
            "prompt_text": row["prompt_text"],
            "source_chapters": _as_string_list(json_load(row["source_chapters_json"], [])),
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _character_index(
        self, rows: list[Any]
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        rows_by_id = {row["id"]: row for row in rows}
        token_to_ids: dict[str, list[str]] = {}
        for row in rows:
            for token in _standard_name_tokens(row["name"]):
                token_to_ids.setdefault(token, [])
                if row["id"] not in token_to_ids[token]:
                    token_to_ids[token].append(row["id"])
        return rows_by_id, token_to_ids

    def _matching_character_ids(
        self,
        item: dict[str, Any],
        rows_by_id: dict[str, Any],
        token_to_ids: dict[str, list[str]],
    ) -> list[str]:
        explicit_id = str(item.get("id") or "").strip()
        if explicit_id and explicit_id in rows_by_id:
            row = rows_by_id[explicit_id]
            return [explicit_id] if _standard_names_match(row["name"], item.get("name")) else []
        tokens = _standard_name_tokens(item.get("name"))
        matches: list[str] = []
        for token in tokens:
            for row_id in token_to_ids.get(token, []):
                if row_id not in matches:
                    matches.append(row_id)
        return matches

    def _write_character_item(
        self,
        connection: Any,
        document_id: str,
        item: dict[str, Any],
        *,
        existing_id: str | None = None,
        duplicate_ids: list[str] | None = None,
        now: str | None = None,
    ) -> str:
        now = now or utc_now()
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError("character_name_required")
        aliases = _unique_strings([
            alias for alias in _as_string_list(item.get("aliases"))
            if normalize_character_key(alias) != normalize_character_key(name)
        ])
        sources = _as_string_list(item.get("source_chapters"))
        card = _character_payload(item)
        if existing_id:
            connection.execute(
                """
                UPDATE document_characters SET name = ?, aliases_json = ?, card_json = ?,
                    source_chapters_json = ?, updated_at = ? WHERE id = ?
                """,
                (
                    name,
                    json.dumps(aliases, ensure_ascii=False),
                    json.dumps(card, ensure_ascii=False),
                    json.dumps(sources, ensure_ascii=False),
                    now,
                    existing_id,
                ),
            )
            character_id = existing_id
        else:
            character_id = new_id()
            connection.execute(
                """
                INSERT INTO document_characters
                    (id, document_id, name, aliases_json, card_json, prompt_text,
                     source_chapters_json, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, '', ?, 1, ?, ?)
                """,
                (
                    character_id,
                    document_id,
                    name,
                    json.dumps(aliases, ensure_ascii=False),
                    json.dumps(card, ensure_ascii=False),
                    json.dumps(sources, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        for duplicate_id in duplicate_ids or []:
            if duplicate_id != character_id:
                connection.execute(
                    "DELETE FROM document_characters WHERE id = ? AND document_id = ?",
                    (duplicate_id, document_id),
                )
        return character_id

    def get_document_character_observations(
        self, document_id: str
    ) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT title, character_observations_json
                FROM chapters
                WHERE document_id = ? AND status = 'completed'
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()
        observations: list[dict[str, Any]] = []
        for row in rows:
            for item in json_load(row["character_observations_json"], []):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                sources = _as_string_list(item.get("source_chapters")) or [row["title"]]
                observations.append({**item, "source_chapters": sources})
        return observations

    def get_relevant_character_cards(
        self, document_id: str, observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not observations:
            return []
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM document_characters WHERE document_id = ? ORDER BY name",
                (document_id,),
            ).fetchall()
        rows_by_id, token_to_ids = self._character_index(rows)
        matched_ids: list[str] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            for row_id in self._matching_character_ids(item, rows_by_id, token_to_ids):
                if row_id not in matched_ids:
                    matched_ids.append(row_id)
        return [self._character(rows_by_id[row_id]) for row_id in matched_ids if row_id in rows_by_id]

    def save_document_summary(self, document_id: str, summary: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE source_documents SET global_summary = ? WHERE id = ?",
                (summary, document_id),
            )

    def _invalidate_document_derived(self, connection: Any, document_id: str) -> None:
        connection.execute(
            "UPDATE source_documents SET global_summary = '' WHERE id = ?",
            (document_id,),
        )
        connection.execute("DELETE FROM document_characters WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM story_facts WHERE document_id = ?", (document_id,))

    def delete_document(self, document_id: str) -> str:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT project_id FROM source_documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("document_not_found")
            connection.execute(
                "UPDATE conversations SET document_id = NULL WHERE document_id = ?",
                (document_id,),
            )
            connection.execute("DELETE FROM source_documents WHERE id = ?", (document_id,))
            connection.commit()
        return row["project_id"]

    def delete_chapter(self, chapter_id: str) -> str:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT project_id, document_id FROM chapters WHERE id = ?", (chapter_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("chapter_not_found")
            connection.execute("DELETE FROM chapters WHERE id = ?", (chapter_id,))
            remaining = connection.execute(
                "SELECT COUNT(*) FROM chapters WHERE document_id = ?", (row["document_id"],)
            ).fetchone()[0]
            self._invalidate_document_derived(connection, row["document_id"])
            connection.commit()
        return row["document_id"]

    def delete_character(self, character_id: str) -> str:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT document_id FROM document_characters WHERE id = ?", (character_id,)
            ).fetchone()
            if row is None:
                raise KeyError("character_not_found")
            connection.execute("DELETE FROM document_characters WHERE id = ?", (character_id,))
        return row["document_id"]

    def clear_project_library(self, project_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM projects WHERE id = ?", (project_id,)
            ).fetchone() is None:
                connection.rollback()
                raise KeyError("project_not_found")
            connection.execute("DELETE FROM source_documents WHERE project_id = ?", (project_id,))
            connection.execute(
                "UPDATE conversations SET document_id = NULL WHERE project_id = ?",
                (project_id,),
            )
            connection.execute(
                "UPDATE projects SET global_summary = '', updated_at = ? WHERE id = ?",
                (utc_now(), project_id),
            )
            connection.commit()

    def export_project_text(self, project_id: str) -> tuple[str, str]:
        project = self.get_project(project_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT title, content FROM chapters WHERE project_id = ? ORDER BY position",
                (project_id,),
            ).fetchall()
        sections = []
        for row in rows:
            title = str(row["title"] or "").strip()
            content = str(row["content"] or "").strip()
            sections.append(f"{title}\n\n{content}".strip())
        return project["name"], "\n\n\n".join(sections).rstrip() + ("\n" if sections else "")

    def export_document_text(self, document_id: str) -> tuple[str, str]:
        workspace = self.get_document_workspace(document_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT title, content FROM chapters WHERE document_id = ? ORDER BY position",
                (document_id,),
            ).fetchall()
        sections = [f"{row['title']}\n\n{row['content']}".strip() for row in rows]
        filename = workspace["filename"].rsplit(".", 1)[0]
        return filename, "\n\n\n".join(sections).rstrip() + ("\n" if sections else "")

    def replace_characters(self, document_id: str, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_rows = connection.execute(
                "SELECT * FROM document_characters WHERE document_id = ? ORDER BY created_at, name",
                (document_id,),
            ).fetchall()
            rows_by_id, token_to_ids = self._character_index(existing_rows)
            pending_by_token: dict[str, str] = {}
            written_items: dict[str, dict[str, Any]] = {}
            for item in cards:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                matches = self._matching_character_ids(item, rows_by_id, token_to_ids)
                tokens = _standard_name_tokens(name)
                pending_match = next(
                    (pending_by_token[token] for token in tokens if token in pending_by_token),
                    None,
                )
                if matches:
                    primary_id = matches[0]
                    base = self._character_item_from_row(rows_by_id[primary_id])
                    for duplicate_id in matches[1:]:
                        base = _merge_character_item(
                            base,
                            self._character_item_from_row(rows_by_id[duplicate_id]),
                        )
                    merged = _merge_character_item(base, item)
                    character_id = self._write_character_item(
                        connection,
                        document_id,
                        merged,
                        existing_id=primary_id,
                        duplicate_ids=matches[1:],
                        now=now,
                    )
                elif pending_match:
                    merged = _merge_character_item(written_items[pending_match], item)
                    character_id = self._write_character_item(
                        connection,
                        document_id,
                        merged,
                        existing_id=pending_match,
                        now=now,
                    )
                else:
                    character_id = self._write_character_item(
                        connection, document_id, item, now=now
                    )
                    merged = {**item, "id": character_id}
                persisted = connection.execute(
                    "SELECT * FROM document_characters WHERE id = ?", (character_id,)
                ).fetchone()
                if persisted:
                    rows_by_id[character_id] = persisted
                    written_items[character_id] = self._character_item_from_row(persisted)
                    for token in _standard_name_tokens(persisted["name"]):
                        token_to_ids.setdefault(token, [])
                        if character_id not in token_to_ids[token]:
                            token_to_ids[token].append(character_id)
                        pending_by_token[token] = character_id
                    for duplicate_id in (matches[1:] if matches else []):
                        rows_by_id.pop(duplicate_id, None)
                        for indexed_ids in token_to_ids.values():
                            if duplicate_id in indexed_ids:
                                indexed_ids.remove(duplicate_id)
            connection.commit()
        return self.get_document_workspace(document_id)["characters"]

    def update_character(self, character_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "aliases", "card", "prompt_text", "enabled"}
        assignments = []
        values: list[Any] = []
        for key, value in changes.items():
            if key not in allowed:
                continue
            column = {"aliases": "aliases_json", "card": "card_json"}.get(key, key)
            assignments.append(f"{column} = ?")
            if key in {"aliases", "card"}:
                values.append(json.dumps(value, ensure_ascii=False))
            elif key == "enabled":
                values.append(int(value))
            else:
                values.append(value)
        assignments.append("updated_at = ?")
        values.append(utc_now())
        values.append(character_id)
        with self.database.connect() as connection:
            cursor = connection.execute(
                f"UPDATE document_characters SET {', '.join(assignments)} WHERE id = ?", values
            )
            row = connection.execute("SELECT * FROM document_characters WHERE id = ?", (character_id,)).fetchone()
        if cursor.rowcount == 0 or row is None:
            raise KeyError("character_not_found")
        return self._character(row)

    def save_chunk_analysis(
        self,
        chunk_id: str,
        summary: dict[str, Any],
        character_observations: list[dict[str, Any]],
        *,
        completed: bool = True,
    ) -> None:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE chapter_chunks SET summary_json = ?, character_observations_json = ?,
                    status = ?, error_message = NULL, updated_at = ? WHERE id = ?
                """,
                (
                    json.dumps(summary, ensure_ascii=False),
                    json.dumps(character_observations, ensure_ascii=False),
                    "completed" if completed else "processing", utc_now(), chunk_id,
                ),
            )
        if cursor.rowcount == 0:
            raise KeyError("chunk_not_found")

    def set_chunk_status(self, chunk_id: str, status: str, error: str | None = None) -> None:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE chapter_chunks SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
                (status, error, utc_now(), chunk_id),
            )
        if cursor.rowcount == 0:
            raise KeyError("chunk_not_found")

    def set_chunk_facts_status(self, chunk_id: str, status: str) -> None:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE chapter_chunks SET facts_status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), chunk_id),
            )
        if cursor.rowcount == 0:
            raise KeyError("chunk_not_found")

    def reset_chapter_analysis(self, chapter_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chunk_ids = [
                row["id"] for row in connection.execute(
                    "SELECT id FROM chapter_chunks WHERE chapter_id = ?", (chapter_id,)
                ).fetchall()
            ]
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                connection.execute(
                    f"DELETE FROM fact_sources WHERE chunk_id IN ({placeholders})", chunk_ids
                )
            connection.execute(
                """
                UPDATE chapter_chunks SET summary_json = '', character_observations_json = '[]',
                    facts_status = 'pending', status = 'pending', error_message = NULL,
                    updated_at = ? WHERE chapter_id = ?
                """,
                (utc_now(), chapter_id),
            )
            connection.execute(
                """
                UPDATE chapters SET summary_json = '', character_observations_json = '[]',
                    edited_summary = '', status = 'pending', error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), chapter_id),
            )
            connection.execute(
                "DELETE FROM story_facts WHERE id NOT IN (SELECT DISTINCT fact_id FROM fact_sources)"
            )
            connection.commit()

    def save_story_facts(
        self,
        document_id: str,
        chapter_id: str,
        chunk_id: str,
        facts: list[dict[str, Any]],
    ) -> None:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for item in facts:
                fact_type = str(item.get("fact_type") or item.get("type") or "other").strip()
                subject = str(item.get("subject") or "").strip()
                predicate = str(item.get("predicate") or "").strip()
                obj = str(item.get("object") or "").strip()
                state = str(item.get("state") or "").strip()
                raw_key = str(item.get("fact_key") or "").strip()
                if not raw_key:
                    basis = "|".join([fact_type, subject, predicate, obj if fact_type == "timeline" else ""])
                    raw_key = hashlib.sha1(basis.encode("utf-8")).hexdigest()
                fact_id = new_id()
                existing = connection.execute(
                    "SELECT id FROM story_facts WHERE document_id = ? AND fact_key = ?",
                    (document_id, raw_key),
                ).fetchone()
                if existing:
                    fact_id = existing["id"]
                    connection.execute(
                        """
                        UPDATE story_facts SET fact_type = ?, subject = ?, predicate = ?,
                            object = ?, state = ?, status = ?, event_time = ?,
                            last_chapter_id = ?, confidence = ?, updated_at = ? WHERE id = ?
                        """,
                        (
                            fact_type, subject, predicate, obj, state,
                            str(item.get("status") or "active"),
                            str(item.get("event_time") or ""), chapter_id,
                            float(item.get("confidence") or 0.7), now, fact_id,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO story_facts
                            (id, document_id, fact_key, fact_type, subject, predicate,
                             object, state, status, event_time, first_chapter_id,
                             last_chapter_id, confidence, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fact_id, document_id, raw_key, fact_type, subject, predicate,
                            obj, state, str(item.get("status") or "active"),
                            str(item.get("event_time") or ""), chapter_id, chapter_id,
                            float(item.get("confidence") or 0.7), now, now,
                        ),
                    )
                evidence = str(item.get("evidence") or "").strip()[:2000]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fact_sources
                        (id, fact_id, chapter_id, chunk_id, evidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (new_id(), fact_id, chapter_id, chunk_id, evidence, now),
                )
            connection.execute(
                "UPDATE chapter_chunks SET facts_status = 'completed', updated_at = ? WHERE id = ?",
                (now, chunk_id),
            )
            connection.commit()

    def relevant_story_facts(
        self, document_id: str, query_text: str, limit: int = 40
    ) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT sf.*, c1.title AS first_chapter, c2.title AS last_chapter
                FROM story_facts sf
                LEFT JOIN chapters c1 ON c1.id = sf.first_chapter_id
                LEFT JOIN chapters c2 ON c2.id = sf.last_chapter_id
                WHERE sf.document_id = ?
                """,
                (document_id,),
            ).fetchall()
        terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", query_text))
        scored = []
        for row in rows:
            item = dict(row)
            text = " ".join(str(item.get(key, "")) for key in ("subject", "predicate", "object", "state"))
            overlap = sum(1 for term in terms if term in text)
            score = overlap * 10
            if item["fact_type"] == "foreshadowing" and item["status"] in {"active", "open"}:
                score += 20
            if item["status"] in {"active", "open"}:
                score += 3
            score += float(item.get("confidence") or 0)
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1]["updated_at"]), reverse=True)
        return [item for _score, item in scored[:limit]]

    def update_story_fact(self, fact_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"state", "status"}
        assignments, values = [], []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(value)
        assignments.append("updated_at = ?")
        values.extend([utc_now(), fact_id])
        with self.database.connect() as connection:
            cursor = connection.execute(
                f"UPDATE story_facts SET {', '.join(assignments)} WHERE id = ?", values
            )
            row = connection.execute("SELECT * FROM story_facts WHERE id = ?", (fact_id,)).fetchone()
        if cursor.rowcount == 0 or row is None:
            raise KeyError("fact_not_found")
        return dict(row)

    def delete_story_fact(self, fact_id: str) -> str:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT document_id FROM story_facts WHERE id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                raise KeyError("fact_not_found")
            connection.execute("DELETE FROM story_facts WHERE id = ?", (fact_id,))
        return row["document_id"]

    def create_analysis_job(
        self, document_id: str, start_position: int, end_position: int,
        total_chapters: int, regenerate: bool, max_tokens: int,
    ) -> dict[str, Any]:
        job_id, now = new_id(), utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_jobs
                    (id, document_id, start_position, end_position, total_chapters,
                     status, regenerate, max_tokens, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (job_id, document_id, start_position, end_position, total_chapters,
                 int(regenerate), max_tokens, now, now),
            )
        return self.get_analysis_job(job_id)

    def get_analysis_job(self, job_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("analysis_job_not_found")
        return dict(row)

    def update_analysis_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {"current_chapter_id", "current_chunk_position", "processed_chapters", "status", "error_message"}
        assignments, values = [], []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(value)
        assignments.append("updated_at = ?")
        values.extend([utc_now(), job_id])
        with self.database.connect() as connection:
            connection.execute(
                f"UPDATE analysis_jobs SET {', '.join(assignments)} WHERE id = ?", values
            )
        return self.get_analysis_job(job_id)

    def get_prompt_context(
        self, conversation_id: str, *, include_outline: bool = True, query_text: str = ""
    ) -> dict[str, str]:
        with self.database.connect() as connection:
            conversation = connection.execute(
                "SELECT project_id, document_id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation is None:
                raise KeyError("conversation_not_found")
            document = None
            if conversation["document_id"]:
                document = connection.execute(
                    "SELECT * FROM source_documents WHERE id = ?",
                    (conversation["document_id"],),
                ).fetchone()
            library_enabled = bool(document and document["library_enabled"])
            recent_chapters = connection.execute(
                """
                SELECT title, summary_json, edited_summary FROM chapters
                WHERE document_id = ? AND status = 'completed'
                ORDER BY position DESC LIMIT 4
                """,
                (document["id"] if document else "",),
            ).fetchall() if library_enabled and document["recent_chapters_enabled"] else []
            characters = connection.execute(
                "SELECT * FROM document_characters WHERE document_id = ? AND enabled = 1 ORDER BY name",
                (document["id"],),
            ).fetchall() if library_enabled and document and document["characters_enabled"] else []
            outline_text = ""
            if include_outline:
                outline = connection.execute(
                    """
                    SELECT o.selected_candidate_id, oc.content, oc.edited_content
                    FROM outlines o
                    LEFT JOIN outline_candidates oc ON oc.id = o.selected_candidate_id
                    WHERE o.conversation_id = ? AND o.enabled = 1
                    ORDER BY o.created_at DESC LIMIT 1
                    """,
                    (conversation_id,),
                ).fetchone()
                if outline:
                    outline_text = outline["edited_content"] or outline["content"] or ""
        chapter_texts = []
        for row in reversed(recent_chapters):
            summary = json_load(row["summary_json"], {})
            chapter_texts.append(row["edited_summary"] or format_chapter_summary(summary))
        character_texts = []
        for row in characters:
            character_texts.append(
                row["prompt_text"]
                or format_character_card(
                    row["name"], json_load(row["card_json"], {}), json_load(row["aliases_json"], [])
                )
            )
        facts = (
            self.relevant_story_facts(document["id"], query_text)
            if library_enabled and document and document["facts_enabled"]
            else []
        )
        fact_lines = []
        for item in facts:
            source = f"首次：{item.get('first_chapter') or '未知'}；最近：{item.get('last_chapter') or '未知'}"
            fact_lines.append(
                f"- [{item['fact_type']}/{item['status']}] {item['subject']} {item['predicate']} "
                f"{item['object']}；状态：{item['state']}；{source}"
            )
        return {
            "project_summary": document["global_summary"] if library_enabled and document and document["summary_enabled"] else "",
            "recent_chapters": "\n\n".join(text for text in chapter_texts if text),
            "characters": "\n\n".join(text for text in character_texts if text),
            "facts": "\n".join(fact_lines),
            "outline": outline_text,
        }

    def get_or_create_outline(
        self,
        conversation_id: str,
        instruction: str | None = None,
        *,
        force_new: bool = False,
    ) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = None
            if not force_new:
                row = connection.execute(
                    "SELECT id FROM outlines WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
                    (conversation_id,),
                ).fetchone()
            if row is None:
                outline_id = new_id()
                now = utc_now()
                if force_new:
                    connection.execute(
                        "UPDATE outlines SET enabled = 0, updated_at = ? WHERE conversation_id = ?",
                        (now, conversation_id),
                    )
                connection.execute(
                    """
                    INSERT INTO outlines
                        (id, conversation_id, instruction, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (
                        outline_id,
                        conversation_id,
                        instruction or "请规划紧接当前进度的下一章。",
                        now,
                        now,
                    ),
                )
            else:
                outline_id = row["id"]
                if instruction:
                    connection.execute(
                        "UPDATE outlines SET instruction = ?, updated_at = ? WHERE id = ?",
                        (instruction, utc_now(), outline_id),
                    )
        return self.get_outline(outline_id)

    def find_latest_outline(self, conversation_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT id FROM outlines WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return self.get_outline(row["id"]) if row else None

    def get_outline(self, outline_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            outline = connection.execute("SELECT * FROM outlines WHERE id = ?", (outline_id,)).fetchone()
            if outline is None:
                raise KeyError("outline_not_found")
            candidates = connection.execute(
                "SELECT * FROM outline_candidates WHERE outline_id = ? ORDER BY candidate_index",
                (outline_id,),
            ).fetchall()
        return {
            "id": outline["id"],
            "conversation_id": outline["conversation_id"],
            "instruction": outline["instruction"],
            "selected_candidate_id": outline["selected_candidate_id"],
            "enabled": bool(outline["enabled"]),
            "created_at": outline["created_at"],
            "updated_at": outline["updated_at"],
            "candidates": [
                {
                    "id": row["id"], "outline_id": row["outline_id"],
                    "candidate_index": row["candidate_index"], "content": row["content"],
                    "edited_content": row["edited_content"], "status": row["status"],
                    "settings_snapshot": json_load(row["settings_snapshot"], {}),
                    "seed": row["seed"], "error_message": row["error_message"],
                    "created_at": row["created_at"], "completed_at": row["completed_at"],
                }
                for row in candidates
            ],
        }

    def create_outline_candidate(
        self, outline_id: str, settings: dict[str, Any], seed: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_id = new_id()
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM outlines WHERE id = ?", (outline_id,)).fetchone() is None:
                connection.rollback()
                raise KeyError("outline_not_found")
            index = connection.execute(
                "SELECT COALESCE(MAX(candidate_index), 0) + 1 FROM outline_candidates WHERE outline_id = ?",
                (outline_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO outline_candidates
                    (id, outline_id, candidate_index, status, settings_snapshot, seed, created_at)
                VALUES (?, ?, ?, 'streaming', ?, ?, ?)
                """,
                (candidate_id, outline_id, index, json.dumps(settings, ensure_ascii=False), seed, now),
            )
            connection.execute("UPDATE outlines SET updated_at = ? WHERE id = ?", (now, outline_id))
            connection.commit()
        outline = self.get_outline(outline_id)
        return outline, next(item for item in outline["candidates"] if item["id"] == candidate_id)

    def save_outline_candidate(
        self,
        conversation_id: str,
        *,
        outline_id: str | None,
        instruction: str,
        content: str,
        settings: dict[str, Any],
        seed: int,
        select: bool = False,
    ) -> dict[str, Any]:
        outline = (
            self.get_outline(outline_id)
            if outline_id
            else self.get_or_create_outline(
                conversation_id, instruction, force_new=True
            )
        )
        if outline["conversation_id"] != conversation_id:
            raise ValueError("outline_conversation_mismatch")
        candidate_id = new_id()
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            index = connection.execute(
                "SELECT COALESCE(MAX(candidate_index), 0) + 1 FROM outline_candidates WHERE outline_id = ?",
                (outline["id"],),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO outline_candidates
                    (id, outline_id, candidate_index, content, edited_content, status,
                     settings_snapshot, seed, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    outline["id"],
                    index,
                    content,
                    content,
                    json.dumps(settings, ensure_ascii=False),
                    seed,
                    now,
                    now,
                ),
            )
            assignments = ["instruction = ?", "updated_at = ?"]
            values: list[Any] = [instruction, now]
            if select:
                assignments.append("selected_candidate_id = ?")
                values.append(candidate_id)
            values.append(outline["id"])
            connection.execute(
                f"UPDATE outlines SET {', '.join(assignments)} WHERE id = ?", values
            )
            connection.commit()
        return self.get_outline(outline["id"])

    def update_outline_draft(self, candidate_id: str, content: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE outline_candidates SET content = ? WHERE id = ? AND status = 'streaming'",
                (content, candidate_id),
            )

    def finalize_outline_candidate(
        self, candidate_id: str, status: str, content: str, error: str | None = None
    ) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute(
                "SELECT outline_id FROM outline_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if candidate is None:
                connection.rollback()
                raise KeyError("outline_candidate_not_found")
            now = utc_now()
            connection.execute(
                """
                UPDATE outline_candidates SET status = ?, content = ?, error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, content, error, now, candidate_id),
            )
            outline = connection.execute(
                "SELECT selected_candidate_id FROM outlines WHERE id = ?", (candidate["outline_id"],)
            ).fetchone()
            if status == "completed" and outline["selected_candidate_id"] is None:
                connection.execute(
                    "UPDATE outlines SET selected_candidate_id = ?, updated_at = ? WHERE id = ?",
                    (candidate_id, now, candidate["outline_id"]),
                )
            connection.commit()
        return self.get_outline(candidate["outline_id"])

    def update_outline(
        self,
        outline_id: str,
        enabled: bool | None = None,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        if enabled is not None or instruction is not None:
            assignments = ["updated_at = ?"]
            values: list[Any] = [utc_now()]
            if enabled is not None:
                assignments.append("enabled = ?")
                values.append(int(enabled))
            if instruction is not None:
                assignments.append("instruction = ?")
                values.append(instruction)
            values.append(outline_id)
            with self.database.connect() as connection:
                if enabled:
                    outline = connection.execute(
                        "SELECT conversation_id FROM outlines WHERE id = ?", (outline_id,)
                    ).fetchone()
                    if outline is None:
                        raise KeyError("outline_not_found")
                    connection.execute(
                        "UPDATE outlines SET enabled = 0, updated_at = ? WHERE conversation_id = ? AND id != ?",
                        (utc_now(), outline["conversation_id"], outline_id),
                    )
                connection.execute(
                    f"UPDATE outlines SET {', '.join(assignments)} WHERE id = ?",
                    values,
                )
        return self.get_outline(outline_id)

    def select_outline_candidate(self, outline_id: str, candidate_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            candidate = connection.execute(
                "SELECT 1 FROM outline_candidates WHERE id = ? AND outline_id = ? AND status = 'completed'",
                (candidate_id, outline_id),
            ).fetchone()
            if candidate is None:
                raise ValueError("outline_candidate_not_selectable")
            connection.execute(
                "UPDATE outlines SET selected_candidate_id = ?, updated_at = ? WHERE id = ?",
                (candidate_id, utc_now(), outline_id),
            )
        return self.get_outline(outline_id)

    def edit_outline_candidate(self, candidate_id: str, content: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT outline_id FROM outline_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise KeyError("outline_candidate_not_found")
            connection.execute(
                "UPDATE outline_candidates SET edited_content = ? WHERE id = ?",
                (content, candidate_id),
            )
        return self.get_outline(row["outline_id"])

    def delete_outline_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT outline_id FROM outline_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("outline_candidate_not_found")
            outline = connection.execute(
                "SELECT selected_candidate_id FROM outlines WHERE id = ?", (row["outline_id"],)
            ).fetchone()
            connection.execute("DELETE FROM outline_candidates WHERE id = ?", (candidate_id,))
            if outline and outline["selected_candidate_id"] == candidate_id:
                connection.execute(
                    "UPDATE outlines SET selected_candidate_id = NULL, enabled = 0, updated_at = ? WHERE id = ?",
                    (utc_now(), row["outline_id"]),
                )
            connection.commit()
        return self.get_outline(row["outline_id"])

    def delete_outline(self, outline_id: str) -> None:
        with self.database.connect() as connection:
            cursor = connection.execute("DELETE FROM outlines WHERE id = ?", (outline_id,))
        if cursor.rowcount == 0:
            raise KeyError("outline_not_found")
