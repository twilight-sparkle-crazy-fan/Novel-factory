from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .config import DEFAULT_GENERATION_SETTINGS, DEFAULT_SYSTEM_PROMPT
from .material_utils import stable_text_hash
from .text_import import split_long_text


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _generation_settings(value: str | None) -> dict[str, Any]:
    loaded = _json_loads(value, {})
    return {**DEFAULT_GENERATION_SETTINGS, **loaded} if isinstance(loaded, dict) else DEFAULT_GENERATION_SETTINGS.copy()


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    global_summary TEXT NOT NULL DEFAULT '',
                    summary_enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    pinned_context TEXT NOT NULL DEFAULT '',
                    style_guide TEXT NOT NULL DEFAULT '',
                    style_lexicon TEXT NOT NULL DEFAULT '',
                    generation_settings TEXT NOT NULL,
                    project_id TEXT,
                    document_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS exchanges (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    user_content TEXT NOT NULL,
                    selected_candidate_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(conversation_id, position)
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    exchange_id TEXT NOT NULL REFERENCES exchanges(id) ON DELETE CASCADE,
                    candidate_index INTEGER NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    reasoning_content TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK(status IN ('streaming', 'completed', 'cancelled', 'failed')),
                    settings_snapshot TEXT NOT NULL,
                    seed INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    duration_ms INTEGER,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(exchange_id, candidate_index)
                );

                CREATE TABLE IF NOT EXISTS source_documents (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    encoding TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    raw_text_hash TEXT NOT NULL DEFAULT '',
                    global_summary TEXT NOT NULL DEFAULT '',
                    library_enabled INTEGER NOT NULL DEFAULT 1,
                    summary_enabled INTEGER NOT NULL DEFAULT 1,
                    recent_chapters_enabled INTEGER NOT NULL DEFAULT 1,
                    characters_enabled INTEGER NOT NULL DEFAULT 1,
                    facts_enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chapters (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL DEFAULT '',
                    summary_json TEXT NOT NULL DEFAULT '',
                    character_observations_json TEXT NOT NULL DEFAULT '[]',
                    edited_summary TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(document_id, position)
                );

                CREATE TABLE IF NOT EXISTS characters (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    card_json TEXT NOT NULL DEFAULT '{}',
                    prompt_text TEXT NOT NULL DEFAULT '',
                    source_chapters_json TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, name)
                );

                CREATE TABLE IF NOT EXISTS chapter_chunks (
                    id TEXT PRIMARY KEY,
                    chapter_id TEXT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL DEFAULT '',
                    summary_json TEXT NOT NULL DEFAULT '',
                    character_observations_json TEXT NOT NULL DEFAULT '[]',
                    facts_status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(facts_status IN ('pending', 'processing', 'completed', 'failed')),
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(chapter_id, position)
                );

                CREATE TABLE IF NOT EXISTS outlines (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    instruction TEXT NOT NULL DEFAULT '请规划紧接当前进度的下一章。',
                    selected_candidate_id TEXT,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS outline_candidates (
                    id TEXT PRIMARY KEY,
                    outline_id TEXT NOT NULL REFERENCES outlines(id) ON DELETE CASCADE,
                    candidate_index INTEGER NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    edited_content TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL
                        CHECK(status IN ('streaming', 'completed', 'cancelled', 'failed')),
                    settings_snapshot TEXT NOT NULL,
                    seed INTEGER,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(outline_id, candidate_index)
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS library_increments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    chapter_id TEXT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
                    source_candidate_id TEXT UNIQUE REFERENCES candidates(id) ON DELETE SET NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_characters (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    card_json TEXT NOT NULL DEFAULT '{}',
                    prompt_text TEXT NOT NULL DEFAULT '',
                    source_chapters_json TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(document_id, name)
                );

                CREATE TABLE IF NOT EXISTS story_facts (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    fact_key TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    predicate TEXT NOT NULL DEFAULT '',
                    object TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    event_time TEXT NOT NULL DEFAULT '',
                    first_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    last_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(document_id, fact_key)
                );

                CREATE TABLE IF NOT EXISTS fact_sources (
                    id TEXT PRIMARY KEY,
                    fact_id TEXT NOT NULL REFERENCES story_facts(id) ON DELETE CASCADE,
                    chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chapter_chunks(id) ON DELETE SET NULL,
                    evidence TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(fact_id, chapter_id, chunk_id, evidence)
                );

                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    start_position INTEGER NOT NULL,
                    end_position INTEGER NOT NULL,
                    current_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    current_chunk_position INTEGER NOT NULL DEFAULT 0,
                    processed_chapters INTEGER NOT NULL DEFAULT 0,
                    total_chapters INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'running', 'paused', 'completed', 'failed')),
                    regenerate INTEGER NOT NULL DEFAULT 0,
                    max_tokens INTEGER NOT NULL DEFAULT 8192,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS material_provenance (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL DEFAULT '',
                    analysis_version TEXT NOT NULL DEFAULT '',
                    prompt_version TEXT NOT NULL DEFAULT '',
                    model_id TEXT NOT NULL DEFAULT '',
                    generated_at TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.7
                );

                CREATE TABLE IF NOT EXISTS semantic_observations (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chapter_chunks(id) ON DELETE SET NULL,
                    observation_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    normalized_key TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    manually_edited INTEGER NOT NULL DEFAULT 0,
                    provenance_id TEXT REFERENCES material_provenance(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS timeline_nodes (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    parent_id TEXT,
                    node_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    start_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    end_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    position INTEGER NOT NULL DEFAULT 0,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_version TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    manually_edited INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS timeline_events (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chapter_chunks(id) ON DELETE SET NULL,
                    sequence INTEGER NOT NULL DEFAULT 0,
                    participants_json TEXT NOT NULL DEFAULT '[]',
                    location_id TEXT,
                    causes_json TEXT NOT NULL DEFAULT '[]',
                    consequences_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'active',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    provenance_id TEXT REFERENCES material_provenance(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS character_entities (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'person',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    manually_confirmed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(document_id, canonical_name)
                );

                CREATE TABLE IF NOT EXISTS character_aliases (
                    id TEXT PRIMARY KEY,
                    character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    alias TEXT NOT NULL,
                    alias_type TEXT NOT NULL DEFAULT 'name',
                    first_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    last_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    manually_confirmed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(character_id, alias)
                );

                CREATE TABLE IF NOT EXISTS character_profiles (
                    id TEXT PRIMARY KEY,
                    character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    start_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    end_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    identity TEXT NOT NULL DEFAULT '',
                    personality TEXT NOT NULL DEFAULT '',
                    goals TEXT NOT NULL DEFAULT '',
                    behavior_pattern TEXT NOT NULL DEFAULT '',
                    ability_stage TEXT NOT NULL DEFAULT '',
                    social_status TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    manually_edited INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS character_facts (
                    id TEXT PRIMARY KEY,
                    character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    field TEXT NOT NULL,
                    value TEXT NOT NULL,
                    valid_from_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    valid_to_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    certainty REAL NOT NULL DEFAULT 0.7,
                    provenance_id TEXT REFERENCES material_provenance(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS character_events (
                    id TEXT PRIMARY KEY,
                    character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    value TEXT NOT NULL DEFAULT '',
                    chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chapter_chunks(id) ON DELETE SET NULL,
                    sequence INTEGER NOT NULL DEFAULT 0,
                    provenance_id TEXT REFERENCES material_provenance(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relationship_events (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    source_character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    target_character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chapter_chunks(id) ON DELETE SET NULL,
                    sequence INTEGER NOT NULL DEFAULT 0,
                    strength_delta REAL NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    provenance_id TEXT REFERENCES material_provenance(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS character_relationships (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    source_character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    target_character_id TEXT NOT NULL REFERENCES character_entities(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    direction TEXT NOT NULL DEFAULT 'directed',
                    status TEXT NOT NULL DEFAULT 'active',
                    strength REAL NOT NULL DEFAULT 0.5,
                    start_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    end_chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    manually_edited INTEGER NOT NULL DEFAULT 0,
                    provenance_id TEXT REFERENCES material_provenance(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS material_review_items (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    review_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    resolution_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auxiliary_records (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    record_type TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    chapter_id TEXT REFERENCES chapters(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chapter_chunks(id) ON DELETE SET NULL,
                    sequence INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    manually_edited INTEGER NOT NULL DEFAULT 0,
                    provenance_id TEXT REFERENCES material_provenance(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompt_budget_profiles (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(deleted_at, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_exchanges_conversation
                    ON exchanges(conversation_id, position);
                CREATE INDEX IF NOT EXISTS idx_candidates_exchange
                    ON candidates(exchange_id, candidate_index);
                CREATE INDEX IF NOT EXISTS idx_documents_project
                    ON source_documents(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_chapters_project
                    ON chapters(project_id, position);
                CREATE INDEX IF NOT EXISTS idx_characters_project
                    ON characters(project_id, name);
                CREATE INDEX IF NOT EXISTS idx_chapter_chunks_chapter
                    ON chapter_chunks(chapter_id, position);
                CREATE INDEX IF NOT EXISTS idx_outlines_conversation
                    ON outlines(conversation_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_library_increments_chapter
                    ON library_increments(chapter_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_document_characters_document
                    ON document_characters(document_id, name);
                CREATE INDEX IF NOT EXISTS idx_story_facts_document
                    ON story_facts(document_id, fact_type, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_fact_sources_fact
                    ON fact_sources(fact_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_jobs_document
                    ON analysis_jobs(document_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_material_provenance_document
                    ON material_provenance(document_id, source_type, source_id);
                CREATE INDEX IF NOT EXISTS idx_semantic_observations_document
                    ON semantic_observations(document_id, observation_type, status);
                CREATE INDEX IF NOT EXISTS idx_timeline_nodes_document
                    ON timeline_nodes(document_id, node_type, position);
                CREATE INDEX IF NOT EXISTS idx_timeline_events_document
                    ON timeline_events(document_id, chapter_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_character_entities_document
                    ON character_entities(document_id, canonical_name);
                CREATE INDEX IF NOT EXISTS idx_character_aliases_character
                    ON character_aliases(character_id, alias);
                CREATE INDEX IF NOT EXISTS idx_relationships_document
                    ON character_relationships(document_id, status, relation_type);
                CREATE INDEX IF NOT EXISTS idx_review_items_document
                    ON material_review_items(document_id, status, review_type);
                CREATE INDEX IF NOT EXISTS idx_auxiliary_records_document
                    ON auxiliary_records(document_id, record_type, status);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "project_id" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN project_id TEXT")
            if "document_id" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN document_id TEXT")
            if "style_guide" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN style_guide TEXT NOT NULL DEFAULT ''")
            if "style_lexicon" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN style_lexicon TEXT NOT NULL DEFAULT ''")
            document_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(source_documents)").fetchall()
            }
            document_defaults = {
                "raw_text_hash": "TEXT NOT NULL DEFAULT ''",
                "global_summary": "TEXT NOT NULL DEFAULT ''",
                "library_enabled": "INTEGER NOT NULL DEFAULT 1",
                "summary_enabled": "INTEGER NOT NULL DEFAULT 1",
                "recent_chapters_enabled": "INTEGER NOT NULL DEFAULT 1",
                "characters_enabled": "INTEGER NOT NULL DEFAULT 1",
                "facts_enabled": "INTEGER NOT NULL DEFAULT 1",
            }
            for column, definition in document_defaults.items():
                if column not in document_columns:
                    connection.execute(
                        f"ALTER TABLE source_documents ADD COLUMN {column} {definition}"
                    )
            outline_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(outlines)").fetchall()
            }
            if "instruction" not in outline_columns:
                connection.execute(
                    "ALTER TABLE outlines ADD COLUMN instruction TEXT NOT NULL DEFAULT '请规划紧接当前进度的下一章。'"
                )
            chapter_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(chapters)").fetchall()
            }
            if "character_observations_json" not in chapter_columns:
                connection.execute(
                    "ALTER TABLE chapters ADD COLUMN character_observations_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "content_hash" not in chapter_columns:
                connection.execute(
                    "ALTER TABLE chapters ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
                )
            chunk_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(chapter_chunks)").fetchall()
            }
            if "content_hash" not in chunk_columns:
                connection.execute(
                    "ALTER TABLE chapter_chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
                )
            if "character_observations_json" not in chunk_columns:
                connection.execute(
                    "ALTER TABLE chapter_chunks ADD COLUMN character_observations_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "facts_status" not in chunk_columns:
                connection.execute(
                    "ALTER TABLE chapter_chunks ADD COLUMN facts_status TEXT NOT NULL DEFAULT 'pending'"
                )
            timeline_event_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(timeline_events)").fetchall()
            }
            if "manually_edited" not in timeline_event_columns:
                connection.execute(
                    "ALTER TABLE timeline_events ADD COLUMN manually_edited INTEGER NOT NULL DEFAULT 0"
                )
            relationship_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(character_relationships)").fetchall()
            }
            if "manually_edited" not in relationship_columns:
                connection.execute(
                    "ALTER TABLE character_relationships ADD COLUMN manually_edited INTEGER NOT NULL DEFAULT 0"
                )
            legacy_summaries = connection.execute(
                """
                SELECT id, summary_json FROM chapters
                WHERE character_observations_json = '[]' AND summary_json != ''
                """
            ).fetchall()
            for chapter in legacy_summaries:
                legacy_characters = _json_loads(chapter["summary_json"], {}).get(
                    "characters", []
                )
                if legacy_characters:
                    connection.execute(
                        "UPDATE chapters SET character_observations_json = ? WHERE id = ?",
                        (json.dumps(legacy_characters, ensure_ascii=False), chapter["id"]),
                    )
            now = utc_now()
            connection.execute(
                """
                INSERT OR IGNORE INTO projects
                    (id, name, global_summary, summary_enabled, created_at, updated_at)
                VALUES ('default', '我的小说', '', 1, ?, ?)
                """,
                (now, now),
            )
            connection.execute(
                "UPDATE conversations SET project_id = 'default' WHERE project_id IS NULL"
            )
            connection.execute(
                """
                UPDATE conversations
                SET document_id = (
                    SELECT d.id FROM source_documents d
                    WHERE d.project_id = conversations.project_id
                    ORDER BY d.created_at LIMIT 1
                )
                WHERE document_id IS NULL
                """
            )
            project_row = connection.execute(
                "SELECT global_summary, summary_enabled FROM projects WHERE id = 'default'"
            ).fetchone()
            document_count = connection.execute(
                "SELECT COUNT(*) FROM source_documents WHERE project_id = 'default'"
            ).fetchone()[0]
            if project_row and document_count == 1:
                connection.execute(
                    """
                    UPDATE source_documents SET global_summary = ?, summary_enabled = ?
                    WHERE project_id = 'default' AND global_summary = ''
                    """,
                    (project_row["global_summary"], project_row["summary_enabled"]),
                )
            legacy_characters = connection.execute(
                "SELECT * FROM characters"
            ).fetchall()
            for character in legacy_characters:
                document = connection.execute(
                    """
                    SELECT id FROM source_documents WHERE project_id = ?
                    ORDER BY created_at LIMIT 1
                    """,
                    (character["project_id"],),
                ).fetchone()
                if document:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO document_characters
                            (id, document_id, name, aliases_json, card_json, prompt_text,
                             source_chapters_json, enabled, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            character["id"], document["id"], character["name"],
                            character["aliases_json"], character["card_json"],
                            character["prompt_text"], character["source_chapters_json"],
                            character["enabled"], character["created_at"], character["updated_at"],
                        ),
                    )
            connection.execute(
                """
                UPDATE candidates
                SET status = 'cancelled',
                    error_message = COALESCE(error_message, '应用上次退出时生成尚未完成'),
                    completed_at = COALESCE(completed_at, ?)
                WHERE status = 'streaming'
                """,
                (utc_now(),),
            )
            connection.execute(
                """
                UPDATE outline_candidates
                SET status = 'cancelled',
                    error_message = COALESCE(error_message, '应用上次退出时大纲生成尚未完成'),
                    completed_at = COALESCE(completed_at, ?)
                WHERE status = 'streaming'
                """,
                (utc_now(),),
            )
            connection.execute(
                """
                UPDATE chapters
                SET status = 'pending',
                    error_message = COALESCE(error_message, '应用上次退出时总结尚未完成'),
                    updated_at = ?
                WHERE status = 'processing'
                """,
                (utc_now(),),
            )
            connection.execute(
                """
                UPDATE chapter_chunks
                SET status = 'pending',
                    error_message = COALESCE(error_message, '应用上次退出时片段总结尚未完成'),
                    updated_at = ?
                WHERE status = 'processing'
                """,
                (utc_now(),),
            )
            connection.execute(
                "UPDATE chapter_chunks SET facts_status = 'pending' WHERE facts_status = 'processing'"
            )
            connection.execute(
                """
                UPDATE analysis_jobs SET status = 'paused',
                    error_message = COALESCE(error_message, '应用退出，等待断点续行'),
                    updated_at = ? WHERE status = 'running'
                """,
                (utc_now(),),
            )
            unchunked = connection.execute(
                """
                SELECT c.id, c.content, c.created_at
                FROM chapters c
                WHERE NOT EXISTS (
                    SELECT 1 FROM chapter_chunks cc WHERE cc.chapter_id = c.id
                )
                """
            ).fetchall()
            for chapter in unchunked:
                for position, content in enumerate(split_long_text(chapter["content"]), start=1):
                    connection.execute(
                        """
                        INSERT INTO chapter_chunks
                            (id, chapter_id, position, content, content_hash, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id(), chapter["id"], position, content,
                            stable_text_hash(content),
                            chapter["created_at"], utc_now(),
                        ),
                    )
            self._backfill_hashes(connection)

    def _backfill_hashes(self, connection: sqlite3.Connection) -> None:
        documents = connection.execute(
            "SELECT id, raw_text FROM source_documents WHERE raw_text_hash = ''"
        ).fetchall()
        for document in documents:
            connection.execute(
                "UPDATE source_documents SET raw_text_hash = ? WHERE id = ?",
                (stable_text_hash(document["raw_text"]), document["id"]),
            )
        chapters = connection.execute(
            "SELECT id, content FROM chapters WHERE content_hash = ''"
        ).fetchall()
        for chapter in chapters:
            connection.execute(
                "UPDATE chapters SET content_hash = ? WHERE id = ?",
                (stable_text_hash(chapter["content"]), chapter["id"]),
            )
        chunks = connection.execute(
            "SELECT id, content FROM chapter_chunks WHERE content_hash = ''"
        ).fetchall()
        for chunk in chunks:
            connection.execute(
                "UPDATE chapter_chunks SET content_hash = ? WHERE id = ?",
                (stable_text_hash(chunk["content"]), chunk["id"]),
            )

    def create_conversation(
        self,
        title: str = "新对话",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        pinned_context: str = "",
        generation_settings: dict[str, Any] | None = None,
        project_id: str = "default",
        document_id: str | None = None,
    ) -> dict[str, Any]:
        conversation_id = new_id()
        now = utc_now()
        settings = generation_settings or DEFAULT_GENERATION_SETTINGS
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if document_id is None:
                document = connection.execute(
                    "SELECT id FROM source_documents WHERE project_id = ? ORDER BY created_at LIMIT 1",
                    (project_id,),
                ).fetchone()
                document_id = document["id"] if document else None
            connection.execute(
                """
                INSERT INTO conversations
                    (id, title, system_prompt, pinned_context, style_guide, style_lexicon, generation_settings,
                     created_at, updated_at, project_id, document_id)
                VALUES (?, ?, ?, ?, '', '', ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title.strip() or "新对话",
                    system_prompt,
                    pinned_context,
                    json.dumps(settings, ensure_ascii=False),
                    now,
                    now,
                    project_id,
                    document_id,
                ),
            )
            connection.commit()
        return self.get_conversation(conversation_id)

    def list_conversations(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*,
                       (SELECT COUNT(*) FROM exchanges e WHERE e.conversation_id = c.id) AS exchange_count
                FROM conversations c
                WHERE c.deleted_at IS NULL
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [self._conversation_summary(row) for row in rows]

    def _conversation_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        keys = set(row.keys())
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "exchange_count": row["exchange_count"] if "exchange_count" in keys else 0,
        }

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND deleted_at IS NULL",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise KeyError("conversation_not_found")
            exchange_rows = connection.execute(
                "SELECT * FROM exchanges WHERE conversation_id = ? ORDER BY position",
                (conversation_id,),
            ).fetchall()
            exchanges: list[dict[str, Any]] = []
            for exchange in exchange_rows:
                candidate_rows = connection.execute(
                    "SELECT * FROM candidates WHERE exchange_id = ? ORDER BY candidate_index",
                    (exchange["id"],),
                ).fetchall()
                exchanges.append(self._exchange_dict(exchange, candidate_rows))
        return {
            "id": conversation["id"],
            "title": conversation["title"],
            "system_prompt": conversation["system_prompt"],
            "pinned_context": conversation["pinned_context"],
            "style_guide": conversation["style_guide"],
            "style_lexicon": conversation["style_lexicon"],
            "generation_settings": _generation_settings(conversation["generation_settings"]),
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "project_id": conversation["project_id"] or "default",
            "document_id": conversation["document_id"],
            "exchanges": exchanges,
        }

    def get_app_setting(self, key: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_app_setting(self, key: str, value: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def _exchange_dict(
        self, exchange: sqlite3.Row, candidate_rows: list[sqlite3.Row]
    ) -> dict[str, Any]:
        return {
            "id": exchange["id"],
            "conversation_id": exchange["conversation_id"],
            "position": exchange["position"],
            "user_content": exchange["user_content"],
            "selected_candidate_id": exchange["selected_candidate_id"],
            "created_at": exchange["created_at"],
            "candidates": [self._candidate_dict(row) for row in candidate_rows],
        }

    def _candidate_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "exchange_id": row["exchange_id"],
            "candidate_index": row["candidate_index"],
            "content": row["content"],
            "reasoning_content": row["reasoning_content"],
            "status": row["status"],
            "settings_snapshot": _json_loads(row["settings_snapshot"], {}),
            "seed": row["seed"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "duration_ms": row["duration_ms"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }

    def update_conversation(self, conversation_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "title", "system_prompt", "pinned_context", "style_guide",
            "style_lexicon", "generation_settings", "document_id",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key not in allowed:
                continue
            assignments.append(f"{key} = ?")
            values.append(
                json.dumps(value, ensure_ascii=False)
                if key == "generation_settings"
                else value
            )
        if not assignments:
            return self.get_conversation(conversation_id)
        assignments.append("updated_at = ?")
        values.append(utc_now())
        values.append(conversation_id)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"UPDATE conversations SET {', '.join(assignments)} "
                "WHERE id = ? AND deleted_at IS NULL",
                values,
            )
            if cursor.rowcount == 0:
                connection.rollback()
                raise KeyError("conversation_not_found")
            connection.commit()
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE conversations SET deleted_at = ?, updated_at = ? "
                "WHERE id = ? AND deleted_at IS NULL",
                (utc_now(), utc_now(), conversation_id),
            )
            if cursor.rowcount == 0:
                connection.rollback()
                raise KeyError("conversation_not_found")
            connection.commit()

    def create_exchange_with_candidate(
        self,
        conversation_id: str,
        user_content: str,
        settings_snapshot: dict[str, Any],
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        exchange_id = new_id()
        candidate_id = new_id()
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            conversation = connection.execute(
                "SELECT title FROM conversations WHERE id = ? AND deleted_at IS NULL",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                connection.rollback()
                raise KeyError("conversation_not_found")
            position = connection.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM exchanges WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO exchanges
                    (id, conversation_id, position, user_content, selected_candidate_id, created_at)
                VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (exchange_id, conversation_id, position, user_content, now),
            )
            connection.execute(
                """
                INSERT INTO candidates
                    (id, exchange_id, candidate_index, status, settings_snapshot, seed, created_at)
                VALUES (?, ?, 1, 'streaming', ?, ?, ?)
                """,
                (
                    candidate_id,
                    exchange_id,
                    json.dumps(settings_snapshot, ensure_ascii=False),
                    seed,
                    now,
                ),
            )
            title = conversation["title"]
            if position == 1 and title == "新对话":
                compact_title = " ".join(user_content.split())[:26] or "新对话"
                connection.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (compact_title, now, conversation_id),
                )
            else:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
            connection.commit()
        exchange = self.get_exchange(exchange_id)
        candidate = next(item for item in exchange["candidates"] if item["id"] == candidate_id)
        return exchange, candidate

    def create_candidate(
        self,
        exchange_id: str,
        settings_snapshot: dict[str, Any],
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_id = new_id()
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exchange = connection.execute(
                "SELECT conversation_id FROM exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
            if exchange is None:
                connection.rollback()
                raise KeyError("exchange_not_found")
            candidate_index = connection.execute(
                "SELECT COALESCE(MAX(candidate_index), 0) + 1 FROM candidates WHERE exchange_id = ?",
                (exchange_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO candidates
                    (id, exchange_id, candidate_index, status, settings_snapshot, seed, created_at)
                VALUES (?, ?, ?, 'streaming', ?, ?, ?)
                """,
                (
                    candidate_id,
                    exchange_id,
                    candidate_index,
                    json.dumps(settings_snapshot, ensure_ascii=False),
                    seed,
                    now,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, exchange["conversation_id"]),
            )
            connection.commit()
        result = self.get_exchange(exchange_id)
        candidate = next(item for item in result["candidates"] if item["id"] == candidate_id)
        return result, candidate

    def get_exchange(self, exchange_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            exchange = connection.execute(
                "SELECT * FROM exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
            if exchange is None:
                raise KeyError("exchange_not_found")
            candidate_rows = connection.execute(
                "SELECT * FROM candidates WHERE exchange_id = ? ORDER BY candidate_index",
                (exchange_id,),
            ).fetchall()
        return self._exchange_dict(exchange, candidate_rows)

    def get_context_source(self, exchange_id: str) -> tuple[dict[str, Any], list[dict[str, str]], str]:
        with self.connect() as connection:
            current = connection.execute(
                "SELECT * FROM exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
            if current is None:
                raise KeyError("exchange_not_found")
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND deleted_at IS NULL",
                (current["conversation_id"],),
            ).fetchone()
            if conversation is None:
                raise KeyError("conversation_not_found")
            history_rows = connection.execute(
                """
                SELECT e.user_content, c.content
                FROM exchanges e
                JOIN candidates c ON c.id = e.selected_candidate_id
                WHERE e.conversation_id = ? AND e.position < ? AND c.status = 'completed'
                ORDER BY e.position
                """,
                (current["conversation_id"], current["position"]),
            ).fetchall()
        conversation_data = {
            "id": conversation["id"],
            "system_prompt": conversation["system_prompt"],
            "pinned_context": conversation["pinned_context"],
            "style_guide": conversation["style_guide"],
            "style_lexicon": conversation["style_lexicon"],
            "generation_settings": _generation_settings(conversation["generation_settings"]),
            "project_id": conversation["project_id"] or "default",
        }
        history: list[dict[str, str]] = []
        for row in history_rows:
            history.append({"role": "user", "content": row["user_content"]})
            history.append({"role": "assistant", "content": row["content"]})
        return conversation_data, history, current["user_content"]

    def update_candidate_draft(self, candidate_id: str, content: str, reasoning: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE candidates SET content = ?, reasoning_content = ? "
                "WHERE id = ? AND status = 'streaming'",
                (content, reasoning, candidate_id),
            )

    def finalize_candidate(
        self,
        candidate_id: str,
        *,
        status: str,
        content: str,
        reasoning: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "cancelled", "failed"}:
            raise ValueError("invalid_candidate_status")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute(
                "SELECT exchange_id FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if candidate is None:
                connection.rollback()
                raise KeyError("candidate_not_found")
            connection.execute(
                """
                UPDATE candidates
                SET status = ?, content = ?, reasoning_content = ?,
                    prompt_tokens = ?, completion_tokens = ?, duration_ms = ?,
                    error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    content,
                    reasoning,
                    prompt_tokens,
                    completion_tokens,
                    duration_ms,
                    error_message,
                    now,
                    candidate_id,
                ),
            )
            exchange = connection.execute(
                "SELECT conversation_id, selected_candidate_id FROM exchanges WHERE id = ?",
                (candidate["exchange_id"],),
            ).fetchone()
            if status == "completed" and exchange["selected_candidate_id"] is None:
                connection.execute(
                    "UPDATE exchanges SET selected_candidate_id = ? WHERE id = ?",
                    (candidate_id, candidate["exchange_id"]),
                )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, exchange["conversation_id"]),
            )
            connection.commit()
        return self.get_exchange(candidate["exchange_id"])

    def count_completed_candidates(self, exchange_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE exchange_id = ? AND status = 'completed'",
                (exchange_id,),
            ).fetchone()
        return int(row[0])

    def select_candidate(self, exchange_id: str, candidate_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exchange = connection.execute(
                "SELECT * FROM exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
            if exchange is None:
                connection.rollback()
                raise KeyError("exchange_not_found")
            candidate = connection.execute(
                "SELECT id FROM candidates WHERE id = ? AND exchange_id = ? AND status = 'completed'",
                (candidate_id, exchange_id),
            ).fetchone()
            if candidate is None:
                connection.rollback()
                raise ValueError("candidate_not_selectable")
            later_count = connection.execute(
                "SELECT COUNT(*) FROM exchanges WHERE conversation_id = ? AND position > ?",
                (exchange["conversation_id"], exchange["position"]),
            ).fetchone()[0]
            if later_count:
                connection.rollback()
                raise RuntimeError("branch_required")
            now = utc_now()
            connection.execute(
                "UPDATE exchanges SET selected_candidate_id = ? WHERE id = ?",
                (candidate_id, exchange_id),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, exchange["conversation_id"]),
            )
            connection.commit()
        return self.get_exchange(exchange_id)

    def create_branch(self, exchange_id: str, candidate_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source_exchange = connection.execute(
                "SELECT * FROM exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
            if source_exchange is None:
                connection.rollback()
                raise KeyError("exchange_not_found")
            source_conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND deleted_at IS NULL",
                (source_exchange["conversation_id"],),
            ).fetchone()
            chosen = connection.execute(
                "SELECT id FROM candidates WHERE id = ? AND exchange_id = ? AND status = 'completed'",
                (candidate_id, exchange_id),
            ).fetchone()
            if source_conversation is None or chosen is None:
                connection.rollback()
                raise ValueError("candidate_not_selectable")

            new_conversation_id = new_id()
            now = utc_now()
            connection.execute(
                """
                INSERT INTO conversations
                    (id, title, system_prompt, pinned_context, style_guide, style_lexicon,
                     generation_settings, created_at, updated_at, project_id, document_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_conversation_id,
                    f"{source_conversation['title']} · 分支",
                    source_conversation["system_prompt"],
                    source_conversation["pinned_context"],
                    source_conversation["style_guide"],
                    source_conversation["style_lexicon"],
                    source_conversation["generation_settings"],
                    now,
                    now,
                    source_conversation["project_id"] or "default",
                    source_conversation["document_id"],
                ),
            )
            source_exchanges = connection.execute(
                """
                SELECT * FROM exchanges
                WHERE conversation_id = ? AND position <= ?
                ORDER BY position
                """,
                (source_exchange["conversation_id"], source_exchange["position"]),
            ).fetchall()
            for old_exchange in source_exchanges:
                new_exchange_id = new_id()
                selected_old_id = (
                    candidate_id
                    if old_exchange["id"] == exchange_id
                    else old_exchange["selected_candidate_id"]
                )
                connection.execute(
                    """
                    INSERT INTO exchanges
                        (id, conversation_id, position, user_content, selected_candidate_id, created_at)
                    VALUES (?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        new_exchange_id,
                        new_conversation_id,
                        old_exchange["position"],
                        old_exchange["user_content"],
                        old_exchange["created_at"],
                    ),
                )
                old_candidates = connection.execute(
                    "SELECT * FROM candidates WHERE exchange_id = ? ORDER BY candidate_index",
                    (old_exchange["id"],),
                ).fetchall()
                selected_new_id: str | None = None
                for old_candidate in old_candidates:
                    new_candidate_id = new_id()
                    connection.execute(
                        """
                        INSERT INTO candidates
                            (id, exchange_id, candidate_index, content, reasoning_content, status,
                             settings_snapshot, seed, prompt_tokens, completion_tokens, duration_ms,
                             error_message, created_at, completed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_candidate_id,
                            new_exchange_id,
                            old_candidate["candidate_index"],
                            old_candidate["content"],
                            old_candidate["reasoning_content"],
                            old_candidate["status"],
                            old_candidate["settings_snapshot"],
                            old_candidate["seed"],
                            old_candidate["prompt_tokens"],
                            old_candidate["completion_tokens"],
                            old_candidate["duration_ms"],
                            old_candidate["error_message"],
                            old_candidate["created_at"],
                            old_candidate["completed_at"],
                        ),
                    )
                    if old_candidate["id"] == selected_old_id:
                        selected_new_id = new_candidate_id
                connection.execute(
                    "UPDATE exchanges SET selected_candidate_id = ? WHERE id = ?",
                    (selected_new_id, new_exchange_id),
                )
            connection.commit()
        return self.get_conversation(new_conversation_id)

    def import_conversation_backup(self, backup: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(backup, dict) or not isinstance(backup.get("exchanges"), list):
            raise ValueError("不是有效的对话 JSON 备份")

        allowed_statuses = {"streaming", "completed", "cancelled", "failed"}

        def text(value: Any, default: str = "") -> str:
            return str(value if value is not None else default)

        def imported_status(value: Any, *, outline: bool = False) -> str:
            status = text(value, "completed").strip() or "completed"
            if status not in allowed_statuses:
                raise ValueError("备份包含未知候选状态")
            return "cancelled" if status == "streaming" else status

        def imported_settings(value: Any) -> dict[str, Any]:
            return value if isinstance(value, dict) else {}

        def imported_seed(value: Any) -> int | None:
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        def imported_metric(value: Any) -> int | None:
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        title_base = text(backup.get("title"), "新对话").strip() or "新对话"
        title_suffix = " · 备份恢复"
        title = f"{title_base[: max(1, 100 - len(title_suffix))]}{title_suffix}"
        generation_settings = {
            **DEFAULT_GENERATION_SETTINGS,
            **imported_settings(backup.get("generation_settings")),
        }
        now = utc_now()
        conversation_id = new_id()

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            project_id = text(backup.get("project_id"), "default").strip() or "default"
            if connection.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
                project_id = "default"
            document_id = text(backup.get("document_id")).strip() or None
            if document_id and connection.execute(
                "SELECT 1 FROM source_documents WHERE id = ?", (document_id,)
            ).fetchone() is None:
                document_id = None

            connection.execute(
                """
                INSERT INTO conversations
                    (id, title, system_prompt, pinned_context, style_guide, style_lexicon,
                     generation_settings, created_at, updated_at, project_id, document_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    text(backup.get("system_prompt"), DEFAULT_SYSTEM_PROMPT),
                    text(backup.get("pinned_context")),
                    text(backup.get("style_guide")),
                    text(backup.get("style_lexicon")),
                    json.dumps(generation_settings, ensure_ascii=False),
                    now,
                    now,
                    project_id,
                    document_id,
                ),
            )

            for position, exchange in enumerate(backup["exchanges"], start=1):
                if not isinstance(exchange, dict):
                    connection.rollback()
                    raise ValueError("备份包含无效对话轮次")
                user_content = text(exchange.get("user_content")).strip()
                if not user_content:
                    connection.rollback()
                    raise ValueError("备份包含空的用户输入")
                exchange_id = new_id()
                selected_old_id = text(exchange.get("selected_candidate_id")).strip()
                connection.execute(
                    """
                    INSERT INTO exchanges
                        (id, conversation_id, position, user_content, selected_candidate_id, created_at)
                    VALUES (?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        exchange_id,
                        conversation_id,
                        position,
                        user_content,
                        text(exchange.get("created_at"), now) or now,
                    ),
                )
                selected_new_id: str | None = None
                candidates = exchange.get("candidates") if isinstance(exchange.get("candidates"), list) else []
                for index, candidate in enumerate(candidates, start=1):
                    if not isinstance(candidate, dict):
                        connection.rollback()
                        raise ValueError("备份包含无效候选版本")
                    candidate_id = new_id()
                    status = imported_status(candidate.get("status"))
                    old_candidate_id = text(candidate.get("id")).strip()
                    completed_at = text(candidate.get("completed_at")) or (
                        now if status in {"completed", "cancelled", "failed"} else None
                    )
                    error_message = candidate.get("error_message")
                    if candidate.get("status") == "streaming" and not error_message:
                        error_message = "备份恢复时生成尚未完成"
                    connection.execute(
                        """
                        INSERT INTO candidates
                            (id, exchange_id, candidate_index, content, reasoning_content, status,
                             settings_snapshot, seed, prompt_tokens, completion_tokens, duration_ms,
                             error_message, created_at, completed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            candidate_id,
                            exchange_id,
                            index,
                            text(candidate.get("content")),
                            text(candidate.get("reasoning_content")),
                            status,
                            json.dumps(imported_settings(candidate.get("settings_snapshot")), ensure_ascii=False),
                            imported_seed(candidate.get("seed")),
                            imported_metric(candidate.get("prompt_tokens")),
                            imported_metric(candidate.get("completion_tokens")),
                            imported_metric(candidate.get("duration_ms")),
                            text(error_message) if error_message else None,
                            text(candidate.get("created_at"), now) or now,
                            completed_at,
                        ),
                    )
                    if old_candidate_id and old_candidate_id == selected_old_id and status == "completed":
                        selected_new_id = candidate_id
                connection.execute(
                    "UPDATE exchanges SET selected_candidate_id = ? WHERE id = ?",
                    (selected_new_id, exchange_id),
                )

            outline = backup.get("outline")
            if isinstance(outline, dict):
                outline_candidates = (
                    outline.get("candidates") if isinstance(outline.get("candidates"), list) else []
                )
                if outline_candidates or outline.get("instruction") or outline.get("selected_candidate_id"):
                    outline_id = new_id()
                    selected_old_id = text(outline.get("selected_candidate_id")).strip()
                    connection.execute(
                        """
                        INSERT INTO outlines
                            (id, conversation_id, instruction, selected_candidate_id, enabled, created_at, updated_at)
                        VALUES (?, ?, ?, NULL, 0, ?, ?)
                        """,
                        (
                            outline_id,
                            conversation_id,
                            text(outline.get("instruction"), "请规划紧接当前进度的下一章。"),
                            now,
                            now,
                        ),
                    )
                    selected_new_id = None
                    for index, candidate in enumerate(outline_candidates, start=1):
                        if not isinstance(candidate, dict):
                            connection.rollback()
                            raise ValueError("备份包含无效大纲候选版本")
                        candidate_id = new_id()
                        status = imported_status(candidate.get("status"), outline=True)
                        old_candidate_id = text(candidate.get("id")).strip()
                        completed_at = text(candidate.get("completed_at")) or (
                            now if status in {"completed", "cancelled", "failed"} else None
                        )
                        error_message = candidate.get("error_message")
                        if candidate.get("status") == "streaming" and not error_message:
                            error_message = "备份恢复时大纲生成尚未完成"
                        connection.execute(
                            """
                            INSERT INTO outline_candidates
                                (id, outline_id, candidate_index, content, edited_content, status,
                                 settings_snapshot, seed, error_message, created_at, completed_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                candidate_id,
                                outline_id,
                                index,
                                text(candidate.get("content")),
                                text(candidate.get("edited_content")),
                                status,
                                json.dumps(imported_settings(candidate.get("settings_snapshot")), ensure_ascii=False),
                                imported_seed(candidate.get("seed")),
                                text(error_message) if error_message else None,
                                text(candidate.get("created_at"), now) or now,
                                completed_at,
                            ),
                        )
                        if old_candidate_id and old_candidate_id == selected_old_id and status == "completed":
                            selected_new_id = candidate_id
                    connection.execute(
                        """
                        UPDATE outlines
                        SET selected_candidate_id = ?, enabled = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            selected_new_id,
                            int(bool(outline.get("enabled")) and selected_new_id is not None),
                            now,
                            outline_id,
                        ),
                    )
            connection.commit()
        return self.get_conversation(conversation_id)
