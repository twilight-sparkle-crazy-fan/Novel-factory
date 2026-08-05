from __future__ import annotations

import asyncio
import json
import re
import secrets
from collections.abc import Callable
from typing import Any

from .config import DEFAULT_GENERATION_SETTINGS
from .llama_client import GenerationCancelled, LlamaClient
from .novel_repository import format_chapter_summary


JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
ProgressCallback = Callable[[str, int, int], None]


def split_text_chunks(text: str, max_characters: int = 12_000) -> list[str]:
    """Compatibility helper: active analysis now sends a chapter in one request."""
    _ = max_characters
    value = text.strip()
    return [value] if value else []


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = JSON_BLOCK.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(candidate[start : end + 1])
                return value if isinstance(value, dict) else {"value": value}
            except json.JSONDecodeError:
                pass
    return {"summary": text.strip(), "parse_warning": "模型未返回有效 JSON"}


def flatten_character_observations(
    character_observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for item in character_observations:
        if not isinstance(item, dict):
            continue
        # Accept the old chapter-summary shape during migration, while all new
        # analyses pass direct observations from the dedicated character call.
        if "characters" in item:
            title = item.get("title", "未知章节")
            for character in item.get("characters") or []:
                if isinstance(character, dict) and character.get("name"):
                    observations.append(
                        {
                            **character,
                            "source_chapters": character.get("source_chapters")
                            or [title],
                        }
                    )
        elif item.get("name"):
            observations.append(item)
    return observations


def compact_character_card(card: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "id": card.get("id"),
        "name": card.get("name"),
        "aliases": card.get("aliases") or [],
        "source_chapters": card.get("source_chapters") or [],
    }
    if isinstance(card.get("card"), dict):
        compact.update(card["card"])
    else:
        for key, value in card.items():
            if key not in {
                "id", "project_id", "document_id", "name", "aliases", "card",
                "prompt_text", "enabled", "updated_at", "source_chapters",
            }:
                compact[key] = value
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


class NovelAnalysisService:
    def __init__(self, client: LlamaClient):
        self.client = client

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stop_event: asyncio.Event,
        temperature: float = 0.2,
    ) -> str:
        settings = {
            **DEFAULT_GENERATION_SETTINGS,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "repeat_penalty": 1.05,
            "seed": secrets.randbelow(2_147_483_647),
        }
        output = ""
        async for event in self.client.stream_chat(
            messages,
            settings,
            stop_event,
            buffer_for_retry=True,
        ):
            if event["type"] == "content_delta":
                output += event["text"]
        return output.strip()

    async def analyze_chunk(
        self,
        title: str,
        chunk: str,
        previous_summary: str,
        index: int,
        total: int,
        stop_event: asyncio.Event,
        *,
        max_tokens: int = 8192,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        summary_prompt = f"""请分析小说章节《{title}》的第 {index}/{total} 个片段。
只返回 JSON。字段：title, summary, time, location, pov, key_events,
conflicts, worldbuilding, clues, unresolved, ending_state, character_changes。
只整理情节与章节结构，不要输出 characters 或人物卡，不得编造。

上一片段摘要：
{previous_summary or '（无）'}

片段正文：
{chunk}"""
        summary_raw = await self.complete(
            [{"role": "system", "content": "你是严谨的中文小说情节资料编辑。"},
             {"role": "user", "content": summary_prompt}],
            max_tokens=max(4096, min(max_tokens, 10_000)), stop_event=stop_event,
        )
        summary = parse_json_object(summary_raw)
        summary.pop("characters", None)
        return summary, []

    async def extract_story_facts(
        self,
        title: str,
        chunk: str,
        stop_event: asyncio.Event,
        *,
        max_tokens: int = 8192,
    ) -> list[dict[str, Any]]:
        prompt = f"""从小说章节《{title}》片段中提取可长期追踪的结构化事实。
只返回 JSON：{{"facts": [...]}}。fact_type 只能是 timeline、foreshadowing、
item、location、relationship。每条包含 fact_key, fact_type, subject, predicate,
object, state, status, event_time, confidence, evidence。
fact_key 对同一持续事实保持稳定，例如“item|钥匙|holder”；时间线事件则包含简短事件标识。
status 使用 active、open、resolved、superseded。evidence 为支持事实的简短原文。
不要把纯文风或无长期价值的描述列为事实，不得编造。

片段正文：
{chunk}"""
        raw = await self.complete(
            [{"role": "system", "content": "你是小说连续性事实管理员。"},
             {"role": "user", "content": prompt}],
            max_tokens=max(4096, min(max_tokens, 8192)), stop_event=stop_event,
        )
        return [item for item in parse_json_object(raw).get("facts", []) if isinstance(item, dict)]

    async def extract_unified_events(
        self,
        title: str,
        chunk: str,
        stop_event: asyncio.Event,
        *,
        max_tokens: int = 8192,
    ) -> dict[str, list[dict[str, Any]]]:
        prompt = f"""从小说章节《{title}》片段中做一次统一语义抽取。
只返回 JSON 对象，字段必须是 plot_events, character_events, relationship_events,
location_events, ability_events, object_events, unresolved_entities。
每个事件保留 title/description/event_type/confidence；人物和关系要保留 source/target/relation_type。
只抽取片段明确支持的信息，不要编造，不要输出章节摘要或人物卡。

片段正文：
{chunk}"""
        raw = await self.complete(
            [{"role": "system", "content": "你是小说统一事件账本抽取器。"},
             {"role": "user", "content": prompt}],
            max_tokens=max(4096, min(max_tokens, 8192)),
            stop_event=stop_event,
        )
        parsed = parse_json_object(raw)
        return {
            key: [item for item in parsed.get(key, []) if isinstance(item, dict)]
            for key in (
                "plot_events", "character_events", "relationship_events",
                "location_events", "ability_events", "object_events",
                "unresolved_entities",
            )
        }

    async def merge_chapter_summaries(
        self,
        title: str,
        partials: list[dict[str, Any]],
        stop_event: asyncio.Event,
        *,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        if not partials:
            return {"title": title, "summary": "（本章没有正文）"}
        if len(partials) == 1:
            return {**partials[0], "title": title}
        prompt = f"""合并《{title}》的分片摘要为完整章节摘要。只返回 JSON。
保持事件顺序和未解决问题；不要输出人物卡或 characters，不得编造。

{json.dumps(partials, ensure_ascii=False)}"""
        raw = await self.complete(
            [{"role": "system", "content": "你是严谨的小说连续性编辑。"},
             {"role": "user", "content": prompt}],
            max_tokens=max(4096, min(max_tokens, 12_000)), stop_event=stop_event,
        )
        result = parse_json_object(raw)
        result.pop("characters", None)
        result["title"] = title
        return result

    async def summarize_chapter(
        self,
        title: str,
        content: str,
        stop_event: asyncio.Event,
        on_progress: ProgressCallback | None = None,
        max_tokens: int = 6144,
    ) -> dict[str, Any]:
        if not content.strip():
            return {
                "title": title,
                "summary": "（本章没有正文）",
            }
        if stop_event.is_set():
            raise GenerationCancelled("用户停止了总结")
        if on_progress:
            on_progress("summary_started", 1, 1)
        prompt = f"""请为小说章节《{title}》生成一份完整章节摘要。
只返回 JSON 对象，不要 Markdown 代码围栏。未知信息写空字符串或空数组，不得编造。
字段必须包含：title, summary, time, location, pov, key_events, conflicts,
worldbuilding, clues, unresolved, ending_state, character_changes。
本次只整理情节与章节结构，不要输出人物卡、人物完整资料或 characters 字段。

章节正文：
{content}"""
        raw = await self.complete(
            [
                {"role": "system", "content": "你是严谨的中文小说资料整理员。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max(4096, min(max_tokens, 12_000)),
            stop_event=stop_event,
        )
        result = parse_json_object(raw)
        result.pop("characters", None)
        result["title"] = title
        if on_progress:
            on_progress("summary_completed", 1, 1)
        return result

    async def build_project_summary(
        self,
        summaries: list[dict[str, Any]],
        stop_event: asyncio.Event,
        max_tokens: int = 6144,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        rendered = [format_chapter_summary(item) for item in summaries]
        if not rendered:
            return ""
        if on_progress:
            on_progress("summary_started", 1, 1)
        prompt = f"""请基于下面的全部章节摘要，生成小说前文总览。
必须保持时间顺序和因果连续。
只保留对后续写作有用的信息：主线进展、核心人物状态、重要关系变化、世界观规则、关键物品、已揭示秘密、未解决矛盾和伏笔。
不要添加原摘要没有的信息，不要展开成章节流水账。使用清晰的中文 Markdown。

全部章节摘要：
{chr(10).join(rendered)}"""
        result = await self.complete(
            [{"role": "system", "content": "你是小说连续性编辑，负责维护前文总览。"}, {"role": "user", "content": prompt}],
            max_tokens=max(3072, min(max_tokens, 12_000)),
            stop_event=stop_event,
        )
        if on_progress:
            on_progress("summary_completed", 1, 1)
        return result

    async def summarize_increment(
        self,
        title: str,
        previous_summary: str,
        new_content: str,
        stop_event: asyncio.Event,
        *,
        max_tokens: int = 6144,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not new_content.strip():
            return {
                "title": title,
                "summary": previous_summary or "（本章没有正文）",
            }
        if stop_event.is_set():
            raise GenerationCancelled("用户停止了增量整理")
        if on_progress:
            on_progress("summary_started", 1, 1)
        prompt = f"""请把旧章节摘要与本次新增正文整合成《{title}》最新的完整章节摘要。
只返回 JSON；保持事件顺序，更新当前状态，不得丢失仍有效的伏笔，也不得编造。
字段包含 title, summary, time, location, pov, key_events, conflicts,
worldbuilding, clues, unresolved, ending_state, character_changes。不要输出 characters 或人物卡。

旧章节摘要：
{previous_summary or '（无）'}

本次新增正文：
{new_content}"""
        raw = await self.complete(
            [
                {"role": "system", "content": "你是小说连续性与资料编辑。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max(4096, min(max_tokens, 12_000)),
            stop_event=stop_event,
        )
        result = parse_json_object(raw)
        result.pop("characters", None)
        result["title"] = title
        if on_progress:
            on_progress("summary_completed", 1, 1)
        return result

    async def extract_character_cards_from_summaries(
        self,
        summaries: list[dict[str, str]],
        existing_cards: list[dict[str, Any]],
        stop_event: asyncio.Event,
        *,
        max_tokens: int = 8192,
        on_progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        if not summaries:
            return []
        if stop_event.is_set():
            raise GenerationCancelled("用户停止了人物卡提炼")
        if on_progress:
            on_progress("extract_started", 1, 1)
        known = [
            {
                "character_name": item.get("name"),
                "aliases": item.get("aliases") or [],
            }
            for item in existing_cards
        ]
        schema_example = {
            "character_name": "人物标准名",
            "character_title": "称号或职位",
            "full_name": "完整姓名",
            "aliases": ["明确属于此人的别名"],
            "basic_info": {
                "identity": "身份",
                "birth_origin": "出身",
                "current_residence": "现居地",
                "appearance": "外貌",
            },
            "core_personality": {"trait_key": "具体表现"},
            "behavior_habits": ["行为习惯"],
            "world_setting": "与人物直接有关的世界观设定",
        }
        prompt = f"""请只依据用户选择的章节摘要提炼人物卡。
只返回 JSON 对象：{{"characters": [...]}}，不要代码围栏。结构必须严格参照：
{json.dumps(schema_example, ensure_ascii=False, indent=2)}

要求：
1. 只写摘要明确支持的信息；未知字段保持空字符串、空对象或空数组，不得编造。
2. character_name 使用稳定标准名；aliases 只含明确属于同一人的姓名或外号。
3. 已有人物若再次出现，可以输出同名卡用于补充；不要仅因称谓相似就合并人物。
4. 不生成事件记录、人物经历、章节流水账或额外字段。

已有人物索引：
{json.dumps(known, ensure_ascii=False)}

所选章节摘要：
{json.dumps(summaries, ensure_ascii=False, indent=2)}"""
        raw = await self.complete(
            [
                {"role": "system", "content": "你是严谨的中文小说人物设定编辑。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max(4096, min(max_tokens, 12_000)),
            stop_event=stop_event,
        )
        cards = [
            item for item in parse_json_object(raw).get("characters", [])
            if isinstance(item, dict) and str(item.get("character_name") or "").strip()
        ]
        source_chapters = [item["title"] for item in summaries]
        result = [
            {
                "name": item["character_name"],
                "aliases": item.get("aliases") or [],
                "card": item,
                "source_chapters": source_chapters,
            }
            for item in cards
        ]
        if on_progress:
            on_progress("extract_completed", 1, 1)
        return result

    async def extract_new_character_cards(
        self,
        title: str,
        content: str,
        existing_cards: list[dict[str, Any]],
        stop_event: asyncio.Event,
        max_tokens: int = 8192,
        on_progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Create cards only for characters not already known to this document."""
        chunks = split_text_chunks(content, 20_000)
        if not chunks:
            return []

        known_names: list[str] = []
        known_keys: set[str] = set()

        def remember(value: Any) -> None:
            name = str(value or "").strip()
            key = re.sub(r"\s+", "", name).casefold()
            if name and key and key not in known_keys:
                known_keys.add(key)
                known_names.append(name)

        for card in existing_cards:
            if not isinstance(card, dict):
                continue
            remember(card.get("name"))
            for alias in card.get("aliases") or []:
                remember(alias)

        created: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            if stop_event.is_set():
                raise GenerationCancelled("用户停止了人物卡提取")
            if on_progress:
                on_progress("batch_started", index, len(chunks))
            prompt = f"""请从小说章节《{title}》正文中找出首次出现、且不在“已有人物名”中的人物，并为他们创建一次性人物卡。
只返回 JSON：{{"characters": [...]}}。如果没有新人物，返回空数组。

每张新卡包含 name, aliases, identity, age, core_personality, behavior_logic,
long_term_desire, core_fear, speech_style, stable_abilities, long_arc,
hard_constraints, facts, inferences, uncertainties, source_chapters。

严格要求：
1. 已有人物名及其别名对应的人物绝对不能再次输出，也不能更新其任何字段。
2. name 使用人物第一次出现时可确认的标准名；有明确姓名就用姓名，没有姓名才用本章最主要、最可区分的称号。
3. aliases 只包含正文明确指向同一人的姓名或外号，不得加入关系对象、同场人物或普通称谓。
4. 只记录首次出场即可确定的稳定信息，不生成 event_records 或人物经历。
5. 事实、推断、不确定信息分开，不得编造。

已有人物名：
{json.dumps(known_names, ensure_ascii=False)}

本章正文（第 {index}/{len(chunks)} 段）：
{chunk}"""
            raw = await self.complete(
                [
                    {"role": "system", "content": "你只创建首次出场人物的初始人物卡。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max(4096, min(max_tokens, 12_000)),
                stop_event=stop_event,
            )
            for item in parse_json_object(raw).get("characters", []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                key = re.sub(r"\s+", "", name).casefold()
                if not name or not key or key in known_keys:
                    continue
                card = {
                    **item,
                    "name": name,
                    "source_chapters": item.get("source_chapters") or [title],
                }
                card.pop("event_records", None)
                card.pop("events", None)
                created.append(card)
                remember(name)
                for alias in card.get("aliases") or []:
                    remember(alias)
            if on_progress:
                on_progress("batch_completed", index, len(chunks))
        return created

    async def extract_character_cards(
        self,
        character_observations: list[dict[str, Any]],
        stop_event: asyncio.Event,
        max_tokens: int = 8192,
        on_progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        observations = flatten_character_observations(character_observations)
        if not observations:
            return []

        chunks = split_text_chunks(json.dumps(observations, ensure_ascii=False), 12_000)
        merged_cards: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress("batch_started", index, len(chunks))
            prompt = """请把下面的小说人物观察合并成人物卡。按人物标准名合并，去重，但不要擅自合并不确定的人物。
只返回 JSON：{"characters": [...]}。
每张卡包含 name, aliases, identity, age, core_personality, behavior_logic,
long_term_desire, core_fear, speech_style, stable_abilities, long_arc,
hard_constraints, event_records, facts, inferences, uncertainties, source_chapters。
人物核心卡只保留长期稳定信息；事件变化放入 event_records，注入提示词时主要使用 consequences.Abstract。
name 是人物标准名：有明确姓名时必须用姓名；没有姓名时，用首次出现章节的主要称号作为标准名。
只有标准名相同，或正文明确说明两个姓名/外号指向同一人，才可以合并。
aliases 只能保存同一人物的其他姓名/外号；亲属称谓、师徒称谓、恋人称呼、关系对象、同场人物和敌对人物不得写入 aliases，应写入 relationships 或 uncertainties。
event_records 按 chapter/event/consequences.Abstract 去重合并；不要把事件摘要提升成长期核心设定。
事实、推断和不确定信息必须分开；不得编造。\n\n""" + chunk
            raw = await self.complete(
                [{"role": "system", "content": "你是严谨的人物设定编辑。"}, {"role": "user", "content": prompt}],
                max_tokens=max(4096, min(max_tokens, 12_000)),
                stop_event=stop_event,
            )
            value = parse_json_object(raw)
            merged_cards.extend(item for item in value.get("characters", []) if isinstance(item, dict))
            if on_progress:
                on_progress("batch_completed", index, len(chunks))

        if len(chunks) == 1:
            return merged_cards
        final_prompt = """再次合并以下人物卡，只能按标准名合并；除非正文明确说明两个姓名/外号属于同一人，否则不要因为 aliases、称谓或关系对象相似而合并。只返回 {"characters": [...]}，保留所有字段和来源章节，不得编造。\n\n""" + json.dumps(merged_cards, ensure_ascii=False)
        if on_progress:
            on_progress("merge_started", len(chunks), len(chunks))
        raw = await self.complete(
            [{"role": "system", "content": "你是严谨的人物设定编辑。"}, {"role": "user", "content": final_prompt}],
            max_tokens=max(6144, min(max_tokens, 16_384)),
            stop_event=stop_event,
        )
        if on_progress:
            on_progress("merge_completed", len(chunks), len(chunks))
        return [item for item in parse_json_object(raw).get("characters", []) if isinstance(item, dict)]

    async def merge_character_updates(
        self,
        existing_cards: list[dict[str, Any]],
        character_observations: list[dict[str, Any]],
        stop_event: asyncio.Event,
        max_tokens: int = 8192,
        on_progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        observations = flatten_character_observations(character_observations)
        if not observations:
            return []
        compact_existing = [compact_character_card(card) for card in existing_cards if isinstance(card, dict)]
        if not compact_existing:
            return await self.extract_character_cards(
                observations,
                stop_event,
                max_tokens=max_tokens,
                on_progress=on_progress,
            )

        chunks = split_text_chunks(json.dumps(observations, ensure_ascii=False), 12_000)
        updated_cards: list[dict[str, Any]] = []
        existing_json = json.dumps(compact_existing, ensure_ascii=False)
        for index, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress("batch_started", index, len(chunks))
            prompt = f"""请把“新增人物观察”合并进“已有相关人物卡”。
只返回 JSON：{{"characters": [...]}}。不要 Markdown 代码围栏。

严格要求：
1. 只输出被新增观察影响的人物，未受影响的人物不要输出。
2. 如果新增观察对应已有人物，必须保留已有人物的 id。
3. 只有新增观察的 name 标准名与已有人物标准名一致时，才更新该人物；找不到对应标准名时必须作为新人物输出，不能借 aliases 强行更新。
4. aliases 只加入明确属于同一人物的其他姓名/外号；亲属称谓、师徒称谓、恋人称呼、关系对象、同场人物和敌对人物不得加入 aliases。
5. event_records、facts、inferences、uncertainties、source_chapters 去重合并；不要把事件摘要提升成长期核心设定。
6. facts 只能写正文明确支持的事实；推测放进 inferences；不确定内容放进 uncertainties。
7. 不得编造正文没有支持的信息。

已有相关人物卡：
{existing_json}

新增人物观察（第 {index}/{len(chunks)} 批）：
{chunk}"""
            raw = await self.complete(
                [
                    {"role": "system", "content": "你是严谨的小说人物档案管理员，负责增量维护人物卡。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max(4096, min(max_tokens, 12_000)),
                stop_event=stop_event,
            )
            updated_cards.extend(
                item for item in parse_json_object(raw).get("characters", []) if isinstance(item, dict)
            )
            if on_progress:
                on_progress("batch_completed", index, len(chunks))

        if len(chunks) == 1:
            return updated_cards
        final_prompt = """再次合并以下增量人物卡，只能处理同一 id 或同一标准名对应的重复项；不得因为 aliases、称谓或关系对象相似而合并。
只返回 {"characters": [...]}；已有人物必须保留 id；不得输出未受新增观察影响的人物；不得编造。\n\n""" + json.dumps(updated_cards, ensure_ascii=False)
        if on_progress:
            on_progress("merge_started", len(chunks), len(chunks))
        raw = await self.complete(
            [
                {"role": "system", "content": "你是严谨的小说人物档案管理员。"},
                {"role": "user", "content": final_prompt},
            ],
            max_tokens=max(6144, min(max_tokens, 16_384)),
            stop_event=stop_event,
        )
        if on_progress:
            on_progress("merge_completed", len(chunks), len(chunks))
        return [item for item in parse_json_object(raw).get("characters", []) if isinstance(item, dict)]
