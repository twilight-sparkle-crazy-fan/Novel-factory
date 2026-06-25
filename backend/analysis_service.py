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
from .text_import import split_long_text


JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
ProgressCallback = Callable[[str, int, int], None]


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


def split_text_chunks(text: str, max_characters: int = 12_000) -> list[str]:
    return split_long_text(text, max_characters)


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
        async for event in self.client.stream_chat(messages, settings, stop_event):
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
        character_prompt = f"""请只提取小说章节《{title}》第 {index}/{total} 个片段里的人物信息。
只返回 JSON：{{"characters": [...]}}。每个人物包含 name, aliases, identity,
age, appearance, personality, goals, fears, secrets, speech_style, abilities,
relationships, arc, current_state, facts, inferences, uncertainties, source_chapters。
事实、推断和不确定信息必须分开，不要输出情节摘要。

片段正文：
{chunk}"""
        character_raw = await self.complete(
            [{"role": "system", "content": "你只负责提取小说人物资料。"},
             {"role": "user", "content": character_prompt}],
            max_tokens=max(8192, min(max_tokens + 2048, 12_000)), stop_event=stop_event,
        )
        value = parse_json_object(character_raw)
        observations = [
            {**item, "source_chapters": item.get("source_chapters") or [title]}
            for item in value.get("characters", [])
            if isinstance(item, dict) and item.get("name")
        ]
        return summary, observations

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
        chunks = split_text_chunks(content)
        if not chunks:
            return {
                "title": title,
                "summary": "（本章没有正文）",
                "_chunk_summaries": [],
                "_character_observations": [],
            }
        partials: list[dict[str, Any]] = []
        character_observations: list[dict[str, Any]] = []
        previous_summary = "（这是第一段，没有上段摘要。）"
        for index, chunk in enumerate(chunks, start=1):
            if stop_event.is_set():
                raise GenerationCancelled("用户停止了总结")
            if on_progress:
                on_progress("summary_chunk_started", index, len(chunks))
            prompt = f"""请分析小说章节《{title}》的第 {index}/{len(chunks)} 个片段。
只返回 JSON 对象，不要 Markdown 代码围栏。未知信息写空字符串或空数组，不得编造。
字段必须包含：title, summary, time, location, pov, key_events, conflicts,
worldbuilding, clues, unresolved, ending_state, character_changes。
本次只整理情节与章节结构，不要输出人物卡、人物完整资料或 characters 字段。

上一段的结构化摘要（只用于保持人物、时间与因果连续，不得当作本段新事实）：
{previous_summary}

片段正文：
{chunk}"""
            raw = await self.complete(
                [
                    {"role": "system", "content": "你是严谨的中文小说资料整理员。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max(4096, min(max_tokens, 10_000)),
                stop_event=stop_event,
            )
            partial = parse_json_object(raw)
            partial.pop("characters", None)
            partials.append(partial)
            previous_summary = format_chapter_summary(partial)
            if on_progress:
                on_progress("summary_chunk_completed", index, len(chunks))

            if on_progress:
                on_progress("character_chunk_started", index, len(chunks))
            character_prompt = f"""请只提取小说章节《{title}》第 {index}/{len(chunks)} 个片段里的人物信息。
只返回 JSON：{{"characters": [...]}}，不要输出情节摘要或章节总结。
每个人物包含 name, aliases, identity, age, appearance, personality, goals,
fears, secrets, speech_style, abilities, relationships, arc, current_state,
facts, inferences, uncertainties, source_chapters。
facts 只能写正文明确支持的事实；推测必须放进 inferences；不确定内容放进 uncertainties。

本片段情节摘要（仅用于定位，不得代替正文证据）：
{format_chapter_summary(partial)}

片段正文：
{chunk}"""
            character_raw = await self.complete(
                [
                    {"role": "system", "content": "你是严谨的中文小说人物资料编辑，只负责人物信息。"},
                    {"role": "user", "content": character_prompt},
                ],
                max_tokens=max(8192, min(max_tokens + 2048, 12_000)),
                stop_event=stop_event,
            )
            character_value = parse_json_object(character_raw)
            for item in character_value.get("characters", []):
                if isinstance(item, dict) and item.get("name"):
                    character_observations.append(
                        {**item, "source_chapters": item.get("source_chapters") or [title]}
                    )
            if on_progress:
                on_progress("character_chunk_completed", index, len(chunks))
        if len(partials) == 1:
            result = partials[0]
            result["title"] = title
            result["_chunk_summaries"] = partials
            result["_character_observations"] = character_observations
            return result

        merge_prompt = f"""下面是《{title}》各片段的结构化摘要。请合并为一份章节摘要。
去重、保持事件先后顺序，不得把推断写成事实。只返回 JSON。
字段与输入保持一致。本次只合并章节情节摘要，不要输出 characters 或人物卡。

{json.dumps(partials, ensure_ascii=False)}"""
        if on_progress:
            on_progress("merge_started", len(chunks), len(chunks))
        raw = await self.complete(
            [
                {"role": "system", "content": "你是严谨的中文小说资料整理员。"},
                {"role": "user", "content": merge_prompt},
            ],
            max_tokens=max(4096, min(max_tokens, 12_000)),
            stop_event=stop_event,
        )
        result = parse_json_object(raw)
        result.pop("characters", None)
        if on_progress:
            on_progress("merge_completed", len(chunks), len(chunks))
        result["title"] = title
        result["_chunk_summaries"] = partials
        result["_character_observations"] = character_observations
        return result

    async def build_project_summary(
        self,
        summaries: list[dict[str, Any]],
        stop_event: asyncio.Event,
        max_tokens: int = 6144,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        rendered = [format_chapter_summary(item) for item in summaries]
        groups: list[str] = []
        current = ""
        for summary in rendered:
            if current and len(current) + len(summary) > 12_000:
                groups.append(current)
                current = ""
            current += summary + "\n\n"
        if current:
            groups.append(current)

        arc_summaries = []
        for index, group in enumerate(groups, start=1):
            if on_progress:
                on_progress("batch_started", index, len(groups))
            prompt = f"""把下面第 {index}/{len(groups)} 组章节摘要压缩成小说阶段总结。
保留时间线、主线目标、关键转折、人物关系变化、已揭示秘密、未回收伏笔和当前局面。
不要添加原摘要没有的信息。使用清晰的中文 Markdown。

{group}"""
            arc_summaries.append(
                await self.complete(
                    [{"role": "system", "content": "你是小说连续性编辑。"}, {"role": "user", "content": prompt}],
                    max_tokens=max(2048, min(max_tokens, 8192)),
                    stop_event=stop_event,
                )
            )
            if on_progress:
                on_progress("batch_completed", index, len(groups))
        if len(arc_summaries) == 1:
            return arc_summaries[0]
        final_prompt = """将下面各阶段总结合并为一份紧凑的全书前文总览。
保持时间顺序，并包含：主线、重要支线、核心人物当前状态、关键关系、世界观规则、未解决矛盾和伏笔。
不得编造。使用中文 Markdown。\n\n""" + "\n\n---\n\n".join(arc_summaries)
        if on_progress:
            on_progress("merge_started", len(groups), len(groups))
        result = await self.complete(
            [{"role": "system", "content": "你是小说连续性编辑。"}, {"role": "user", "content": final_prompt}],
            max_tokens=max(3072, min(max_tokens, 12_000)),
            stop_event=stop_event,
        )
        if on_progress:
            on_progress("merge_completed", len(groups), len(groups))
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
        chunks = split_text_chunks(new_content)
        if not chunks:
            return {
                "title": title,
                "summary": previous_summary or "（本章没有正文）",
                "_chunk_summaries": [],
                "_character_observations": [],
            }
        partials: list[dict[str, Any]] = []
        character_observations: list[dict[str, Any]] = []
        carry = previous_summary or "（此前没有章节摘要，这是本章开头。）"
        for index, chunk in enumerate(chunks, start=1):
            if stop_event.is_set():
                raise GenerationCancelled("用户停止了增量整理")
            if on_progress:
                on_progress("summary_chunk_started", index, len(chunks))
            prompt = f"""请分析小说《{title}》本次新增正文的第 {index}/{len(chunks)} 段。
只返回 JSON，不要代码围栏。字段包含 title, summary, time, location, pov,
key_events, conflicts, worldbuilding, clues, unresolved, ending_state, character_changes。
本次只输出新增情节摘要，不要输出 characters 或完整人物卡。
不要重复旧事件，重点记录新增变化，但必须承接上段状态。

此前/上段摘要：
{carry}

新增正文：
{chunk}"""
            raw = await self.complete(
                [
                    {"role": "system", "content": "你是严谨的中文小说增量资料编辑。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max(4096, min(max_tokens, 10_000)),
                stop_event=stop_event,
            )
            partial = parse_json_object(raw)
            partial.pop("characters", None)
            partials.append(partial)
            carry = format_chapter_summary(partial)
            if on_progress:
                on_progress("summary_chunk_completed", index, len(chunks))

            if on_progress:
                on_progress("character_chunk_started", index, len(chunks))
            character_prompt = f"""请只提取《{title}》本次新增正文第 {index}/{len(chunks)} 段中的人物资料变化。
只返回 JSON：{{"characters": [...]}}，不要输出情节总结。
人物字段包含 name, aliases, identity, age, appearance, personality, goals,
fears, secrets, speech_style, abilities, relationships, arc, current_state,
facts, inferences, uncertainties, source_chapters。事实、推断、不确定项必须分开。

新增正文：
{chunk}"""
            character_raw = await self.complete(
                [
                    {"role": "system", "content": "你只负责提取小说人物资料变化。"},
                    {"role": "user", "content": character_prompt},
                ],
                max_tokens=max(8192, min(max_tokens + 2048, 12_000)),
                stop_event=stop_event,
            )
            character_value = parse_json_object(character_raw)
            for item in character_value.get("characters", []):
                if isinstance(item, dict) and item.get("name"):
                    character_observations.append(
                        {**item, "source_chapters": item.get("source_chapters") or [title]}
                    )
            if on_progress:
                on_progress("character_chunk_completed", index, len(chunks))

        merge_prompt = f"""请把旧章节摘要与新增片段摘要合并成《{title}》最新的完整章节摘要。
只返回 JSON；保持事件顺序，更新人物当前状态，不得丢失仍有效的伏笔，也不得编造。
字段包含 title, summary, time, location, pov, key_events, conflicts,
worldbuilding, clues, unresolved, ending_state, character_changes。不要输出 characters 或人物卡。

旧章节摘要：
{previous_summary or '（无）'}

新增片段摘要：
{json.dumps(partials, ensure_ascii=False)}"""
        if on_progress:
            on_progress("merge_started", len(chunks), len(chunks))
        raw = await self.complete(
            [
                {"role": "system", "content": "你是小说连续性与资料编辑。"},
                {"role": "user", "content": merge_prompt},
            ],
            max_tokens=max(4096, min(max_tokens, 12_000)),
            stop_event=stop_event,
        )
        result = parse_json_object(raw)
        result.pop("characters", None)
        if on_progress:
            on_progress("merge_completed", len(chunks), len(chunks))
        result["title"] = title
        result["_chunk_summaries"] = partials
        result["_character_observations"] = character_observations
        return result

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
            prompt = """请把下面的小说人物观察合并成人物卡。识别同一人的别名，去重，但不要擅自合并不确定的人物。
只返回 JSON：{"characters": [...]}。
每张卡包含 name, aliases, identity, age, appearance, personality, goals,
fears, secrets, speech_style, abilities, relationships, arc, current_state,
facts, inferences, uncertainties, source_chapters。
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
        final_prompt = """再次合并以下人物卡，处理跨批次的同一人物和别名。只返回 {"characters": [...]}，保留所有字段和来源章节，不得编造。\n\n""" + json.dumps(merged_cards, ensure_ascii=False)
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
3. 若出现新别名或称谓，加入 aliases；不要因为称谓不同就新建人物。
4. facts、inferences、uncertainties、source_chapters 去重合并。
5. facts 只能写正文明确支持的事实；推测放进 inferences；不确定内容放进 uncertainties。
6. 不得编造正文没有支持的信息。

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
        final_prompt = """再次合并以下增量人物卡，处理同一 id、同一姓名或别名对应的重复项。
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
