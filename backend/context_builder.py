from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ContextResult:
    messages: list[dict[str, str]]
    trimmed_exchange_count: int
    estimated_characters: int
    prompt_tokens: int | None = None


def build_messages(
    *,
    system_prompt: str,
    pinned_context: str,
    history: list[dict[str, str]],
    current_user_content: str,
    n_ctx: int,
    project_context: dict[str, str] | None = None,
    trim_by_characters: bool = True,
) -> ContextResult:
    system_parts = [system_prompt.strip()]
    if pinned_context.strip():
        system_parts.append("固定创作资料（请在创作时保持一致）：\n" + pinned_context.strip())
    assets = project_context or {}
    if assets.get("project_summary", "").strip():
        system_parts.append("小说前文总览（仅作连续性参考）：\n" + assets["project_summary"].strip())
    if assets.get("recent_chapters", "").strip():
        system_parts.append("最近章节结构化摘要：\n" + assets["recent_chapters"].strip())
    if assets.get("characters", "").strip():
        system_parts.append("核心人物卡（事实优先，不确定项不得擅自写死）：\n" + assets["characters"].strip())
    if assets.get("facts", "").strip():
        system_parts.append("与本轮相关的结构化事实（保留来源与状态）：\n" + assets["facts"].strip())
    if assets.get("outline", "").strip():
        system_parts.append("已选用的下一章大纲（正文应遵循，可在不违背关键节点时自然发挥）：\n" + assets["outline"].strip())
    system_content = "\n\n".join(part for part in system_parts if part)

    # Chinese fiction often approaches one token per character. Keep a conservative
    # reserve for template tokens and model output while still using recent history.
    input_character_budget = max(2048, int(n_ctx * 0.72))
    fixed_characters = len(system_content) + len(current_user_content)
    remaining = max(0, input_character_budget - fixed_characters)

    pairs = [history[index : index + 2] for index in range(0, len(history), 2)]
    kept_pairs: list[list[dict[str, str]]] = []
    used = 0
    if trim_by_characters:
        for pair in reversed(pairs):
            pair_size = sum(len(message.get("content", "")) for message in pair)
            if kept_pairs and used + pair_size > remaining:
                break
            if not kept_pairs and pair_size > remaining:
                break
            kept_pairs.append(pair)
            used += pair_size
        kept_pairs.reverse()
    else:
        kept_pairs = pairs

    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    for pair in kept_pairs:
        messages.extend(pair)
    messages.append({"role": "user", "content": current_user_content})

    return ContextResult(
        messages=messages,
        trimmed_exchange_count=len(pairs) - len(kept_pairs),
        estimated_characters=sum(len(message["content"]) for message in messages),
    )
