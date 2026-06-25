import asyncio
import json
from typing import Any

from backend.analysis_service import NovelAnalysisService


class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.call_count = 0
        self.summary_count = 0
        self.max_tokens: list[int] = []

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        _settings: dict[str, Any],
        _stop_event: asyncio.Event,
    ):
        self.prompts.append(messages[-1]["content"])
        self.max_tokens.append(int(_settings["max_tokens"]))
        self.call_count += 1
        prompt = messages[-1]["content"]
        if "请只提取" in prompt:
            value = {
                "characters": [
                    {"name": "林舟", "facts": ["经历停电"], "source_chapters": ["长章"]}
                ]
            }
        elif "合并为一份章节摘要" in prompt:
            value = {"title": "长章", "summary": "停电后众人点亮蜡烛"}
        elif self.summary_count == 0:
            self.summary_count += 1
            value = {"title": "长章", "summary": "第一段发生停电", "characters": []}
        else:
            self.summary_count += 1
            value = {"title": "长章", "summary": "第二段点亮蜡烛", "characters": []}
        yield {"type": "content_delta", "text": json.dumps(value, ensure_ascii=False)}
        yield {"type": "done"}


def test_long_summary_carries_previous_chunk_summary() -> None:
    client = FakeClient()
    service = NovelAnalysisService(client)  # type: ignore[arg-type]
    content = "停电。" + "雨" * 11_995 + "\n\n" + "点亮蜡烛。" + "风" * 100
    progress: list[tuple[str, int, int]] = []

    result = asyncio.run(
        service.summarize_chapter(
            "长章",
            content,
            asyncio.Event(),
            on_progress=lambda stage, index, total: progress.append((stage, index, total)),
            max_tokens=4096,
        )
    )

    assert len(result["_chunk_summaries"]) == 2
    second_summary_prompt = next(prompt for prompt in client.prompts if "第 2/2 个片段" in prompt)
    assert "第一段发生停电" in second_summary_prompt
    assert result["summary"] == "停电后众人点亮蜡烛"
    assert len(result["_character_observations"]) == 2
    character_call_tokens = [
        tokens
        for prompt, tokens in zip(client.prompts, client.max_tokens, strict=True)
        if "请只提取" in prompt
    ]
    assert character_call_tokens == [8192, 8192]
    assert progress[0] == ("summary_chunk_started", 1, 2)
    assert ("character_chunk_completed", 2, 2) in progress
    assert progress[-1] == ("merge_completed", 2, 2)
