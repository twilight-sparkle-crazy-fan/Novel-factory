from __future__ import annotations

import re
from dataclasses import dataclass


CHAPTER_HEADING = re.compile(
    r"(?m)^[\t ]*("
    r"第[0-9０-９零〇一二三四五六七八九十百千万两]+[章回节卷篇]"
    r"(?:[^\n]{0,50})?"
    r"|(?:序章|楔子|引子|尾声|后记|番外(?:[^\n]{0,30})?))"
    r"[\t ]*$"
)


@dataclass(slots=True)
class ImportedText:
    text: str
    encoding: str


@dataclass(slots=True)
class ChapterPart:
    title: str
    content: str


def decode_text(data: bytes) -> ImportedText:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return ImportedText(data.decode("utf-16"), "utf-16")
    for encoding in ("utf-8-sig", "gb18030", "big5"):
        try:
            return ImportedText(data.decode(encoding), encoding)
        except UnicodeDecodeError:
            continue
    return ImportedText(data.decode("utf-8", errors="replace"), "utf-8-replace")


def normalize_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def split_chapters(text: str) -> list[ChapterPart]:
    normalized = normalize_text(text)
    matches = list(CHAPTER_HEADING.finditer(normalized))
    if not matches:
        return [ChapterPart("正文", normalized)]

    parts: list[ChapterPart] = []
    prefix = normalized[: matches[0].start()].strip()
    if prefix:
        parts.append(ChapterPart("序章", prefix))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        title = match.group(1).strip()
        body = normalized[match.end() : end].strip()
        parts.append(ChapterPart(title, body))
    return [part for part in parts if part.content or part.title]


def split_long_text(text: str, max_characters: int = 12_000) -> list[str]:
    """Split long prose on paragraph boundaries, with a hard character fallback."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_characters:
        return [normalized]
    paragraphs = re.split(r"\n\s*\n", normalized)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_characters:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_size = 0
            for start in range(0, len(paragraph), max_characters):
                chunks.append(paragraph[start : start + max_characters])
            continue
        separator = 2 if current else 0
        if current and current_size + separator + len(paragraph) > max_characters:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0
        current.append(paragraph)
        current_size += (2 if len(current) > 1 else 0) + len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks
