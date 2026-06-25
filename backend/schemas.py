from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .config import DEFAULT_GENERATION_SETTINGS


class GenerationSettings(BaseModel):
    temperature: float = Field(default=0.9, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_tokens: int = Field(default=1600, ge=16, le=16384)
    repeat_penalty: float = Field(default=1.08, ge=0.5, le=2)
    seed: int | None = None


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", max_length=100)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=100)
    system_prompt: str | None = Field(default=None, max_length=100_000)
    pinned_context: str | None = Field(default=None, max_length=500_000)
    generation_settings: GenerationSettings | None = None
    document_id: str | None = None

    def changes(self) -> dict[str, Any]:
        values = self.model_dump(exclude_none=True)
        if "document_id" in self.model_fields_set:
            values["document_id"] = self.document_id
        if "generation_settings" in values:
            values["generation_settings"] = {
                **DEFAULT_GENERATION_SETTINGS,
                **values["generation_settings"],
            }
        return values


class GenerateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1_000_000)
    settings: GenerationSettings | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("输入内容不能为空")
        return value.strip()


class RegenerateRequest(BaseModel):
    settings: GenerationSettings | None = None


class SelectionRequest(BaseModel):
    candidate_id: str


class BranchRequest(BaseModel):
    candidate_id: str


class RuntimeContextRequest(BaseModel):
    context_size: int

    @field_validator("context_size")
    @classmethod
    def supported_context_size(cls, value: int) -> int:
        if value not in {32768, 65536}:
            raise ValueError("上下文只支持 32768 或 65536")
        return value


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    global_summary: str | None = Field(default=None, max_length=1_000_000)
    summary_enabled: bool | None = None


class ChapterUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    content: str | None = Field(default=None, max_length=5_000_000)
    edited_summary: str | None = Field(default=None, max_length=500_000)


class CharacterUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    aliases: list[str] | None = None
    card: dict[str, Any] | None = None
    prompt_text: str | None = Field(default=None, max_length=500_000)
    enabled: bool | None = None


class SummarizeRequest(BaseModel):
    document_id: str | None = None
    chapter_ids: list[str] | None = None
    start_position: int | None = Field(default=None, ge=1)
    end_position: int | None = Field(default=None, ge=1)
    resume_job_id: str | None = None
    regenerate: bool = False
    max_tokens: int = Field(default=8192, ge=1024, le=16384)


class ContextCountRequest(BaseModel):
    content: str = Field(default="", max_length=1_000_000)


class OutlineGenerateRequest(BaseModel):
    instruction: str = Field(
        default="请规划紧接当前进度的下一章。",
        min_length=1,
        max_length=20_000,
    )
    settings: GenerationSettings | None = None


class OutlineUpdateRequest(BaseModel):
    enabled: bool


class OutlineCandidateEditRequest(BaseModel):
    content: str = Field(max_length=500_000)


class OutlineCandidateSaveRequest(BaseModel):
    outline_id: str | None = None
    instruction: str = Field(
        default="请规划紧接当前进度的下一章。",
        min_length=1,
        max_length=20_000,
    )
    content: str = Field(min_length=1, max_length=500_000)
    select: bool = False
    settings: GenerationSettings | None = None


class ProjectAppendRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2_000_000)
    chapter_id: str | None = None
    document_id: str | None = None
    title: str | None = Field(default=None, max_length=200)
    source_candidate_id: str | None = None
    max_tokens: int = Field(default=8192, ge=1024, le=16384)
    summarize_now: bool = False

    @field_validator("content")
    @classmethod
    def append_content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("增量正文不能为空")
        return value.strip()


class DocumentUpdate(BaseModel):
    filename: str | None = Field(default=None, max_length=255)
    global_summary: str | None = Field(default=None, max_length=1_000_000)
    library_enabled: bool | None = None
    summary_enabled: bool | None = None
    recent_chapters_enabled: bool | None = None
    characters_enabled: bool | None = None
    facts_enabled: bool | None = None


class StoryFactUpdate(BaseModel):
    state: str | None = Field(default=None, max_length=10_000)
    status: str | None = Field(default=None, max_length=50)
