from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT = """你是一位专注于中文小说创作的写作助手。
严格遵循用户给出的题材、视角、语气、人物设定和篇幅要求；保持人物行为、称谓、时间线与世界观一致。
当用户要求续写、改写或扩写时，直接输出可用的小说正文，不添加无关的解释、总结或创作说明。
信息不足时采用克制、可延展的处理，不擅自改变核心设定。"""

DEFAULT_GENERATION_SETTINGS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 1600,
    "repeat_penalty": 1.08,
    "seed": None,
}


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    model_path: Path
    llama_server_bin: str
    llama_host: str
    llama_port: int
    app_host: str
    app_port: int
    n_ctx: int
    n_gpu_layers: str
    n_parallel: int
    prompt_cache_mb: int
    cache_type_k: str
    cache_type_v: str
    reasoning: str
    database_path: Path
    auto_start_llama: bool
    max_candidates: int
    llama_start_timeout: float
    llama_log_max_bytes: int
    llama_log_backup_count: int
    experimental_material_system: bool

    @property
    def llama_base_url(self) -> str:
        return f"http://{self.llama_host}:{self.llama_port}"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            project_root=PROJECT_ROOT,
            model_path=_resolve_path(
                os.getenv(
                    "MODEL_PATH",
                    "model/your-model.gguf",
                )
            ),
            llama_server_bin=os.getenv("LLAMA_SERVER_BIN", "llama-server"),
            llama_host=os.getenv("LLAMA_HOST", "127.0.0.1"),
            llama_port=int(os.getenv("LLAMA_PORT", "8080")),
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=int(os.getenv("APP_PORT", "8000")),
            n_ctx=int(os.getenv("N_CTX", "32768")),
            n_gpu_layers=os.getenv("N_GPU_LAYERS", "auto"),
            n_parallel=int(os.getenv("N_PARALLEL", "1")),
            prompt_cache_mb=int(os.getenv("PROMPT_CACHE_MB", "512")),
            cache_type_k=os.getenv("CACHE_TYPE_K", "q8_0"),
            cache_type_v=os.getenv("CACHE_TYPE_V", "q8_0"),
            reasoning=os.getenv("REASONING", "off"),
            database_path=_resolve_path(os.getenv("DATABASE_PATH", "data/novel-factory.db")),
            auto_start_llama=_as_bool(os.getenv("AUTO_START_LLAMA"), True),
            max_candidates=int(os.getenv("MAX_CANDIDATES_PER_EXCHANGE", "20")),
            llama_start_timeout=float(os.getenv("LLAMA_START_TIMEOUT", "180")),
            llama_log_max_bytes=int(os.getenv("LLAMA_LOG_MAX_BYTES", str(5 * 1024 * 1024))),
            llama_log_backup_count=int(os.getenv("LLAMA_LOG_BACKUP_COUNT", "3")),
            experimental_material_system=_as_bool(os.getenv("EXPERIMENTAL_MATERIAL_SYSTEM"), False),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
