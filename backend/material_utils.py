from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_json_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
