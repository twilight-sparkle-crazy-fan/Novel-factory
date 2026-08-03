#!/usr/bin/env python3
"""LLM4chat 端到端功能测试脚本。

依次执行：
1. 导入测试 TXT
2. 触发资料分析（章节摘要 + 人物卡 + 全书总览 + 结构化事实）
3. 检查分析结果
4. 生成下一章场景卡（大纲）
5. 续写下一章正文

每一步都会把请求/响应/状态写入 test-logs/ 目录。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from urllib.parse import quote

import httpx

BASE = "http://127.0.0.1:8000"
LOG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LOG_DIR.parent
TXT_PATH = PROJECT_ROOT / "异世界：符文刻印师.txt"

# 累积的问题清单
issues: list[dict] = []


def log(name: str, content: str) -> None:
    (LOG_DIR / name).write_text(content, encoding="utf-8")
    print(f"  [log] {name} ({len(content)} bytes)")


def record_issue(category: str, severity: str, title: str, detail: str) -> None:
    issue = {"category": category, "severity": severity, "title": title, "detail": detail}
    issues.append(issue)
    tag = "BUG" if severity == "bug" else "WARN"
    print(f"  [{tag}] {title}: {detail[:120]}")


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def http_get(path: str, **kwargs) -> httpx.Response:
    with httpx.Client(timeout=30) as client:
        return client.get(BASE + path, **kwargs)


def http_post(path: str, **kwargs) -> httpx.Response:
    timeout = kwargs.pop("timeout", 60)
    with httpx.Client(timeout=timeout) as client:
        return client.post(BASE + path, **kwargs)


def http_put(path: str, **kwargs) -> httpx.Response:
    with httpx.Client(timeout=30) as client:
        return client.put(BASE + path, **kwargs)


def http_patch(path: str, **kwargs) -> httpx.Response:
    with httpx.Client(timeout=30) as client:
        return client.patch(BASE + path, **kwargs)


# ---------------------------------------------------------------------------
# Step 0: 健康检查
# ---------------------------------------------------------------------------
step("Step 0: 健康检查与运行时状态")
resp = http_get("/api/health")
health = resp.json()
print(f"  health: {health}")
log("00-health.json", json.dumps(health, ensure_ascii=False, indent=2))

resp = http_get("/api/runtime")
runtime = resp.json()
print(f"  runtime: status={runtime['status']} model={runtime['model_name']} ctx={runtime['context_size']}")
log("00-runtime.json", json.dumps(runtime, ensure_ascii=False, indent=2))

if runtime.get("app") or health.get("app") == "Novel-factory":
    record_issue(
        "naming", "warn", "应用标识名与项目名不一致",
        f"/api/health 返回 app='Novel-factory'，但项目目录和文档均为 LLM4chat",
    )

# ---------------------------------------------------------------------------
# Step 1: 导入 TXT
# ---------------------------------------------------------------------------
step("Step 1: 导入测试 TXT")
if not TXT_PATH.is_file():
    print(f"  测试 TXT 不存在: {TXT_PATH}")
    sys.exit(1)

txt_bytes = TXT_PATH.read_bytes()
print(f"  TXT 文件: {TXT_PATH.name} ({len(txt_bytes)} bytes)")
t0 = time.time()
resp = http_post(
    "/api/projects/default/import-txt",
    content=txt_bytes,
    headers={"x-filename": quote(TXT_PATH.name)},
    timeout=60,
)
t1 = time.time()
import_result = resp.json()
print(f"  import: status={resp.status_code} 耗时={t1-t0:.2f}s")
log("01-import-response.json", json.dumps(import_result, ensure_ascii=False, indent=2))

if resp.status_code != 201:
    record_issue("import", "bug", "导入 TXT 失败", f"HTTP {resp.status_code}: {import_result}")
    log("99-issues.json", json.dumps(issues, ensure_ascii=False, indent=2))
    sys.exit(1)

document_id = import_result.get("document_id") or import_result.get("id")
if not document_id and isinstance(import_result.get("document"), dict):
    document_id = import_result["document"].get("id")
project_id = import_result.get("project_id") or import_result.get("document", {}).get("project_id", "default")
chapters = import_result.get("chapters", [])
print(f"  document_id={document_id} chapters={len(chapters)}")
if not document_id:
    record_issue("import", "bug", "导入响应缺少 document_id", str(import_result))
if len(chapters) != 1:
    record_issue(
        "import", "bug" if len(chapters) == 0 else "warn",
        "章节切分数量不是 1",
        f"单章 TXT 切出 {len(chapters)} 章",
    )

# 检查工作区
resp = http_get(f"/api/documents/{document_id}/workspace")
workspace = resp.json()
log("01-workspace-after-import.json", json.dumps(workspace, ensure_ascii=False, indent=2))
print(f"  workspace: chapters={len(workspace.get('chapters', []))} characters={len(workspace.get('characters', []))} facts={len(workspace.get('facts', []))}")

# ---------------------------------------------------------------------------
# Step 2: 触发资料分析（总结 + 人物卡 + 全书总览 + 事实）
# ---------------------------------------------------------------------------
step("Step 2: 触发资料分析（SSE 流式）")
chapter_ids = [c["id"] for c in workspace.get("chapters", [])]
summarize_payload = {
    "document_id": document_id,
    "chapter_ids": chapter_ids,
    "max_tokens": 8192,
}
print(f"  payload: {summarize_payload}")
log("02-summarize-request.json", json.dumps(summarize_payload, ensure_ascii=False, indent=2))

t0 = time.time()
events: list[dict] = []
event_types: dict[str, int] = {}
final_status = "unknown"
chars_completed_payload = None
job_completed_payload = None
error_payload = None

with httpx.Client(timeout=1800) as client:
    with client.stream(
        "POST", BASE + f"/api/projects/{project_id}/summarize",
        json=summarize_payload,
    ) as response:
        print(f"  SSE 响应状态: {response.status_code}")
        if response.status_code != 200:
            body = response.read().decode("utf-8", errors="replace")
            print(f"  错误响应: {body[:500]}")
            record_issue("summarize", "bug", "总结请求返回非 200", f"HTTP {response.status_code}: {body[:300]}")
            log("02-summarize-error.txt", body)
        else:
            event_name = ""
            data_buf = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_buf = line[5:].strip()
                elif line == "":
                    if event_name:
                        try:
                            data = json.loads(data_buf) if data_buf else {}
                        except json.JSONDecodeError:
                            data = {"_raw": data_buf}
                        events.append({"event": event_name, "data": data})
                        event_types[event_name] = event_types.get(event_name, 0) + 1
                        if event_name == "characters_completed":
                            chars_completed_payload = data
                        elif event_name == "job_completed":
                            job_completed_payload = data
                        elif event_name == "error":
                            error_payload = data
                        # 实时打印关键事件
                        if event_name in {"chapter_started", "chapter_completed", "characters_started", "characters_completed", "job_completed", "error", "analysis_progress"}:
                            summary = {k: v for k, v in data.items() if k != "summary"}
                            print(f"    {event_name}: {json.dumps(summary, ensure_ascii=False)[:160]}")
                    event_name = ""
                    data_buf = ""

t1 = time.time()
print(f"  总结耗时: {t1-t0:.1f}s, 事件总数: {len(events)}")
print(f"  事件类型分布: {event_types}")
log("02-summarize-events.json", json.dumps(events, ensure_ascii=False, indent=2))
log("02-summarize-event-types.json", json.dumps(event_types, ensure_ascii=False, indent=2))

if error_payload:
    record_issue("summarize", "bug", "总结过程中收到 error 事件", json.dumps(error_payload, ensure_ascii=False)[:300])
    final_status = "error"
elif job_completed_payload:
    final_status = "completed"
    print(f"  job_completed: {json.dumps(job_completed_payload, ensure_ascii=False)[:200]}")
else:
    record_issue("summarize", "bug", "总结未收到 job_completed 事件", f"最后事件类型: {event_types}")
    final_status = "incomplete"

# ---------------------------------------------------------------------------
# Step 3: 检查分析结果
# ---------------------------------------------------------------------------
step("Step 3: 检查分析结果")
resp = http_get(f"/api/documents/{document_id}/workspace")
workspace_after = resp.json()
log("03-workspace-after-summarize.json", json.dumps(workspace_after, ensure_ascii=False, indent=2))

chapters_after = workspace_after.get("chapters", [])
characters_after = workspace_after.get("characters", [])
facts_after = workspace_after.get("facts", [])
print(f"  chapters={len(chapters_after)} characters={len(characters_after)} facts={len(facts_after)}")

for ch in chapters_after:
    print(f"  章节 [{ch.get('position')}] {ch.get('title')}: status={ch.get('status')} chunks={len(ch.get('chunks', []))}")
    if ch.get("status") != "completed":
        record_issue("summarize", "bug", "章节状态非 completed", f"章节 {ch.get('title')} status={ch.get('status')}")
    summary = ch.get("summary") or {}
    if isinstance(summary, dict):
        summary_text = summary.get("summary", "")
        print(f"    摘要长度: {len(summary_text)} 字")
        if len(summary_text) < 50:
            record_issue("summarize", "warn", "章节摘要过短", f"章节 {ch.get('title')} 摘要仅 {len(summary_text)} 字")
    obs = ch.get("character_observations") or []
    print(f"    人物观察: {len(obs)} 条")

print(f"\n  人物卡 ({len(characters_after)}):")
for char in characters_after:
    name = char.get("name")
    aliases = char.get("aliases", [])
    enabled = char.get("enabled")
    card = char.get("card") or {}
    print(f"    - {name} (别名={aliases}, enabled={enabled})")
    card_keys = [k for k in card.keys() if card.get(k)]
    print(f"      卡片字段: {card_keys}")
    if not name:
        record_issue("characters", "bug", "人物卡缺少 name", str(char)[:200])
    if not card.get("facts") and not card.get("identity") and not card.get("personality"):
        record_issue("characters", "warn", "人物卡字段稀疏", f"{name}: card 字段几乎为空")

print(f"\n  结构化事实 ({len(facts_after)}):")
for fact in facts_after[:10]:
    print(f"    - [{fact.get('fact_type')}] {fact.get('state', '')[:60]}")

# 全书总览
document = workspace_after.get("document", {})
global_summary = document.get("global_summary", "")
print(f"\n  全书总览长度: {len(global_summary)} 字")
if not global_summary:
    record_issue("summarize", "warn", "全书总览为空", "global_summary 为空字符串")
elif len(global_summary) < 100:
    record_issue("summarize", "warn", "全书总览过短", f"仅 {len(global_summary)} 字")

# ---------------------------------------------------------------------------
# Step 4: 生成场景卡（大纲）
# ---------------------------------------------------------------------------
step("Step 4: 生成场景卡（大纲）")
# 先创建一个对话，绑定 document
conv_resp = http_post(
    "/api/conversations",
    json={"title": "测试-场景卡与续写"},
)
print(f"  创建对话: status={conv_resp.status_code}")
conversation = conv_resp.json()
conversation_id = conversation["id"]
log("04-conversation-create.json", json.dumps(conversation, ensure_ascii=False, indent=2))

# 绑定 document
patch_resp = http_patch(
    f"/api/conversations/{conversation_id}",
    json={"document_id": document_id},
)
print(f"  绑定 document: status={patch_resp.status_code}")

# 生成场景卡（大纲）
outline_payload = {
    "instruction": "请把紧接当前进度的下一章拆成 JSON 场景卡。",
    "settings": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 2000, "repeat_penalty": 1.08, "seed": None},
}
log("04-outline-request.json", json.dumps(outline_payload, ensure_ascii=False, indent=2))

t0 = time.time()
outline_events: list[dict] = []
outline_content = ""
outline_candidate_id = None
outline_error = None
with httpx.Client(timeout=600) as client:
    with client.stream(
        "POST", BASE + f"/api/conversations/{conversation_id}/outline/generate",
        json=outline_payload,
    ) as response:
        print(f"  outline SSE 状态: {response.status_code}")
        if response.status_code != 200:
            body = response.read().decode("utf-8", errors="replace")
            print(f"  错误: {body[:500]}")
            record_issue("outline", "bug", "场景卡生成返回非 200", f"HTTP {response.status_code}: {body[:300]}")
            log("04-outline-error.txt", body)
        else:
            event_name = ""
            data_buf = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_buf = line[5:].strip()
                elif line == "":
                    if event_name:
                        try:
                            data = json.loads(data_buf) if data_buf else {}
                        except json.JSONDecodeError:
                            data = {"_raw": data_buf}
                        outline_events.append({"event": event_name, "data": data})
                        if event_name == "content_delta":
                            outline_content += data.get("text", "")
                        elif event_name == "candidate_created":
                            outline_candidate_id = data.get("candidate", {}).get("id")
                        elif event_name == "done":
                            outline_candidate_id = data.get("candidate_id", outline_candidate_id)
                        elif event_name == "error":
                            outline_error = data
                        if event_name in {"candidate_created", "done", "error"}:
                            print(f"    {event_name}: {json.dumps(data, ensure_ascii=False)[:160]}")
                    event_name = ""
                    data_buf = ""

t1 = time.time()
print(f"  场景卡耗时: {t1-t0:.1f}s, 内容长度: {len(outline_content)}")
log("04-outline-events.json", json.dumps(outline_events, ensure_ascii=False, indent=2))
log("04-outline-content.txt", outline_content)

if outline_error:
    record_issue("outline", "bug", "场景卡生成收到 error", json.dumps(outline_error, ensure_ascii=False)[:300])
elif not outline_content:
    record_issue("outline", "bug", "场景卡内容为空", "未收到任何 content_delta")
elif len(outline_content) < 100:
    record_issue("outline", "warn", "场景卡内容过短", f"仅 {len(outline_content)} 字")

# 检查场景卡是否是有效 JSON
try:
    parsed = json.loads(outline_content)
    if isinstance(parsed, dict) and "scenes" in parsed:
        print(f"  场景卡解析成功: {len(parsed['scenes'])} 个场景")
    elif isinstance(parsed, list):
        print(f"  场景卡解析成功: {len(parsed)} 个场景（列表格式）")
    else:
        record_issue("outline", "warn", "场景卡 JSON 结构非预期", f"顶层类型: {type(parsed).__name__}")
except json.JSONDecodeError as e:
    record_issue("outline", "warn", "场景卡不是有效 JSON", f"解析错误: {e}")

# ---------------------------------------------------------------------------
# Step 5: 续写
# ---------------------------------------------------------------------------
step("Step 5: 续写下一章")
generate_payload = {
    "content": "请续写第二章，承接第一章结尾林默与苏月前往风语镇的旅程。约 1200 字，保持林默视角，延续第一人称心理描写。",
    "settings": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1600, "repeat_penalty": 1.08, "seed": None},
}
log("05-generate-request.json", json.dumps(generate_payload, ensure_ascii=False, indent=2))

t0 = time.time()
gen_events: list[dict] = []
gen_content = ""
gen_reasoning = ""
gen_candidate_id = None
gen_exchange_id = None
gen_error = None
gen_prompt_tokens = None
gen_completion_tokens = None
trimmed_count = None
with httpx.Client(timeout=600) as client:
    with client.stream(
        "POST", BASE + f"/api/conversations/{conversation_id}/generate",
        json=generate_payload,
    ) as response:
        print(f"  generate SSE 状态: {response.status_code}")
        if response.status_code != 200:
            body = response.read().decode("utf-8", errors="replace")
            print(f"  错误: {body[:500]}")
            record_issue("generation", "bug", "续写返回非 200", f"HTTP {response.status_code}: {body[:300]}")
            log("05-generate-error.txt", body)
        else:
            event_name = ""
            data_buf = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_buf = line[5:].strip()
                elif line == "":
                    if event_name:
                        try:
                            data = json.loads(data_buf) if data_buf else {}
                        except json.JSONDecodeError:
                            data = {"_raw": data_buf}
                        gen_events.append({"event": event_name, "data": data})
                        if event_name == "candidate_created":
                            gen_candidate_id = data.get("candidate", {}).get("id")
                            gen_exchange_id = data.get("exchange_id")
                            trimmed_count = data.get("trimmed_exchange_count")
                            gen_prompt_tokens = data.get("prompt_tokens")
                        elif event_name == "content_delta":
                            gen_content += data.get("text", "")
                        elif event_name == "reasoning_delta":
                            gen_reasoning += data.get("text", "")
                        elif event_name == "done":
                            gen_candidate_id = data.get("candidate_id", gen_candidate_id)
                            exch = data.get("exchange", {})
                            gen_prompt_tokens = gen_prompt_tokens or exch.get("prompt_tokens")
                            gen_completion_tokens = exch.get("completion_tokens")
                        elif event_name == "error":
                            gen_error = data
                        if event_name in {"candidate_created", "done", "error", "cancelled"}:
                            summary = {k: v for k, v in data.items() if k not in {"candidate", "exchange", "user_content"}}
                            print(f"    {event_name}: {json.dumps(summary, ensure_ascii=False)[:160]}")
                    event_name = ""
                    data_buf = ""

t1 = time.time()
print(f"  续写耗时: {t1-t0:.1f}s")
print(f"  正文长度: {len(gen_content)} 字")
print(f"  推理长度: {len(gen_reasoning)} 字")
print(f"  prompt_tokens={gen_prompt_tokens} completion_tokens={gen_completion_tokens} trimmed={trimmed_count}")
log("05-generate-events.json", json.dumps(gen_events, ensure_ascii=False, indent=2))
log("05-generate-content.txt", gen_content)
if gen_reasoning:
    log("05-generate-reasoning.txt", gen_reasoning)

if gen_error:
    record_issue("generation", "bug", "续写收到 error", json.dumps(gen_error, ensure_ascii=False)[:300])
elif not gen_content:
    record_issue("generation", "bug", "续写内容为空", "未收到任何 content_delta")
elif len(gen_content) < 200:
    record_issue("generation", "warn", "续写内容过短", f"仅 {len(gen_content)} 字")

if gen_prompt_tokens and gen_prompt_tokens > 60000:
    record_issue("generation", "warn", "prompt_tokens 过高", f"prompt_tokens={gen_prompt_tokens}，上下文接近上限")

# 检查续写后对话状态
resp = http_get(f"/api/conversations/{conversation_id}")
conv_after = resp.json()
log("05-conversation-after.json", json.dumps(conv_after, ensure_ascii=False, indent=2))
exchanges = conv_after.get("exchanges", [])
print(f"  对话 exchanges: {len(exchanges)}")
for ex in exchanges:
    cands = ex.get("candidates", [])
    print(f"    exchange {ex.get('position')}: candidates={len(cands)} selected={ex.get('selected_candidate_id')}")
    for c in cands:
        print(f"      candidate {c.get('candidate_index')}: status={c.get('status')} len={len(c.get('content',''))} prompt={c.get('prompt_tokens')} comp={c.get('completion_tokens')}")

# ---------------------------------------------------------------------------
# Step 6: 汇总
# ---------------------------------------------------------------------------
step("Step 6: 汇总问题")
log("99-issues.json", json.dumps(issues, ensure_ascii=False, indent=2))
print(f"\n共记录 {len(issues)} 个问题:")
for i, iss in enumerate(issues, 1):
    tag = "BUG" if iss["severity"] == "bug" else "WARN"
    print(f"  {i}. [{tag}][{iss['category']}] {iss['title']}")
    print(f"     {iss['detail'][:150]}")

print(f"\n日志文件保存在: {LOG_DIR}")
print("完成。")
