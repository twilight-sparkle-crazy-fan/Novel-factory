---
name: novel-factory-writing
description: Operate a running Novel-factory instance through its MCP tools to inspect novel projects, manage conversations and materials, generate or continue fiction, build scene-card outlines, run scene workflows, update character cards, summarize chapters, and preview or apply local selection rewrites. Use when an agent is asked to write, revise, organize, query, or maintain a novel stored in Novel-factory.
---

# Novel-factory Writing

Use the `novel_*` MCP tools against the running Novel-factory application. Keep the browser UI and agent on the same application instance so the user can observe changes.

## Start safely

1. Call `novel_status`.
2. If the runtime is not ready:
   - In local mode, ask the user to start or inspect the local model.
   - In DeepSeek mode, tell the user to enter the key in the page header. Never request, read, echo, store, or pass the API key through MCP.
3. Call `novel_list_projects` and `novel_list_conversations`.
4. Resolve names to exact IDs before any write. Do not invent IDs.
5. Read the relevant object again immediately before a destructive or content-replacing action.

## Choose the workflow

### Continue or draft prose

1. Call `novel_get_conversation` and inspect its selected candidates.
2. Call `novel_get_prompt_preview` when material injection or continuity is uncertain.
3. Call `novel_generate` with one concrete instruction.
4. Read the completed exchange and report which candidate was created. Do not silently select, append, or overwrite unrelated content.
5. Use `novel_regenerate_candidate` for alternatives, `novel_select_candidate` for an explicit choice, and `novel_branch_candidate` when changing an earlier selected reply.

### Plan and write a chapter with scenes

1. Call `novel_get_outline`.
2. If no suitable selected outline exists, call `novel_generate_outline`.
3. Inspect the returned JSON scene cards. Use `novel_save_outline` only after the structure is valid; set `select` only when the user asked to use it.
4. Call `novel_run_scene_workflow`.
5. Stop at fragment review. Let the user review in the front end; use `novel_regenerate_scene_fragment` for a focused retry.
6. Call `novel_polish_scene_workflow` only after the user accepts the fragments or explicitly authorizes autonomous completion.

### Rewrite part of a chapter

1. Call `novel_get_chapter`.
2. Calculate `start` and `end` as Python/Unicode string character offsets into the exact returned `content`.
3. Call `novel_preview_selection_rewrite` with focused guidance.
4. Show or inspect the preview. Do not apply it automatically unless the user asked for an actual change rather than a preview.
5. Call `novel_apply_selection_rewrite` with the exact `source_hash`, `original_text`, range, and replacement returned by the preview.
6. If conflict validation fails, fetch the chapter again and redo the preview. Never force an outdated replacement.

### Maintain the novel library

- Use `novel_get_document` before editing chapter, character, event, or summary data.
- Use `novel_append_chapter` to add generated prose to the library.
- Use `novel_summarize_chapters` after content changes when refreshed summaries and character experiences are desired.
- Use `novel_update_character` only for intentional manual card corrections.
- Use `novel_update_character_event` only to change experience injection.
- Use `novel_merge_characters` only when the two cards are confirmed to represent the same person.
- Use `novel_import_text` only with text the user supplied or explicitly placed in scope.

## Writing rules

- Preserve point of view, tense, naming, chronology, world rules, and character motivation.
- Prefer observable action, dialogue, and specific sensory detail over explanatory summaries.
- Advance the scene; do not restate the previous scene using different words.
- Treat summaries and cards as constraints, not prose to copy.
- When facts conflict, pause and surface the conflict instead of inventing a resolution.
- For local rewrites, preserve the facts and transitions immediately outside the selection.
- Keep the user's guidance higher priority than stylistic defaults.

## Mutation rules

- Queries are safe to perform without confirmation.
- Preview operations may call a model but must not persist prose.
- Applying rewrites, updating cards, importing text, appending chapters, summarizing, and generating candidates are writes within the named project.
- Before `novel_delete_item`, obtain explicit user confirmation and pass `confirmed: true`. Never infer confirmation.
- Do not edit the SQLite database or project files directly. Use MCP tools so the front end stays synchronized.
- If a tool reports another generation in progress, wait or ask the user; do not bypass the coordinator.

## Finish

Read the affected conversation, document, or chapter again. Report:

- what changed;
- which IDs were affected;
- whether generated text is only a preview or already persisted;
- whether summaries now require regeneration;
- any unresolved continuity issue or user review step.
