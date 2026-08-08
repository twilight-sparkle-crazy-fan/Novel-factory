import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(str(values["id"]))


def test_frontend_ids_are_unique_and_static_selectors_exist() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    parser = IdCollector()
    parser.feed(html)

    duplicates = [value for value, count in Counter(parser.ids).items() if count > 1]
    assert duplicates == []

    referenced_ids = set(re.findall(r'document\.querySelector\("#([a-zA-Z0-9_-]+)"\)', javascript))
    missing = referenced_ids - set(parser.ids)
    assert missing == set()


def test_frontend_uses_tab_scoped_conversation_claims() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    assert "BroadcastChannel(\"llm4chat-window-isolation-v1\")" in javascript
    assert "sessionStorage.setItem(TAB_CONVERSATION_KEY" in javascript
    assert "conversationOpenElsewhere" in javascript


def test_frontend_renders_material_relationship_network_graph() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    stylesheet = (ROOT / "frontend/styles/app.css").read_text(encoding="utf-8")

    assert "function renderMaterialRelationshipGraph" in javascript
    assert "renderMaterialRelationshipGraph(relationshipNetwork)" in javascript
    assert "material-network-graph" in stylesheet
    assert "material-network-nodes" in stylesheet


def test_frontend_supports_field_level_import_conflict_resolution() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    stylesheet = (ROOT / "frontend/styles/app.css").read_text(encoding="utf-8")

    assert "function renderMaterialImportConflictFields" in javascript
    assert "material-conflict-field:checked" in javascript
    assert "resolution.fields = fields" in javascript
    assert "material-conflict-row" in stylesheet


def test_frontend_keeps_material_package_migration_backend_compatibility_hidden() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    api = (ROOT / "frontend/scripts/api.js").read_text(encoding="utf-8")

    assert 'id="migrate-material-package"' not in html
    assert 'id="rebuild-material-system"' not in html
    assert 'id="preview-material-plan"' not in html
    assert 'id="preview-material-snapshot"' in html
    assert "previewMaterialSnapshot" in javascript
    assert "migrateMaterialPackageFile" in javascript
    assert "promptMaterialPackageMigration" in javascript
    assert "report.checks?.schema !== \"needs_migration\"" in javascript
    assert "/api/experimental/material-system/packages/migrate" in api


def test_frontend_renders_material_budget_summary() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")

    assert "function formatMaterialBudgetSummary" in javascript
    assert "plan.budget_summary" in javascript
    assert "full_prompt" in javascript


def test_frontend_can_select_visible_material_review_items() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")

    assert "function setVisibleMaterialReviewSelection" in javascript
    assert "function updateMaterialReviewSelectionState" in javascript
    assert "select-visible-material-reviews" in javascript
    assert "clear-material-review-selection" in javascript
    assert "material-review-selection-count" in javascript


def test_frontend_explains_material_package_import_block_reason() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")

    assert "function materialPackageImportBlockReason" in javascript
    assert "materialPackageImportBlockReason(report, mode)" in javascript
    assert '["merge", "replace_material"].includes(mode)' in javascript
    assert "分析包暂不能作为纯新文件导入" not in javascript


def test_character_editor_uses_fields_without_exposing_json() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")

    assert 'id="character-editor-dialog"' in html
    assert 'id="character-name"' in html
    assert 'id="character-trait-list"' in html
    assert 'id="character-world-setting"' in html
    assert "JSON.stringify(character.card" not in javascript
    assert "保存 JSON 人物卡" not in javascript
    assert "renderCharacterProfile(character)" in javascript


def test_api_token_warning_is_prominent() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    stylesheet = (ROOT / "frontend/styles/app.css").read_text(encoding="utf-8")

    assert 'class="api-token-warning"' in html
    assert ".api-token-warning" in stylesheet
    assert "color: #000" in stylesheet
    assert "font-weight: 700" in stylesheet


def test_scene_workflow_can_be_accepted_without_polishing() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    api = (ROOT / "frontend/scripts/api.js").read_text(encoding="utf-8")

    assert 'id="scene-fragment-accept"' in html
    assert "acceptSceneWorkflowDraft" in javascript
    assert "acceptSceneWorkflow" in api
    assert "/scene-workflow/accept" in api
    assert 'event === "model_retry"' in javascript


def test_scene_workflow_results_use_full_workflow_regeneration() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")

    assert 'const regenerateLabel = isSceneWorkflowExchange(exchange) ? "重新编排" : "重新生成"' in javascript
    assert "if (isSceneWorkflowExchange(exchange)) rerunSceneWorkflow(exchange);" in javascript
    assert "await runSceneWorkflow(sceneWorkflowInstruction(exchange));" in javascript


def test_outline_editor_hides_json_behind_structured_scene_cards() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    stylesheet = (ROOT / "frontend/styles/app.css").read_text(encoding="utf-8")

    assert 'id="outline-structured-editor"' in html
    assert 'id="outline-content" hidden' in html
    assert "function renderOutlineStructuredEditor" in javascript
    assert "function collectOutlineEditorData" in javascript
    assert "function handleOutlineEditorClick" in javascript
    assert "outline-scene-card" in stylesheet


def test_prompt_snapshot_button_is_a_show_hide_toggle() -> None:
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")

    assert 'dataset.view === "prompt-snapshot"' in javascript
    assert '"隐藏当前快照"' in javascript
    assert 'elements.previewMaterialSnapshot.textContent = "当前提示词快照"' in javascript
