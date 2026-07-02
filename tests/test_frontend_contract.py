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


def test_frontend_exposes_material_package_schema_migration() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend/scripts/app.js").read_text(encoding="utf-8")
    api = (ROOT / "frontend/scripts/api.js").read_text(encoding="utf-8")

    assert 'id="migrate-material-package"' in html
    assert "migrateMaterialPackageFile" in javascript
    assert "/api/experimental/material-system/packages/migrate" in api
