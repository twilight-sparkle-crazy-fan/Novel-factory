from __future__ import annotations

from agent import novel_factory_mcp


def test_mcp_initializes_and_lists_unique_tools() -> None:
    initialized = novel_factory_mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    assert initialized["result"]["serverInfo"]["name"] == "novel-factory"
    assert initialized["result"]["capabilities"]["tools"] == {"listChanged": False}

    listed = novel_factory_mcp.handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    names = [tool["name"] for tool in listed["result"]["tools"]]
    assert len(names) == len(set(names))
    assert {
        "novel_generate",
        "novel_get_document",
        "novel_create_document",
        "novel_create_character",
        "novel_extract_characters",
        "novel_preview_selection_rewrite",
        "novel_apply_selection_rewrite",
        "novel_run_scene_workflow",
        "novel_regenerate_scene_fragment",
        "novel_accept_scene_workflow",
        "novel_polish_scene_workflow",
        "novel_branch_candidate",
        "novel_update_project",
        "novel_update_document",
        "novel_select_candidate",
        "novel_select_outline",
        "novel_edit_outline_candidate",
    }.issubset(names)
