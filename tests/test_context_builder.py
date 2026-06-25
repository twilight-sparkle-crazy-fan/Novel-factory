from backend.context_builder import build_messages


def test_context_keeps_recent_pairs_and_fixed_material() -> None:
    history = []
    for index in range(8):
        history.extend(
            [
                {"role": "user", "content": f"用户{index}" + "甲" * 300},
                {"role": "assistant", "content": f"助手{index}" + "乙" * 300},
            ]
        )

    result = build_messages(
        system_prompt="写作助手",
        pinned_context="人物设定",
        history=history,
        current_user_content="继续写",
        n_ctx=4096,
    )

    assert result.messages[0]["role"] == "system"
    assert "人物设定" in result.messages[0]["content"]
    assert result.messages[-1] == {"role": "user", "content": "继续写"}
    assert result.trimmed_exchange_count > 0
    assert "用户7" in str(result.messages)
    assert "用户0" not in str(result.messages)


def test_context_does_not_modify_current_user_text() -> None:
    value = "第一行\n第二行"
    result = build_messages(
        system_prompt="",
        pinned_context="",
        history=[],
        current_user_content=value,
        n_ctx=8192,
    )
    assert result.messages == [{"role": "user", "content": value}]
