"""History import preserves conversation signals and never reads other groups."""

from copy import deepcopy

import pytest

from app.services.group_history import fetch_group_history, normalize_message


def event():
    return {
        "group_id": 10,
        "message_type": "group",
        "post_type": "message",
        "self_id": 99,
        "user_id": 2,
        "message_id": -12,
        "time": 123,
        "sender": {"nickname": "小明", "card": "阿明"},
        "message": [
            {"type": "text", "data": {"text": "笑死😂"}},
            {"type": "at", "data": {"qq": "3"}},
            {"type": "reply", "data": {"id": "-10"}},
            {"type": "face", "data": {"id": "14"}},
            {"type": "image", "data": {"url": "private-media-url"}},
        ],
    }


def test_normalizer_retains_alias_context_emoji_and_no_media_locations():
    source = event()
    original = deepcopy(source)
    row = normalize_message(source, "10")
    assert row["message_id"] == "-12"
    assert row["nickname"] == "阿明"
    assert row["mentions"] == ["3"] and row["reply_to"] == "-10"
    assert row["emoji"] == ["😂", "QQ表情#14", "图片/表情包"]
    assert "未识别画面" in row["text"]
    assert "private-media-url" not in str(row)
    assert row["is_bot"] is False
    assert source == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"group_id": 20},
        {"message_type": "private"},
        {"post_type": "notice"},
        {"user_id": True},
        {"user_id": 0},
        {"message_id": None},
        {"message": None},
    ],
)
def test_normalizer_rejects_unrelated_or_malformed_payloads(overrides):
    assert normalize_message({**event(), **overrides}, "10") is None


def test_napcat_sent_history_keeps_only_our_bot_as_context():
    source = {**event(), "post_type": "message_sent", "user_id": 99}
    result = normalize_message(source, "10", self_id="99")
    assert result is not None and result["is_bot"] is True
    assert normalize_message({**source, "user_id": 2}, "10", self_id="99") is None


async def test_history_queries_only_requested_group_and_filters_response():
    class Client:
        calls = []

        async def call_action(self, action, params, **kwargs):
            self.calls.append((action, params))
            if action == "get_login_info":
                return {"data": {"user_id": 99}}
            if action == "get_group_member_list":
                return {
                    "data": [
                        {"user_id": 99},
                        {
                            "user_id": 2,
                            "nickname": "小明",
                            "age": 123,
                            "location": "not-for-profile",
                        },
                    ]
                }
            return {"data": {"messages": [event(), {**event(), "group_id": 20}]}}

    client = Client()
    bot, roster, records = await fetch_group_history(client, "10")
    assert bot == "99"
    assert roster == [{"user_id": "2", "nickname": "小明", "card": ""}]
    assert len(records) == 1
    assert all(p["group_id"] == 10 for a, p in client.calls if a != "get_login_info")
    assert all(not a.startswith("send") for a, _ in client.calls)
