import pytest

from app.onebot.messages import extract_plain_text, is_at_bot, make_reply, split_text
from app.onebot.models import MessageSegment


def segment(kind: str, **data) -> MessageSegment:
    return MessageSegment(type=kind, data=data)


@pytest.mark.parametrize("qq", ["12345", 12345])
def test_recognizes_exact_bot_mention(qq):
    assert is_at_bot([segment("at", qq=qq)], 12345)


@pytest.mark.parametrize("qq", ["all", "bot nickname", "67890", "012345", " 12345", None, True])
def test_nonbot_mentions_never_trigger(qq):
    assert not is_at_bot([segment("at", qq=qq)], 12345)


def test_plain_text_and_reply_metadata_do_not_count_as_mentions():
    assert not is_at_bot(
        [segment("text", text="@12345 @机器人 [CQ:at,qq=12345]"), segment("reply", qq="12345")],
        12345,
    )
    assert not is_at_bot([], 12345)


def test_extracts_multiple_text_segments_without_introducing_spaces():
    segments = [
        segment("at", qq="12345"),
        segment("text", text=" 什么是"),
        segment("text", text=" Transformer？🐱 "),
    ]
    assert extract_plain_text(segments, 12345) == "什么是 Transformer？🐱"


def test_extracts_only_text_ignoring_unknown_and_malformed_payloads():
    segments = [
        segment("at", qq="67890"),
        segment("reply", id="99"),
        segment("image", file="image.jpg"),
        segment("face", id="1"),
        segment("extension", text="not text"),
        segment("text", text=None),
        segment("text", text=123),
        segment("text"),
        segment("text", text=" [CQ:at,qq=12345] 原样保留 "),
    ]
    assert extract_plain_text(segments, 12345) == "[CQ:at,qq=12345] 原样保留"


@pytest.mark.parametrize("segments", [[], [segment("text", text=" \n\t ")]])
def test_empty_text(segments):
    assert extract_plain_text(segments, 12345) == ""


def test_reply_segments_keep_unicode_and_cq_content_literal():
    text = "你好🐱 [CQ:at,qq=all]"
    assert make_reply(-99, text) == [
        {"type": "reply", "data": {"id": "-99"}},
        {"type": "text", "data": {"text": text}},
    ]


@pytest.mark.parametrize("text", ["", " ", "\n\t\r\n"])
def test_splitting_empty_content_returns_no_messages(text):
    assert split_text(text, 20) == []


def test_exact_character_boundary_does_not_truncate():
    assert split_text("  你好🐱世界  ", 5) == ["你好🐱世界"]
    assert split_text("甲" * 15, 5) == ["甲" * 5] * 3


def test_splitting_prefers_paragraphs_over_later_sentence_boundaries():
    assert split_text("第一段\n\n第二句。第三句。\n最后一段", 10) == [
        "第一段",
        "第二句。第三句。",
        "最后一段",
    ]


def test_splitting_prefers_sentence_ends_to_hard_cuts():
    assert split_text("第一句。第二句。第三句。", 5) == ["第一句。", "第二句。", "第三句。"]
    assert split_text("One. Two? Three!", 7) == ["One.", "Two?", "Three!"]


def test_sentence_splitting_does_not_treat_decimal_points_as_full_stops():
    assert split_text("12.3456789", 5) == ["12.34", "56789"]


def test_hard_splitting_counts_unicode_characters():
    assert split_text("🐱你好世界", 2) == ["🐱你", "好世", "界"]


def test_paragraph_boundary_preserves_following_code_indentation():
    assert split_text("开始\n    print(1)\n结束", 12) == ["开始", "    print(1)", "结束"]


def test_reply_truncation_is_visible_and_respects_all_limits():
    chunks = split_text("问题回答" * 100, 20)
    assert len(chunks) == 3
    assert all(0 < len(chunk) <= 20 for chunk in chunks)
    assert chunks[-1].endswith("…（内容已截断）")
    assert chunks[0] == "问题回答" * 5


@pytest.mark.parametrize("max_chars", range(1, 12))
def test_even_tiny_limits_include_a_bounded_truncation_marker(max_chars):
    chunks = split_text("🦊" * 100, max_chars)
    assert len(chunks) == 3
    assert all(0 < len(chunk) <= max_chars for chunk in chunks)
    assert "…" in chunks[-1]


@pytest.mark.parametrize("max_chunks", [1, 2, 3])
def test_maximum_chunk_count_is_enforced(max_chunks):
    chunks = split_text("内容" * 100, 20, max_chunks)
    assert len(chunks) == max_chunks
    assert "截断" in chunks[-1]


@pytest.mark.parametrize("max_chars", [0, -1, 1.5, True])
def test_invalid_character_limits_are_rejected(max_chars):
    with pytest.raises(ValueError):
        split_text("内容", max_chars)


@pytest.mark.parametrize("max_chunks", [0, -1, 4, 1.5, True])
def test_invalid_chunk_limits_are_rejected(max_chunks):
    with pytest.raises(ValueError):
        split_text("内容", 20, max_chunks)
