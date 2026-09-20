"""Parse OneBot segments and construct bounded, literal-text replies."""

import re

from app.onebot.models import MessageSegment

_SENTENCE_END = re.compile(r"[。！？!?]|\.(?=\s|$)")
_TRUNCATION_NOTICE = "\n…（内容已截断）"


def is_at_bot(segments: list[MessageSegment], bot_id: int) -> bool:
    """Match a concrete at.qq identifier; nicknames and at-all never trigger."""
    target = str(bot_id)
    for segment in segments:
        if segment.type != "at":
            continue
        qq = segment.data.get("qq")
        if isinstance(qq, (str, int)) and not isinstance(qq, bool) and str(qq) == target:
            return True
    return False


def extract_plain_text(segments: list[MessageSegment], bot_id: int) -> str:
    """Join text segments; mentions, replies and unsupported content add no text."""
    return "".join(
        text
        for segment in segments
        if segment.type == "text" and isinstance(text := segment.data.get("text"), str)
    ).strip()


def make_reply(message_id: int, text: str) -> list[dict]:
    """Use segment arrays so text resembling CQ codes stays literal text."""
    return [
        {"type": "reply", "data": {"id": str(message_id)}},
        {"type": "text", "data": {"text": text}},
    ]


def _take_chunk(text: str, max_chars: int) -> tuple[str, str]:
    """Prefer a paragraph boundary, then a sentence end, then a hard cut."""
    if len(text) <= max_chars:
        return text, ""
    paragraph_end = text.rfind("\n", 0, max_chars + 1)
    if paragraph_end > 0 and text[:paragraph_end].strip():
        return text[:paragraph_end].rstrip("\r\n"), text[paragraph_end:].lstrip("\r\n")
    sentence_ends = list(_SENTENCE_END.finditer(text, 0, max_chars))
    if sentence_ends:
        boundary = sentence_ends[-1].end()
        return text[:boundary], text[boundary:].lstrip()
    return text[:max_chars], text[max_chars:]


def split_text(text: str, max_chars: int, max_chunks: int = 3) -> list[str]:
    """Split by character count, marking omitted content in the final chunk.

    Paragraph separators are omitted at message boundaries. Newlines and code
    indentation inside a chunk are preserved. At most three messages are emitted.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or not 1 <= max_chunks <= 3:
        raise ValueError("max_chunks must be an integer from 1 to 3")
    remaining = text.strip()
    chunks = []
    while remaining:
        if len(chunks) == max_chunks - 1 and len(remaining) > max_chars:
            notice = _TRUNCATION_NOTICE if max_chars > len(_TRUNCATION_NOTICE) else "…"
            capacity = max_chars - len(notice)
            prefix, _ = _take_chunk(remaining, capacity) if capacity else ("", remaining)
            chunks.append(prefix.rstrip() + notice)
            break
        chunk, remaining = _take_chunk(remaining, max_chars)
        if chunk.strip():
            chunks.append(chunk)
        # A hard split can leave a paragraph separator at the next boundary.
        remaining = remaining.lstrip("\r\n")
    return chunks
