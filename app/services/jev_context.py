"""Build a bounded, identity-consistent decision state from untrusted chat data.

QQ identifiers become per-request labels. Nicknames and message prose are still
present, so this is identifier minimization, not full anonymization.
"""

import json
import re

_MAX_MESSAGES = 25
_MAX_TEXT = 500
_MAX_PROFILE = 3000
_MAX_MENTIONS = 10
_MAX_STATE_CHARS = 20000
_SPEAKER_PREFIX = re.compile(r"^\[(.*?)/([1-9][0-9]{0,19})\]\s*", re.DOTALL)
_AT_REFERENCE = re.compile(r"@([1-9][0-9]{0,19})(?![0-9])")
_UNKNOWN_MEDIA = re.compile(r"\[(?:图片[^\]]*|表情包[^\]]*|语音|视频|文件|转发|消息段)\]")
_PROFILE_ID = re.compile(r'"user_id"\s*:\s*"?([1-9][0-9]{0,19})(?![0-9])')
_PROFILE_IDS = re.compile(r'"user_ids"\s*:\s*\[([^\]]*)\]')


def _user_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    text = str(value)
    return text if re.fullmatch(r"[1-9][0-9]{0,19}", text) else ""


def _message_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    text = str(value)
    return text if re.fullmatch(r"-?[0-9]{1,20}", text) else ""


def build_jev_state(
    history: list[dict],
    *,
    new_message_count: int,
    bot_id: str,
    profile_context: str = "",
    addressed_to_bot: bool = False,
) -> dict:
    """Select the last N user messages without losing intervening bot replies.

    ``context_incomplete`` means the new batch lost messages or text, rather
    than merely reaching the normal old-context limit. Callers should defer to
    the existing full-context decision path when that flag is true.
    ``reply_to`` identifies a message, never a QQ member.
    """
    if isinstance(new_message_count, bool) or not isinstance(new_message_count, int):
        raise ValueError("new_message_count must be an integer")
    requested_count = max(0, new_message_count)
    identities: dict[str, str] = {}
    if actual_bot_id := _user_id(bot_id):
        identities[actual_bot_id] = "bot"
    next_member = 1

    def register(value: object) -> str:
        nonlocal next_member
        uid = _user_id(value)
        if not uid:
            return "unknown_member"
        if uid not in identities:
            identities[uid] = f"member_{next_member}"
            next_member += 1
        return identities[uid]

    rows = []
    message_targets = {}
    for item in history:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        role = item["role"]
        text = item.get("content", "")
        if role == "assistant" and isinstance(item.get("reply_text"), str):
            text = item["reply_text"]
        if not isinstance(text, str):
            continue
        prefix = _SPEAKER_PREFIX.match(text) if role == "user" else None
        uid = _user_id(item.get("user_id")) or (prefix[2] if prefix else "")
        nickname = item.get("nickname")
        if not isinstance(nickname, str):
            nickname = prefix[1] if prefix else ""
        if prefix:
            text = text[prefix.end() :]
        speaker = "bot" if role == "assistant" else register(uid)
        mentions = item.get("mentions")
        if not isinstance(mentions, list):
            mentions = _AT_REFERENCE.findall(text)
        if role == "assistant" and (mention := _user_id(item.get("mention_user_id"))):
            mentions = [*mentions, mention]
        mentions = list(dict.fromkeys(uid for value in mentions if (uid := _user_id(value))))
        for mention in mentions[:_MAX_MENTIONS]:
            register(mention)
        # Text references can exist even when a transport metadata list is empty.
        for mention in _AT_REFERENCE.findall(text)[:_MAX_MENTIONS]:
            register(mention)
        mid = _message_id(item.get("message_id"))
        if mid and mid not in message_targets:
            message_targets[mid] = (f"message_{len(message_targets) + 1}", speaker)
        rows.append(
            {
                "role": role,
                "speaker": speaker,
                "nickname": nickname[:80],
                "text": text,
                "mentions": mentions[:_MAX_MENTIONS],
                "message_id": mid,
                "reply_to": _message_id(item.get("reply_to")),
                "unknown_media": bool(item.get("has_unknown_media"))
                or bool(_UNKNOWN_MEDIA.search(text)),
                "truncated": len(text) > _MAX_TEXT or len(mentions) > _MAX_MENTIONS,
            }
        )

    profile = profile_context[:_MAX_PROFILE] if isinstance(profile_context, str) else ""
    for uid in _PROFILE_ID.findall(profile):
        register(uid)
    for values in _PROFILE_IDS.findall(profile):
        for uid in re.findall(r"(?<![0-9])[1-9][0-9]{0,19}(?![0-9])", values):
            register(uid)

    def replace_ids(text: str) -> str:
        return re.sub(r"[0-9]+", lambda match: identities.get(match[0], match[0]), text)

    user_indices = [index for index, row in enumerate(rows) if row["role"] == "user"]
    count = min(requested_count, len(user_indices))
    boundary = user_indices[-count] if count else len(rows)
    retained_from = max(0, len(rows) - _MAX_MESSAGES)
    new_rows = rows[boundary:]
    context_incomplete = (
        requested_count > len(user_indices)
        or (count > 0 and retained_from > boundary)
        or any(row["truncated"] or len(replace_ids(row["text"])) > _MAX_TEXT for row in new_rows)
    )

    def render(row: dict) -> dict:
        result = {
            "speaker": row["speaker"],
            "text": replace_ids(row["text"])[:_MAX_TEXT],
            "mentions": [identities[uid] for uid in row["mentions"]],
            "reply_to": None,
        }
        if row["message_id"]:
            result["message_id"] = message_targets[row["message_id"]][0]
        if row["reply_to"]:
            target = message_targets.get(row["reply_to"])
            result["reply_to"] = target[0] if target else "unknown_message"
            if target:
                result["reply_to_speaker"] = target[1]
        return result

    recent_context, new_messages, aliases = [], [], []
    seen_aliases = set()
    for index, row in enumerate(rows[retained_from:], start=retained_from):
        destination = recent_context if index < boundary else new_messages
        destination.append(render(row))
        nickname = replace_ids(row["nickname"])[:80]
        alias = (row["speaker"], nickname)
        if nickname and alias not in seen_aliases:
            aliases.append({"speaker": row["speaker"], "nickname": nickname})
            seen_aliases.add(alias)

    state = {
        "bot_id": "bot",
        "recent_context": recent_context,
        "new_messages": new_messages,
        "recent_bot_replies": [
            render(row) for row in [r for r in rows if r["role"] == "assistant"][-3:]
        ],
        "relevant_aliases": aliases,
        "relevant_memories": (
            [
                {
                    "source": "group_profiles_and_rag",
                    "untrusted": True,
                    "text": replace_ids(profile)[:_MAX_PROFILE],
                }
            ]
            if profile
            else []
        ),
        "addressed_to_bot": bool(addressed_to_bot),
        "unknown_media_present": any(row["unknown_media"] for row in new_rows),
        "context_incomplete": context_incomplete,
    }
    # Metadata and escaping also count toward the bound. Normal old-context
    # trimming is expected; dropping anything from the new batch needs fallback.
    while len(json.dumps(state, ensure_ascii=False)) > _MAX_STATE_CHARS:
        if state["recent_context"]:
            state["recent_context"].pop(0)
        elif state["new_messages"]:
            state["new_messages"].pop(0)
            state["context_incomplete"] = True
        elif state["relevant_aliases"]:
            state["relevant_aliases"].pop(0)
        elif state["recent_bot_replies"]:
            state["recent_bot_replies"].pop(0)
        elif state["relevant_memories"]:
            state["relevant_memories"].clear()
        else:
            break
    return state
