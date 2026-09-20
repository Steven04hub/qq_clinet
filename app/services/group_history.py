"""Normalize only a requested group's messages, without downloading media."""

import re
import time
from typing import Any

_EMOJI = re.compile(
    r"[\U0001f1e6-\U0001f1ff]{2}|"
    r"[\U0001f300-\U0001faff\u2600-\u27bf]"
    r"(?:[\ufe0e\ufe0f\U0001f3fb-\U0001f3ff]|\u200d[\U0001f300-\U0001faff\u2600-\u27bf])*"
)


def normalize_message(payload: dict, group_id: str, *, self_id: str = "") -> dict | None:
    """Keep speaker, literal text, mentions, replies and observable emoji metadata."""
    if (
        str(payload.get("group_id")) != group_id
        or payload.get("message_type") != "group"
        or payload.get("post_type", "message") not in {"message", "message_sent"}
    ):
        return None
    message_id, user_id = payload.get("message_id"), payload.get("user_id")
    bot_id = self_id or str(payload.get("self_id", ""))
    if payload.get("post_type") == "message_sent" and str(user_id) != bot_id:
        return None
    if (
        isinstance(message_id, bool)
        or not isinstance(message_id, (int, str))
        or not str(message_id).lstrip("-").isascii()
        or not str(message_id).lstrip("-").isdigit()
        or isinstance(user_id, bool)
        or not isinstance(user_id, (int, str))
        or not str(user_id).isascii()
        or not str(user_id).isdigit()
        or int(user_id) <= 0
    ):
        return None
    sender = payload.get("sender")
    sender = sender if isinstance(sender, dict) else {}
    nickname = sender.get("card") or sender.get("nickname") or str(user_id)
    nickname = str(nickname)[:100]
    segments = payload.get("message")
    if isinstance(segments, str):
        # A legacy CQ string is kept as text, with media payloads stripped.
        segments = [
            {"type": "text", "data": {"text": re.sub(r"\[CQ:[^\]]*\]", "[消息段]", segments)}}
        ]
    if not isinstance(segments, list):
        return None
    parts, mentions, emoji = [], [], []
    reply_to = None
    for segment in segments[:100]:
        if not isinstance(segment, dict) or not isinstance(segment.get("data"), dict):
            continue
        kind, data = segment.get("type"), segment["data"]
        if kind == "text" and isinstance(data.get("text"), str):
            part = data["text"][:4000]
            parts.append(part)
            emoji.extend(_EMOJI.findall(part))
        elif kind == "at":
            target = str(data.get("qq", ""))
            if target.isascii() and target.isdigit() and int(target) > 0:
                mentions.append(target)
                parts.append(f"[@{target}]")
            elif target == "all":
                parts.append("[@全体]")
        elif kind == "reply":
            candidate = str(data.get("id", ""))
            if candidate.lstrip("-").isascii() and candidate.lstrip("-").isdigit():
                reply_to = candidate
        elif kind == "face":
            face_id = str(data.get("id", ""))[:20]
            if face_id.isascii() and face_id.isdigit():
                label = f"QQ表情#{face_id}"
                parts.append(f"[{label}]")
                emoji.append(label)
        elif kind in {"image", "mface"}:
            parts.append("[图片或表情包，未识别画面]")
            emoji.append("图片/表情包")
        elif kind in {"record", "video", "file", "forward"}:
            parts.append(
                {"record": "[语音]", "video": "[视频]", "file": "[文件]", "forward": "[转发]"}[kind]
            )
    text = "".join(parts).strip()[:4000]
    if not text:
        return None
    timestamp = payload.get("time")
    timestamp = int(timestamp) if isinstance(timestamp, (int, float)) else int(time.time())
    return {
        "message_id": str(message_id),
        "user_id": str(user_id),
        "nickname": nickname,
        "text": text,
        "time": timestamp,
        "mentions": list(dict.fromkeys(mentions)),
        "reply_to": reply_to,
        "emoji": emoji[:50],
        "is_bot": str(user_id) == bot_id,
    }


async def fetch_group_history(client: Any, group_id: str, *, count: int = 1000) -> tuple:
    """Only query the named group and return available records, not a claimed full archive."""
    login = await client.call_action("get_login_info", {})
    self_id = str((login.get("data") or {}).get("user_id", ""))
    roster_response = await client.call_action("get_group_member_list", {"group_id": int(group_id)})
    roster = []
    data = roster_response.get("data")
    if isinstance(data, list):
        for member in data:
            if not isinstance(member, dict):
                continue
            uid = str(member.get("user_id", ""))
            if not uid.isascii() or not uid.isdigit() or int(uid) <= 0 or uid == self_id:
                continue
            roster.append(
                {
                    "user_id": uid,
                    "nickname": str(member.get("nickname") or "")[:100],
                    "card": str(member.get("card") or "")[:100],
                }
            )
    history = await client.call_action(
        "get_group_msg_history",
        {
            "group_id": int(group_id),
            "count": min(max(count, 1), 1000),
            "message_seq": "0",
            "reverseOrder": False,
        },
        timeout=30,
    )
    data = history.get("data")
    messages = data.get("messages", []) if isinstance(data, dict) else []
    if not isinstance(messages, list):
        raise ValueError("Invalid group history response")
    normalized = []
    for payload in messages[:1000]:
        if isinstance(payload, dict):
            record = normalize_message(payload, group_id, self_id=self_id)
            if record:
                normalized.append(record)
    normalized.sort(key=lambda record: record["time"])
    return self_id, roster, normalized
