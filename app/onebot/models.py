"""The small subset of OneBot 11 models needed for group message events."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class MessageSegment(BaseModel):
    """Keep extension segment types usable without modelling their payloads."""

    model_config = ConfigDict(extra="ignore")

    type: str
    data: dict[str, Any]


class GroupSender(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    nickname: str = ""
    card: str = ""
    role: str = "member"


class GroupMessageEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    post_type: Literal["message"]
    message_type: Literal["group"]
    message_id: int
    group_id: int
    user_id: int
    self_id: int
    message: list[MessageSegment]
    raw_message: str = ""
    sender: GroupSender
