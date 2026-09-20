"""Phase 2 replies are bounded literal text and quote the source message."""

import pytest

from app.handlers.chat import ChatHandler
from app.handlers.commands import CommandHandler
from tests.test_router import make_event


class RecordingClient:
    def __init__(self):
        self.messages = []

    async def send_group_message(self, group_id, message):
        self.messages.append((group_id, message))
        return len(self.messages)


def reply_text(message):
    assert message[0] == {"type": "reply", "data": {"id": "1"}}
    assert message[1]["type"] == "text"
    return message[1]["data"]["text"]


async def test_echo_quotes_original_message_and_preserves_literal_cq_text():
    client = RecordingClient()
    handler = ChatHandler(client)
    await handler.handle(make_event(text="echo hello [CQ:at,qq=all]"))
    assert len(client.messages) == 1
    group_id, message = client.messages[0]
    assert group_id == 10
    assert reply_text(message) == "hello [CQ:at,qq=all]"


@pytest.mark.parametrize("text", ["hi", "echo", "echohello"])
async def test_non_echo_request_explains_current_mode(text):
    client = RecordingClient()
    await ChatHandler(client).handle(make_event(text=text))
    assert "echo" in reply_text(client.messages[0][1])
    assert "@机器人 echo hello" in reply_text(client.messages[0][1])


async def test_empty_request_produces_no_reply():
    client = RecordingClient()
    await ChatHandler(client).handle(make_event(text=" \n "))
    assert client.messages == []


async def test_long_echo_is_limited_to_three_bounded_quoted_messages():
    client = RecordingClient()
    await ChatHandler(client, max_reply_chars=8).handle(make_event(text="echo " + "a" * 100))
    assert len(client.messages) == 3
    for group_id, message in client.messages:
        assert group_id == 10
        assert 0 < len(reply_text(message)) <= 8


@pytest.mark.parametrize(
    "command,expected",
    [("/help", "@机器人 echo hello"), ("/status", "echo"), ("/clear", "记忆暂未开启")],
)
async def test_commands_reply_with_phase_two_information(command, expected):
    client = RecordingClient()
    await CommandHandler(client).handle(make_event(text=command, mention=False))
    assert len(client.messages) == 1
    assert client.messages[0][0] == 10
    assert expected in reply_text(client.messages[0][1])


async def test_unknown_command_produces_no_reply():
    client = RecordingClient()
    await CommandHandler(client).handle(make_event(text="/unknown"))
    assert client.messages == []


async def test_send_failure_propagates_without_retrying():
    class FailingClient(RecordingClient):
        async def send_group_message(self, group_id, message):
            await super().send_group_message(group_id, message)
            raise ConnectionError("socket disconnected")

    client = FailingClient()
    with pytest.raises(ConnectionError):
        await ChatHandler(client).handle(make_event())
    assert len(client.messages) == 1
