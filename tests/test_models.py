from malcolm.models import Conversation, Message, ToolCall


def test_message_text_only():
    msg = Message(role="user", text="hello")
    assert msg.role == "user"
    assert msg.text == "hello"
    assert msg.tool_calls == []
    assert msg.tool_result is None


def test_message_with_tool_calls():
    tc = ToolCall(name="read_file", arguments='{"path": "foo.py"}', id="call_1")
    msg = Message(role="assistant", tool_calls=[tc])
    assert msg.tool_calls[0].name == "read_file"
    assert msg.tool_calls[0].id == "call_1"


def test_message_with_tool_result():
    msg = Message(role="tool", tool_result="file contents here")
    assert msg.tool_result == "file contents here"
    assert msg.text is None


def test_message_raw_preserved():
    raw = {"role": "user", "content": "hello"}
    msg = Message(role="user", text="hello", raw=raw)
    assert msg.raw == raw


def test_conversation_defaults():
    conv = Conversation()
    assert conv.messages == []
    assert conv.model == ""
    assert conv.stream is False
    assert conv.status_code is None


def test_conversation_with_messages():
    msgs = [
        Message(role="user", text="hello"),
        Message(role="assistant", text="hi"),
    ]
    conv = Conversation(messages=msgs, model="gpt-4", timestamp="2026-01-01T00:00:00")
    assert len(conv.messages) == 2
    assert conv.model == "gpt-4"
