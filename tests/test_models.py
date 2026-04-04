from malcolm.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage


def test_chat_message_basic():
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_chat_message_extra_fields():
    msg = ChatMessage(role="assistant", content="hi", custom_field="value")
    assert msg.custom_field == "value"


def test_chat_message_multimodal_content():
    content = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    msg = ChatMessage(role="user", content=content)
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2


def test_chat_completion_request_minimal():
    req = ChatCompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="hello")],
    )
    assert req.model == "gpt-4"
    assert req.stream is False
    assert len(req.messages) == 1


def test_chat_completion_request_extra_fields():
    req = ChatCompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
    )
    assert req.temperature == 0.7
    assert req.max_tokens == 100


def test_chat_completion_request_stream():
    req = ChatCompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="hello")],
        stream=True,
    )
    assert req.stream is True


def test_chat_completion_response():
    resp = ChatCompletionResponse(
        id="chatcmpl-123",
        created=1700000000,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    assert resp.id == "chatcmpl-123"
    assert resp.choices[0].message.content == "Hello!"
    assert resp.usage.total_tokens == 15
