"""Bidirectional protocol translation between Anthropic and OpenAI API formats."""

from __future__ import annotations

import json
import time
import uuid


# ---------------------------------------------------------------------------
# Finish/stop reason mappings
# ---------------------------------------------------------------------------

_ANTHROPIC_TO_OPENAI_STOP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}

_OPENAI_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


# ===================================================================
# Anthropic → OpenAI
# ===================================================================


def anthropic_request_to_openai(body: dict) -> dict:
    """Translate an Anthropic /v1/messages request to OpenAI /v1/chat/completions."""
    result: dict = {}

    messages: list[dict] = []

    # System prompt → system message
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be list of content blocks
            text = " ".join(
                block.get("text", "") for block in system if block.get("type") == "text"
            )
            if text:
                messages.append({"role": "system", "content": text})

    # Convert messages
    for msg in body.get("messages", []):
        messages.extend(_anthropic_msg_to_openai(msg))

    result["messages"] = messages
    result["model"] = body.get("model", "")

    # Shared fields
    for key in ("stream", "temperature", "top_p", "max_tokens"):
        if key in body:
            result[key] = body[key]

    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    # Tools
    if "tools" in body:
        result["tools"] = [_anthropic_tool_to_openai(t) for t in body["tools"]]

    return result


def _anthropic_msg_to_openai(msg: dict) -> list[dict]:
    """Convert a single Anthropic message to one or more OpenAI messages."""
    role = msg.get("role", "user")
    content = msg.get("content")

    # Simple string content
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return [{"role": role, "content": content}]

    # Array content blocks — need to handle tool_use and tool_result specially
    if role == "assistant":
        return [_anthropic_assistant_blocks_to_openai(content)]

    # User messages may contain text, images, and tool_result blocks
    result_messages: list[dict] = []
    non_tool_parts: list[dict] = []

    for block in content:
        btype = block.get("type")
        if btype == "tool_result":
            # Flush accumulated non-tool parts as a user message first
            if non_tool_parts:
                result_messages.append({"role": "user", "content": non_tool_parts})
                non_tool_parts = []
            # Each tool_result becomes a separate tool message
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = " ".join(
                    b.get("text", "") for b in tool_content if b.get("type") == "text"
                )
            result_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": str(tool_content),
            })
        elif btype == "text":
            non_tool_parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            non_tool_parts.append(_anthropic_image_to_openai(block))
        else:
            non_tool_parts.append(block)

    if non_tool_parts:
        # If there's only one text part, simplify to string
        if len(non_tool_parts) == 1 and non_tool_parts[0].get("type") == "text":
            result_messages.append({"role": "user", "content": non_tool_parts[0]["text"]})
        else:
            result_messages.append({"role": "user", "content": non_tool_parts})

    return result_messages or [{"role": "user", "content": ""}]


def _anthropic_assistant_blocks_to_openai(blocks: list[dict]) -> dict:
    """Convert Anthropic assistant content blocks to a single OpenAI assistant message."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    msg: dict = {"role": "assistant"}
    msg["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _anthropic_image_to_openai(block: dict) -> dict:
    source = block.get("source", {})
    if source.get("type") == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    # URL source — pass through
    if source.get("type") == "url":
        return {
            "type": "image_url",
            "image_url": {"url": source.get("url", "")},
        }
    return block


def _anthropic_tool_to_openai(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


# --- Response translation (OpenAI response → Anthropic response) ---


def openai_response_to_anthropic(body: dict, model: str) -> dict:
    """Translate an OpenAI chat completion response to Anthropic message format."""
    # Error bodies ({"error": {...}}) are passed through unchanged so the
    # client sees the real upstream error instead of a fake empty message.
    if "error" in body:
        return body

    choice = {}
    if body.get("choices"):
        choice = body["choices"][0]

    message = choice.get("message", {})
    content_blocks: list[dict] = []

    # Text content
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {"raw": func.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": input_data,
        })

    finish = choice.get("finish_reason", "stop")
    stop_reason = _OPENAI_TO_ANTHROPIC_STOP.get(finish, "end_turn")

    usage_in = body.get("usage", {})
    usage = {
        "input_tokens": usage_in.get("prompt_tokens", 0),
        "output_tokens": usage_in.get("completion_tokens", 0),
    }

    return {
        "id": body.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model or body.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


# --- Streaming translation (OpenAI SSE → Anthropic SSE) ---


def openai_stream_to_anthropic_events(line: str, state: dict) -> list[str]:
    """Translate one OpenAI SSE line into Anthropic SSE event lines.

    ``state`` is a mutable dict that persists across calls for one stream.
    """
    line = line.strip()
    if not line:
        return []

    if line == "data: [DONE]":
        events: list[str] = []
        # Close any open content block
        if state.get("block_open"):
            events.append(
                f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': state.get('block_index', 0)})}\n"
            )
            state["block_open"] = False
        events.append(
            f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': state.get('stop_reason', 'end_turn')}, 'usage': {'output_tokens': state.get('output_tokens', 0)}})}\n"
        )
        events.append(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n")
        return events

    if not line.startswith("data: "):
        return []

    try:
        chunk = json.loads(line[6:])
    except json.JSONDecodeError:
        return []

    # Mid-stream errors: OpenAI sends {"error": {...}} on a data line.
    # Forward as an Anthropic error event so the client sees the failure
    # instead of a silently-truncated stream.
    if "error" in chunk:
        return [
            f"event: error\ndata: {json.dumps({'type': 'error', 'error': chunk['error']})}\n"
        ]

    events = []
    model = chunk.get("model", state.get("model", ""))
    state["model"] = model

    # Emit message_start on first chunk
    if not state.get("started"):
        state["started"] = True
        state["message_id"] = chunk.get("id", f"msg_{uuid.uuid4().hex[:24]}")
        state["block_index"] = 0
        state["block_open"] = False
        msg_start = {
            "type": "message_start",
            "message": {
                "id": state["message_id"],
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        events.append(f"event: message_start\ndata: {json.dumps(msg_start)}\n")

    for choice in chunk.get("choices", []):
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Text content
        content = delta.get("content")
        if content is not None:
            if not state.get("block_open") or state.get("block_type") != "text":
                # Close previous block if different type
                if state.get("block_open"):
                    events.append(
                        f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': state['block_index']})}\n"
                    )
                    state["block_index"] += 1
                # Open new text block
                events.append(
                    f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': state['block_index'], 'content_block': {'type': 'text', 'text': ''}})}\n"
                )
                state["block_open"] = True
                state["block_type"] = "text"

            if content:
                events.append(
                    f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': state['block_index'], 'delta': {'type': 'text_delta', 'text': content}})}\n"
                )

        # Tool calls
        for tc in delta.get("tool_calls", []):
            tc_index = tc.get("index", 0)
            tc_key = f"tool_{tc_index}"

            if tc.get("id"):
                # New tool call — close previous block, open new one
                if state.get("block_open"):
                    events.append(
                        f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': state['block_index']})}\n"
                    )
                    state["block_index"] += 1

                state[tc_key] = {"id": tc["id"], "name": tc.get("function", {}).get("name", "")}
                events.append(
                    f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': state['block_index'], 'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': state[tc_key]['name'], 'input': {}}})}\n"
                )
                state["block_open"] = True
                state["block_type"] = "tool_use"

            args = tc.get("function", {}).get("arguments", "")
            if args:
                events.append(
                    f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': state['block_index'], 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n"
                )

        if finish_reason:
            state["stop_reason"] = _OPENAI_TO_ANTHROPIC_STOP.get(finish_reason, "end_turn")

    # Capture usage from chunk if present
    usage = chunk.get("usage", {})
    if usage.get("completion_tokens"):
        state["output_tokens"] = usage["completion_tokens"]
    if usage.get("prompt_tokens"):
        state["input_tokens"] = usage["prompt_tokens"]

    return events


# ===================================================================
# OpenAI → Anthropic
# ===================================================================


def openai_request_to_anthropic(body: dict) -> dict:
    """Translate an OpenAI /v1/chat/completions request to Anthropic /v1/messages."""
    result: dict = {}

    messages_in = body.get("messages", [])
    system_parts: list[str] = []
    messages: list[dict] = []

    i = 0
    while i < len(messages_in):
        msg = messages_in[i]
        role = msg.get("role", "user")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.append(
                    " ".join(p.get("text", "") for p in content if p.get("type") == "text")
                )
            i += 1
            continue

        if role == "assistant":
            messages.append(_openai_assistant_to_anthropic(msg))
            i += 1
            continue

        if role == "tool":
            # Collect consecutive tool messages and merge into a user message
            tool_results: list[dict] = []
            while i < len(messages_in) and messages_in[i].get("role") == "tool":
                tmsg = messages_in[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tmsg.get("tool_call_id", ""),
                    "content": tmsg.get("content", ""),
                })
                i += 1
            messages.append({"role": "user", "content": tool_results})
            continue

        # Regular user message
        messages.append(_openai_user_to_anthropic(msg))
        i += 1

    if system_parts:
        result["system"] = "\n".join(system_parts)

    result["messages"] = messages
    result["model"] = body.get("model", "")

    # max_tokens is required in Anthropic API
    result["max_tokens"] = body.get("max_tokens", 4096)

    for key in ("stream", "temperature", "top_p"):
        if key in body:
            result[key] = body[key]

    if "stop" in body:
        result["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    if "tools" in body:
        result["tools"] = [_openai_tool_to_anthropic(t) for t in body["tools"]]

    return result


def _openai_user_to_anthropic(msg: dict) -> dict:
    content = msg.get("content")
    if isinstance(content, str):
        return {"role": "user", "content": content}

    if isinstance(content, list):
        blocks: list[dict] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                blocks.append(_openai_image_to_anthropic(part))
            else:
                blocks.append(part)
        return {"role": "user", "content": blocks}

    return {"role": "user", "content": content}


def _openai_assistant_to_anthropic(msg: dict) -> dict:
    blocks: list[dict] = []

    content = msg.get("content")
    if content:
        blocks.append({"type": "text", "text": content})

    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {"raw": func.get("arguments", "")}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": input_data,
        })

    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return {"role": "assistant", "content": blocks[0]["text"]}

    return {"role": "assistant", "content": blocks}


def _openai_image_to_anthropic(part: dict) -> dict:
    url = part.get("image_url", {}).get("url", "")
    if url.startswith("data:"):
        # Parse data URI: data:media_type;base64,data
        header, _, data = url.partition(",")
        media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


def _openai_tool_to_anthropic(tool: dict) -> dict:
    func = tool.get("function", {})
    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "input_schema": func.get("parameters", {}),
    }


# --- Response translation (Anthropic response → OpenAI response) ---


def anthropic_response_to_openai(body: dict) -> dict:
    """Translate an Anthropic message response to OpenAI chat completion format."""
    # Error bodies ({"error": {...}} or {"type": "error", ...}) are passed
    # through unchanged so the client sees the real upstream error.
    if "error" in body or body.get("type") == "error":
        return body

    content_blocks = body.get("content", [])
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    message: dict = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = body.get("stop_reason", "end_turn")
    finish_reason = _ANTHROPIC_TO_OPENAI_STOP.get(stop_reason, "stop")

    usage_in = body.get("usage", {})
    prompt_tokens = usage_in.get("input_tokens", 0)
    completion_tokens = usage_in.get("output_tokens", 0)

    return {
        "id": body.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", ""),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# --- Streaming translation (Anthropic SSE → OpenAI SSE) ---


def anthropic_stream_to_openai_lines(line: str, state: dict) -> list[str]:
    """Translate one Anthropic SSE line into OpenAI SSE data lines.

    ``state`` is a mutable dict that persists across calls for one stream.
    Anthropic SSE uses ``event:`` lines followed by ``data:`` lines.
    """
    line = line.strip()
    if not line:
        return []

    # Track event type
    if line.startswith("event: "):
        state["_pending_event"] = line[7:].strip()
        return []

    if not line.startswith("data: "):
        return []

    try:
        data = json.loads(line[6:])
    except json.JSONDecodeError:
        return []

    event_type = state.pop("_pending_event", data.get("type", ""))
    lines: list[str] = []

    # Mid-stream errors: Anthropic sends `event: error\ndata: {"type":"error",
    # "error": {...}}`.  Forward as an OpenAI-shaped error data line so the
    # client sees the failure instead of a silently-truncated stream.
    if event_type == "error" or data.get("type") == "error":
        err = data.get("error", data)
        return [f"data: {json.dumps({'error': err})}\n"]

    if event_type == "message_start":
        msg = data.get("message", {})
        state["message_id"] = msg.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}")
        state["model"] = msg.get("model", "")
        state["created"] = int(time.time())
        # Emit first chunk with role
        chunk = _openai_stream_chunk(state, delta={"role": "assistant"})
        lines.append(f"data: {json.dumps(chunk)}\n")

    elif event_type == "content_block_start":
        block = data.get("content_block", {})
        if block.get("type") == "tool_use":
            state["_current_tool_index"] = data.get("index", 0)
            chunk = _openai_stream_chunk(state, delta={
                "tool_calls": [{
                    "index": state["_current_tool_index"],
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }]
            })
            lines.append(f"data: {json.dumps(chunk)}\n")

    elif event_type == "content_block_delta":
        delta_data = data.get("delta", {})
        dtype = delta_data.get("type")

        if dtype == "text_delta":
            chunk = _openai_stream_chunk(state, delta={"content": delta_data.get("text", "")})
            lines.append(f"data: {json.dumps(chunk)}\n")
        elif dtype == "input_json_delta":
            tc_index = state.get("_current_tool_index", 0)
            chunk = _openai_stream_chunk(state, delta={
                "tool_calls": [{
                    "index": tc_index,
                    "function": {"arguments": delta_data.get("partial_json", "")},
                }]
            })
            lines.append(f"data: {json.dumps(chunk)}\n")

    elif event_type == "message_delta":
        delta_data = data.get("delta", {})
        stop_reason = delta_data.get("stop_reason", "end_turn")
        finish_reason = _ANTHROPIC_TO_OPENAI_STOP.get(stop_reason, "stop")
        usage = data.get("usage", {})
        chunk = _openai_stream_chunk(state, delta={}, finish_reason=finish_reason)
        if usage:
            chunk["usage"] = {
                "prompt_tokens": state.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": state.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
        lines.append(f"data: {json.dumps(chunk)}\n")

    elif event_type == "message_stop":
        lines.append("data: [DONE]\n")

    return lines


def _openai_stream_chunk(
    state: dict,
    delta: dict,
    finish_reason: str | None = None,
) -> dict:
    return {
        "id": state.get("message_id", ""),
        "object": "chat.completion.chunk",
        "created": state.get("created", int(time.time())),
        "model": state.get("model", ""),
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


# ===================================================================
# Path rewriting
# ===================================================================


def rewrite_path(path: str, translate: str) -> str:
    """Rewrite the request path when translation is active."""
    if translate == "anthropic_to_openai":
        # /v1/messages → /chat/completions (base URL has /v1)
        if "/messages" in path:
            return path.replace("/messages", "/chat/completions").split("?")[0]
    elif translate == "openai_to_anthropic":
        # /v1/chat/completions → /messages
        if "/chat/completions" in path:
            return path.replace("/chat/completions", "/messages").split("?")[0]
    return path
