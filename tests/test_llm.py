import pytest

from app.llm import CHAT, REASON, build_payload, parse_sse_line

MESSAGES = [{"role": "user", "content": "สวัสดี"}]


def test_voice_chat_turns_thinking_off_with_low_effort():
    p = build_payload(MESSAGES, CHAT, 256)
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["reasoning_effort"] == "low"
    assert p["stream"] is True
    assert p["cache_prompt"] is True


def test_reasoning_turns_thinking_on_with_medium_effort():
    p = build_payload(MESSAGES, REASON, 4096)
    assert p["chat_template_kwargs"] == {"enable_thinking": True}
    assert p["reasoning_effort"] == "medium"
    assert p["max_tokens"] == 4096


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_payload(MESSAGES, "fast", 10)


def test_parse_content_and_reasoning_deltas():
    d = parse_sse_line('data: {"choices":[{"delta":{"content":"สวัสดี"}}]}')
    assert d.content == "สวัสดี" and d.reasoning == ""
    d = parse_sse_line('data: {"choices":[{"delta":{"reasoning_content":"คิด"}}]}')
    assert d.reasoning == "คิด" and d.content == ""
    d = parse_sse_line('data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"predicted_ms":1}}')
    assert d.finish_reason == "stop" and d.timings == {"predicted_ms": 1}


def test_parse_ignores_keepalives_and_done():
    assert parse_sse_line("") is None
    assert parse_sse_line(": ping") is None
    assert parse_sse_line("data: [DONE]") is None


def test_parse_raises_on_server_error():
    with pytest.raises(RuntimeError):
        parse_sse_line('data: {"error":{"message":"boom"}}')
