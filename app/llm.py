"""Streaming client for llama-server's OpenAI-compatible chat endpoint.

One model serves both paths:

* ``chat``   – thinking off, ``reasoning_effort: low``; the fast voice path.
* ``reason`` – thinking on, ``reasoning_effort: medium``; calculations, plans,
  comparisons.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

CHAT = "chat"
REASON = "reason"

_SAMPLING = {
    # Values recommended for this model family in each mode.
    CHAT: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    REASON: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
}


def build_payload(messages: list[dict], mode: str, max_tokens: int, stream: bool = True) -> dict:
    if mode not in _SAMPLING:
        raise ValueError(f"unknown mode {mode!r}")
    thinking = mode == REASON
    return {
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "cache_prompt": True,
        "reasoning_effort": "medium" if thinking else "low",
        "chat_template_kwargs": {"enable_thinking": thinking},
        **_SAMPLING[mode],
    }


@dataclass
class Delta:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    timings: dict | None = None


def parse_sse_line(line: str) -> Delta | None:
    """One ``data: {...}`` line of the stream, or None for keep-alives and [DONE]."""
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    chunk = json.loads(data)
    if "error" in chunk:
        raise RuntimeError(f"llama-server error: {chunk['error']}")
    choices = chunk.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    return Delta(
        content=delta.get("content") or "",
        reasoning=delta.get("reasoning_content") or "",
        finish_reason=choices[0].get("finish_reason"),
        timings=chunk.get("timings"),
    )


class LLMClient:
    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=600),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def healthy(self) -> bool:
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def stream(self, messages: list[dict], mode: str, max_tokens: int) -> AsyncIterator[Delta]:
        payload = build_payload(messages, mode, max_tokens)
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"llama-server returned {resp.status_code}: {body[:300]}")
            async for line in resp.aiter_lines():
                delta = parse_sse_line(line)
                if delta is not None:
                    yield delta

    async def prefill(self, messages: list[dict], mode: str = CHAT) -> None:
        """Process a prompt and discard the output, to leave it in the KV cache.

        llama-server checkpoints the state at the start of the last user
        message, so the real request that follows only has to process the new
        words.
        """
        payload = build_payload(messages, mode, max_tokens=1, stream=False)
        await self._client.post("/v1/chat/completions", json=payload)
