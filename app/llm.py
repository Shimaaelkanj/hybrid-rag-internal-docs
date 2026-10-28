from __future__ import annotations

import hashlib
import json
import os
import re
import time

from openai import OpenAI

from app.config import (CACHE_FILE, JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL,
                        LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT)

_clients: dict[bool, OpenAI] = {}
_cache: dict[str, str] | None = None


def _client(judge: bool) -> OpenAI:
    if judge not in _clients:
        base, key = (JUDGE_BASE_URL, JUDGE_API_KEY) if judge else (LLM_BASE_URL, LLM_API_KEY)
        _clients[judge] = OpenAI(base_url=base, api_key=key, timeout=LLM_TIMEOUT)
    return _clients[judge]


def _disk_cache() -> dict[str, str]:
    """Append-only JSONL cache so repeated eval runs do not repeat identical LLM calls."""
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_FILE.exists():
            for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    _cache[item["k"]] = item["v"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return _cache


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 700,
         judge: bool = False) -> str:
    """Call the generator model, or the judge model when judge=True.

    Caching is only active when the env var LLM_CACHE=1 (the eval suite sets it) and the call
    is deterministic (temperature 0). The cache key includes model, endpoint and the full
    prompt, so changing any of them automatically invalidates old entries.
    """
    model = JUDGE_MODEL if judge else LLM_MODEL
    base = JUDGE_BASE_URL if judge else LLM_BASE_URL
    
    if judge:
        max_tokens = max(max_tokens, 800)     # leave room for reasoning models to think


    key = None
    if temperature == 0 and os.getenv("LLM_CACHE") == "1":
        payload = json.dumps([base, model, max_tokens, messages], sort_keys=True)
        key = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        cached = _disk_cache().get(key)
        if cached is not None:
            return cached

    text, last_error = None, None
    for attempt in range(4):
        try:
            resp = _client(judge).chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:                       # noqa: BLE001
            last_error = e
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
                time.sleep(5 * (attempt + 1))        # free tiers: back off and retry
                continue
            break

    if text is None:
        hint = (f" If you use Ollama: is it running and did you run 'ollama pull {model}'?"
                if "11434" in base else "")
        raise RuntimeError(f"LLM call failed (url={base}, model={model}).{hint} "
                           f"Original error: {last_error}") from last_error

    if key and text:
        _disk_cache()[key] = text
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": text}, ensure_ascii=False) + "\n")
    return text


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from a model reply (models often add extra words)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
