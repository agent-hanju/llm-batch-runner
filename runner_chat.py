from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from typing import Any

from models import Block, LLMRequest, LLMResponse
from utils import RateLimiter, normalize_base_url

logger = logging.getLogger(__name__)

_TTL_RE = re.compile(r"^\d+(\.\d+)?s$")


def _parse_ttl_seconds(ttl: str) -> float:
    """Parse a Gemini Duration string (e.g. ``"3600s"``) and return seconds as float.

    Only the ``"NNNs"`` / ``"NNN.NNNs"`` format is accepted by the API.
    """
    if not _TTL_RE.match(ttl):
        raise ValueError(
            f"Invalid TTL {ttl!r}: Gemini only accepts seconds-based Duration strings "
            f"like '3600s' or '300.5s'. Convert minutes/hours manually (e.g. 1h = '3600s')."
        )
    return float(ttl[:-1])


def _build_gemini_user_content(req: LLMRequest) -> str | list[dict[str, Any]]:
    """Build Gemini user content array with documents inlined as text blocks."""
    all_blocks = [b for seg in req.user_segments for b in seg]
    has_cache = any(c for _, c in all_blocks)
    if not has_cache:
        return req.flat_user
    content: list[dict[str, Any]] = []

    def _add(blocks: list[Block]) -> None:
        for text, cached in blocks:
            item: dict[str, Any] = {"type": "text", "text": text}
            if cached:
                item["cache_control"] = {"type": "ephemeral"}
            content.append(item)

    doc_block = {"type": "text", "text": "\n\n".join(LLMRequest._doc_text(d) for d in req.documents)} if req.documents else None

    has_marker = len(req.user_segments) > 1
    if has_marker:
        for i, seg in enumerate(req.user_segments):
            _add(seg)
            if i < len(req.user_segments) - 1 and doc_block:
                content.append(doc_block)
    else:
        if doc_block:
            content.append(doc_block)
        _add(all_blocks)
    return content


def _to_chat_content(blocks: list[Block], gemini: bool) -> str | list[dict[str, Any]]:
    """Convert block list to a chat content value.

    For Gemini endpoints, blocks with ``cached=True`` are wrapped in a content
    array with ``cache_control: {type: "ephemeral"}``. For all other providers
    the blocks are concatenated into a plain string (cache hints are ignored).
    """
    if not gemini or not any(c for _, c in blocks):
        return "".join(t for t, _ in blocks)
    content: list[dict[str, Any]] = []
    for text, cached in blocks:
        item: dict[str, Any] = {"type": "text", "text": text}
        if cached:
            item["cache_control"] = {"type": "ephemeral"}
        content.append(item)
    return content


class GeminiCacheClient:
    """
    Thin wrapper around Gemini's ``/v1beta/cachedContents`` REST API.

    ``get_or_create`` checks a local JSON store before hitting the API so the
    same content is never uploaded twice within its TTL.  The store is keyed by
    a SHA-256 hash of ``(model, contents, system_instruction, ttl)``.

    A threading lock serialises store reads and writes; concurrent callers with
    different content hashes are serialised only during the file I/O window, not
    during the API call itself.
    """

    _ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/cachedContents"

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        cache_store: str | os.PathLike[str] = ".gemini_cache_store.json",
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._store_path = Path(cache_store)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        ttl: str = "3600s",
        display_name: str | None = None,
    ) -> str:
        """Return an existing cache name if still valid, otherwise create one."""
        ttl_seconds = _parse_ttl_seconds(ttl)
        key = self._hash(model, contents, system_instruction, ttl)

        with self._lock:
            store = self._load_store()
            entry = store.get(key)
            if entry and entry["expires_at"] > time.time():
                logger.debug("gemini cache hit key=%s name=%s", key[:8], entry["name"])
                return entry["name"]
            logger.info("gemini cache miss — creating (model=%s ttl=%s)", model, ttl)
            name = self.create(model, contents, system_instruction, ttl, display_name)
            logger.info("gemini cache created name=%s", name)
            store[key] = {"name": name, "expires_at": time.time() + ttl_seconds}
            self._save_store(store)

        return name

    def create(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        ttl: str = "3600s",
        display_name: str | None = None,
    ) -> str:
        """Create a cache and return its name (e.g. ``"cachedContents/abc123"``)."""
        _parse_ttl_seconds(ttl)  # validate before sending
        body: dict[str, Any] = {"model": model, "contents": contents, "ttl": ttl}
        if system_instruction is not None:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if display_name is not None:
            body["displayName"] = display_name
        url = f"{self._ENDPOINT}?key={self.api_key}"
        http_req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout) as resp:
                result: dict[str, Any] = json.loads(resp.read())
            return result["name"]
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Gemini cache creation failed — HTTP {exc.code}: "
                f"{exc.read().decode(errors='replace')}"
            ) from exc

    def delete(self, cache_name: str) -> None:
        """Delete a cache by its resource name (e.g. ``"cachedContents/abc123"``)."""
        url = f"https://generativelanguage.googleapis.com/v1beta/{cache_name}?key={self.api_key}"
        http_req = urllib.request.Request(url, method="DELETE")
        try:
            urllib.request.urlopen(http_req, timeout=self.timeout).close()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Gemini cache deletion failed — HTTP {exc.code}: "
                f"{exc.read().decode(errors='replace')}"
            ) from exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hash(
        self,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: str | None,
        ttl: str,
    ) -> str:
        key = json.dumps([model, contents, system_instruction, ttl], sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()

    def _load_store(self) -> dict[str, Any]:
        if self._store_path.exists():
            try:
                return json.loads(self._store_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}

    def _save_store(self, store: dict[str, Any]) -> None:
        now = time.time()
        pruned = {k: v for k, v in store.items() if v["expires_at"] > now}
        self._store_path.write_text(json.dumps(pruned, indent=2), encoding="utf-8")


class ChatCompletionsRunner:
    """
    OpenAI-compatible Chat Completions endpoint.

    Uses only Python stdlib (urllib + threading) — no pip installs required.
    Works with local LLMs (Ollama, LM Studio, vLLM) and closed networks.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        max_concurrent: int = 5,
        requests_per_minute: int | None = None,
        timeout: int = 300,
        max_tokens: int = 4096,
        service_tier: str | None = None,
        cached_content: str | None = None,
        gemini_cache_client: GeminiCacheClient | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        normalized = normalize_base_url(base_url)
        self._is_gemini = "generativelanguage.googleapis.com" in normalized
        if cached_content is not None and not self._is_gemini:
            raise ValueError(f"cached_content requires a Gemini base_url, got {base_url!r}")
        if gemini_cache_client is not None and not self._is_gemini:
            raise ValueError(f"gemini_cache_client requires a Gemini base_url, got {base_url!r}")
        self.endpoint = normalized + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.service_tier = service_tier
        self.cached_content = cached_content
        self._gemini_cache_client = gemini_cache_client
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.repetition_penalty = repetition_penalty
        self._limiter = RateLimiter(requests_per_minute) if requests_per_minute else None

    def _resolve_cached_content(self, req: LLMRequest) -> str | None:
        """Return the cached_content name for this request, creating it if needed."""
        if not self._gemini_cache_client:
            return self.cached_content
        system_text = req.flat_system
        if not system_text:
            return self.cached_content
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": "_"}]}]
        return self._gemini_cache_client.get_or_create(
            model=self.model,
            contents=contents,
            system_instruction=system_text,
        )

    def _call_one(self, req: LLMRequest) -> LLMResponse:
        logger.debug("_call_one start id=%s", req.custom_id)
        if self._limiter:
            self._limiter.acquire()
        messages: list[dict[str, Any]] = []
        sys_blocks = [b for seg in (req.system_segments or []) for b in seg]
        if sys_blocks:
            messages.append({"role": "system", "content": _to_chat_content(sys_blocks, self._is_gemini)})
        if self._is_gemini:
            user_content: Any = _build_gemini_user_content(req)
        else:
            user_content = req.flat_user
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if req.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": req.json_schema},
            }
        if self.service_tier is not None:
            payload["service_tier"] = self.service_tier
        cached_content = self._resolve_cached_content(req)
        if cached_content is not None:
            payload["google"] = {"cached_content": cached_content}
        for key in ("temperature", "top_p", "top_k", "seed", "repetition_penalty"):
            val = getattr(self, key)
            if val is not None:
                payload[key] = val
        http_req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        _MAX_RETRIES = 3
        for attempt in range(_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(http_req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                content = body["choices"][0]["message"]["content"]
                logger.debug("_call_one done id=%s content_len=%d", req.custom_id, len(content or ""))
                return LLMResponse(id=req.output_id, content=content, source_file=req.source_file)
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
                if exc.code in {400, 401, 403, 404, 422}:
                    # Non-retryable: bad request, auth, or not found
                    logger.warning("request failed (non-retryable) id=%s: %s", req.custom_id, repr(error)[:200])
                    return LLMResponse(id=req.output_id, content=None, error=error, source_file=req.source_file)
                if attempt < _MAX_RETRIES:
                    logger.warning("request failed id=%s (attempt %d/%d): %s — retrying", req.custom_id, attempt + 1, _MAX_RETRIES, repr(error)[:200])
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("request failed id=%s: %s", req.custom_id, repr(error)[:200])
                return LLMResponse(id=req.output_id, content=None, error=error, source_file=req.source_file)
            except (KeyError, json.JSONDecodeError) as exc:
                # Non-retryable: unexpected response shape
                logger.warning("request failed (non-retryable) id=%s: %s", req.custom_id, repr(str(exc))[:200])
                return LLMResponse(id=req.output_id, content=None, error=str(exc), source_file=req.source_file)
            except urllib.error.URLError as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning("request failed id=%s (attempt %d/%d): %s — retrying", req.custom_id, attempt + 1, _MAX_RETRIES, repr(str(exc))[:200])
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("request failed id=%s: %s", req.custom_id, repr(str(exc))[:200])
                return LLMResponse(id=req.output_id, content=None, error=str(exc), source_file=req.source_file)
        # unreachable
        return LLMResponse(id=req.output_id, content=None, error="unexpected retry exhaustion", source_file=req.source_file)

    def run(self, req: LLMRequest) -> LLMResponse:
        """Send a single request and return its response."""
        return self._call_one(req)

    def stream(self, requests: list[LLMRequest]):
        """Yield each LLMResponse as it completes (completion order, not input order)."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(self._call_one, req): req for req in requests}
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as exc:
                    req = futures[future]
                    yield LLMResponse(id=req.output_id, content=None, error=str(exc), source_file=req.source_file)

