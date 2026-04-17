from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from models import LLMRequest, LLMResponse
from runner_base import BaseRunner
from utils import RateLimiter

logger = logging.getLogger(__name__)

_CLI_TYPE = Literal["claude", "codex", "gemini"]


def _build_inline_prompt(req: LLMRequest) -> str:
    """Flatten system and user blocks into a single prompt string for CLIs."""
    parts: list[str] = []
    if req.system_segments:
        parts.append(f"[System instructions: {req.flat_system}]")
    parts.append(req.flat_user)
    if req.json_schema:
        parts.append(
            f"Respond with valid JSON that strictly matches this schema:\n"
            f"{json.dumps(req.json_schema, ensure_ascii=False)}"
        )
    return "\n\n".join(parts)


class CliRunner(BaseRunner):
    """
    Subprocess runner for local AI CLIs: ``claude``, ``codex``, or ``gemini``.

    Use this when you have an account-based CLI installed but no API key.

    **Claude** uses native CLI flags for all fields.
    **Codex** and **Gemini** inline system_prompt and json_schema into the prompt.
    """

    def __init__(
        self,
        cli: _CLI_TYPE = "claude",
        max_concurrent: int = 3,
        requests_per_minute: int | None = None,
        timeout: int = 300,
        cli_bin: str | None = None,
    ) -> None:
        self.cli = cli
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.cli_bin = cli_bin or cli
        self._limiter = RateLimiter(requests_per_minute) if requests_per_minute else None
        if shutil.which(self.cli_bin) is None:
            raise FileNotFoundError(f"'{self.cli_bin}' not found — is the CLI installed and on PATH?")

    def _build_cmd(self, req: LLMRequest) -> list[str]:
        if self.cli == "claude":
            cmd: list[str] = [self.cli_bin, "-p", req.flat_user, "--output-format", "json"]
            if req.system_segments:
                cmd += ["--system-prompt", req.flat_system or ""]
            if req.json_schema is not None:
                cmd += ["--json-schema", json.dumps(req.json_schema)]
            return cmd
        if self.cli == "codex":
            return [self.cli_bin, "exec", _build_inline_prompt(req)]
        # gemini
        return [self.cli_bin, "-p", _build_inline_prompt(req)]

    def _parse_output(self, raw: str) -> tuple[str | None, str | None]:
        """Return (content, error). content is None on error, error is None on success."""
        if self.cli != "claude":
            return raw, None
        try:
            outer: dict[str, Any] = json.loads(raw)
            if outer.get("is_error"):
                return None, outer.get("result") or "claude CLI returned is_error=true"
            result = outer.get("result")
            return (result if isinstance(result, str) else raw), None
        except (json.JSONDecodeError, TypeError):
            return raw, None

    # Exit codes that indicate a transient error worth retrying (rate limit, server error)
    _RETRYABLE_EXIT_CODES = {1}

    def run(self, req: LLMRequest) -> LLMResponse:
        logger.debug("run start cli=%s id=%s", self.cli, req.custom_id)
        if self._limiter:
            self._limiter.acquire()
        _MAX_RETRIES = 3
        for attempt in range(_MAX_RETRIES + 1):
            try:
                proc = subprocess.run(
                    self._build_cmd(req),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                if attempt < _MAX_RETRIES:
                    logger.warning("timeout id=%s (attempt %d/%d) — retrying", req.custom_id, attempt + 1, _MAX_RETRIES)
                    continue
                logger.warning("timeout id=%s", req.custom_id)
                return LLMResponse(id=req.output_id, content=None, error="timeout", source_file=req.source_file)

            if proc.returncode != 0:
                error = proc.stderr or f"exit code {proc.returncode}"
                if proc.returncode in self._RETRYABLE_EXIT_CODES and attempt < _MAX_RETRIES:
                    logger.warning("cli error id=%s (attempt %d/%d): %s — retrying", req.custom_id, attempt + 1, _MAX_RETRIES, repr(error)[:200])
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("cli error id=%s: %s", req.custom_id, repr(error)[:200])
                return LLMResponse(id=req.output_id, content=None, error=error, source_file=req.source_file)

            content, parse_error = self._parse_output(proc.stdout.strip())
            if parse_error:
                # Non-retryable: CLI returned success but output was malformed
                logger.warning("cli error id=%s: %s", req.custom_id, repr(parse_error)[:200])
                return LLMResponse(id=req.output_id, content=None, error=parse_error, source_file=req.source_file)
            logger.debug("_call_one done id=%s content_len=%d", req.custom_id, len(content or ""))
            return LLMResponse(id=req.output_id, content=content, source_file=req.source_file)
        # unreachable
        return LLMResponse(id=req.output_id, content=None, error="unexpected retry exhaustion", source_file=req.source_file)

    def stream(self, requests: list[LLMRequest]):
        """Yield each LLMResponse as it completes (completion order, not input order)."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(self.run, req): req for req in requests}
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as exc:
                    req = futures[future]
                    yield LLMResponse(id=req.output_id, content=None, error=str(exc), source_file=req.source_file)

