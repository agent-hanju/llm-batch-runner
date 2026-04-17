from __future__ import annotations

import logging
import time
from typing import Any

from models import Block, DocumentSpec, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


def _to_anthropic_doc_block(spec: DocumentSpec) -> dict[str, Any]:
    source: Any = (
        {"type": "plain_text", "media_type": "text/plain", "data": spec.source}
        if isinstance(spec.source, str)
        else spec.source
    )
    block: dict[str, Any] = {"type": "document", "source": source}
    if spec.title:
        block["title"] = spec.title
    if spec.cached:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def _to_anthropic_blocks(blocks: list[Block]) -> list[Any] | str:
    """Convert block list to Anthropic TextBlockParam list (or plain str if trivial)."""
    from anthropic.types import TextBlockParam

    result: list[Any] = []
    for text, cached in blocks:
        if cached:
            result.append(TextBlockParam(type="text", text=text, cache_control={"type": "ephemeral"}))
        else:
            result.append(TextBlockParam(type="text", text=text))

    # Single uncached block → plain string (simpler wire format)
    if len(result) == 1 and not blocks[0][1]:
        return blocks[0][0]
    return result


class AnthropicBatchRunner:
    """
    Anthropic Message Batches API with prompt caching.

    - Batches are 50% cheaper than real-time requests.
    - Place ``<cache />`` in system/user templates to mark cache breakpoints.
      Everything before each marker gets ``cache_control: ephemeral``; text
      after the last marker is sent uncached.
    - Polls until the batch reaches ``processing_status == "ended"``.

    Requires: ``pip install anthropic``
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 4096,
        poll_interval: int = 30,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError("Install the Anthropic SDK: pip install anthropic") from exc

        self._anthropic = _anthropic
        self.client = (
            _anthropic.Anthropic(api_key=api_key) if api_key else _anthropic.Anthropic()
        )
        self.model = model
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.repetition_penalty = repetition_penalty

    def _make_batch_req(self, req: LLMRequest) -> Any:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        doc_blocks = [_to_anthropic_doc_block(d) for d in (req.documents or [])]
        has_marker = len(req.user_segments) > 1
        all_user_blocks = [b for seg in req.user_segments for b in seg]
        has_user_cache = any(c for _, c in all_user_blocks)

        if doc_blocks or has_user_cache:
            combined: list[Any] = []
            if has_marker:
                for i, seg in enumerate(req.user_segments):
                    seg_blocks = _to_anthropic_blocks(seg) if seg else []
                    combined += [seg_blocks] if isinstance(seg_blocks, str) else seg_blocks
                    if i < len(req.user_segments) - 1:
                        combined += doc_blocks
            else:
                combined = doc_blocks + (_to_anthropic_blocks(all_user_blocks) if all_user_blocks else [])  # type: ignore[operator]
                if isinstance(combined, str):
                    combined = [combined]
            user_content: Any = combined if combined else req.flat_user
        else:
            user_content = req.flat_user

        params_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user_content}],
        }

        # System content — if <documents /> marker present, inline doc text at that position
        if req.system_segments:
            sys_segs = req.system_segments
            if len(sys_segs) > 1 and req.documents:
                doc_text = "\n\n".join(LLMRequest._doc_text(d) for d in req.documents)
                sys_blocks: list[Block] = []
                for i, seg in enumerate(sys_segs):
                    sys_blocks.extend(seg)
                    if i < len(sys_segs) - 1 and doc_text:
                        sys_blocks.append((doc_text, False))
            else:
                sys_blocks = [b for seg in sys_segs for b in seg]
            if sys_blocks:
                has_sys_cache = any(c for _, c in sys_blocks)
                if has_sys_cache:
                    params_kwargs["system"] = _to_anthropic_blocks(sys_blocks)
                else:
                    params_kwargs["system"] = "".join(t for t, _ in sys_blocks)

        for key in ("temperature", "top_p", "top_k", "seed", "repetition_penalty"):
            val = getattr(self, key)
            if val is not None:
                params_kwargs[key] = val

        if req.json_schema is not None:
            from anthropic.types import JSONOutputFormatParam
            params_kwargs["output_config"] = {"format": JSONOutputFormatParam(
                type="json_schema",
                schema=req.json_schema,
            )}

        return Request(
            custom_id=req.custom_id,
            params=MessageCreateParamsNonStreaming(**params_kwargs),  # type: ignore[typeddict-item]
        )

    def _wait_for_batch(self, batch: Any) -> Any:
        """Poll until batch processing_status == 'ended', return final batch object."""
        _MAX_RETRIES = 5
        retries = 0
        while True:
            try:
                batch = self.client.messages.batches.retrieve(batch.id)
                retries = 0
            except (
                self._anthropic.AuthenticationError,
                self._anthropic.PermissionDeniedError,
                self._anthropic.NotFoundError,
                self._anthropic.BadRequestError,
            ) as exc:
                # Non-retryable: config/auth problem, batch not found, etc.
                raise RuntimeError(f"batch {batch.id}: unrecoverable error — {exc}") from exc
            except (
                self._anthropic.APIConnectionError,
                self._anthropic.APITimeoutError,
                self._anthropic.RateLimitError,
                self._anthropic.InternalServerError,
            ) as exc:
                retries += 1
                if retries > _MAX_RETRIES:
                    raise RuntimeError(
                        f"batch {batch.id}: polling failed {_MAX_RETRIES} times in a row — last error: {exc}"
                    ) from exc
                logger.warning(
                    "batch %s: retrieve failed (attempt %d/%d): %s — retrying in %ds",
                    batch.id, retries, _MAX_RETRIES, exc, self.poll_interval,
                )
                time.sleep(self.poll_interval)
                continue
            c = batch.request_counts
            logger.info(
                "batch %s: %s — processing=%d succeeded=%d errored=%d",
                batch.id, batch.processing_status,
                c.processing, c.succeeded, c.errored,
            )
            if batch.processing_status == "ended":
                return batch
            time.sleep(self.poll_interval)

    def run(self, req: LLMRequest) -> LLMResponse:
        """Submit a single-item batch and return its response."""
        return next(iter(self.stream([req])))

    def stream(self, requests: list[LLMRequest]):
        """Submit an Anthropic batch, poll until done, yield each LLMResponse as results arrive."""
        batch_reqs = []
        for req in requests:
            try:
                batch_reqs.append(self._make_batch_req(req))
            except Exception as exc:
                raise ValueError(f"failed to build batch request for id={req.output_id!r}: {exc}") from exc
        batch = self.client.messages.batches.create(requests=batch_reqs)
        logger.info("submitted batch %s (%d requests)", batch.id, len(requests))
        batch = self._wait_for_batch(batch)
        id_to_req: dict[str, LLMRequest] = {req.custom_id: req for req in requests}
        for result in self.client.messages.batches.results(batch.id):
            req = id_to_req[result.custom_id]
            yield self._parse_result(result, req.output_id, req.source_file)


    def _parse_result(self, result: Any, output_id: str | None = None, source_file: str | None = None) -> LLMResponse:
        rid = output_id or result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            content = next((b.text for b in msg.content if b.type == "text"), None)
            return LLMResponse(id=rid, content=content, source_file=source_file)
        return LLMResponse(
            id=rid,
            content=None,
            error=str(getattr(result.result, "error", result.result)),
            source_file=source_file,
        )
