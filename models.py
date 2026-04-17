from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Tuple

logger = logging.getLogger(__name__)

# (text, cached) — cached=True means this block gets a cache breakpoint
Block = Tuple[str, bool]


@dataclass
class DocumentSpec:
    """A single document to attach to a request.

    Comes from ``_documents`` in the input JSONL row.

    ``source`` is either:
    - ``str``: plain text content (shorthand for ``plain_text`` source type)
    - ``dict``: a full Anthropic document source object (``type``, ``media_type``,
      ``data`` / ``url`` / ``content`` fields) — passed as-is to the API.
      ``_cache`` and ``title`` are stripped before storage.
    """

    source: str | dict[str, Any]
    cached: bool = False
    title: str | None = None


@dataclass
class LLMRequest:
    """A single rendered request ready to send to any backend.

    Both system and user content are stored as segment lists so callers can place
    cache breakpoints via ``<cache />`` and document injection points via
    ``<documents />``. Each segment is a ``list[Block]``; segments are separated
    by ``<documents />`` markers. Documents are injected between each adjacent
    pair of segments. If there is only one segment (no marker), documents are
    prepended to the user message.

    Helpers
    -------
    flat_system / flat_user : plain concatenated strings with documents inlined
        (used by backends that don't support native document blocks).
    """

    custom_id: str
    user_segments: list[list[Block]]
    system_segments: list[list[Block]] | None = None
    json_schema: dict[str, Any] | None = None
    source_file: str | None = None
    output_id: str = ""
    documents: list[DocumentSpec] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.json_schema, dict) and not self.json_schema:
            self.json_schema = None
        if not self.output_id:
            self.output_id = self.custom_id

    @staticmethod
    def _doc_text(spec: DocumentSpec) -> str:
        if isinstance(spec.source, str):
            return spec.source
        if "data" not in spec.source:
            logger.warning("document type=%r has no 'data' field — cannot inline as text; skipping", spec.source.get("type"))
            return ""
        return str(spec.source["data"])

    def _flatten_segments(self, segments: list[list[Block]], has_marker: bool) -> str:
        seg_texts = ["".join(t for t, _ in seg) for seg in segments]
        if not self.documents:
            return "".join(seg_texts)
        doc_text = "\n\n".join(self._doc_text(d) for d in self.documents)
        if has_marker:
            parts = []
            for i, text in enumerate(seg_texts):
                if text.strip():
                    parts.append(text)
                if i < len(seg_texts) - 1:
                    parts.append(doc_text)
            return "\n\n".join(parts)
        return "\n\n".join(p for p in [doc_text, "".join(seg_texts)] if p.strip())

    @property
    def flat_user(self) -> str:
        return self._flatten_segments(self.user_segments, len(self.user_segments) > 1)

    @property
    def flat_system(self) -> str | None:
        if self.system_segments is None:
            return None
        result = self._flatten_segments(self.system_segments, len(self.system_segments) > 1)
        return result or None


@dataclass
class LLMResponse:
    """Result from any backend."""

    id: str
    content: str | None  # raw text from the model, None on error
    error: str | None = None
    source_file: str | None = None
