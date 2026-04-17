from __future__ import annotations

import logging
import re
from typing import Any

from models import Block, DocumentSpec, LLMRequest

logger = logging.getLogger(__name__)

_CACHE_MARKER_RE = re.compile(r"<cache\s*/>")
_DOC_MARKER_RE = re.compile(r"<documents\s*/>")


def _parse_blocks(text: str) -> list[Block]:
    """Split *text* on ``<cache />`` markers into ``(text, cached)`` blocks.

    Each segment before a marker gets ``cached=True``; the final segment has
    ``cached=False``. Empty segments are dropped.
    """
    parts = _CACHE_MARKER_RE.split(text)
    last = len(parts) - 1
    return [
        (part.strip(), i < last)
        for i, part in enumerate(parts)
        if part.strip()
    ]


def _parse_segments(text: str, label: str = "") -> list[list[Block]]:
    """Split *text* on ``<documents />`` markers; parse each part into blocks.

    Returns a list of block-lists. Length == number of markers + 1.
    Length 1 means no marker was present.
    """
    parts = _DOC_MARKER_RE.split(text)
    if len(parts) > 2:
        logger.warning(
            "%s<documents /> marker appears %d times — documents will be inserted at each position",
            f"{label}: " if label else "",
            len(parts) - 1,
        )
    return [_parse_blocks(part) for part in parts]


def _parse_documents(raw: list[Any]) -> list[DocumentSpec]:
    """Parse ``_documents`` entries from an input row into ``DocumentSpec`` objects."""
    specs: list[DocumentSpec] = []
    for item in raw:
        if isinstance(item, dict) and item.get("type") in {"plain_text", "content"}:
            specs.append(DocumentSpec(
                source={k: v for k, v in item.items() if k not in ("_cache", "title")},
                cached=bool(item.get("_cache", False)),
                title=item.get("title") or None,
            ))
        else:
            logger.warning("_documents: unsupported item %r — expected dict with type 'plain_text' or 'content'", item)
    return specs


def build_requests(
    user_template: str,
    variable_list: list[dict[str, Any]],
    system_template: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> list[LLMRequest]:
    """Build one LLMRequest per entry in *variable_list*.

    ``<cache />`` markers are parsed into block lists with cache breakpoints.
    ``<documents />`` markers split a template into segments; documents from
    ``_documents`` in each input row are injected between each adjacent segment.
    If no marker is present, documents are prepended to the user message.
    """
    requests: list[LLMRequest] = []
    for i, variables in enumerate(variable_list):
        rid = str(variables.get("_id", i))
        try:
            user_rendered = user_template.format_map(variables)
            system_rendered = system_template.format_map(variables) if system_template else None
        except KeyError as exc:
            raise ValueError(f"row {rid}: template variable {exc} not found in input") from exc

        user_segments = _parse_segments(user_rendered, f"row {rid} user")
        system_segments = _parse_segments(system_rendered, f"row {rid} system") if system_rendered is not None else None

        raw_docs = variables.get("_documents")
        documents = _parse_documents(raw_docs) if isinstance(raw_docs, list) and raw_docs else None

        requests.append(LLMRequest(
            custom_id=str(i),
            output_id=str(variables.get("_id", i)),
            user_segments=user_segments,
            system_segments=system_segments,
            json_schema=json_schema,
            source_file=variables.get("_source_file"),
            documents=documents,
        ))
    return requests
