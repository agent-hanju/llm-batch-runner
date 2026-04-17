from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from models import Block, DocumentSpec, LLMRequest, LLMResponse


def _req_to_dict(req: LLMRequest) -> dict[str, Any]:
    return asdict(req)


def _dict_to_req(d: dict[str, Any]) -> LLMRequest:
    user_segments: list[list[Block]] = [[tuple(b) for b in seg] for seg in d["user_segments"]]  # type: ignore[misc]
    system_segments: list[list[Block]] | None = (
        [[tuple(b) for b in seg] for seg in d["system_segments"]]  # type: ignore[misc]
        if d.get("system_segments") is not None else None
    )
    documents = (
        [DocumentSpec(**doc) for doc in d["documents"]]
        if d.get("documents") else None
    )
    return LLMRequest(
        custom_id=d["custom_id"],
        output_id=d["output_id"],
        user_segments=user_segments,
        system_segments=system_segments,
        json_schema=d.get("json_schema"),
        source_file=d.get("source_file"),
        documents=documents,
    )


class BatchCollector:
    """
    Accumulate LLMRequests on disk and flush them as a single batch.

    Requests are serialised to a JSONL spool file on ``add()`` so memory
    usage stays flat regardless of batch size.  ``flush()`` reads the spool,
    submits all requests to the runner, yields each LLMResponse, then clears
    the spool.

    When ``max_size`` is set, ``add()`` automatically flushes and yields
    responses once the accumulated count reaches the limit — useful for
    backends with a per-batch cap (e.g. Anthropic's 10,000 limit).

    Usage::

        collector = BatchCollector(runner, max_size=10_000)
        for item in source:
            for resp in collector.add(build_request(item)):
                process(resp)
        # sentinel reached
        for resp in collector.flush():
            process(resp)
    """

    def __init__(
        self,
        runner: Any,
        spool_path: str | Path | None = None,
        max_size: int | None = None,
    ) -> None:
        self._runner = runner
        self._max_size = max_size
        if spool_path:
            self._spool = Path(spool_path)
        else:
            fd, tmp = tempfile.mkstemp(suffix=".jsonl")
            os.close(fd)
            self._spool = Path(tmp)
        self._count = 0

    # ------------------------------------------------------------------

    def add(self, req: LLMRequest) -> Iterator[LLMResponse]:
        """Append a request to the spool; auto-flush and yield responses if max_size is reached."""
        with self._spool.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_req_to_dict(req), ensure_ascii=False) + "\n")
        self._count += 1
        if self._max_size is not None and self._count >= self._max_size:
            yield from self.flush()

    def flush(self) -> Iterator[LLMResponse]:
        """Submit all buffered requests, yield responses, then clear the spool."""
        if self._count == 0:
            return
        requests = self._load_spool()
        self._clear_spool()
        yield from self._runner.stream(requests)

    def __len__(self) -> int:
        return self._count

    # ------------------------------------------------------------------

    def _load_spool(self) -> list[LLMRequest]:
        requests: list[LLMRequest] = []
        for line in self._spool.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                requests.append(_dict_to_req(json.loads(line)))
        return requests

    def _clear_spool(self) -> None:
        self._spool.unlink(missing_ok=True)
        self._count = 0
