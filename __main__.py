"""CLI entry point: python -m llm_batch_runner (or python __main__.py)"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from models import LLMRequest, LLMResponse  # noqa: F401
from template import build_requests
from runner_chat import ChatCompletionsRunner
from runner_anthropic import AnthropicBatchRunner
from runner_cli import CliRunner

_BACKEND_KEYS = ("chat", "anthropic", "cli")

_CONFIG_EXAMPLE = """\
Example config.json
-------------------
Chat Completions (local / OpenAI / Gemini):
{
  "backend": "chat",
  "base_url": "http://localhost:11434",
  "model": "qwen2.5:7b",
  "api_key": "none",
  "max_concurrent": 5,
  "requests_per_minute": null,
  "timeout": 300,
  "max_tokens": 4096,
  "service_tier": null,
  "cached_content": null,
  "temperature": null,
  "top_p": null,
  "top_k": null,
  "seed": null,
  "repetition_penalty": null
}

Anthropic Batch API:
{
  "backend": "anthropic",
  "api_key": null,
  "model": "claude-haiku-4-5",
  "max_tokens": 4096,
  "poll_interval": 30,
  "temperature": null,
  "top_p": null,
  "top_k": null,
  "seed": null,
  "repetition_penalty": null
}

CLI runner (claude / codex / gemini):
{
  "backend": "cli",
  "cli": "claude",
  "cli_bin": null,
  "max_concurrent": 3,
  "requests_per_minute": null,
  "timeout": 300
}

Template JSON (--template):
{
  "user_template": "Answer this question: {question}",
  "system_template": "You are a {role}.",
  "json_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": false}
}

Input JSONL (--input, one variable dict per line):
{"_id": "q1", "role": "teacher", "question": "What is osmosis?"}
{"_id": "q2", "role": "teacher", "question": "What is photosynthesis?"}
"""

logger = logging.getLogger(__name__)


def _load_file(path: Path) -> list[dict[str, Any]]:
    """Load variable dicts from a single .json or .jsonl file."""
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []

    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON — {exc}") from exc
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError(f"{path}: expected a JSON object or array, got {type(parsed).__name__}")
        for i, obj in enumerate(parsed):
            if not isinstance(obj, dict):
                raise ValueError(f"{path}[{i}]: expected a JSON object, got {type(obj).__name__}")
            rows.append(obj)
    else:
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}")
            rows.append(obj)

    source = str(path.resolve())
    for row in rows:
        row["_source_file"] = source

    return rows


def _load_variables(input_path: str) -> list[dict[str, Any]]:
    """Load variable dicts from a .json file, .jsonl file, or directory of such files."""
    p = Path(input_path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".json", ".jsonl"))
        if not files:
            raise ValueError(f"{p}: directory contains no .json or .jsonl files")
        rows: list[dict[str, Any]] = []
        for f in files:
            rows.extend(_load_file(f))
        return rows
    return _load_file(p)


def _load_template(template_path: str) -> dict[str, Any]:
    """Load a template JSON with user_template, optional system_template, optional json_schema."""
    try:
        tmpl = json.loads(Path(template_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{template_path}: invalid JSON — {exc}") from exc
    if "user_template" not in tmpl:
        raise ValueError(f"{template_path}: missing required field 'user_template'")
    return tmpl


def _read_output(output_path: Path) -> tuple[set[str], set[str]]:
    """Parse existing output file. Returns (succeeded_ids, errored_ids)."""
    succeeded: set[str] = set()
    errored: set[str] = set()
    if not output_path.exists():
        return succeeded, errored
    for lineno, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("output:%d: invalid JSON — skipping", lineno)
            continue
        rid = row.get("id")
        if rid is None:
            continue
        if row.get("error") is None:
            succeeded.add(str(rid))
        else:
            errored.add(str(rid))
    return succeeded, errored


def _strip_error_rows(output_path: Path, ids: set[str]) -> None:
    """Rewrite output file removing rows whose id is in *ids*."""
    lines = output_path.read_text(encoding="utf-8").splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if str(row.get("id")) not in ids:
            kept.append(line)
    output_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _build_runner(cfg: dict[str, Any]) -> ChatCompletionsRunner | AnthropicBatchRunner | CliRunner:
    backend = cfg.get("backend", "chat")
    if backend == "chat":
        return ChatCompletionsRunner(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg.get("api_key", "none"),
            max_concurrent=cfg.get("max_concurrent", 5),
            requests_per_minute=cfg.get("requests_per_minute"),
            timeout=cfg.get("timeout", 120),
            max_tokens=cfg.get("max_tokens", 4096),
            service_tier=cfg.get("service_tier"),
            cached_content=cfg.get("cached_content"),
            temperature=cfg.get("temperature"),
            top_p=cfg.get("top_p"),
            top_k=cfg.get("top_k"),
            seed=cfg.get("seed"),
            repetition_penalty=cfg.get("repetition_penalty"),
        )
    if backend == "anthropic":
        return AnthropicBatchRunner(
            api_key=cfg.get("api_key"),
            model=cfg.get("model", "claude-haiku-4-5"),
            max_tokens=cfg.get("max_tokens", 4096),
            poll_interval=cfg.get("poll_interval", 30),
            temperature=cfg.get("temperature"),
            top_p=cfg.get("top_p"),
            top_k=cfg.get("top_k"),
            seed=cfg.get("seed"),
            repetition_penalty=cfg.get("repetition_penalty"),
        )
    if backend == "cli":
        return CliRunner(
            cli=cfg.get("cli", "claude"),
            max_concurrent=cfg.get("max_concurrent", 3),
            requests_per_minute=cfg.get("requests_per_minute"),
            timeout=cfg.get("timeout", 300),
            cli_bin=cfg.get("cli_bin"),
        )
    raise ValueError(f"Unknown backend {backend!r}. Must be one of: {_BACKEND_KEYS}")


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="llm_batch_runner",
        description="Run batched LLM requests from JSONL input to JSONL output.",
        epilog=_CONFIG_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, metavar="PATH",
                        help="JSON file with backend/model/endpoint settings.")
    parser.add_argument("--template", required=True, metavar="PATH",
                        help="JSON file with user_template, system_template, json_schema.")
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="JSONL file of variable dicts.")
    parser.add_argument("--output", required=True, metavar="PATH",
                        help="JSONL file to append results to.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all requests even if output file already exists.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config file not found: %s", config_path)
        sys.exit(1)

    cfg: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

    output_path = Path(args.output)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("input file not found: %s", input_path)
        sys.exit(1)
    template_path = Path(args.template)
    if not template_path.exists():
        logger.error("template file not found: %s", template_path)
        sys.exit(1)

    try:
        runner = _build_runner(cfg)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        variables = _load_variables(str(input_path))
        tmpl = _load_template(str(template_path))
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    requests = build_requests(
        user_template=tmpl["user_template"],
        variable_list=variables,
        system_template=tmpl.get("system_template"),
        json_schema=tmpl.get("json_schema"),
    )

    if not args.force and output_path.exists():
        succeeded_ids, errored_ids = _read_output(output_path)
        rerun_ids = errored_ids - succeeded_ids  # succeeded on retry → don't rerun
        if rerun_ids:
            logger.info("resume: removing %d error row(s) from output for re-run", len(rerun_ids))
            _strip_error_rows(output_path, rerun_ids)
        skip_ids = succeeded_ids
        before = len(requests)
        requests = [r for r in requests if r.output_id not in skip_ids]
        skipped = before - len(requests)
        if skipped:
            logger.info("resume: %d/%d already succeeded — %d remaining", skipped, before, len(requests))

    if not requests:
        logger.info("nothing to do")
        sys.exit(0)

    logger.info("%d requests → %s", len(requests), args.output)

    total = len(requests)
    errors = 0
    done = 0
    try:
        with open(output_path, "a", encoding="utf-8") as fout:
            for resp in runner.stream(requests):
                done += 1
                if resp.error:
                    errors += 1
                    logger.warning("error on %s: %s", resp.id, repr(resp.error)[:200])
                logger.info("%d/%d done (errors=%d)", done, total, errors)
                fout.write(json.dumps(asdict(resp), ensure_ascii=False) + "\n")
                fout.flush()
    except (ValueError, RuntimeError) as exc:
        logger.error("fatal: %s", exc)
        sys.exit(1)

    logger.info("done. errors=%d/%d", errors, total)
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _main()
