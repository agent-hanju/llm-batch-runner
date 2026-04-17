# llm_batch_runner

Batch LLM requests from a JSONL input file to a JSONL output file.
Supports OpenAI-compatible endpoints (local/cloud), Anthropic Batch API, and local AI CLIs.
Uses only Python stdlib — no pip installs required except for the Anthropic backend.

## Requirements

- Python 3.7+
- `anthropic` SDK only if using `backend: "anthropic"` (`pip install anthropic`)

## Backends

| Backend | Description |
|---|---|
| `chat` | OpenAI-compatible Chat Completions (Ollama, vLLM, LM Studio, OpenAI, Gemini, etc.) |
| `anthropic` | Anthropic Message Batches API (50% cheaper, async polling) |
| `cli` | Local AI CLIs: `claude`, `codex`, or `gemini` |

## Usage

```bash
python __main__.py \
  --config config.json \
  --template template.json \
  --input input.jsonl \
  --output output.jsonl
```

All four arguments are required. Results are appended to `--output` as JSONL.
Exit code is `2` if any request errored, `0` otherwise.

### Resume and --force

By default, if `--output` already exists, succeeded rows are skipped and errored rows are re-run:

```bash
# Re-run only failed requests (default)
python __main__.py --config config.json --template template.json \
  --input input.jsonl --output output.jsonl

# Re-run everything, ignoring existing output
python __main__.py --config config.json --template template.json \
  --input input.jsonl --output output.jsonl --force
```

Resume behavior:
1. Rows with `"error": null` in the output are skipped.
2. Rows with errors are removed from the output file and re-run.
3. A row that previously errored but later succeeded won't be re-run again.

To run in the background and monitor progress:

```bash
python __main__.py --config config.json --template template.json \
  --input input.jsonl --output output.jsonl > run.log 2>&1 &
tail -f run.log
```

## Config

### Chat Completions

```json
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
```

`base_url` accepts any OpenAI-compatible endpoint. `/v1` is appended automatically if not present.
`api_key: "none"` is fine for local endpoints that don't require authentication.
`cached_content` is a Gemini-specific pre-created cache resource name (e.g. `"cachedContents/abc123"`).
See `GeminiCacheClient` in `runner_chat.py` to create one, or use it programmatically for automatic per-request caching.

### Anthropic Batch API

```json
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
```

`api_key: null` falls back to the `ANTHROPIC_API_KEY` environment variable.
`poll_interval` is in seconds. The batch is submitted, polled until `processing_status == "ended"`, then results are streamed to file.

### CLI Runner

```json
{
  "backend": "cli",
  "cli": "claude",
  "cli_bin": null,
  "max_concurrent": 3,
  "requests_per_minute": null,
  "timeout": 300
}
```

`cli` controls argument/output parsing — must be `"claude"`, `"codex"`, or `"gemini"`.
`cli_bin` overrides the executable path (e.g. `"/usr/local/bin/claude-dev"`); defaults to the value of `cli`.

## Template

```json
{
  "user_template": "Answer this question: {question}",
  "system_template": "You are a {role}.",
  "json_schema": {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

`system_template` and `json_schema` are optional.
Templates use Python `str.format_map` — variables from each input row are substituted by name.

### Prompt caching with `<cache />`

Place `<cache />` (or `<cache/>`) in templates to set cache breakpoints.
Everything before each marker gets a cache-control checkpoint; the final segment is not cached.

```
You are an expert. <cache />

Here is the document:
{document} <cache />

Answer this question: {question}
```

- Supported in both `system_template` and `user_template`
- Works with `backend: "anthropic"` and `backend: "chat"` (Gemini only)
- Ignored by the CLI backend — markers are stripped and content is flattened to plain text

## Input JSONL

One variable dict per line. Use `_id` for a custom request ID; otherwise the line index is used.
`--input` also accepts a directory — all `.json` and `.jsonl` files are loaded in sorted order.

```jsonl
{"_id": "q1", "role": "teacher", "question": "What is osmosis?"}
{"_id": "q2", "role": "teacher", "question": "What is photosynthesis?"}
```

### Documents (`_documents`)

Attach documents to a request with the `_documents` field. Each entry must be an Anthropic source object:

```jsonl
{"_id": "q1", "question": "Summarise this.", "_documents": [
  {"type": "plain_text", "data": "The document text here.", "title": "Doc 1"},
  {"type": "plain_text", "data": "Another document.", "_cache": true}
]}
```

Supported types: `plain_text` (`data` field) and `content` (`content` field, for structured blocks e.g. from DoclingDocument).
Add `"_cache": true` to cache a document block. `"title"` is optional.

Where `<documents />` appears in a template determines how documents are handled for the `anthropic` backend:

- **`user_template`**: documents become native `document` blocks in the user message `content` array at the marker position. Without the marker, they are prepended as `document` blocks before the user text.
- **`system_template`**: documents are inlined as plain text at the marker position inside the system prompt string.

All other backends always inline document `data` text at the marker position regardless of which template it appears in; without a marker, text is prepended to the user message.

#### Example (`anthropic` backend, `user_template` marker)

Template:
```json
{
  "user_template": "Here are the documents:\n\n<documents />\n\nQuestion: {question}"
}
```

Input:
```jsonl
{"_id": "q1", "question": "What are the findings?", "_documents": [
  {"type": "plain_text", "data": "Study A text.", "title": "Study A", "_cache": true},
  {"type": "plain_text", "data": "Study B text.", "title": "Study B"}
]}
```

The user message sent to the API:
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Here are the documents:\n\n"},
    {"type": "document", "source": {"type": "plain_text", "data": "Study A text."}, "title": "Study A", "cache_control": {"type": "ephemeral"}},
    {"type": "document", "source": {"type": "plain_text", "data": "Study B text."}, "title": "Study B"},
    {"type": "text", "text": "\n\nQuestion: What are the findings?"}
  ]
}
```

The cache breakpoint after Study A means the model can reuse the cached KV state up through that document across requests that share the same prefix.

## Output JSONL

```jsonl
{"id": "q1", "content": "Osmosis is ...", "error": null, "source_file": "/abs/path/to/input.jsonl"}
{"id": "q2", "content": null, "error": "HTTP 429: ...", "source_file": "/abs/path/to/input.jsonl"}
```

`content` is `null` on error; `error` is `null` on success.
`id` is the original `_id` from the input row.
`source_file` is the absolute path of the source file. When `--input` is a directory, it identifies which file each row came from.

For `chat` and `cli` backends, results are written immediately as each request completes.
For `anthropic`, results are written after the entire batch ends.

## Logging

Progress is logged to stderr at `INFO` level:

```
16:42:01 INFO 10 requests → output.jsonl
16:42:03 INFO 1/10 done (errors=0)
16:42:04 INFO 2/10 done (errors=0)
...
16:42:09 INFO 10/10 done (errors=1)
16:42:09 INFO done. errors=1/10
```

For `anthropic`, batch polling status is logged before results arrive:

```
16:42:01 INFO submitted batch msgbatch_abc123 (10 requests)
16:42:31 INFO batch msgbatch_abc123: processing — processing=8 succeeded=2 errored=0
16:43:01 INFO batch msgbatch_abc123: ended — processing=0 succeeded=9 errored=1
16:43:01 INFO 1/10 done (errors=0)
...
16:43:02 INFO 10/10 done (errors=1)
16:43:02 INFO done. errors=1/10
```

Set log level to `DEBUG` to see per-request detail and Gemini cache hit/miss events.

## Programmatic use

All runners share the same interface:

```python
from runner_anthropic import AnthropicBatchRunner
from runner_chat import ChatCompletionsRunner
from template import build_requests

runner = AnthropicBatchRunner(model="claude-haiku-4-5")
requests = build_requests(user_template="Summarise: {text}", variable_list=[...])

# Batch: yield responses as they complete
for resp in runner.stream(requests):
    print(resp.id, resp.content)

# Single request
resp = runner.run(requests[0])
```

### BatchCollector — accumulate then flush

Use `BatchCollector` when requests arrive one at a time and you want to submit them as a single batch once all are ready:

```python
from collector import BatchCollector

collector = BatchCollector(runner, max_size=10_000)
for item in source:
    for resp in collector.add(build_request(item)):  # auto-flushes at max_size
        process(resp)
# sentinel reached — flush remainder
for resp in collector.flush():
    process(resp)
```

Requests are serialized to a temp JSONL spool file on `add()`, so memory usage stays flat regardless of batch size. `flush()` reads the spool, calls `runner.stream()`, yields each response, then clears the spool.

`max_size` triggers an automatic flush (and response yield) when the accumulated count reaches the limit. Use `max_size=10_000` with the Anthropic backend to stay within its per-batch cap. Omit it when there is no limit.
