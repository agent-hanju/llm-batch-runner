# llm_batch_runner

JSONL 입력 파일로부터 LLM 요청을 일괄 처리해 JSONL 출력 파일로 저장합니다.
OpenAI 호환 엔드포인트(로컬/클라우드), Anthropic Batch API, 로컬 AI CLI를 지원합니다.
Anthropic 백엔드를 제외하면 Python 표준 라이브러리만 사용합니다.

## 요구사항

- Python 3.7+
- `backend: "anthropic"` 사용 시에만 `anthropic` SDK 필요 (`pip install anthropic`)

## 백엔드

| 백엔드 | 설명 |
|---|---|
| `chat` | OpenAI 호환 Chat Completions (Ollama, vLLM, LM Studio, OpenAI, Gemini 등) |
| `anthropic` | Anthropic Message Batches API (50% 할인, 비동기 폴링) |
| `cli` | 로컬 AI CLI: `claude`, `codex`, `gemini` |

## 사용법

```bash
python __main__.py \
  --config config.json \
  --template template.json \
  --input input.jsonl \
  --output output.jsonl
```

네 인수 모두 필수입니다. 결과는 `--output` 파일에 JSONL 형식으로 추가(append)됩니다.
오류가 하나라도 있으면 종료 코드 `2`, 없으면 `0`입니다.

### 이어서 실행 / --force

`--output` 파일이 이미 존재하면 기본적으로 성공한 행은 건너뛰고 오류 행만 재실행합니다:

```bash
# 실패한 요청만 재실행 (기본 동작)
python __main__.py --config config.json --template template.json \
  --input input.jsonl --output output.jsonl

# 기존 output을 무시하고 전체 재실행
python __main__.py --config config.json --template template.json \
  --input input.jsonl --output output.jsonl --force
```

재실행 동작 상세:
1. output에서 `"error": null`인 행의 ID는 skip합니다.
2. 오류 행은 output 파일에서 제거한 뒤 재실행합니다.
3. 이전에 오류가 났다가 이후 성공한 행은 다시 실행되지 않습니다.

백그라운드 실행 및 로그 모니터링:

```bash
python __main__.py --config config.json --template template.json \
  --input input.jsonl --output output.jsonl > run.log 2>&1 &
tail -f run.log
```

## 설정 (config.json)

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

- `base_url`: OpenAI 호환 엔드포인트 URL. 경로에 `/v1`이 없으면 자동으로 추가됩니다.
- `api_key`: 인증이 필요 없는 로컬 엔드포인트는 `"none"`으로 설정합니다.
- `cached_content`: Gemini 전용. 사전에 생성한 캐시 리소스 이름 (예: `"cachedContents/abc123"`). `runner_chat.py`의 `GeminiCacheClient`로 생성하거나, 프로그래밍 방식으로 요청별 자동 캐싱을 사용할 수 있습니다.

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

- `api_key`: `null`이면 환경변수 `ANTHROPIC_API_KEY`를 사용합니다.
- `poll_interval`: 초 단위. 배치 제출 후 `processing_status == "ended"`가 될 때까지 폴링합니다.

### CLI 러너

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

- `cli`: 인수 형식 및 출력 파싱 방식을 결정합니다. `"claude"`, `"codex"`, `"gemini"` 중 하나.
- `cli_bin`: 실행 파일 경로를 직접 지정할 때 사용합니다 (예: `"/usr/local/bin/claude-dev"`). 생략하면 `cli` 값을 그대로 사용합니다.

## 템플릿 (template.json)

```json
{
  "user_template": "다음 질문에 답하세요: {question}",
  "system_template": "당신은 {role}입니다.",
  "json_schema": {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

`system_template`과 `json_schema`는 선택 사항입니다.
템플릿은 Python `str.format_map`을 사용합니다 — 입력 행의 변수가 이름으로 치환됩니다.
템플릿에 존재하지 않는 변수가 있으면 명확한 오류 메시지가 출력됩니다.

### 프롬프트 캐싱: `<cache />`

템플릿에 `<cache />` (또는 `<cache/>`)를 삽입해 캐시 구분점을 지정합니다.
마커 앞의 내용은 cache-control 체크포인트와 함께 전송되고, 마지막 구간은 캐싱되지 않습니다.

```
당신은 전문가입니다. <cache />

다음은 문서 내용입니다:
{document} <cache />

질문: {question}
```

- `system_template`과 `user_template` 모두 지원
- `backend: "anthropic"` 및 `backend: "chat"` (Gemini 전용)에서만 동작
- CLI 백엔드는 마커를 제거하고 일반 텍스트로 평탄화합니다

## 입력 JSONL

한 줄에 변수 딕셔너리 하나. `_id`를 지정하면 커스텀 요청 ID로 사용되고, 없으면 줄 번호(0-indexed)가 사용됩니다.
`--input`에는 디렉토리도 지정 가능합니다 — `.json`/`.jsonl` 파일 전체를 정렬 순서로 로드합니다.

```jsonl
{"_id": "q1", "role": "선생님", "question": "삼투압이란 무엇인가요?"}
{"_id": "q2", "role": "선생님", "question": "광합성이란 무엇인가요?"}
```

### 문서 첨부 (`_documents`)

`_documents` 필드로 요청에 문서를 첨부할 수 있습니다. 각 항목은 Anthropic source 객체여야 합니다:

```jsonl
{"_id": "q1", "question": "요약해주세요.", "_documents": [
  {"type": "plain_text", "data": "문서 내용입니다.", "title": "문서 1"},
  {"type": "plain_text", "data": "다른 문서입니다.", "_cache": true}
]}
```

지원 타입: `plain_text` (`data` 필드) 및 `content` (`content` 필드, DoclingDocument 등 구조화된 블록).
`"_cache": true`를 추가하면 해당 문서 블록에 캐싱이 적용됩니다. `"title"`은 선택 사항입니다.

템플릿에서 `<documents />`의 위치에 따라 `anthropic` 백엔드의 처리 방식이 달라집니다:

- **`user_template`**: 마커 위치에 네이티브 `document` 블록이 user 메시지 `content` 배열에 삽입됩니다. 마커가 없으면 user 텍스트 앞에 `document` 블록으로 자동 삽입됩니다.
- **`system_template`**: 마커 위치에 문서 텍스트가 system 프롬프트 문자열 안에 인라인으로 삽입됩니다.

나머지 백엔드는 어느 템플릿에 마커가 있든 항상 `data` 텍스트를 인라인으로 삽입합니다. 마커가 없으면 user 메시지 앞에 붙습니다.

#### 예시 (`anthropic` 백엔드, `user_template` 마커)

템플릿:
```json
{
  "user_template": "참고 문서:\n\n<documents />\n\n질문: {question}"
}
```

입력:
```jsonl
{"_id": "q1", "question": "주요 결과는 무엇인가요?", "_documents": [
  {"type": "plain_text", "data": "연구 A 내용입니다.", "title": "연구 A", "_cache": true},
  {"type": "plain_text", "data": "연구 B 내용입니다.", "title": "연구 B"}
]}
```

API로 전송되는 user 메시지:
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "참고 문서:\n\n"},
    {"type": "document", "source": {"type": "plain_text", "data": "연구 A 내용입니다."}, "title": "연구 A", "cache_control": {"type": "ephemeral"}},
    {"type": "document", "source": {"type": "plain_text", "data": "연구 B 내용입니다."}, "title": "연구 B"},
    {"type": "text", "text": "\n\n질문: 주요 결과는 무엇인가요?"}
  ]
}
```

연구 A 이후의 캐시 체크포인트 덕분에 동일한 프리픽스를 공유하는 요청들은 해당 문서까지의 KV 캐시를 재사용할 수 있습니다.

## 출력 JSONL

```jsonl
{"id": "q1", "content": "삼투압은 ...", "error": null, "source_file": "/절대/경로/input.jsonl"}
{"id": "q2", "content": null, "error": "HTTP 429: ...", "source_file": "/절대/경로/input.jsonl"}
```

- `content`: 오류 시 `null`
- `error`: 성공 시 `null`
- `id`: 입력 행의 원본 `_id`
- `source_file`: 소스 파일의 절대 경로. 디렉토리 입력 시 각 행의 원본 파일을 추적합니다.

`chat`과 `cli` 백엔드는 각 요청이 완료될 때마다 즉시 결과를 파일에 씁니다.
`anthropic` 백엔드는 배치 전체가 끝난 후 결과를 씁니다.

## 로그

진행 상황은 `INFO` 레벨로 stderr에 출력됩니다:

```
16:42:01 INFO 10 requests → output.jsonl
16:42:03 INFO 1/10 done (errors=0)
16:42:04 INFO 2/10 done (errors=0)
...
16:42:09 INFO 10/10 done (errors=1)
16:42:09 INFO done. errors=1/10
```

`anthropic` 백엔드는 배치 폴링 상태가 먼저 출력됩니다:

```
16:42:01 INFO submitted batch msgbatch_abc123 (10 requests)
16:42:31 INFO batch msgbatch_abc123: processing — processing=8 succeeded=2 errored=0
16:43:01 INFO batch msgbatch_abc123: ended — processing=0 succeeded=9 errored=1
16:43:01 INFO 1/10 done (errors=0)
...
16:43:02 INFO 10/10 done (errors=1)
16:43:02 INFO done. errors=1/10
```

`DEBUG` 레벨로 설정하면 요청별 상세 정보와 Gemini 캐시 hit/miss 이벤트도 확인할 수 있습니다.

## 프로그래밍 방식 사용

모든 runner는 동일한 인터페이스를 구현합니다:

```python
from runner_anthropic import AnthropicBatchRunner
from template import build_requests

runner = AnthropicBatchRunner(model="claude-haiku-4-5")
requests = build_requests(user_template="요약해줘: {text}", variable_list=[...])

# 배치: 완료 순서대로 yield
for resp in runner.stream(requests):
    print(resp.id, resp.content)

# 단건 실행
resp = runner.run(requests[0])
```

### BatchCollector — 모아서 한꺼번에 배치 제출

요청이 하나씩 들어오다가 sentinel 도달 시 한꺼번에 제출하고 싶을 때 사용합니다:

```python
from collector import BatchCollector

collector = BatchCollector(runner, max_size=10_000)
for item in source:
    for resp in collector.add(build_request(item)):  # max_size 도달 시 자동 flush
        process(resp)
# sentinel 도달 — 나머지 flush
for resp in collector.flush():
    process(resp)
```

`add()`는 요청을 디스크 JSONL 스풀 파일에 직렬화하므로 대량 누적 시에도 메모리가 늘어나지 않습니다.
`flush()`는 스풀을 읽어 `runner.stream()`을 호출하고, 응답을 yield한 뒤 스풀을 초기화합니다.

`max_size`는 누적 건수가 한도에 도달하면 자동으로 flush하고 응답을 yield합니다. Anthropic 백엔드의 배치당 10,000건 제한에 맞추려면 `max_size=10_000`을 사용하세요. 제한이 없는 경우 생략합니다.
