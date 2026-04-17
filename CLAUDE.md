# llm_batch_runner — Claude context

## Quick start

```bash
pip install anthropic        # anthropic 백엔드 사용 시
python __main__.py --config config.json --template template.json --input input.jsonl --output out.jsonl
python __main__.py --input inputs/   # --input에 디렉터리 지정 가능 (.json/.jsonl 파일 전체 로드)
python __main__.py ... --force       # 기존 output 무시하고 전체 재실행
```

## Backends

| backend | 설명 | 필수 config 키 |
|---------|------|----------------|
| `chat` | Chat Completions (OpenAI/Ollama/Gemini 등) | `base_url`, `model` |
| `anthropic` | Anthropic Batch API | `model` |
| `cli` | claude / codex / gemini CLI 래퍼 | `cli` |

## Architecture decisions

- `LLMRequest.custom_id` = 전역 enumerate 인덱스(str); 고유성 보장용, 출력에 사용 금지
- `LLMRequest.output_id` = 입력 행의 `_id` (없으면 행 인덱스); 출력 `id` 필드에 사용
- `LLMRequest.source_file` = 절대 경로 (`str(path.resolve())`), 항상 설정됨
- `<cache />` 마커 → 프롬프트 캐시 브레이크포인트; `<documents />` 마커 → 문서 삽입 위치 (여러 개 허용)
- `_documents` 항목: `{"type": "plain_text", "data": "...", "_cache": true, "title": "..."}` 또는 `{"type": "content", "content": [...]}` — 그 외 형식은 WARNING 로그
- `<documents />` 마커 없으면 문서를 user 메시지 앞에 prepend
- `anthropic` 백엔드: native `document` 블록으로 렌더링; 그 외 백엔드: `data` 텍스트 인라인 삽입
- `LLMRequest.user_segments`/`system_segments: list[list[Block]]` — 외부 리스트 = `<documents />`로 분리된 세그먼트, 내부 리스트 = `<cache />`로 분리된 블록

## Runner interface

모든 runner는 `runner_base.BaseRunner` ABC를 상속한다:
- `run(req: LLMRequest) -> LLMResponse` — 단건 실행 (retry 로직 포함)
- `stream(requests: list[LLMRequest]) -> Iterator[LLMResponse]` — 완료 순서대로 yield; `chat`/`cli`는 내부적으로 `self.run()`을 ThreadPoolExecutor로 병렬 실행

`__main__._build_runner()`가 union type을 반환하지만 `BaseRunner` 타입으로 받아도 무방하다.

## BatchCollector (`collector.py`)

요청을 디스크 스풀 파일에 누적하다가 sentinel 도달 시 한꺼번에 배치 제출:

```python
from collector import BatchCollector
collector = BatchCollector(runner, max_size=10_000)
for item in source:
    for resp in collector.add(build_request(item)):  # max_size 도달 시 yield
        process(resp)
for resp in collector.flush():
    process(resp)
```

- `add(req)`: JSONL 스풀 파일에 직렬화; `max_size` 도달 시 자동 flush하고 `LLMResponse`를 yield (평상시엔 빈 iterator)
- `flush()`: 스풀 로드 → `runner.stream()` → yield → 스풀 초기화
- `max_size`: Anthropic 배치 10,000건 제한 등에 사용; 미설정 시 자동 flush 없음
- `len(collector)`: 현재 누적 건수

## Resume / --force

`--force` 없이 output 파일이 이미 존재하면:
1. 성공 행(`error == null`) ID는 skip
2. 오류 행은 output에서 제거 후 재실행
3. `rerun_ids = errored_ids - succeeded_ids` — 재시도 성공한 건은 재실행 방지

## Output JSONL fields

`id` (= output_id), `content`, `error`, `source_file` — `custom_id`는 절대 출력하지 않음

## Features out of scope

- `cache_system_prompt` 옵션 추가 금지 — `<cache />` 마커로 충분
- `--resume-batch-id` 추가 금지

## Key files

| 파일 | 역할 |
|------|------|
| `__main__.py` | CLI 진입점, resume 로직 |
| `runner_base.py` | `BaseRunner` ABC |
| `runner_chat.py` | Chat Completions + Gemini |
| `runner_anthropic.py` | Anthropic Batch API |
| `runner_cli.py` | claude/codex/gemini CLI 래퍼 |
| `template.py` | `<cache />` / `<documents />` 마커 파싱, `build_requests()` |
| `models.py` | `LLMRequest`, `LLMResponse`, `Block`, `DocumentSpec` |
| `collector.py` | `BatchCollector` |
| `utils.py` | `RateLimiter`, `normalize_base_url` |

## Code style

- 코드가 하는 일(WHAT)은 주석 금지; 비자명한 이유(WHY)만 허용
- tqdm/progress bar 금지 — `logger.info`로 진행 상황 출력 (`N/M done (errors=K)`)
- 각 runner는 `BaseRunner` ABC 상속, `run()`에 retry 로직 인라인, `stream()`은 `self.run()` 호출
