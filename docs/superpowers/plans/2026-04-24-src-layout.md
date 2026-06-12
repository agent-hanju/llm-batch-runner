# src layout 패키지화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 루트에 flat하게 놓인 `.py` 파일들을 `src/llm_batch_runner/` 패키지로 이동하고 `pyproject.toml`로 관리 가능한 구조로 만든다.

**Architecture:** `src/` 레이아웃을 채택해 `src/llm_batch_runner/` 아래 모든 소스를 배치. 내부 import는 relative import(`from .models import ...`)로 전환. `pyproject.toml`의 `[project.scripts]`로 `llm-batch-runner` CLI 엔트리포인트 등록.

**Tech Stack:** Python ≥ 3.10, setuptools(build backend), anthropic(optional extra)

---

### Task 1: src 패키지 디렉터리 생성

**Files:**
- Create: `src/llm_batch_runner/__init__.py`

- [ ] `src/llm_batch_runner/` 디렉터리 생성

```bash
mkdir -p src/llm_batch_runner
```

- [ ] 빈 `__init__.py` 생성

```python
# src/llm_batch_runner/__init__.py
```

---

### Task 2: 소스 파일 이동 (의존성 없는 파일 먼저)

**Files:**
- Move: `models.py` → `src/llm_batch_runner/models.py`
- Move: `utils.py` → `src/llm_batch_runner/utils.py`

두 파일은 내부 의존성이 없으므로 import 수정 불필요.

- [ ] `models.py` 이동

```bash
mv models.py src/llm_batch_runner/models.py
mv utils.py src/llm_batch_runner/utils.py
```

---

### Task 3: runner_base / template 이동 + import 수정

**Files:**
- Move: `runner_base.py` → `src/llm_batch_runner/runner_base.py`
- Move: `template.py` → `src/llm_batch_runner/template.py`

`runner_base.py` 수정:
```python
# 변경 전
from models import LLMRequest, LLMResponse
# 변경 후
from .models import LLMRequest, LLMResponse
```

`template.py` 수정:
```python
# 변경 전
from models import Block, DocumentSpec, LLMRequest
# 변경 후
from .models import Block, DocumentSpec, LLMRequest
```

---

### Task 4: runner_chat / runner_cli / runner_anthropic 이동 + import 수정

**Files:**
- Move: `runner_chat.py` → `src/llm_batch_runner/runner_chat.py`
- Move: `runner_cli.py` → `src/llm_batch_runner/runner_cli.py`
- Move: `runner_anthropic.py` → `src/llm_batch_runner/runner_anthropic.py`

각 파일의 내부 import를 relative로 변경:

`runner_chat.py`:
```python
from .models import Block, LLMRequest, LLMResponse
from .runner_base import BaseRunner
from .utils import RateLimiter, normalize_base_url
```

`runner_cli.py`:
```python
from .models import LLMRequest, LLMResponse
from .runner_base import BaseRunner
from .utils import RateLimiter
```

`runner_anthropic.py`:
```python
from .models import Block, DocumentSpec, LLMRequest, LLMResponse
from .runner_base import BaseRunner
```

---

### Task 5: collector / __main__ 이동 + import 수정

**Files:**
- Move: `collector.py` → `src/llm_batch_runner/collector.py`
- Move: `__main__.py` → `src/llm_batch_runner/__main__.py`

`collector.py`:
```python
from .models import Block, DocumentSpec, LLMRequest, LLMResponse
from .runner_base import BaseRunner
```

`__main__.py`:
```python
from .models import LLMRequest, LLMResponse
from .template import build_requests
from .runner_chat import ChatCompletionsRunner
from .runner_anthropic import AnthropicBatchRunner
from .runner_cli import CliRunner
```

---

### Task 6: pyproject.toml 생성

**Files:**
- Create: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "llm-batch-runner"
version = "0.1.0"
description = "LLM batch runner supporting chat, Anthropic Batch API, and CLI backends"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]

[project.scripts]
llm-batch-runner = "llm_batch_runner.__main__:_main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.ruff]
target-version = "py310"
line-length = 100
```

---

### Task 7: 루트 정리

**Files:**
- Delete: `run.py` (console_scripts로 대체)
- Delete: `requirements.txt` (pyproject.toml로 통합)
- Update: `.gitignore`

`.gitignore`에 추가:
```
.mypy_cache/
*.egg-info/
```

---

### Task 8: 설치 및 동작 확인

- [ ] 개발 모드 설치

```bash
pip install -e .
```

- [ ] CLI 동작 확인

```bash
llm-batch-runner --help
```

- [ ] anthropic 선택 설치 확인

```bash
pip install -e ".[anthropic]"
```

- [ ] 커밋

```bash
git add -A
git commit -m "refactor: src layout 패키지화 + pyproject.toml 도입"
```
