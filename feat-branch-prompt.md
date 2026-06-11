# Feature Branch Prompt

Paste this at the start of any `feat/` branch session, replacing `<feature-name>` and the
placeholders below.

---

## feat/<feature-name> — session start

I am working on branch `feat/<feature-name>`.

Read these files first before proposing or writing anything:

1. `production-ready.md` — bottom section `## feature/<feature-name>` (if it exists) gives the
   agreed Understanding, design, and step-by-step plan.
2. `CHANGELOG.md` — top entry shows the expected changed-files table and test table for this
   feature.
3. Every source file listed under **Relevant files** in the `production-ready.md` entry.
4. Every test file listed in the CHANGELOG entry.

---

## Which markdown files to update — and when

### Before implementation (plan phase)

| File | What to add | Where |
|------|-------------|-------|
| `production-ready.md` | New `## feature/<feature-name>` section | Append at the bottom |
| `CHANGELOG.md` | New `## [feat/<feature-name>]` entry | Prepend at the top (above all existing entries) |

#### `production-ready.md` section template

```markdown
## feature/<feature-name>

### Understanding

**Current implementation:**

- <what exists today — stubs, bare raises, missing wrapping, etc.>

**Relevant files:**

- `<path>` — <what to do>

**Risks:**

- <wrapping too broadly, breaking changes, silent fallbacks to leave alone, etc.>

### Design

<hierarchy diagram, schema, interface, or key decisions — depends on feature type>

### Steps

**1 — <first atomic step>**

<code snippet if needed>

**2 — <second step>**

...

### Expected outcome

- <what passes after implementation>
- <what is importable / callable>
- <test count>
```

#### `CHANGELOG.md` entry template

```markdown
## [feat/<feature-name>] — <Short Title> — <YYYY-MM-DD>

**Commit:** `TBD`

<One-paragraph summary: what changed and why.>

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `<path>` | Created / Updated | <what changed> |

### Tests created — `tests/unit/test_<feature>.py` (<N> tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `Test<X>` | `test_<name>` | <what it asserts> |
```

---

### After implementation (close-out phase)

| File | What to update | Detail |
|------|---------------|--------|
| `CHANGELOG.md` | Replace `TBD` commit hash | Run `git log --oneline -1` after merge to get hash |
| `production-ready.md` library map | Change `stub` → `✅ done` | For every module that was a stub and is now complete |
| `notebook-to-production-steps.md` | Add a row to the session log | Only if this feature touched a notebook |
| `docs/adr/` | Create a new ADR | Only if the feature introduced a significant architectural decision (new dep, new pattern, swap of component) |

---

## Rules

- Never write to `production-ready.md` or `CHANGELOG.md` without reading them first.
- Append `production-ready.md` entries at the **bottom**; prepend `CHANGELOG.md` entries at
  the **top**.
- Keep `production-ready.md` entries focused on design intent — not session narrative.
- Keep `CHANGELOG.md` entries factual — changed files and tests only, no opinions.
- Do not update `README.md` unless the public API or install instructions changed.
- Do not create an ADR unless a non-obvious architectural decision was made.

---

---

## chore/docker-healthchecks — session start

I am working on branch `chore/docker-healthchecks`.

Read these files first before proposing or writing anything:

1. `production-ready.md` — bottom section `## chore/docker-healthchecks` gives the agreed
   Understanding, design, and step-by-step plan.
2. `CHANGELOG.md` — top entry shows the expected changed-files table for this chore.
3. `docker/docker-compose.yml` — current service definitions and bare `depends_on` blocks.
4. `docker/Dockerfile.api` — current API image definition.
5. `docker/Dockerfile.streamlit` — current Streamlit image definition.

---

## Which markdown files to update — and when

### Before implementation (plan phase)

| File | What to add | Where |
|------|-------------|-------|
| `production-ready.md` | New `## chore/docker-healthchecks` section | Append at the bottom |
| `CHANGELOG.md` | New `## [chore/docker-healthchecks]` entry | Prepend at the top |

#### `production-ready.md` section to add

```markdown
## chore/docker-healthchecks

### Understanding

**Current implementation:**

- `docker-compose.yml` uses bare `depends_on` — waits for a container to *start*, not for
  the service inside to be *ready*. Ollama can take 10–30 s to load a model; the API starts
  immediately and fails its first embed/chat call if Ollama is not yet up.
- No `HEALTHCHECK` instruction in `Dockerfile.api` or `Dockerfile.streamlit`.
- No `healthcheck` blocks in `docker-compose.yml`.
- Result: `docker compose up` in local dev is unreliable — API often starts before Ollama,
  and Streamlit sometimes starts before the API `/health` route is live.

**Relevant files:**

- `docker/docker-compose.yml` — add healthcheck blocks; upgrade depends_on to condition-based
- `docker/Dockerfile.api` — add HEALTHCHECK instruction
- `docker/Dockerfile.streamlit` — add HEALTHCHECK instruction

**Dependencies:**

- `python:3.11-slim` does not include `curl`. Use a Python one-liner for HEALTHCHECK to
  avoid adding a new system package:
  `python -c "import urllib.request; urllib.request.urlopen('http://localhost:PORT/PATH')"`
- Ollama image (`ollama/ollama:latest`) does include curl — use it for the Ollama healthcheck.
- Streamlit's built-in health endpoint is `/_stcore/health`, not `/health`.
- Ollama's health endpoint is `GET /api/tags` (returns model list; 200 when ready).

**Risks:**

- `start-period` must be generous for Ollama (model loading can exceed 30 s on first pull).
  Use `--start-period=60s` for Ollama, `--start-period=30s` for API and Streamlit.
- `condition: service_healthy` requires Compose v2 — already satisfied (no `version:` key
  in the file; Docker Compose v2 is the default since Docker Desktop 4.x).
- Do not set `interval` too short — 15 s is safe for local dev; CI uses its own smoke test.

### Design

```text
ollama   (healthcheck: GET /api/tags → 200)
  └── api        (depends_on: ollama condition: service_healthy)
                 (healthcheck: GET /health → 200)
        └── streamlit  (depends_on: api condition: service_healthy)
                       (healthcheck: GET /_stcore/health → 200)
```

HEALTHCHECK timing per service:

| Service | interval | timeout | start-period | retries |
|---------|----------|---------|--------------|---------|
| ollama (compose) | 15s | 5s | 60s | 5 |
| api (Dockerfile + compose) | 15s | 5s | 30s | 3 |
| streamlit (Dockerfile + compose) | 15s | 5s | 30s | 3 |

### Steps

**1 — `docker/Dockerfile.api`** — add HEALTHCHECK after EXPOSE:

```dockerfile
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

**2 — `docker/Dockerfile.streamlit`** — add HEALTHCHECK after EXPOSE:

```dockerfile
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
```

**3 — `docker/docker-compose.yml`** — add `healthcheck` block to each service and upgrade
`depends_on` to condition-based:

```yaml
  ollama:
    image: ollama/ollama:latest
    ...
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 15s
      timeout: 5s
      start_period: 60s
      retries: 5

  api:
    ...
    depends_on:
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 15s
      timeout: 5s
      start_period: 30s
      retries: 3

  streamlit:
    ...
    depends_on:
      api:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]
      interval: 15s
      timeout: 5s
      start_period: 30s
      retries: 3
```

### Expected outcome

- `docker compose up` starts services in dependency order and waits for each to pass its
  healthcheck before starting the next.
- `docker compose ps` shows `(healthy)` for all three services once fully up.
- API never attempts an Ollama call before Ollama is ready.
- No new tests — this is pure Docker configuration.

```

#### `CHANGELOG.md` entry to add

```markdown
## [chore/docker-healthchecks] — Docker Healthchecks & Startup Ordering — 2026-06-11

**Commit:** `TBD`

Adds `HEALTHCHECK` instructions to both Dockerfiles and `healthcheck` blocks to
`docker-compose.yml`. Upgrades bare `depends_on` to condition-based (`service_healthy`)
so the API waits for Ollama to be ready before starting, and Streamlit waits for the
API. Fixes unreliable `docker compose up` in local development.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `docker/Dockerfile.api` | Updated | `HEALTHCHECK` added — polls `GET /health` via Python urllib |
| `docker/Dockerfile.streamlit` | Updated | `HEALTHCHECK` added — polls `GET /_stcore/health` via Python urllib |
| `docker/docker-compose.yml` | Updated | `healthcheck` blocks on all 3 services; `depends_on` upgraded to `condition: service_healthy` |

### No new test files

Pure Docker/Compose configuration — no library or API code changed.
```

---

### After implementation (close-out phase)

| File | What to update | Detail |
|------|---------------|--------|
| `CHANGELOG.md` | Replace `TBD` with merge commit hash | `git log --oneline -1` after merge |
| `production-ready.md` | Replace "Expected outcome" with "Actual outcome (shipped)" | Note any timing tweaks made during testing |

---

## Rules

- Read `docker-compose.yml` and both Dockerfiles before writing anything.
- Do not change any `src/mrta/` or `apps/` code — only Docker files.
- Do not create an ADR — this is configuration, not an architectural decision.
- Test locally with `docker compose up --build` and confirm `docker compose ps` shows `(healthy)` before marking done.
