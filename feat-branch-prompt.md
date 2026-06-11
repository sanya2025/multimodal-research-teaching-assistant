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
