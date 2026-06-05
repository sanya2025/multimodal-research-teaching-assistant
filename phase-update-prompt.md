# Phase Update Prompt

Paste this at the start of any notebook-to-production session, replacing `NN` and `<module>`.

---

```
Convert Phase NN (<name>) tutorial notebook to production.

## Before starting
1. Read `production-ready.md` → Phase NN section — confirm what to extract, target files, interfaces
2. Read `notebook-to-production-steps.md` → check where Phase NN-1 left off
3. Read the tutorial notebook (`notebooks/tutorials/...`) — understand the inline code
4. Read the production notebook (`notebooks/production/...`) — see what stubs are already there

## Implement
5. Add any new schemas to `src/mrta/core/schemas.py`
6. Implement `src/mrta/<module>.py` — extract functions from tutorial notebook, add type hints + docstrings
7. Export new public symbols in `src/mrta/__init__.py` (`__all__`)
8. Activate the production notebook imports — replace inline definitions with `from mrta.X import Y`
9. Write tests in `tests/unit/test_<module>.py`
10. Run `MRTA_ENV=test pytest -q` — all tests must pass

## Update documents
11. `production-ready.md` → mark extracted items ✅ done in Phase NN table; update library map line
12. `notebook-to-production-steps.md` → add Phase NN section:
    - "What's extracted" table (tutorial cell → production import)
    - "What's still inline" table (if anything remains)
    - Running notebook cell status table
    - "Functions implemented" table (signature + what it does)
    - Concept note for any non-obvious technique introduced

## Wrap up
13. Update memory (`save to memory`)
14. Suggest git commit commands
```
