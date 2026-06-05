# ADR-003 — LLM Provider Strategy

**Status:** Accepted  
**Date:** 2026-06-04

---

## Context

The project needs an LLM for generation and a VLM (vision-language model) for figure captioning. Key constraints:

1. **No mandatory API keys** — tutorials must be fully runnable offline.
2. **Portfolio credibility** — must demonstrate awareness of cloud providers (OpenAI, Anthropic, Google).
3. **Consumer hardware** — models must run on a MacBook or mid-range GPU (8–16 GB VRAM).
4. **Swappability** — changing provider should not require touching RAG or evaluation code.

Providers evaluated: Ollama (local), Hugging Face Transformers (local), OpenAI API, Anthropic API, Google Gemini API.

## Decision

**Default to Ollama; support all five providers via a provider-agnostic `llm.py` interface.**

Configuration in `src/mrta/core/config.py`:

```python
llm_provider: Literal["ollama", "huggingface", "openai", "anthropic", "google"] = "ollama"
```

Default models:

- LLM: `llama3.2:3b` (Ollama) — 2 GB, runs on CPU
- VLM: `qwen2.5vl:7b` (Ollama, dev) / `llava:7b` (fallback) — vision + text
- Embedding: `nomic-embed-text` (Ollama, dev) / `sentence-transformers/all-MiniLM-L6-v2` (test/default)

Switching to OpenAI: set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` in `.env`. No code changes.

## Consequences

**Positive:**

- Zero-cost, offline-first tutorials.
- Swapping providers is a config change, not a code change.
- Demonstrates multi-provider awareness for portfolio purposes.

**Negative / Tradeoffs:**

- Ollama models are weaker than GPT-4o for complex reasoning; expected and documented.
- Hugging Face path requires more VRAM than Ollama (no quantization by default).
- API providers require internet + billing — documented as optional.

## References

- [Ollama model library](https://ollama.com/library)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- `src/mrta/core/config.py` — provider field and model defaults
