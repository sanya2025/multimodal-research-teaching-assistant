
# SHA-1
SHA-1 is a hashing algorithm. You feed it any sequence of bytes and it produces a fixed-length 40-character hex string. The same input always produces the same output; any change to the input produces a completely different output.

Why it's used for _doc_id
The problem: you need a stable, unique identifier for each PDF that:

doesn't depend on the filename (user might rename the file)
doesn't depend on when it was uploaded
is the same every time you load the same file
The solution: hash the file's raw bytes.


def _doc_id(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}_{h}"
Step by step:

path.read_bytes() — reads the entire PDF as raw bytes
hashlib.sha1(...) — runs SHA-1 on those bytes
.hexdigest() — converts the result to a 40-char hex string like "a3f2c91b8e..."
[:10] — takes the first 10 characters (enough to be unique in practice)
f"{path.stem}_{h}" — combines the filename stem with the hash, e.g. "attention_is_all_you_need_a3f2c91b8e"
What this means in practice

attention_is_all_you_need.pdf  →  attention_is_all_you_need_a3f2c91b
attention_is_all_you_need.pdf  →  attention_is_all_you_need_a3f2c91b  (same file, same ID)
p2017_AttentionIsAllYouNeed.pdf → p2017_AttentionIsAllYouNeed_a3f2c91b  (same content, different stem — different name but same hash suffix)
attention_is_all_you_need_v2.pdf → attention_is_all_you_need_v2_7d4e110f  (different content = different hash)
Why this matters for RAG
Every PageRecord stores its doc_id. When the LLM cites page 7, the system looks up doc_id to find the source file. If the ID changes between runs, citations break — you'd lose the link between a retrieved chunk and its source document.

SHA-1 of the file content guarantees the ID is stable as long as the file doesn't change.
# `llama3.2:3b`

`llama3.2:3b` is a **small, text-only large language model** from Meta's Llama 3.2 family. The `3b` means it has approximately **3 billion parameters**. It is designed to run efficiently on laptops, desktops, and edge devices while still providing good instruction following, summarization, and chat capabilities. ([Hugging Face][1])

## Breaking down the name

* **Llama 3.2** = Meta's model family released in September 2024.
* **3B** = 3 billion parameters.
* In Ollama, `llama3.2:3b` usually refers to the **instruction-tuned chat model**, optimized for conversation and following user instructions. ([Ollama][2])

### Where it fits in the Llama family

| Model                | Parameters | Type          | Typical Use                          |                                  |
| -------------------- | ---------: | ------------- | ------------------------------------ | -------------------------------- |
| Llama 3.2 1B         |         1B | Text          | Very lightweight local apps          |                                  |
| **Llama 3.2 3B**     |     **3B** | Text          | Chat, RAG, agents, coding assistance |                                  |
| Llama 3.1 8B         |         8B | Text          | Stronger reasoning and coding        |                                  |
| Llama 3.2 11B Vision |        11B | Vision + Text | Image understanding                  |                                  |
| Llama 3.2 90B Vision |        90B | Vision + Text | Advanced multimodal tasks            | ([Amazon Web Services, Inc.][3]) |

### Why people like it

For local AI projects, the 3B model is attractive because:

* Fast inference
* Small memory footprint
* Runs comfortably on consumer hardware
* Supports long contexts (up to 128K tokens)
* Good enough for many RAG and agent workflows ([Meta AI][4])

### On Mac Studio (64 GB RAM)

It will run extremely comfortably.

Typical memory usage:

* Q4 quantized model: ~2–3 GB RAM
* Q8 quantized model: ~4–5 GB RAM
* Full precision: much larger but still manageable on MacStudio machine. ([Medium][5])

### Is it good enough for other learning projects?

For other planned projects:

| Project            | Llama 3.2 3B                     |
| ------------------ | -------------------------------- |
| Learning LangGraph | ✅ Excellent                     |
| Learning RAG       | ✅ Excellent                     |
| AI Tutor           | ✅ Good                          |
| Agent workflows    | ✅ Good                          |
| Research Assistant | ⚠️ OK for prototype              |
| Complex reasoning  | ⚠️ Limited                       |
| Portfolio project  | ⚠️ Use larger model eventually   |

For **ResearchClaw** and **Multimodal AI Research Assistant**, I would use:

* **Llama 3.2 3B** for local experimentation and development
* **Qwen3 8B** or **Llama 3.1 8B** for stronger reasoning
* **Qwen2.5-VL 7B** for vision tasks
* Cloud models (Claude, Gemini, GPT) when evaluating state-of-the-art performance

### Check if you already have it installed

In Terminal:

```bash
ollama list
```

Look for something like:

```text
NAME
llama3.2:3b
qwen2.5vl:7b
nomic-embed-text
```

To run it:

```bash
ollama run llama3.2:3b
```

For the multimodal projects it is better to use **Qwen2.5-VL:7B as a default vision model** and **Qwen3:8B or Llama 3.1:8B as a default reasoning model**, with `llama3.2:3b` serving as a lightweight development and testing model.

[1]: https://huggingface.co/meta-llama/Llama-3.2-3B?utm_source=chatgpt.com "meta-llama/Llama-3.2-3B"
[2]: https://ollama.com/library/llama3.2%3A3b?utm_source=chatgpt.com "llama3.2:3b"
[3]: https://aws.amazon.com/blogs/aws/introducing-llama-3-2-models-from-meta-in-amazon-bedrock-a-new-generation-of-multimodal-vision-and-lightweight-models/?utm_source=chatgpt.com "Introducing Llama 3.2 models from Meta in Amazon Bedrock"
[4]: https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/?utm_source=chatgpt.com "Llama 3.2: Revolutionizing edge AI and vision with open ..."
[5]: https://medium.com/pythoneers/llama-3-2-1b-and-3b-small-but-mighty-23648ca7a431?utm_source=chatgpt.com "LLama 3.2 1B and 3B: small but mighty!"
