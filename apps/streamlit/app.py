"""apps/streamlit/app.py — single-file UI for the RAG assistant."""

import os

import httpx
import streamlit as st

# --- page config ----------------------------------------------------------
# set_page_config must precede every other Streamlit call.
st.set_page_config(page_title="Research & Teaching Assistant", layout="wide")
st.title("Multimodal AI Research & Teaching Assistant")

API = os.getenv("API_URL", "http://localhost:8000")

# Extracted figures are full-resolution page crops — Figure 1 of the Transformer
# paper is 1520x2239 — so rendering at natural size overflows the viewport and
# stretching to a wide container distorts tall diagrams. A bounded width keeps
# the aspect ratio and fits the expander.
FIGURE_DISPLAY_WIDTH = 520


@st.cache_data(show_spinner=False)
def fetch_figure_image(source: str, page: int, figure_index: int | None) -> bytes | None:
    """Return the PNG bytes for one cited figure, or None if it has no raster asset.

    Cached because the same figure is commonly cited across consecutive
    questions, and because re-fetching on every Streamlit rerun would make the
    expander flicker.
    """
    try:
        r = httpx.get(
            f"{API}/figures/image",
            params={
                "source": source,
                "page": page,
                "figure_index": figure_index if figure_index is not None else 1,
            },
            timeout=30,
        )
    except Exception:
        return None
    if r.status_code != 200:
        return None
    return r.content


# --- sidebar: upload + doc list -------------------------------------------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Index this PDF"):
        with st.spinner("Parsing and indexing..."):
            r = httpx.post(
                f"{API}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=600,
            )
        if r.status_code == 200:
            resp = r.json()
            if resp.get("already_indexed"):
                st.info(f"{resp['source']} is already indexed ({resp['n_chunks']} chunks).")
            else:
                st.success(f"Indexed {resp['n_pages']} pages, {resp['n_chunks']} chunks")
        else:
            st.error(f"Upload failed: {r.text}")

    st.divider()
    try:
        docs = httpx.get(f"{API}/documents", timeout=5).json()
    except Exception:
        docs = []
        st.warning("Backend not reachable. Start it with: uvicorn apps.api.main:app --reload")
    for d in docs:
        st.write(f"- {d['source']} ({d['n_pages']}p, {d['n_chunks']}c)")

# --- main panel: retrieval mode + question --------------------------------
retrieval_mode = st.radio(
    "Retrieval mode",
    ["Text RAG", "Multimodal RAG"],
    horizontal=True,
    help="Multimodal RAG sends retrieved figures to the vision model alongside text.",
)

# text-only modes: prompt-prefix based
TEXT_MODE_PREFIXES: dict[str, str] = {
    "Default": "",
    "Beginner": "Explain like I am new to this topic. ",
    "Graduate": "Explain at the level of a graduate student in ML. ",
    "Interview": "Explain as you would in an ML system-design interview. ",
    "Quiz me": "Generate 5 multiple-choice quiz questions (with answers) about: ",
    "Explain figure": "Explain the figure(s) on the relevant page(s) and what they show. ",
}

TEACHING_MODES: dict[str, str | None] = {
    "Direct answer": None,
    "Explain": "explain",
    "Socratic tutor": "socratic",
    "Quiz": "quiz",
    "Compare": "compare",
    "Visual evidence": "visual_evidence",
}

selected_source: str | None = None
mode_prefix = ""
teaching_mode_key: str | None = None

if retrieval_mode == "Text RAG":
    text_mode = st.radio(
        "Mode",
        list(TEXT_MODE_PREFIXES.keys()),
        horizontal=True,
    )
    mode_prefix = TEXT_MODE_PREFIXES[text_mode]

    if text_mode == "Explain figure":
        if docs:
            source_options = [d["source"] for d in docs]
            selected_source = st.selectbox(
                "Document to extract figures from",
                source_options,
                help="Figures will be extracted from the cited pages in this document.",
            )
            st.info(
                "After retrieval, figures on cited pages are extracted and captioned by the "
                "vision model (`ollama_vlm_model`). If the model is not installed the text "
                "answer is shown alone.\n\n"
                "Install the vision model:\n```\nollama pull qwen2.5vl:7b\n```"
            )
        else:
            st.warning("No documents indexed yet. Upload a PDF first.")

else:  # Multimodal RAG
    teaching_label = st.selectbox(
        "Teaching mode",
        list(TEACHING_MODES.keys()),
        help=(
            "Explain: pedagogical explanation · Socratic: guiding questions · "
            "Quiz: generate questions · Compare: compare figures/claims · "
            "Visual evidence: explain what each figure shows"
        ),
    )
    teaching_mode_key = TEACHING_MODES[teaching_label]
    st.info(
        "Multimodal RAG retrieves text chunks and figures, then sends both to the vision "
        "model. Requires Ollama with a vision model:\n```\nollama pull qwen2.5vl:latest\n```"
    )

question = st.text_input("Ask a question about the indexed documents", "")
k = st.slider("Top-k retrieved chunks", 1, 10, 5)

if st.button("Ask", type="primary", disabled=not question):
    # build payload
    if retrieval_mode == "Text RAG":
        payload: dict = {"question": mode_prefix + question, "top_k": k, "retrieval_mode": "text"}
        if text_mode == "Explain figure" and selected_source:
            payload["source"] = selected_source
    else:
        payload = {
            "question": question,
            "top_k": k,
            "retrieval_mode": "multimodal",
        }
        if teaching_mode_key:
            payload["teaching_mode"] = teaching_mode_key

    with st.spinner("Thinking..."):
        try:
            r = httpx.post(f"{API}/ask", json=payload, timeout=180)
            r.raise_for_status()
            resp = r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                st.error(
                    "Multimodal retriever not available on the server. "
                    "Make sure `mrta-rag[multimodal]` is installed and the server restarted."
                )
            else:
                st.error(f"Error {e.response.status_code}: {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # --- answer -----------------------------------------------------------
    st.subheader("Answer")
    st.markdown(resp["answer"])

    pages_by_source: dict[str, list[int]] = {}
    for s in resp["sources"]:
        pages_by_source.setdefault(s["source"], [])
        if s["page"] not in pages_by_source[s["source"]]:
            pages_by_source[s["source"]].append(s["page"])
    if pages_by_source:
        parts = [
            f"{src} · pages: {', '.join(str(p) for p in sorted(pages))}"
            for src, pages in sorted(pages_by_source.items())
        ]
        st.caption("Text sources: " + " | ".join(parts))

    resp_mode = resp.get("retrieval_mode", "text")
    st.caption(f"Latency: {resp['latency_s']:.1f}s · mode: {resp_mode}")

    # --- retrieved text chunks expander -----------------------------------
    if resp["sources"]:
        with st.expander("Retrieved text chunks"):
            for s in resp["sources"]:
                score = s.get("score")
                score_label = ""
                if score is not None:
                    s_rounded = round(score, 3)
                    colour = (
                        "green" if s_rounded >= 0.7 else "orange" if s_rounded >= 0.4 else "red"
                    )
                    score_label = f" :{colour}[score {s_rounded:.3f}]"
                chunk_num = s["chunk_id"].rsplit("_c", 1)[-1] if "_c" in s["chunk_id"] else "0"
                st.markdown(
                    f"**{s['source']} · page {s['page']} · chunk {chunk_num}**{score_label}"
                )
                if s.get("preview"):
                    st.markdown(f"> {s['preview']}")
                st.divider()

    # --- visual evidence expander (multimodal path) -----------------------
    visual_sources = resp.get("visual_sources", [])
    if visual_sources:
        with st.expander("Visual evidence", expanded=True):
            for vs in visual_sources:
                label = vs["label"]
                src = vs["source"]
                page = vs["page"]
                fig_idx = vs.get("figure_index")

                fig_title = f"{label} — {src} · page {page}"
                # `is not None`, not truthiness: figure_index 0 is a real index.
                if fig_idx is not None:
                    fig_title += f" · figure {fig_idx}"
                st.markdown(f"**{fig_title}**")

                # Fetch the PNG through the API rather than pointing the browser
                # at it: this process can always reach the backend, whereas the
                # browser may not when the two run in separate containers.
                image_bytes = fetch_figure_image(src, page, fig_idx)
                if image_bytes is not None:
                    st.image(image_bytes, width=FIGURE_DISPLAY_WIDTH)
                else:
                    st.caption(
                        "Figure image unavailable — the page may contain only vector "
                        "graphics, which the raster extractor does not capture."
                    )

                # The answer already carries the caption the retriever scored this
                # figure on. Showing that, rather than re-captioning via /figures,
                # keeps the displayed text identical to the evidence the model saw
                # and avoids a VLM call on every question.
                caption = vs.get("caption")
                if caption:
                    st.markdown(f"_{caption}_")
                st.divider()

    # --- explain figure: call /figures for cited pages (text RAG path) ---
    source_pages = pages_by_source.get(selected_source, []) if selected_source else []
    if retrieval_mode == "Text RAG" and selected_source and source_pages:
        st.subheader("Figure Captions")
        with st.spinner("Extracting and captioning figures..."):
            try:
                fig_r = httpx.post(
                    f"{API}/figures",
                    json={"source": selected_source, "pages": source_pages},
                    timeout=300,
                )
                fig_r.raise_for_status()
                fig_data = fig_r.json()
            except Exception as e:
                st.warning(f"Figure extraction unavailable: {e}")
                fig_data = None

        if fig_data is not None:
            if not fig_data["vlm_available"]:
                st.info(
                    f"Vision model `{fig_data['model']}` is not installed.\n\n"
                    f"To enable pixel-level figure analysis:\n"
                    f"```\nollama pull {fig_data['model']}\n```"
                )
            elif not fig_data["figures"]:
                st.info(
                    "No embedded raster figures found on the cited pages. "
                    "The document may use vector graphics, which are not captured "
                    "by the current extractor."
                )
            else:
                for fig in fig_data["figures"]:
                    with st.expander(
                        f"Page {fig['page']}, Figure {fig['figure_index']}", expanded=True
                    ):
                        st.markdown(fig["caption"])
                st.caption(f"Figure captioning latency: {fig_data['latency_s']:.2f}s")
