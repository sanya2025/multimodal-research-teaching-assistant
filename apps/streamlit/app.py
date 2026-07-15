"""apps/streamlit/app.py — single-file UI for the RAG assistant."""

import os

import httpx
import streamlit as st

# --- page config ----------------------------------------------------------
# In Docker, the API runs in a sibling container reachable via its service
# name. Override with API_URL env var; fall back to localhost for local dev.
API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Research & Teaching Assistant", layout="wide")
st.title("Multimodal AI Research & Teaching Assistant")

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
            st.success(f"Indexed {r.json()['n_pages']} pages, {r.json()['n_chunks']} chunks")
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

# --- main panel: mode + question + ask ------------------------------------
mode = st.radio(
    "Mode",
    ["Default", "Beginner", "Graduate", "Interview", "Quiz me", "Explain figure"],
    horizontal=True,
)
mode_prefix = {
    "Default": "",
    "Beginner": "Explain like I am new to this topic. ",
    "Graduate": "Explain at the level of a graduate student in ML. ",
    "Interview": "Explain as you would in an ML system-design interview. ",
    "Quiz me": "Generate 5 multiple-choice quiz questions (with answers) about: ",
    "Explain figure": "Explain the figure(s) on the relevant page(s) and what they show. ",
}[mode]

# --- explain figure: doc selector + VLM note ------------------------------
selected_source: str | None = None
if mode == "Explain figure":
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

question = st.text_input("Ask a question about the indexed documents", "")
k = st.slider("Top-k retrieved chunks", 1, 10, 5)

if st.button("Ask", type="primary", disabled=not question):
    payload = {"question": mode_prefix + question, "top_k": k}
    with st.spinner("Thinking..."):
        try:
            r = httpx.post(f"{API}/ask", json=payload, timeout=120)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # --- response: answer + cited pages + retrieved chunks expander -------
    st.subheader("Answer")
    st.markdown(resp["answer"])

    cited_pages = sorted({s["page"] for s in resp["sources"]})
    cited_sources = sorted({s["source"] for s in resp["sources"]})
    if cited_pages:
        sources_label = ", ".join(cited_sources)
        st.caption(f"Sources: {sources_label} · pages: " + ", ".join(str(p) for p in cited_pages))
    st.caption(f"Latency: {resp['latency_s']:.1f}s")

    with st.expander("Retrieved chunks"):
        for s in resp["sources"]:
            score = s.get("score")
            score_label = ""
            if score is not None:
                s_rounded = round(score, 3)
                colour = "green" if s_rounded >= 0.7 else "orange" if s_rounded >= 0.4 else "red"
                score_label = f" :{colour}[score {s_rounded:.3f}]"
            st.markdown(f"**{s['source']} · page {s['page']}**{score_label}  \n_{s['chunk_id']}_")
            st.markdown(f"> {s['preview']}")
            st.divider()

    # --- explain figure: call /figures for cited pages --------------------
    if mode == "Explain figure" and selected_source and cited_pages:
        st.subheader("Figure Captions")
        with st.spinner("Extracting and captioning figures..."):
            try:
                fig_r = httpx.post(
                    f"{API}/figures",
                    json={"source": selected_source, "pages": cited_pages},
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
