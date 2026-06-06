"""apps/streamlit/app.py — single-file UI for the RAG assistant."""

import httpx
import streamlit as st

# --- page config ----------------------------------------------------------
API = "http://localhost:8000"

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
    if cited_pages:
        st.caption("Cited pages: " + ", ".join(str(p) for p in cited_pages))
    st.caption(f"Latency: {resp['latency_s']:.1f}s")

    with st.expander("Retrieved chunks"):
        for s in resp["sources"]:
            st.markdown(f"**page {s['page']}**  \n_{s['chunk_id']}_")
            st.markdown(f"> {s['preview']}")
            st.divider()
