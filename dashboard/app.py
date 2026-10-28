from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from helpers import (VERDICT_ICON, band, confidence_rows, link_citations, overlap,
                     source_label)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Hybrid RAG Dashboard", page_icon="🔎", layout="wide")
st.markdown("""
<style>
a.cite { font-weight: 600; text-decoration: none; padding: 0 2px; border-radius: 4px;
         background: rgba(80,120,255,0.15); }
span.cite-bad { color: #c0392b; font-weight: 600; }
div.answer { font-size: 1.05rem; line-height: 1.6; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API access
def call_api(method: str, path: str, timeout: int = 300, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except ValueError:
            detail = e.response.text
        return None, f"API error {e.response.status_code}: {detail}"
    except requests.RequestException as e:
        return None, f"Cannot reach the API at {API_URL}: {e}"


def run_query(question: str, mode: str, opts: dict, with_answer: bool) -> dict:
    """Retrieval (ranked chunks with all scores) and, optionally, the generated answer."""
    out = {"mode": mode, "answer": None, "search": None, "error": None}
    search, err = call_api("POST", "/v1/search", json={
        "query": question, "mode": mode, "rerank": opts["rerank"],
        "dense_weight": opts["dense_weight"], "top_n": 10})
    out["search"], out["error"] = search, err
    if with_answer and not err:
        answer, err = call_api("POST", "/v1/ask", json={
            "question": question, "verify": opts["verify"], "retrieval_mode": mode,
            "rerank": opts["rerank"], "dense_weight": opts["dense_weight"]})
        out["answer"], out["error"] = answer, err
    return out


# ------------------------------------------------------------------ rendering
def render_result(res: dict, prefix: str) -> None:
    if res["error"]:
        st.error(res["error"])
        return

    answer = res["answer"]
    if answer:
        if answer["status"] == "insufficient_context":
            st.warning("I couldn't find a reliable answer in the documents.")
            st.markdown(f"**What I found:** {answer['found']}")
            st.markdown(f"**What I couldn't find:** {answer['not_found']}")
            if answer["suggested_documents"]:
                st.markdown("**Worth checking manually:** " + ", ".join(answer["suggested_documents"]))
        else:
            st.subheader("Answer")
            st.markdown(f'<div class="answer">{link_citations(answer["answer"], answer["sources"], prefix)}</div>',
                        unsafe_allow_html=True)
            st.caption(f"{answer['latency_ms'] / 1000:.1f} s · retrieval: {answer['retrieval_mode']}")

        conf = answer.get("confidence")
        st.subheader("Confidence")
        if conf:
            st.metric("Composite", f"{conf['composite']:.2f}", band(conf["composite"]))
            for label, value, hint in confidence_rows(conf):
                st.caption(f"{label}: {value:.2f}  ·  {hint}")
                st.progress(min(max(value, 0.0), 1.0))
        else:
            st.metric("Retrieval confidence", f"{answer['retrieval_confidence']:.2f}",
                      band(answer["retrieval_confidence"]))
            if answer["status"] == "answered":
                st.caption("Turn on 'Verify citations' to see the full breakdown.")

        for w in answer["warnings"]:
            st.warning(w)

        if answer["citation_checks"]:
            with st.expander("Citation checks"):
                for c in answer["citation_checks"]:
                    cites = ", ".join(f"[{n}]" for n in c["citations"]) or "none"
                    st.markdown(f"{VERDICT_ICON.get(c['verdict'], '•')} **{c['verdict']}** · cites {cites}  \n"
                                f"{c['claim']}  \n*{c['reason']}*")

        if answer["status"] == "answered":
            st.subheader("Sources")
            for s in answer["sources"]:
                st.markdown(f'<div id="{prefix}-{s["ref"]}"></div>', unsafe_allow_html=True)
                with st.container(border=True):
                    score = f" · rerank score {s['rerank_score']:.2f}" if s.get("rerank_score") is not None else ""
                    st.markdown(f"**[{s['ref']}] {source_label(s)}**{score}")
                    st.write(s["text"])

    search = res["search"]
    if search:
        st.subheader(f"Retrieved chunks ({search['mode']}, ranked)")
        used = {s["text"] for s in answer["sources"]} if answer and answer["status"] == "answered" else set()
        rows = []
        for rank, h in enumerate(search["results"], 1):
            rows.append({"#": rank, "Document": source_label(h),
                         "Dense rank": h["dense_rank"], "BM25 rank": h["sparse_rank"],
                         "RRF": None if h["rrf_score"] is None else round(h["rrf_score"], 4),
                         "Rerank": None if h["rerank_score"] is None else round(h["rerank_score"], 2),
                         "Used in answer": "✓" if h["text"] in used else "",
                         "Preview": h["text"][:110] + ("…" if len(h["text"]) > 110 else "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"{search['latency_ms']} ms")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    health, health_err = call_api("GET", "/v1/health", timeout=5)
    if health_err:
        st.error(health_err)
    else:
        st.success(f"API online · {health['chunks']} chunks indexed")

    verify = st.checkbox("Verify citations (slower)", value=True)
    rerank = st.checkbox("Cross-encoder reranker", value=True)
    dense_weight = st.slider("Dense weight (hybrid)", 0.0, 1.0, 0.7, 0.05,
                             help="0.7 means 70% dense / 30% BM25 in the rank fusion.")
    compare = st.toggle("Compare hybrid vs dense-only", value=False)
    compare_answers = st.checkbox("Also generate both answers", value=False, disabled=not compare)

    st.divider()
    st.subheader("Indexed documents")
    docs, docs_err = call_api("GET", "/v1/documents", timeout=10)
    if docs_err:
        st.caption(docs_err)
    elif docs:
        st.dataframe(pd.DataFrame([{"File": d["source"], "Chunks": d["chunks"]} for d in docs["documents"]]),
                     hide_index=True)

    st.subheader("Add documents")
    uploads = st.file_uploader("Markdown, text, HTML or PDF", accept_multiple_files=True,
                               type=["md", "markdown", "txt", "html", "htm", "pdf"])
    strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "semantic"])
    if st.button("Index uploaded files", disabled=not uploads):
        files = [("files", (u.name, u.getvalue(), "application/octet-stream")) for u in uploads]
        result, err = call_api("POST", "/v1/ingest", files=files, data={"strategy": strategy})
        if err:
            st.error(err)
        else:
            for f in result["files"]:
                st.success(f"{f['source']}: {f['chunks_indexed']} chunks, "
                           f"{f['duplicates_skipped']} duplicates skipped")
            st.rerun()

# ------------------------------------------------------------------ main page
st.title("🔎 Hybrid RAG over internal docs")
question = st.text_input("Ask a question", placeholder="e.g. How many vacation days do employees get?")
go = st.button("Ask", type="primary", disabled=not question.strip())

if go:
    opts = {"verify": verify, "rerank": rerank, "dense_weight": dense_weight}
    with st.spinner("Retrieving and generating (the first request loads the models)..."):
        if compare:
            st.session_state["last"] = {"question": question, "compare": True,
                                        "hybrid": run_query(question, "hybrid", opts, compare_answers),
                                        "dense": run_query(question, "dense", opts, compare_answers)}
        else:
            st.session_state["last"] = {"question": question, "compare": False,
                                        "hybrid": run_query(question, "hybrid", opts, True)}

last = st.session_state.get("last")
if last:
    st.markdown(f"**Question:** {last['question']}")
    if last["compare"]:
        h, d = last["hybrid"], last["dense"]
        if h["search"] and d["search"]:
            ids_h = [r["id"] for r in h["search"]["results"][:5]]
            ids_d = [r["id"] for r in d["search"]["results"][:5]]
            diff = overlap(ids_h, ids_d)
            st.info(f"Top-5 overlap: {len(diff['both'])} chunks in both · "
                    f"{len(diff['only_first'])} only in hybrid · {len(diff['only_second'])} only in dense-only")
        left, right = st.columns(2)
        with left:
            st.header("Hybrid")
            render_result(h, "hybrid")
        with right:
            st.header("Dense-only")
            render_result(d, "dense")
    else:
        render_result(last["hybrid"], "hybrid")
