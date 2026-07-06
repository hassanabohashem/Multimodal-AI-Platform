"""Streamlit demo UI: four tabs mirroring the four gateway endpoints."""
from __future__ import annotations

import httpx
import streamlit as st

import api_client

st.set_page_config(page_title="Multimodal AI Platform", layout="wide")
st.title("Multimodal AI Platform")

tab_cap, tab_vqa, tab_search, tab_docs = st.tabs(["Caption", "VQA", "Search", "Documents"])


def show_error(exc: Exception) -> None:
    """Render gateway problem+json errors readably."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
            st.error(f"{body.get('title', 'error')} — {body.get('detail', '')}")
            return
        except Exception:  # noqa: BLE001
            pass
    st.error(str(exc))


with tab_cap:
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key="cap")
    style = st.radio("Style", ["concise", "detailed"], horizontal=True)
    if up and st.button("Caption it"):
        st.image(up, width=420)
        try:
            res = api_client.caption(up.getvalue(), style)
            st.success(res["caption"])
            st.caption(f"{res['model_version']} · {res['latency_ms']:.0f} ms · {res['request_id']}")
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

with tab_vqa:
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key="vqa")
    question = st.text_input("Question")
    verbose = st.toggle("Conversational answer (base model, no adapter)")
    if up and question and st.button("Ask"):
        st.image(up, width=420)
        try:
            res = api_client.vqa(up.getvalue(), question, verbose)
            st.success(res["answer"])
            st.caption(f"type: {res['answer_type']} · {res['model_version']} · {res['latency_ms']:.0f} ms")
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

with tab_search:
    query = st.text_input("Describe the image you're looking for")
    if query and st.button("Search"):
        try:
            res = api_client.search(query)
            cols = st.columns(4)
            for i, hit in enumerate(res["hits"]):
                with cols[i % 4]:
                    st.image(hit["thumb_uri"] or hit["uri"])
                    st.caption(f"{hit['score']:.3f} — {hit.get('caption') or ''}")
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

with tab_docs:
    up = st.file_uploader("Document", type=["jpg", "jpeg", "png", "pdf"], key="ocr")
    schema = st.text_input("Fields to extract (comma-separated)", "date,total_amount,vendor,invoice_number")
    if up and st.button("Extract"):
        try:
            res = api_client.ocr(up.getvalue(), up.name, schema)
            left, right = st.columns(2)
            with left:
                st.subheader("Extracted fields")
                st.table({f: (v["value"] or "—") for f, v in res["entities"].items()})
                for w in res["warnings"]:
                    st.warning(w)
            with right:
                st.subheader("Document text")
                st.markdown(res["pages"][0]["markdown"][:4000])
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
