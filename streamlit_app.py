"""Streamlit dashboard (Phase 5) — inline-cited clinical Q&A over the FastAPI service.

Run the API first, then the dashboard:
    mise run api
    mise run dashboard          # -> http://localhost:8501

The dashboard talks to the API over HTTP (decoupled, per the architecture), so
it shows exactly what a real client would get back. Set API_URL to point
elsewhere.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

from app.schemas import AskResponse
from app.web.render import (
    answer_to_html,
    format_codes,
    number_citations,
    ordered_sources,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 120.0

EXAMPLES = [
    "Is the patient being treated with insulin?",
    "Does this patient have prediabetes?",
    "What is the patient's blood type?",
    "What is the capital of France?",
]

st.set_page_config(page_title="Clinical Q&A (synthetic FHIR)", page_icon="🩺", layout="centered")


@st.cache_data(ttl=30)
def fetch_patients() -> list[str]:
    r = httpx.get(f"{API_URL}/patients", timeout=10.0)
    r.raise_for_status()
    return r.json().get("patients", [])


def ask(patient_id: str, question: str) -> AskResponse:
    r = httpx.post(
        f"{API_URL}/ask",
        json={"patient_id": patient_id, "question": question},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return AskResponse.model_validate(r.json())


# --- Header ------------------------------------------------------------------
st.title("🩺 Clinical Q&A over synthetic FHIR")
st.caption(
    "Every sentence is traced to the FHIR resource it came from. "
    "**Synthetic data only — not clinically validated, not medical advice.**"
)

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.subheader("Setup")
    st.write(f"API: `{API_URL}`")
    try:
        patients = fetch_patients()
    except Exception as exc:  # noqa: BLE001 — surface any connection issue to the user
        st.error(
            f"Couldn't reach the API at {API_URL}.\n\n"
            f"Start it with `mise run api`, then reload.\n\n`{exc}`"
        )
        st.stop()

    if not patients:
        st.warning("No patients ingested. Run `mise run ingest data/sample/patient_bundle.json`.")
        st.stop()

    patient_id = st.selectbox("Patient", patients)
    st.markdown("**Try:**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

# --- Question ----------------------------------------------------------------
question = st.text_input(
    "Ask a question about this patient",
    key="question",
    placeholder="e.g. What medications is this patient on?",
)
go = st.button("Ask", type="primary")

if go and question.strip():
    with st.spinner("Retrieving evidence and grounding the answer…"):
        try:
            resp = ask(patient_id, question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
            st.stop()

    if resp.abstained:
        # Abstention is a feature, not a failure — present it as such.
        st.warning(f"**The system abstained.** {resp.abstention_reason or ''}")
        st.caption(
            "No sufficiently relevant, citable evidence was found in this patient's "
            "record — the system refuses rather than guessing."
        )
    else:
        numbering = number_citations(resp.sentences)
        st.markdown("#### Answer")
        body = answer_to_html(resp, numbering)
        st.markdown(
            f"<div style='font-size:1.05rem;line-height:1.7'>{body}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Sources")
        for n, src in ordered_sources(resp, numbering):
            codes = format_codes(src)
            header = f"**[{n}] {src.title}**  ·  `{src.source_id}`"
            st.markdown(f"<a name='src-{n}'></a>", unsafe_allow_html=True)
            with st.expander(header, expanded=True):
                st.write(src.text)
                meta = []
                if src.date:
                    meta.append(f"📅 {src.date}")
                if codes:
                    meta.append(f"🏷️ {codes}")
                if meta:
                    st.caption("  ·  ".join(meta))

    with st.expander("Raw response (JSON)"):
        st.json(resp.model_dump())
