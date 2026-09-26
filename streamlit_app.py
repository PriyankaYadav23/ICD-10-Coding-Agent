# streamlit_app.py
# Public demo (Streamlit Community Cloud): runs the agent INSIDE the app, no FastAPI server needed.
# Local run:  streamlit run streamlit_app.py
# Needs GROQ_API_KEY in .env (local) or in the app's Secrets (Streamlit Cloud).

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ICD-10 Coding Agent", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner="Loading models (first start takes ~1-2 minutes)...")
def load_agent():
    from agent import analyze_note   # importing agent loads BioBERT, bge and the data once
    return analyze_note


SAMPLE_NOTES = {
    "(type your own)": "",
    "Colon: diverticulosis with bleeding": "Flexible sigmoidoscopy. Sigmoid and left colon diverticulosis. Rectal bleeding.",
    "Knee: meniscus tear (laterality)": "Right knee pain for 3 weeks. MRI shows a tear of the medial meniscus of the right knee.",
    "Ankle sprain (laterality)": "Twisted left ankle today while running. Exam: sprain of the left ankle. Initial visit.",
    "Type 2 diabetes": "Type 2 diabetes mellitus without complications. Continue metformin.",
    "Dental caries": "Dental caries on the smooth surface of tooth 19, penetrating into dentin.",
    "Not a clinical note": "hello how are you",
}

st.title("🩺 ICD-10 Coding Agent")
st.caption("BioBERT classifier + ICD-10-CM guideline search (RAG) + code search, orchestrated by an LLM agent. "
           "Every suggested code is verified in Python against the official FY2026 code list.")
st.info("Portfolio demo, not for real clinical or billing use. Runs on free tiers, so it may be slow or "
        "temporarily rate-limited. Code: github.com/PriyankaYadav23/ICD-10-Coding-Agent")

analyze_note = load_agent()

# Safe key check (never shows the key itself): helps debug "Invalid API Key" on the cloud
import os
_raw = os.getenv("GROQ_API_KEY") or ""
with st.sidebar:
    st.caption("Diagnostics")
    st.write("Groq key found:", bool(_raw))
    st.write("Starts with gsk_:", _raw.strip().strip('"').strip("'").startswith("gsk_"))
    st.write("Length:", len(_raw.strip().strip('"').strip("'").strip()))
    st.write("Has spaces/quotes around it:", _raw != _raw.strip().strip('"').strip("'"))

choice = st.selectbox("Try a sample note:", list(SAMPLE_NOTES.keys()))
note = st.text_area("Clinical note:", value=SAMPLE_NOTES[choice], height=140)

if st.button("Analyze", type="primary"):
    if not note.strip():
        st.warning("Please enter a note first.")
    else:
        with st.spinner("The agent is working (classifier → code search → verification)..."):
            try:
                report = analyze_note(note)
            except Exception as error:
                msg = repr(error)
                if "401" in msg or "invalid_api_key" in msg:
                    st.error("LLM authentication failed: the GROQ_API_KEY secret is missing or invalid.")
                elif "429" in msg or "rate_limit" in msg:
                    st.error("The free LLM rate limit was reached. Please try again later.")
                else:
                    st.error("The agent could not finish. Please try again in a minute.")
                st.caption(repr(error)[:300])
                st.stop()

        st.subheader("Agent answer")
        st.markdown(report["answer"])

        st.subheader("Code checks (verified in Python, not by the LLM)")
        checks = report["code_checks"]
        if checks:
            for c in checks:
                c["status"] = "OK" if (c["valid"] and c["from_tools"]) else "CHECK MANUALLY"
            st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
        else:
            st.info("No codes in the answer.")

        with st.expander("What the BioBERT classifier said (raw)"):
            st.dataframe(pd.DataFrame(report["classifier_suggestions"]), use_container_width=True, hide_index=True)

        with st.expander("Agent steps"):
            for s in report["steps"]:
                st.write(f"Step {s['step']}: `{s['tool']}` {s['args']}")
