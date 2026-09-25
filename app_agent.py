# app_agent.py
# Streamlit front end for the ICD-10 coding AGENT (calls POST /analyze on main.py).
# Run: streamlit run app_agent.py   (main.py must already be running with uvicorn)

import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000/analyze"

SAMPLE_NOTES = {
    "(type your own)": "",
    "Sigmoidoscopy, short note": "Flexible sigmoidoscopy. Sigmoid and left colon diverticulosis. Rectal bleeding.",
    "Knee pain (laterality)": "Right knee pain for 3 weeks. MRI shows a tear of the medial meniscus of the right knee.",
    "Nonsense input": "hello how are you",
}

st.title("ICD-10 Coding Agent")
st.caption("BioBERT classifier + ICD-10-CM guideline search + code verification, orchestrated by an LLM agent.")

choice = st.selectbox("Try a sample note:", list(SAMPLE_NOTES.keys()))
note = st.text_area("Clinical note:", value=SAMPLE_NOTES[choice], height=150)

if st.button("Analyze"):
    if not note.strip():
        st.warning("Please enter a note first.")
    else:
        with st.spinner("The agent is working (classifier -> verify -> guidelines)..."):
            response = requests.post(API_URL, json={"note": note}, timeout=180)

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            report = response.json()

            st.subheader("Agent answer")
            st.markdown(report["answer"])

            st.subheader("Code checks (verified in Python, not by the LLM)")
            checks = report["code_checks"]
            if checks:
                for c in checks:
                    c["status"] = "OK" if (c["valid"] and c["from_tools"]) else "CHECK MANUALLY"
                st.dataframe(pd.DataFrame(checks), use_container_width=True)
            else:
                st.info("No codes found in the answer.")

            st.subheader("What the classifier said (raw)")
            st.dataframe(pd.DataFrame(report["classifier_suggestions"]), use_container_width=True)

            with st.expander("Agent steps"):
                for s in report["steps"]:
                    st.write(f"Step {s['step']}: {s['tool']} {s['args']}")
