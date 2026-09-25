# main.py
# FastAPI backend for the ICD-10 coding assistant.
#   POST /predict  -> classifier only (top-1 code), same response as before
#   POST /analyze  -> full agent: codes + verification + warnings + citations

from fastapi import FastAPI
from pydantic import BaseModel

from tools import predict_codes          # loads BioBERT + guideline search once
from agent import analyze_note           # the LLM agent (uses the same loaded tools)

app = FastAPI()

print("Models loaded, FastAPI app created")


class NoteInput(BaseModel):
    note: str


@app.post("/predict")
def predict(input: NoteInput):
    top = predict_codes(input.note, k=1)[0]
    return {"predicted_code": top["code"], "confidence": top["confidence"]}


@app.post("/analyze")
def analyze(input: NoteInput):
    return analyze_note(input.note)
