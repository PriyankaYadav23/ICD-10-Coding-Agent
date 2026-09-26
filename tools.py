# tools.py
# The agent's tools for the ICD-10 coding assistant.
#   Tool 1: predict_codes(note)            -> top-k ICD codes from the fine-tuned BioBERT
#   Tool 2: search_coding_guidelines(query) -> top-k passages from the ICD-10-CM guidelines
# Everything heavy (models, data) is loaded ONCE, when this file is imported.

import os
import json
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Folder where this file lives, so paths work no matter where the program is started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "biobert_icd_classifier_weighted")
HF_MODEL_REPO = "PriyankaYadav777/biobert-icd10-classifier"   # public copy on the Hugging Face Hub

# Use the local model if it exists (my laptop); otherwise download it from the Hub (cloud deployment).
# Set FORCE_HUB=1 to test the download path locally.
if os.path.isdir(LOCAL_MODEL_PATH) and os.getenv("FORCE_HUB") != "1":
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    from huggingface_hub import snapshot_download
    MODEL_PATH = snapshot_download(HF_MODEL_REPO)
print("Classifier loaded from:", MODEL_PATH)
CODES_FILE = os.path.join(BASE_DIR, "data", "icd10cm-codes-2026.txt")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "icd_guideline_chunks.json")
VECTORS_FILE = os.path.join(BASE_DIR, "data", "icd_guideline_vectors.npy")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------- Load once: Tool 1 (BioBERT classifier) ----------
clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
clf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
clf_model.eval()

with open(os.path.join(MODEL_PATH, "label_to_code.json")) as f:
    label_to_code_raw = json.load(f)
label_to_code = {int(k): v for k, v in label_to_code_raw.items()}

code_to_description = {}
with open(CODES_FILE) as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        code_to_description[parts[0]] = parts[1]


# ---------- Load once: Tool 2 (guideline search) ----------
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open(CHUNKS_FILE) as f:
    guideline_chunks = json.load(f)

guideline_vectors = np.load(VECTORS_FILE)


# ---------- Tool 1 ----------
def predict_codes(note, k=3):
    inputs = clf_tokenizer(note, truncation=True, padding=True, max_length=256, return_tensors="pt")

    with torch.no_grad():
        outputs = clf_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    top_probs, top_labels = torch.topk(probabilities, k)

    results = []
    for prob, label in zip(top_probs, top_labels):
        code = label_to_code[label.item()]
        results.append({
            "code": code,
            "description": code_to_description[code],
            "confidence": round(prob.item(), 4),
        })
    return results


# ---------- Tool 2 ----------
def search_coding_guidelines(query, k=3):
    query_vector = embed_model.encode([QUERY_PREFIX + query])
    scores = cosine_similarity(query_vector, guideline_vectors)[0]
    top_ids = np.argsort(-scores)[:k]

    results = []
    for i in top_ids:
        results.append({
            "source": guideline_chunks[i]["source"],
            "page": guideline_chunks[i]["page"],
            "score": round(float(scores[i]), 3),
            "text": guideline_chunks[i]["text"],
        })
    return results


# ---------- Tool 3 ----------
def lookup_code(code, max_results=10):
    clean_code = code.replace(".", "").strip().upper()

    if clean_code in code_to_description:
        return {"code": clean_code, "exists": True, "description": code_to_description[clean_code]}

    children = []
    for c, desc in code_to_description.items():
        if c.startswith(clean_code):
            children.append({"code": c, "description": desc})
            if len(children) == max_results:
                break

    return {"code": clean_code, "exists": False, "billable_codes_starting_with_this": children}


# ---------- Tool 4 ----------
STOPWORDS = {"of", "the", "and", "or", "with", "without", "a", "an", "in", "on", "to", "for", "by",
             "due", "not", "other", "unspecified", "icd", "10", "cm", "code"}


def words(text):
    # lowercase words, stopwords removed, simple plural stripping ("tears" -> "tear")
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def search_codes(description, max_results=10):
    query_words = words(description)
    scored = []
    for code, desc in code_to_description.items():
        hits = len(query_words & words(desc))
        if hits:
            scored.append((hits, -len(desc), code, desc))
    scored.sort(reverse=True)
    return [{"code": c, "description": d, "matched_words": h} for h, _, c, d in scored[:max_results]]
