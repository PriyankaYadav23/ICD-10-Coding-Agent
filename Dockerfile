FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py tools.py agent.py ./
COPY saved_model/ saved_model/
COPY data/icd10cm-codes-2026.txt data/icd_guideline_chunks.json data/icd_guideline_vectors.npy data/

EXPOSE 8000

# GROQ_API_KEY must be passed at runtime, e.g. docker run -e GROQ_API_KEY=... (never baked into the image)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
