import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.fasttext.fasttext import get_model as load_fasttext_model
from models.plwordnet_fasttext_hybrid.plwordnet_fasttext_hybrid import group_similar_words_hybrid
from text_extraction.extract_from_pdf import clean_and_extract

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    try:
        load_fasttext_model()
        print("FastText model loaded successfully on startup.")
    except Exception as exc:
        print(f"Failed to load FastText model on startup: {exc}")
        raise

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/group/hybrid")
async def group_hybrid(pdf: UploadFile = File(...), threshold: float = Form(...)) -> Any:
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0.")

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_file = tmp.name
            tmp.write(await pdf.read())

        lemma_mapping = clean_and_extract(tmp_file)

        if lemma_mapping is None:
            raise HTTPException(status_code=500, detail="Failed to extract text from the provided PDF.")

        lemmas = sorted(lemma_mapping.keys())
        groups = group_similar_words_hybrid(lemmas, threshold)
        serialized_groups = [sorted(list(group)) for group in groups]

        return {
            "filename": pdf.filename,
            "threshold": threshold,
            "lemma_count": len(lemmas),
            "groups": serialized_groups,
            "lemmas": lemmas,
        }
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.remove(tmp_file)
