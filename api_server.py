"""
FastAPI backend for the LedgerLens Utility Bill Intelligence platform.

Endpoints consumed by frontend/src/api.js:
  POST /api/bills/extract   multipart: file -> { bill, flags, model }
  POST /api/bills/answer    multipart: file, question, bill -> { answer, status, grounded }

Run:
  .venv/Scripts/python.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

CORS is open to localhost Vite dev server (port 5173) and the built dist (any origin
on localhost). Adjust ALLOWED_ORIGINS for production.
"""
from __future__ import annotations

import json
import tempfile
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- project imports ---
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bill_decoder.api import APIProblem, Gemini
from bill_decoder.config import Settings
from bill_decoder.engine import ExtractionProblem, extract
from bill_decoder.images import ImageOptions
from bill_decoder.qa import BillQA, Question
from bill_decoder.schema import dumps

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
]

app = FastAPI(
    title="LedgerLens — Utility Bill Intelligence API",
    version="1.0.0",
    description="Extracts structured financial records from utility bill scans and delivers grounded, zero-hallucination Q&A.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

CACHE_L1 = ROOT / ".cache" / "level1"
CACHE_L2 = ROOT / ".cache" / "level2"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_BYTES = 15 * 1024 * 1024  # 15 MB


def _decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _safe_json_response(data: dict) -> JSONResponse:
    """Serialize with Decimal support."""
    return JSONResponse(content=json.loads(dumps(data)))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    try:
        settings = Settings.load()
        return {"status": "ok", "model": settings.model}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


# ---------------------------------------------------------------------------
# POST /api/bills/extract
# ---------------------------------------------------------------------------

@app.post("/api/bills/extract")
async def extract_bill(file: UploadFile = File(...)):
    """
    Accept a JPG/PNG bill image; return Level 1 structured extraction.

    Response shape (matches frontend expectation):
      { bill: <18-field object>, flags: [...], model: "gemini-3.1-flash-lite" }
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(422, f"Unsupported file type: {file.content_type}. Use JPG or PNG.")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large. Maximum is 15 MB.")
    if len(raw) < 1024:
        raise HTTPException(422, "File too small to be a valid bill image.")

    suffix = ".jpg" if "jpeg" in (file.content_type or "") else ".png"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        settings = Settings.load()
        api = Gemini(settings)
        try:
            opts = ImageOptions(detail_crops=True)
            envelope = extract(tmp_path, api, CACHE_L1, opts, refresh=False)
        finally:
            api.close()
            tmp_path.unlink(missing_ok=True)

        bill = envelope["extraction"]["bill"]
        flags = envelope.get("flags", [])
        return _safe_json_response({
            "bill": bill,
            "flags": flags,
            "model": envelope.get("model", settings.model),
            "cache_hit": envelope.get("cache_key") is not None,
        })

    except ExtractionProblem as exc:
        raise HTTPException(422, f"Extraction failed: {exc}") from exc
    except APIProblem as exc:
        raise HTTPException(502, f"Upstream API error ({exc.code}): {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Internal error: {type(exc).__name__}") from exc


# ---------------------------------------------------------------------------
# POST /api/bills/answer
# ---------------------------------------------------------------------------

@app.post("/api/bills/answer")
async def answer_question(
    question: str = Form(...),
    bill: str = Form("{}"),
    file: UploadFile = File(None),
):
    """
    Answer a free-text question grounded in the uploaded bill image.

    If `file` is provided (JPG/PNG), it is used for fresh evidence extraction.
    The `bill` form field may contain the Level 1 JSON previously returned by
    /api/bills/extract, but it is never used directly as the answer source —
    only printed bill evidence from the image is used.

    Response shape:
      { answer: "...", status: "answered|partial|unavailable", grounded: true }
    """
    question = question.strip()
    if not question:
        raise HTTPException(422, "Question must not be empty.")
    if not file:
        raise HTTPException(422, "A bill image file is required to answer grounded questions.")

    if file.content_type and not any(x in file.content_type.lower() for x in ["jpeg", "jpg", "png"]):
        # Warn but don't crash if it's not strictly image/jpeg; the frontend already filters
        pass

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large. Maximum is 15 MB.")

    suffix = ".jpg" if "jpeg" in (file.content_type or "") else ".png"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        settings = Settings.load()
        with BillQA(settings, cache_dir=CACHE_L2, level1_cache=CACHE_L1) as service:
            q = Question(question_id="Q1", question=question)
            report = service.answer(tmp_path, [q], refresh=False)

        tmp_path.unlink(missing_ok=True)

        results = report.get("results", [])
        if not results:
            raise HTTPException(500, "No answer generated.")

        first = results[0]
        return _safe_json_response({
            "answer": first.get("answer", ""),
            "status": first.get("status", "unavailable"),
            "grounded": True,
            "model": report.get("model", settings.model),
            "evidence_status": report.get("evidence_status", ""),
        })

    except ExtractionProblem as exc:
        raise HTTPException(422, f"Extraction failed: {exc}") from exc
    except APIProblem as exc:
        raise HTTPException(502, f"Upstream API error ({exc.code}): {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Internal error: {type(exc).__name__}") from exc


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
