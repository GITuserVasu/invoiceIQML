"""
OCR Extraction Microservice
============================
FastAPI service that:
  1. Receives an image path + entity context
  2. Fetches live entity config from the backend API (classes, model, prompts, channels)
  3. Classifies the document
  4. Routes to the correct extractor (E-Way Bill, header+table, table-only)
  5. Posts results back to the backend API (or saves JSON locally as fallback)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8100 --reload

Env vars:
    BACKEND_URL   - Node.js backend  (default: http://localhost:7070)
    OLLAMA_URL    - Ollama server     (default: http://localhost:11434)
    BACKEND_SERVICE_TOKEN - Service token for backend authentication
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import cv2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from models import ProcessRequest, ExtractionResult
from config import fetch_entity_config, get_ocr_prompt, EWAY_VARIANTS
from classifier import classify_document
from extractor import extract_table, extract_header
from poster import post_extracted_document, post_extracted_invoice, post_eway_bill, post_audit
from duplicate_detector import DuplicateDetector

try:
    import pymupdf as fitz
    from PIL import Image
except ImportError:
    fitz = None
    Image = None

app = FastAPI(title="Lexa OCR Extraction Service", version="1.0.0")


def _prepare_image_path(document_path: str, batch_id: str | None) -> str:
    """Render PDF pages into one OCR image when mail sends a PDF attachment."""
    if Path(document_path).suffix.lower() != ".pdf":
        return document_path
    if fitz is None or Image is None:
        raise HTTPException(
            status_code=503,
            detail="PDF extraction requires PyMuPDF and Pillow in the OCR service environment",
        )

    pdf = fitz.open(document_path)
    if pdf.page_count == 0:
        raise HTTPException(status_code=422, detail="PDF has no pages")

    pages = []
    for index in range(min(pdf.page_count, 10)):
        page = pdf.load_page(index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pages.append(Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples))

    width = max(page.width for page in pages)
    height = sum(page.height for page in pages)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for page in pages:
        canvas.paste(page, (0, offset))
        offset += page.height

    rendered_path = f"{document_path}.{batch_id or 'document'}.jpg"
    canvas.save(rendered_path, format="JPEG", quality=92)
    return rendered_path


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "ocr-extraction"}


# ── Process single image ────────────────────────────────────────────────────
@app.post("/process", response_model=ExtractionResult)
def process_image(req: ProcessRequest):
    """
    Process one image: classify → extract → post to backend.
    """
    req.image_path = _prepare_image_path(req.image_path, req.batch_id)
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    img = cv2.imread(req.image_path)
    if img is None:
        raise HTTPException(status_code=422, detail="Cannot read image file")

    # ── 1. Check for duplicates ──────────────────────────────────────────────
    print(f"[DUPLICATE CHECK] Calculating image hashes...")
    detector = DuplicateDetector(req.tenant_id, req.entity_id)
    image_hashes = detector.calculate_hashes(req.image_path)
    print(f"[DUPLICATE CHECK] Hashes calculated - file: {image_hashes['file_hash'][:16]}..., "
          f"perceptual: {image_hashes['perceptual_hash'][:16]}...")
    
    duplicate_result = detector.check_duplicate(req.image_path, threshold=5)
    if duplicate_result:
        print(f"[DUPLICATE FOUND] Similarity: {duplicate_result['similarity']}%, "
              f"Match type: {duplicate_result['match_type']}, "
              f"Matched invoice: {duplicate_result['matched_invoice_number']}")
        
        # Return result with duplicate flag - do not process further
        duplicate_result = ExtractionResult(
            document_type="duplicate",
            confidence=duplicate_result['similarity'],
            routing_key="duplicate",
            image_path=req.image_path,
            entity_id=req.entity_id,
            tenant_id=req.tenant_id,
            batch_id=req.batch_id,
            invoice={
                "duplicate_detected": True,
                "matched_invoice_id": duplicate_result['matched_invoice_id'],
                "matched_invoice_number": duplicate_result['matched_invoice_number'],
                "matched_batch_id": duplicate_result.get('matched_batch_id'),
                "similarity_score": duplicate_result['similarity'],
                "match_type": duplicate_result['match_type']
            }
        )
        duplicate_result.source_channel = req.channel_key or "upload"
        return duplicate_result
    else:
        print(f"[DUPLICATE CHECK] No duplicates found - proceeding with OCR")

    # ── 2. Fetch entity config (live from backend API) ──────────────────────
    cfg = fetch_entity_config(req.tenant_id, req.entity_id)
    print(f"[CONFIG] entity={cfg.entity_key} model={cfg.ocr_model} "
          f"segments={cfg.segment_count} classes={len(cfg.allowed_ocr_classes)}")

    # ── 2.5. Check if Ollama is configured ──────────────────────────────────
    if not cfg.ollama_url or cfg.ollama_url.strip() == "":
        error_msg = (
            "⚠️ OCR System Not Configured\n\n"
            "The AI-powered OCR extraction system (Ollama) is not configured on this server.\n\n"
            "This system requires Ollama to be installed and running for accurate document extraction.\n\n"
            "📋 Required Actions:\n"
            "1. Install Ollama on the server\n"
            "2. Configure the OLLAMA_URL in system settings\n"
            "3. Restart the OCR service\n\n"
            "👤 Please contact your system administrator to complete the setup.\n\n"
            "Technical Details:\n"
            "- Service: OCR Extraction Service\n"
            "- Required: Ollama AI Server\n"
            "- Status: Not Configured\n"
            "- Entity: " + cfg.entity_key + "\n"
        )
        print(f"[ERROR] {error_msg}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OCR System Not Configured",
                "message": error_msg,
                "code": "OLLAMA_NOT_CONFIGURED",
                "technical_info": {
                    "entity_id": cfg.entity_id,
                    "entity_key": cfg.entity_key,
                    "ollama_url": cfg.ollama_url or "not set",
                    "required_action": "Configure Ollama URL in entity settings"
                }
            }
        )

    # ── 3. Classify ─────────────────────────────────────────────────────────
    doc_type, confidence = classify_document(img, cfg, req.image_path)
    print(f"[CLASSIFY] {doc_type} ({confidence:.1f}%)")

    # Resolve routing key from classification config
    routing_key = cfg.fallback_class
    for cls in cfg.classification_classes:
        if cls.get("class_key") == doc_type:
            routing_key = cls.get("routing_key", routing_key)
            break

    # ── 4. Route by document type ────────────────────────────────────────────
    # E-Way Bill — delegate to standalone handler
    if doc_type in cfg.eway_classes or doc_type in EWAY_VARIANTS:
        print(f"[ROUTE] E-Way Bill → eway handler")
        return _handle_eway(img, req, cfg, doc_type, confidence, routing_key)

    # Not in allowed OCR classes — skip table extraction
    if doc_type not in cfg.allowed_ocr_classes:
        print(f"[ROUTE] {doc_type} not in allowed classes — skipping OCR")
        result = ExtractionResult(
            document_type=doc_type,
            confidence=confidence,
            routing_key=routing_key,
            image_path=req.image_path,
            entity_id=req.entity_id,
            tenant_id=req.tenant_id,
            batch_id=req.batch_id,
        )
        result.source_channel = req.channel_key or "upload"
        post_audit(cfg, "document.classified", req.image_path, doc_type,
                   {"skipped": True, "reason": "not_in_allowed_classes"})
        return result

    # ── 5. Extract header (invoice-level metadata) ───────────────────────────
    header_data = extract_header(img, cfg, req.image_path)
    print(f"[HEADER] {header_data}")

    # ── 6. Extract table (segment-wise) ─────────────────────────────────────
    prompt = get_ocr_prompt(doc_type, cfg)
    result = extract_table(img, doc_type, prompt, cfg, req.image_path)
    result.invoice = header_data
    result.confidence = confidence
    result.routing_key = routing_key
    result.batch_id = req.batch_id
    result.source_channel = req.channel_key or "upload"
    
    # ── 6.5. Add image hashes to result ─────────────────────────────────────
    if not result.invoice:
        result.invoice = {}
    result.invoice['file_hash'] = image_hashes['file_hash']
    result.invoice['perceptual_hash'] = image_hashes['perceptual_hash']
    result.invoice['average_hash'] = image_hashes['average_hash']

    # ── 7. Post results to backend ───────────────────────────────────────────
    normalized_type = doc_type.lower()
    is_reference_document = (
        "purchase" in normalized_type
        or normalized_type in {"po", "grn", "migo"}
        or "receipt" in normalized_type
    )
    post_response = (
        post_extracted_document(result, cfg)
        if is_reference_document
        else post_extracted_invoice(result, cfg)
    )
    print(f"[POST] {post_response}")
    post_audit(cfg, "document.extracted", req.image_path, doc_type,
               {"rows": len(result.rows), "confidence": confidence})

    return result


# ── Process entire folder ───────────────────────────────────────────────────
@app.post("/process-folder")
def process_folder(
    tenant_id: str,
    entity_id: str,
    folder_path: str,
    batch_id: str | None = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Walk a folder and enqueue all images for processing.
    Returns immediately; processing runs in background.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    images: list[str] = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    images.sort()

    if not images:
        raise HTTPException(status_code=404, detail="No images found in folder")

    def _run_all():
        cfg = fetch_entity_config(tenant_id, entity_id)
        print(f"[BATCH] {len(images)} images | entity={cfg.entity_key}")
        for idx, img_path in enumerate(images):
            print(f"\n[BATCH] {idx+1}/{len(images)}: {img_path}")
            try:
                req = ProcessRequest(
                    tenant_id=tenant_id,
                    entity_id=entity_id,
                    image_path=img_path,
                    batch_id=batch_id,
                )
                process_image(req)
            except Exception as e:
                print(f"[BATCH] ERROR on {img_path}: {e}")

    if background_tasks:
        background_tasks.add_task(_run_all)
        return {"status": "queued", "image_count": len(images)}

    # Synchronous fallback
    _run_all()
    return {"status": "done", "image_count": len(images)}


# ── E-Way Bill handler ──────────────────────────────────────────────────────
def _handle_eway(img, req: ProcessRequest, cfg, doc_type: str,
                 confidence: float, routing_key: str) -> ExtractionResult:
    """
    Minimal E-Way Bill extraction.
    Pulls key fields (EWBNO, date, vehicle, consignor/ee) from the full image.
    You can drop in the existing process_ewaybill_image() here.
    """
    prompt = """Extract E-Way Bill fields as JSON:
{
  "ewb_no": "",
  "ewb_date": "",
  "valid_upto": "",
  "consignor_gstin": "",
  "consignee_gstin": "",
  "vehicle_no": "",
  "invoice_no": "",
  "invoice_date": "",
  "total_value": "",
  "hsn_code": "",
  "transport_mode": ""
}
Return ONLY valid JSON. Empty string for missing fields."""

    from extractor import _encode, _ocr_segment
    seg_data = _ocr_segment(img, prompt, cfg)

    result = ExtractionResult(
        document_type=doc_type,
        confidence=confidence,
        routing_key=routing_key,
        invoice=seg_data,
        headers=list(seg_data.keys()),
        rows=[seg_data],
        segments_processed=1,
        image_path=req.image_path,
        entity_id=req.entity_id,
        tenant_id=req.tenant_id,
        batch_id=req.batch_id,
    )
    result.source_channel = req.channel_key or "upload"

    post_eway_bill(result, cfg)
    post_audit(cfg, "document.extracted", req.image_path, doc_type,
               {"type": "eway_bill", "confidence": confidence})
    return result


# ── Standalone entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)
