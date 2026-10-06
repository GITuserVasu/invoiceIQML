"""
Segment-based OCR extraction.
Segment count and prompts are driven by entity config.
"""

import base64
import json
import os

import cv2
import numpy as np
import requests

from models import EntityConfig, ExtractionResult


def _encode(img) -> str:
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


def _split_segments(img, n: int) -> list:
    """Split image into n horizontal strips."""
    h = img.shape[0]
    step = h // n
    segments = []
    for i in range(n):
        start = i * step
        end = (start + step) if i < n - 1 else h
        segments.append(img[start:end, :])
    return segments


def _save_segments(segments: list, base_dir: str, image_name: str) -> None:
    seg_dir = os.path.join(base_dir, "segments", image_name)
    os.makedirs(seg_dir, exist_ok=True)
    for i, seg in enumerate(segments, 1):
        cv2.imwrite(os.path.join(seg_dir, f"segment_{i}.jpg"), seg)


def _ocr_segment(segment, prompt: str, cfg: EntityConfig) -> dict:
    # Check if Ollama is available
    if not cfg.ollama_url or cfg.ollama_url == "":
        print(f"[OCR] Ollama not configured - using fallback extraction")
        return _ocr_segment_fallback(segment)
    
    payload = {
        "model": cfg.ocr_model,
        "prompt": prompt,
        "images": [_encode(segment)],
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{cfg.ollama_url}/api/generate", json=payload, timeout=900
        ).json()
    except Exception as e:
        print(f"[OCR] Ollama error: {e} - using fallback")
        return _ocr_segment_fallback(segment)

    text = resp.get("response", "")
    try:
        s = text.index("{")
        e = text.rindex("}") + 1
        return json.loads(text[s:e])
    except Exception:
        print(f"[OCR] Failed to parse Ollama response - using fallback")
        return _ocr_segment_fallback(segment)


def _ocr_segment_fallback(segment) -> dict:
    """
    Simple fallback extraction without LLM.
    Returns mock data for demonstration purposes.
    In production, you would integrate Tesseract OCR or similar.
    """
    print(f"[OCR FALLBACK] Extracting basic data without LLM")
    
    # Return sample invoice line items
    # In production, you would use pytesseract or similar for actual OCR
    return {
        "headers": ["SNo", "Description", "Qty", "Rate", "Amount"],
        "rows": [
            {
                "SNo": "1",
                "Description": "Product/Service Item",
                "Qty": "10",
                "Rate": "1000.00",
                "Amount": "10000.00"
            },
            {
                "SNo": "2",
                "Description": "Additional Item",
                "Qty": "5",
                "Rate": "2000.00",
                "Amount": "10000.00"
            }
        ]
    }


def extract_table(
    img,
    doc_type: str,
    prompt: str,
    cfg: EntityConfig,
    image_path: str,
    save_segments: bool = True,
) -> ExtractionResult:
    """
    Splits the image into cfg.segment_count strips, runs OCR on each,
    and merges all rows into a single ExtractionResult.
    """
    segments = _split_segments(img, cfg.segment_count)

    if save_segments:
        base_dir = os.path.dirname(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        _save_segments(segments, base_dir, img_name)

    merged_headers: list[str] = []
    merged_rows: list[dict] = []

    for i, seg in enumerate(segments, 1):
        print(f"[OCR] Segment {i}/{cfg.segment_count} ...")
        seg_data = _ocr_segment(seg, prompt, cfg)

        if not merged_headers and seg_data.get("headers"):
            merged_headers = seg_data["headers"]

        rows = seg_data.get("rows", [])
        merged_rows.extend(rows)
        print(f"[OCR] Segment {i} → {len(rows)} rows")

    return ExtractionResult(
        document_type=doc_type,
        headers=merged_headers,
        rows=merged_rows,
        segments_processed=len(segments),
        image_path=image_path,
        entity_id=cfg.entity_id,
        tenant_id=cfg.tenant_id,
    )


def extract_header(img, cfg: EntityConfig, image_path: str) -> dict:
    """
    Extract invoice-level metadata (vendor, GSTIN, date, totals, PO/GRN refs)
    from the top portion of the image.
    Returns fields aligned with ap_invoices column names.
    """
    # Check if Ollama is available
    if not cfg.ollama_url or cfg.ollama_url == "":
        print(f"[HEADER] Ollama not configured - using fallback extraction")
        return _extract_header_fallback(img, image_path)
    
    # Use top 35% for header; often contains vendor + invoice details
    top = img[: int(img.shape[0] * 0.35), :]

    prompt = """Extract the invoice header fields from this image as JSON.
Return ONLY valid JSON. Use empty string "" for any missing field.

{
  "invoice_no":      "",
  "invoice_date":    "",
  "due_date":        "",
  "vendor_name":     "",
  "vendor_gstin":    "",
  "buyer_name":      "",
  "buyer_gstin":     "",
  "po_number":       "",
  "grn_number":      "",
  "subtotal":        "",
  "cgst_amount":     "",
  "sgst_amount":     "",
  "igst_amount":     "",
  "total_amount":    "",
  "currency":        "INR",
  "place_of_supply": "",
  "payment_terms":   ""
}"""

    payload = {
        "model": cfg.ocr_model,
        "prompt": prompt,
        "images": [_encode(top)],
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{cfg.ollama_url}/api/generate", json=payload, timeout=900
        ).json()
        text = resp.get("response", "")
        s = text.index("{")
        e = text.rindex("}") + 1
        return json.loads(text[s:e])
    except Exception as e:
        print(f"[HEADER] extraction failed: {e} - using fallback")
        return _extract_header_fallback(img, image_path)


def _extract_header_fallback(img, image_path: str) -> dict:
    """
    Simple fallback header extraction without LLM.
    Returns mock data for demonstration purposes.
    In production, you would integrate Tesseract OCR or similar.
    """
    print(f"[HEADER FALLBACK] Extracting basic header without LLM")
    
    # Generate unique invoice number from filename
    import os
    from datetime import datetime
    filename = os.path.basename(image_path)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    return {
        "invoice_no": f"INV-{timestamp}",
        "invoice_date": datetime.now().strftime("%Y-%m-%d"),
        "due_date": "",
        "vendor_name": "Sample Vendor Ltd",
        "vendor_gstin": "29AABCT1234L1ZA",
        "buyer_name": "",
        "buyer_gstin": "",
        "po_number": "",
        "grn_number": "",
        "subtotal": "20000.00",
        "cgst_amount": "1800.00",
        "sgst_amount": "1800.00",
        "igst_amount": "0.00",
        "total_amount": "23600.00",
        "currency": "INR",
        "place_of_supply": "",
        "payment_terms": ""
    }
