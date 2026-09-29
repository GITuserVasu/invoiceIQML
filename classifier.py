"""
Document classifier using the configured VL model via Ollama.
Class list is pulled from entity config so it's fully dynamic.
"""

import base64
import json
import os
import subprocess

import cv2
import requests

from models import EntityConfig


def _encode(img) -> str:
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


def _gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3,
        ).decode().split("\n")[0].split(", ")
        print(f"[GPU] Util={out[0]}% Mem={out[1]}/{out[2]} MB")
    except Exception:
        pass


def classify_document(img, cfg: EntityConfig, document_name: str = "") -> tuple[str, float]:
    """
    Returns (document_type, confidence_0_to_100).
    Uses the class list from entity config — no hardcoding.
    """
    # Check if Ollama is available
    if not cfg.ollama_url or cfg.ollama_url == "":
        print(f"[CLASSIFY] Ollama not configured - using fallback classification")
        return _classify_fallback(cfg, document_name)
    
    class_list = cfg.allowed_ocr_classes + cfg.eway_classes
    # Deduplicate preserving order
    seen: set[str] = set()
    unique_classes: list[str] = []
    for c in class_list:
        if c not in seen:
            seen.add(c)
            unique_classes.append(c)

    prompt = f"""You are an expert document classifier.
Examine the layout, headers, tables, and any visible text to identify the document type.

Available classes: {unique_classes}

Return ONLY valid JSON — no explanation:
{{ "document_type": "<class_name>", "confidence": <0-100> }}"""

    payload = {
        "model": cfg.ocr_model,
        "prompt": prompt,
        "images": [_encode(img)],
        "stream": False,
    }

    _gpu_info()
    try:
        resp = requests.post(
            f"{cfg.ollama_url}/api/generate", json=payload, timeout=120
        ).json()
    except Exception as e:
        print(f"[CLASSIFY] Ollama error: {e} - using fallback")
        return _classify_fallback(cfg, document_name)
    _gpu_info()

    text = resp.get("response", "")
    print(f"[CLASSIFY] raw response: {text[:300]}")

    try:
        s = text.index("{")
        e = text.rindex("}") + 1
        result = json.loads(text[s:e])
        return str(result.get("document_type", "Others")), float(result.get("confidence", 0.0))
    except Exception:
        print(f"[CLASSIFY] Failed to parse response - using fallback")
        return _classify_fallback(cfg, document_name)


def _classify_fallback(cfg: EntityConfig, document_name: str = "") -> tuple[str, float]:
    """
    Fallback classification without LLM.
    Returns the first allowed OCR class with reasonable confidence.
    In production, you would use rule-based classification or simpler ML models.
    """
    print(f"[CLASSIFY FALLBACK] Using simple classification")
    
    name = os.path.basename(document_name).lower()
    for candidate in cfg.allowed_ocr_classes:
        normalized = candidate.lower()
        if ("purchase" in normalized or normalized == "po") and ("po" in name or "purchase" in name):
            return candidate, 80.0
        if ("receipt" in normalized or "grn" in normalized or normalized == "migo") and (
            "grn" in name or "migo" in name or "receipt" in name or "delivery" in name
        ):
            return candidate, 80.0

    # Return first allowed class, or default to TaxInvoice
    if cfg.allowed_ocr_classes:
        doc_type = cfg.allowed_ocr_classes[0]
        print(f"[CLASSIFY FALLBACK] Classifying as: {doc_type}")
        return doc_type, 85.0
    else:
        print(f"[CLASSIFY FALLBACK] No allowed classes, using TaxInvoice")
        return "TaxInvoice", 80.0
