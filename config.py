"""
Fetches live entity configuration from the Node.js backend API.
All classification classes, prompts, segment counts, and model settings
come from entity_intake_channels.connection_settings and
classification_mapping_classes.metadata stored in the DB.
"""

import os
import requests
from typing import Optional
from models import EntityConfig

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7070")
BACKEND_SERVICE_TOKEN = os.getenv("BACKEND_SERVICE_TOKEN", "ocr-service-token-16cc1f5f")


def backend_headers() -> dict[str, str]:
    return {"x-service-token": BACKEND_SERVICE_TOKEN} if BACKEND_SERVICE_TOKEN else {}

# Default OCR prompts per document type — overridable via DB class metadata
DEFAULT_PROMPTS: dict[str, str] = {
    "TaxInvoice": """
You are an OCR assistant for GST tax invoices.
Extract ALL line items from the table in this image as JSON.
Include GST rates per line if visible.

{
  "headers": ["SNo","Code","Particular","Qty","UOM","Rate","CGST","SGST","IGST","Batch","Amount"],
  "rows": [
    {
      "SNo": "",
      "Code": "",
      "Particular": "",
      "Qty": "",
      "UOM": "",
      "Rate": "",
      "CGST": "",
      "SGST": "",
      "IGST": "",
      "Batch": "",
      "Amount": ""
    }
  ]
}
Rules: one row per line item, empty string for missing fields, return strictly valid JSON only.
""",
    "HospitalBill": """
You are an OCR assistant for hospital bills.
Extract ALL line items from the table as JSON:

{
  "headers": ["SNo","Date","Code","Particular","Rate","Qty","CGST","SGST","Amount"],
  "rows": [
    {
      "SNo": "",
      "Date": "",
      "Code": "",
      "Particular": "",
      "Rate": "",
      "Qty": "",
      "CGST": "",
      "SGST": "",
      "Amount": ""
    }
  ]
}
Rules: one row per line item, empty string for missing fields, valid JSON only.
""",
    "MedicalBills": """
You are an OCR assistant for medical / pharmacy invoices.
Extract ALL line items as JSON:

{
  "headers": ["SNo","Code","Particular","Batch","Qty","Rate","CGST","SGST","IGST","Amount"],
  "rows": [
    {
      "SNo": "",
      "Code": "",
      "Particular": "",
      "Batch": "",
      "Qty": "",
      "Rate": "",
      "CGST": "",
      "SGST": "",
      "IGST": "",
      "Amount": ""
    }
  ]
}
Rules: one row per line item, empty string for missing fields, valid JSON only.
""",
    "PurchaseOrder": """
You are an OCR assistant for purchase orders.
Extract ALL line items as JSON:

{
  "headers": ["SNo","Description","Code","Qty","UOM","UnitPrice","CGST","SGST","IGST","Amount"],
  "rows": [
    {
      "SNo": "",
      "Description": "",
      "Code": "",
      "Qty": "",
      "UOM": "",
      "UnitPrice": "",
      "CGST": "",
      "SGST": "",
      "IGST": "",
      "Amount": ""
    }
  ]
}
Rules: one row per line item, empty string for missing fields, valid JSON only.
""",
    "default": """
You are an OCR assistant for business documents.
Extract ALL table line items as JSON:

{
  "headers": ["SNo","Code","Particular","Qty","Rate","CGST","SGST","IGST","Amount"],
  "rows": [
    {
      "SNo": "",
      "Code": "",
      "Particular": "",
      "Qty": "",
      "Rate": "",
      "CGST": "",
      "SGST": "",
      "IGST": "",
      "Amount": ""
    }
  ]
}
Rules: one row per line item, empty string for missing fields, valid JSON only.
""",
}

# Document types that are E-Way Bill variants (skip table OCR, use eway handler)
EWAY_VARIANTS = {"EWayBill", "E-Way Bill", "EWAY BILL", "e-Way Bill", "eway_bill"}


def fetch_entity_config(tenant_id: str, entity_id: str) -> EntityConfig:
    """
    Pull live entity config from the backend API.
    Falls back to sensible defaults if any call fails.
    """
    base = f"{BACKEND_URL}/api/v1/tenants/{tenant_id}/entities/{entity_id}"
    cfg = EntityConfig(entity_id=entity_id, tenant_id=tenant_id, entity_key=entity_id)

    # ── 1. Dashboard: matching policy + intake channels + modules ───────────
    try:
        dash = requests.get(f"{base}/dashboard", headers=backend_headers(), timeout=10).json()
        print({base})
        print("-vv-")
        print(backend_headers())
        print("-vv-")
        print(dash)
        print("-vv-")
        cfg.entity_key = dash.get("entity", {}).get("entity_key", entity_id)

        mp = dash.get("matchingPolicy") or {}
        cfg.matching_method = mp.get("matching_method", "three_way")

        cfg.active_channels = [
            ch["channel_key"]
            for ch in dash.get("intakeChannels", [])
            if ch.get("is_enabled")
        ]

        # OCR model + segment count from connection_settings of upload channel
        for ch in dash.get("intakeChannels", []):
            if ch.get("channel_key") == "upload":
                cs = ch.get("connection_settings") or {}
                cfg.ocr_model = cs.get("ocr_model", cfg.ocr_model)
                #cfg.ocr_model = "llava"
                cfg.segment_count = int(cs.get("segment_count", cfg.segment_count))
                cfg.ollama_url = cs.get("ollama_url", cfg.ollama_url)
                # print("vv1")
                # print(cfg.ocr_model)
                # print("vv2")
                # print(cfg.ollama_url)
                # print("vv3")

    except Exception as e:
        print(f"[CONFIG] dashboard fetch failed: {e}")

    # ── 2. Classification mapping: allowed classes + per-class prompts ───────
    try:
        cl = requests.get(f"{base}/classification", headers=backend_headers(), timeout=10).json()
        mapping = cl.get("mapping") or {}
        cfg.confidence_threshold = float(mapping.get("confidence_threshold_pct", 90))
        cfg.fallback_class = mapping.get("fallback_class_key", "manual_review")

        classes = cl.get("classes") or []
        cfg.classification_classes = classes

        # Build allowed OCR class list and per-class prompts
        for cls in classes:
            if cls.get("status") in ("active", "pilot") and not cls.get("is_fallback"):
                cfg.allowed_ocr_classes.append(cls["class_key"])
                # Prompt stored in class metadata under "ocr_prompt"
                meta = cls.get("metadata") or {}
                if meta.get("ocr_prompt"):
                    cfg.class_prompts[cls["class_key"]] = meta["ocr_prompt"]
                if cls["class_key"] in EWAY_VARIANTS or meta.get("is_eway_bill"):
                    cfg.eway_classes.append(cls["class_key"])

    except Exception as e:
        print(f"[CONFIG] classification fetch failed: {e}")

    # If no classes configured yet, fall back to hardcoded defaults
    if not cfg.allowed_ocr_classes:
        cfg.allowed_ocr_classes = [
            "HospitalBill", "MedicalBills", "TaxInvoice",
            "BILL FOR APPROVAL DETAILED BREAKUP", "Supply order",
            "Purchase order", "Goods Receipt", "GRN", "Bill of Supply", "Invoice",
        ]
    if not cfg.eway_classes:
        cfg.eway_classes = list(EWAY_VARIANTS)

    return cfg


def get_ocr_prompt(doc_type: str, cfg: EntityConfig) -> str:
    """Return the OCR prompt for a document type.
    Priority: per-class DB override → default per-type → generic default."""
    if doc_type in cfg.class_prompts:
        return cfg.class_prompts[doc_type]
    for key, prompt in DEFAULT_PROMPTS.items():
        if key.lower() in doc_type.lower() or doc_type.lower() in key.lower():
            return prompt
    return DEFAULT_PROMPTS["default"]
