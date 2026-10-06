"""
Posts OCR extraction results to the Node.js backend /invoices/intake endpoint.
Payload matches ap_invoices + ap_invoice_line_items schema exactly.
"""

import os
import json
import requests
from models import ExtractionResult, EntityConfig
from config import backend_headers

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7070")


def _safe_float(val, default=0.0) -> float:
    try:
        return float(str(val).replace(",", "").strip()) if val not in (None, "", "N/A") else default
    except (ValueError, TypeError):
        return default


def _build_line_items(result: ExtractionResult) -> list[dict]:
    """
    Convert OCR rows (arbitrary header names from the VL model) into
    normalized line item dicts matching ap_invoice_line_items columns.

    Handles common OCR field name variants:
      SNo / line_number   → line_number
      Particular / Description / particular → description
      Code / HSN / item_code / hsn_sac_code → item_code
      Qty / Quantity / qty → quantity
      Rate / UnitPrice / unit_price → unit_price
      Amount / line_total → line_total
      Batch / BatchNo / batch_number → batch_number
      CGST / cgst_rate → cgst_rate
      SGST / sgst_rate → sgst_rate
      IGST / igst_rate → igst_rate
    """
    items = []
    for i, row in enumerate(result.rows or []):
        line_num = _safe_float(
            row.get("SNo") or row.get("line_number") or row.get("S.No") or row.get("Sr")
        ) or i + 1

        desc = str(
            row.get("Particular") or row.get("Description") or row.get("particular") or
            row.get("description") or row.get("Item") or row.get("Particulars") or ""
        ).strip() or "Item"

        item_code = str(
            row.get("Code") or row.get("HSN") or row.get("item_code") or
            row.get("hsn_sac_code") or row.get("HSN/SAC") or row.get("HsnCode") or ""
        ).strip() or None

        batch = str(
            row.get("Batch") or row.get("BatchNo") or row.get("batch_number") or
            row.get("Batch No") or ""
        ).strip() or None

        qty   = _safe_float(row.get("Qty") or row.get("Quantity") or row.get("quantity") or row.get("QTY"), 1)
        price = _safe_float(row.get("Rate") or row.get("UnitPrice") or row.get("unit_price") or row.get("Unit Price"), 0)
        cgst_r = _safe_float(row.get("CGST") or row.get("cgst_rate") or row.get("CGSTRate"), 0)
        sgst_r = _safe_float(row.get("SGST") or row.get("sgst_rate") or row.get("SGSTRate"), 0)
        igst_r = _safe_float(row.get("IGST") or row.get("igst_rate") or row.get("IGSTRate"), 0)

        # Compute line_total from Amount if present, else calculate
        line_sub  = qty * price
        cgst_amt  = line_sub * cgst_r / 100
        sgst_amt  = line_sub * sgst_r / 100
        igst_amt  = line_sub * igst_r / 100
        line_total = _safe_float(
            row.get("Amount") or row.get("line_total") or row.get("Total") or row.get("Amt"),
            line_sub + cgst_amt + sgst_amt + igst_amt,
        )

        items.append({
            "line_number":  int(line_num),
            "description":  desc,
            "item_code":    item_code,
            "batch_number": batch,
            "quantity":     qty if qty > 0 else 1,
            "unit":         str(row.get("UOM") or row.get("unit") or "units"),
            "unit_price":   price,
            "cgst_rate":    cgst_r,
            "sgst_rate":    sgst_r,
            "igst_rate":    igst_r,
            "line_total":   line_total,
        })
    return items


def post_extracted_document(result: ExtractionResult, cfg: EntityConfig) -> dict:
    """Post an extracted PO or GRN so later invoices can be matched against it."""
    base = f"{BACKEND_URL}/api/v1/tenants/{cfg.tenant_id}/entities/{cfg.entity_id}"
    doc = result.invoice or {}
    kind = result.document_type.lower()
    document_type = (
        "purchase_order" if "purchase" in kind or kind == "po"
        else "goods_receipt" if "receipt" in kind or "grn" in kind or "migo" in kind
        else "invoice"
    )
    document_number = str(
        doc.get("document_number")
        or doc.get("po_number" if document_type == "purchase_order" else "grn_number")
        or doc.get("invoice_no")
        or doc.get("invoice_number")
        or ""
    ).strip()
    total = _safe_float(doc.get("total_amount") or doc.get("Total") or doc.get("Amount"), 0)
    cgst = _safe_float(doc.get("cgst_amount") or doc.get("cgst") or doc.get("CGST"), 0)
    sgst = _safe_float(doc.get("sgst_amount") or doc.get("sgst") or doc.get("SGST"), 0)
    igst = _safe_float(doc.get("igst_amount") or doc.get("igst") or doc.get("IGST"), 0)
    subtotal = _safe_float(doc.get("subtotal") or doc.get("taxable_value"), total - cgst - sgst - igst)
    payload = {
        "source_channel": result.source_channel,
        "document_type": document_type,
        "document_number": document_number,
        "document_date": str(doc.get("document_date") or doc.get("invoice_date") or "").strip() or None,
        "vendor_code": str(doc.get("vendor_code") or "").strip() or None,
        "vendor_name": str(doc.get("vendor_name") or doc.get("Vendor") or "").strip(),
        "vendor_gstin": str(doc.get("vendor_gstin") or doc.get("GSTIN") or "").strip(),
        "po_number": str(doc.get("po_number") or "").strip() or None,
        "grn_number": str(doc.get("grn_number") or "").strip() or None,
        "currency": str(doc.get("currency") or "INR"),
        "subtotal": subtotal,
        "cgst_amount": cgst,
        "sgst_amount": sgst,
        "igst_amount": igst,
        "total_amount": total,
        "line_items": _build_line_items(result),
    }
    try:
        response = requests.post(
            f"{base}/documents/intake",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-actor-email": "ocr-service@system",
                "x-actor-name": "OCR Service",
                **backend_headers(),
            },
            timeout=900,
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:
        raise RuntimeError(f"Backend document intake failed: {error}") from error


def post_extracted_invoice(result: ExtractionResult, cfg: EntityConfig) -> dict:
    """
    POST to /invoices/intake with the normalized ap_invoices-compatible payload.
    Falls back to local JSON if the backend is unreachable.
    """
    base = f"{BACKEND_URL}/api/v1/tenants/{cfg.tenant_id}/entities/{cfg.entity_id}"
    inv  = result.invoice or {}

    # Header-level amounts
    total  = _safe_float(inv.get("total_amount") or inv.get("Total") or inv.get("Amount"), 0)
    cgst_h = _safe_float(inv.get("cgst")  or inv.get("cgst_amount")  or inv.get("CGST"),  0)
    sgst_h = _safe_float(inv.get("sgst")  or inv.get("sgst_amount")  or inv.get("SGST"),  0)
    igst_h = _safe_float(inv.get("igst")  or inv.get("igst_amount")  or inv.get("IGST"),  0)
    subtotal = _safe_float(inv.get("subtotal") or inv.get("taxable_value"), total - cgst_h - sgst_h - igst_h)

    payload = {
        "source_channel":  result.source_channel,
        "document_type":   result.document_type,
        "intake_batch_id": result.batch_id,

        # Vendor
        "vendor_name":     str(inv.get("vendor_name")  or inv.get("Vendor") or "").strip(),
        "vendor_gstin":    str(inv.get("vendor_gstin") or inv.get("GSTIN")  or "").strip(),

        # Invoice header
        "invoice_number":  str(inv.get("invoice_no")   or inv.get("InvoiceNo") or inv.get("invoice_number") or "").strip(),
        "invoice_date":    str(inv.get("invoice_date") or inv.get("InvoiceDate") or inv.get("Date") or "").strip() or None,
        "due_date":        str(inv.get("due_date")     or "").strip() or None,
        "currency":        str(inv.get("currency")     or "INR"),

        # PO / GRN references (for auto-matching)
        "po_number":       str(inv.get("po_number")    or inv.get("PONumber") or inv.get("PO No") or "").strip() or None,
        "grn_number":      str(inv.get("grn_number")   or inv.get("GRNNumber") or "").strip() or None,

        # Tax amounts
        "subtotal":        subtotal,
        "cgst_amount":     cgst_h,
        "sgst_amount":     sgst_h,
        "igst_amount":     igst_h,
        "total_amount":    total,

        "ocr_confidence":  result.confidence,

        # Image hashes for duplicate detection
        "file_hash":       inv.get("file_hash"),
        "perceptual_hash": inv.get("perceptual_hash"),
        "average_hash":    inv.get("average_hash"),

        # Line items — normalized from OCR rows
        "line_items":      _build_line_items(result),
    }

    last_error = "Backend intake failed"
    try:
        resp = requests.post(
            f"{base}/invoices/intake",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-actor-email": "ocr-service@system",
                "x-actor-name":  "OCR Service",
                **backend_headers(),
            },
            timeout=900,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        body = e.response.text[:400] if e.response is not None else ""
        print(f"[POST] HTTP {code}: {body}")
        if code == 409:
            return {"duplicate": True, "message": body}
        last_error = f"Backend intake returned HTTP {code}: {body}"
    except Exception as e:
        print(f"[POST] request failed: {e}")
        last_error = str(e)

    # Fallback: save JSON locally next to the source image
    out_dir = os.path.join(os.path.dirname(result.image_path), "json")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(result.image_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_extracted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[POST] saved locally → {out_path}")
    raise RuntimeError(f"{last_error}; extracted payload saved to {out_path}")


def post_eway_bill(result: ExtractionResult, cfg: EntityConfig) -> dict:
    """
    POST E-Way Bill data to /eway-bills intake endpoint.
    """
    base = f"{BACKEND_URL}/api/v1/tenants/{cfg.tenant_id}/entities/{cfg.entity_id}"
    eway = result.invoice or {}

    payload = {
        "eway_bill_number":    str(eway.get("ewb_no")           or "").strip(),
        "eway_bill_date":      str(eway.get("ewb_date")         or "").strip() or None,
        "valid_until":         str(eway.get("valid_upto")       or "").strip() or None,
        "gstin_supplier":      str(eway.get("consignor_gstin")  or "").strip(),
        "gstin_recipient":     str(eway.get("consignee_gstin")  or "").strip() or None,
        "vehicle_number":      str(eway.get("vehicle_no")       or "").strip() or None,
        "doc_number":          str(eway.get("invoice_no")       or "").strip() or None,
        "doc_date":            str(eway.get("invoice_date")     or "").strip() or None,
        "total_value":         _safe_float(eway.get("total_value"), 0),
        "transport_mode":      str(eway.get("transport_mode")   or "road").lower(),
        "source_channel":      result.source_channel,
        "ocr_confidence":      result.confidence,
        "metadata":            {"raw": eway},
    }

    last_error = "Backend E-Way Bill intake failed"
    try:
        resp = requests.post(
            f"{base}/eway-bills/intake",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-actor-email": "ocr-service@system",
                "x-actor-name":  "OCR Service",
                **backend_headers(),
            },
            timeout=900,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[POST EWAY] failed: {e}")
        last_error = str(e)
        out_dir = os.path.join(os.path.dirname(result.image_path), "json")
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(result.image_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_eway.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        raise RuntimeError(f"{last_error}; extracted payload saved to {out_path}")


def post_audit(cfg: EntityConfig, action: str, image_path: str, doc_type: str, meta: dict = {}) -> None:
    """Write an audit event for the OCR processing action."""
    base = f"{BACKEND_URL}/api/v1/tenants/{cfg.tenant_id}"
    try:
        requests.post(
            f"{base}/audit-events",
            json={
                "action": action,
                "resource_type": "document",
                "resource_id": None,
                "metadata": {
                    "entityId":      cfg.entity_key,
                    "module":        "Document Extraction",
                    "target":        os.path.basename(image_path),
                    "document_type": doc_type,
                    "severity":      "low",
                    "result":        "success",
                    **meta,
                },
            },
            headers=backend_headers(),
            timeout=10,
        )
    except Exception:
        pass
