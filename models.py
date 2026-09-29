from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os


class ProcessRequest(BaseModel):
    tenant_id: str
    entity_id: str
    image_path: str
    batch_id: Optional[str] = None
    channel_key: Optional[str] = "upload"       # sap | mail | vendor | upload


class SegmentRow(BaseModel):
    headers: List[str] = []
    rows: List[Dict[str, Any]] = []


class ExtractionResult(BaseModel):
    document_type: str
    confidence: float = 0.0
    headers: List[str] = []
    rows: List[Dict[str, Any]] = []
    invoice: Optional[Dict[str, Any]] = None
    routing_key: str = "manual_review"
    segments_processed: int = 0
    image_path: str = ""
    entity_id: str = ""
    tenant_id: str = ""
    batch_id: Optional[str] = None
    source_channel: str = "upload"


class EntityConfig(BaseModel):
    entity_id: str
    tenant_id: str
    entity_key: str
    matching_method: str = "three_way"
    active_channels: List[str] = []
    classification_classes: List[Dict[str, Any]] = []
    confidence_threshold: float = 90.0
    fallback_class: str = "manual_review"
    ocr_model: str = "qwen3-vl:4b-instruct-q4_K_M"
    segment_count: int = 3
    ollama_url: str = os.getenv("OLLAMA_URL", "")  # Empty = use fallback extraction
    allowed_ocr_classes: List[str] = []
    class_prompts: Dict[str, str] = {}         # doc_type → OCR prompt override
    eway_classes: List[str] = []
