"""
duplicate_detector.py
======================
Image-based duplicate detection using perceptual hashing.
Detects visually similar documents even if filenames differ.
"""

import os
import hashlib
import imagehash
from PIL import Image
import requests
from typing import Optional, Dict, Any

# Backend URL from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7070")
BACKEND_SERVICE_TOKEN = os.getenv("BACKEND_SERVICE_TOKEN", "")


class DuplicateDetector:
    """
    Detects duplicate invoices using perceptual image hashing.
    
    Uses multiple hash algorithms:
    - MD5: Exact file duplicate (byte-for-byte)
    - Perceptual Hash: Similar images (rotation/resize resistant)
    - Average Hash: Fast similarity check
    """
    
    def __init__(self, tenant_id: str, entity_id: str):
        self.tenant_id = tenant_id
        self.entity_id = entity_id
        self.backend_url = BACKEND_URL
    
    def calculate_hashes(self, image_path: str) -> Dict[str, str]:
        """
        Calculate multiple hashes for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'file_hash', 'perceptual_hash', 'average_hash'
        """
        # File MD5 hash (exact duplicate detection)
        file_hash = self._file_md5(image_path)
        
        # Image perceptual hashes
        img = Image.open(image_path)
        perceptual_hash = str(imagehash.phash(img, hash_size=16))  # 16x16 = 256 bits
        average_hash = str(imagehash.average_hash(img, hash_size=8))  # 8x8 = 64 bits
        
        return {
            'file_hash': file_hash,
            'perceptual_hash': perceptual_hash,
            'average_hash': average_hash
        }
    
    def check_duplicate(self, image_path: str, threshold: int = 5) -> Optional[Dict[str, Any]]:
        """
        Check if image is a duplicate of any existing invoice.
        
        Args:
            image_path: Path to the image file
            threshold: Hamming distance threshold (0=identical, 5=very similar, 10=similar)
            
        Returns:
            None if not duplicate, otherwise dict with:
            {
                'is_duplicate': True,
                'matched_invoice_id': 'uuid',
                'matched_invoice_number': 'INV-001',
                'similarity': 95.5,
                'match_type': 'exact_file' | 'perceptual' | 'average'
            }
        """
        # Calculate hashes for uploaded image
        current_hashes = self.calculate_hashes(image_path)
        
        # Get recent invoice hashes from backend
        existing_hashes = self._fetch_recent_hashes(days=90)
        
        if not existing_hashes:
            print("vasu -- No existing hashes")
            return None
        
        # Check 1: Exact file match (MD5)
        for inv in existing_hashes['data']:
            print("File Hash")
            print(current_hashes['file_hash'])
            print(inv.get('file_hash'))
            if inv.get('file_hash') == current_hashes['file_hash']:
                return {
                    'is_duplicate': True,
                    'matched_invoice_id': inv['id'],
                    'matched_invoice_number': inv['invoice_number'],
                    'matched_batch_id': inv.get('intake_batch_id'),
                    'similarity': 100.0,
                    'match_type': 'exact_file'
                }
        
        # Check 2: Perceptual hash (similar images)
        current_phash = imagehash.hex_to_hash(current_hashes['perceptual_hash'])
        
        for inv in existing_hashes['data']:
            if inv.get('perceptual_hash'):
                try:
                    stored_phash = imagehash.hex_to_hash(inv['perceptual_hash'])
                    distance = current_phash - stored_phash  # Hamming distance
                    
                    if distance <= threshold:
                        # Convert distance to similarity percentage
                        # Distance 0 = 100% similar, distance 10 = 50% similar
                        similarity = max(0, 100 - (distance * 5))
                        
                        return {
                            'is_duplicate': True,
                            'matched_invoice_id': inv['id'],
                            'matched_invoice_number': inv['invoice_number'],
                            'matched_batch_id': inv.get('intake_batch_id'),
                            'similarity': round(similarity, 1),
                            'match_type': 'perceptual',
                            'hamming_distance': distance
                        }
                except Exception as e:
                    print(f"[WARN] Failed to compare perceptual hash: {e}")
                    continue
        
        # Check 3: Average hash (fallback for heavily processed images)
        current_ahash = imagehash.hex_to_hash(current_hashes['average_hash'])
        
        for inv in existing_hashes['data']:
            if inv.get('average_hash'):
                try:
                    stored_ahash = imagehash.hex_to_hash(inv['average_hash'])
                    distance = current_ahash - stored_ahash
                    
                    if distance <= threshold:
                        similarity = max(0, 100 - (distance * 5))
                        
                        return {
                            'is_duplicate': True,
                            'matched_invoice_id': inv['id'],
                            'matched_invoice_number': inv['invoice_number'],
                            'matched_batch_id': inv.get('intake_batch_id'),
                            'similarity': round(similarity, 1),
                            'match_type': 'average',
                            'hamming_distance': distance
                        }
                except Exception as e:
                    print(f"[WARN] Failed to compare average hash: {e}")
                    continue
        
        return None
    
    def _file_md5(self, file_path: str) -> str:
        """Calculate MD5 hash of file content."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _fetch_recent_hashes(self, days: int = 90) -> list:
        """
        Fetch invoice hashes from backend.
        
        Args:
            days: Number of days to look back for duplicates
            
        Returns:
            List of invoice records with hashes
        """
        try:
            headers = {}
            if BACKEND_SERVICE_TOKEN:
                headers['x-service-token'] = BACKEND_SERVICE_TOKEN
            
            url = f"{self.backend_url}/api/v1/tenants/{self.tenant_id}/entities/{self.entity_id}/invoice-hashes"
            response = requests.get(
                url,
                params={'days': days},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print("VASU")
                print(data)
                # return data.get('hashes', [])
                return data
            else:
                print(f"[WARN] Backend returned {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch invoice hashes: {e}")
            return []


def check_duplicate(tenant_id: str, entity_id: str, image_path: str, threshold: int = 5) -> Optional[Dict[str, Any]]:
    """
    Convenience function for duplicate checking.
    
    Args:
        tenant_id: Tenant UUID
        entity_id: Entity UUID or key
        image_path: Path to image file
        threshold: Similarity threshold (default: 5)
        
    Returns:
        None if not duplicate, otherwise duplicate info dict
    """
    detector = DuplicateDetector(tenant_id, entity_id)
    return detector.check_duplicate(image_path, threshold)
