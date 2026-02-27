"""
Crop Processing Pipeline
Handles detection crops: save to temp, extract OCR, match to SKUs via FAISS.

Workflow:
1. Holo detects 4 products → 4 crop boxes
2. Crop each region & save to tmp/ (4 separate .jpg files)
3. Process crops IN PARALLEL:
   - Extract OCR text + keywords from each crop
   - Generate embedding and match to SKU via FAISS
   - Match OCR keywords to SKU names (text similarity)
4. Combine scores:
   - FAISS similarity (image matching)
   - OCR keyword match (text matching)
5. Return HIGH CONFIDENCE results:
   - confidence_combined >= threshold → Product assigned SKU
   - confidence_combined < threshold → No SKU (invalid)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent


class CropProcessor:
    """Process detected crops through OCR + FAISS pipeline."""
    
    def __init__(self, tmp_dir: Optional[str] = None, matcher=None, ocr_extractor=None):
        """Initialize crop processor.
        
        Args:
            tmp_dir: Directory to save temporary crops (default: PROJECT/tmp/)
            matcher: SKUEmbeddingMatcher instance for FAISS matching
            ocr_extractor: Function that extracts OCR text from image
        """
        self.tmp_dir = Path(tmp_dir) if tmp_dir else PROJECT / 'tmp'
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.matcher = matcher
        self.ocr_extractor = ocr_extractor
        self.crop_metadata = {}  # Track all processed crops
    
    def crop_from_detection(self, image: np.ndarray, detection: Dict, 
                          crop_index: int) -> Optional[Tuple[np.ndarray, str]]:
        """Crop image region from detection box.
        
        Args:
            image: Input image (numpy array)
            detection: Detection dict with 'box' [x, y, w, h]
            crop_index: Index of this crop for naming
            
        Returns:
            Tuple of (crop_image, temp_crop_filename) or None if invalid
        """
        try:
            box = detection.get('box', [])
            if len(box) < 4:
                logger.warning(f"Invalid box format: {box}")
                return None
            
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure coordinates are within bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            
            # Validate crop dimensions
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid crop dimensions: ({x1},{y1}) to ({x2},{y2})")
                return None
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning(f"Empty crop for detection {crop_index}")
                return None
            
            # Save to temp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            crop_filename = f'crop_{timestamp}_{crop_index}.jpg'
            crop_path = self.tmp_dir / crop_filename
            
            cv2.imwrite(str(crop_path), crop)
            logger.debug(f"✓ Saved crop {crop_index}: {crop_filename}")
            
            return crop, crop_filename
            
        except Exception as e:
            logger.error(f"Error cropping detection {crop_index}: {e}")
            return None
    
    def extract_crop_ocr(self, crop_image: np.ndarray, crop_index: int) -> Dict:
        """Extract OCR text from crop image.
        
        Args:
            crop_image: Cropped image (numpy array)
            crop_index: Index for logging
            
        Returns:
            Dict with OCR data: {'success': bool, 'text': str, 'keywords': [], ...}
        """
        if self.ocr_extractor is None:
            return {'success': False, 'text': '', 'keywords': [], 'error': 'OCR not configured'}
        
        try:
            ocr_data = self.ocr_extractor(crop_image)
            
            # Ensure expected keys
            if not isinstance(ocr_data, dict):
                return {'success': False, 'text': '', 'keywords': []}
            
            # Extract relevant OCR fields
            return {
                'success': ocr_data.get('success', False),
                'text': ocr_data.get('full_text', '').strip(),
                'keywords': ocr_data.get('keywords', []),
                'high_confidence_keywords': ocr_data.get('high_confidence_keywords', []),
                'keyword_count': ocr_data.get('keyword_count', 0),
                'average_confidence': ocr_data.get('average_confidence', 0.0)
            }
        except Exception as e:
            logger.error(f"Error extracting OCR from crop {crop_index}: {e}")
            return {'success': False, 'text': '', 'keywords': [], 'error': str(e)}
    
    def match_crop_to_sku(self, crop_image: np.ndarray, crop_index: int,
                         sku_embeddings: Optional[Dict] = None,
                         use_accuracy_mode: bool = False) -> Dict:
        """Match crop to SKU using FAISS embedding.
        
        Args:
            crop_image: Cropped image
            crop_index: Index for logging
            sku_embeddings: Pre-computed SKU embeddings dict
            use_accuracy_mode: If True, search all embeddings instead of FAISS
            
        Returns:
            Dict with matching results: {'matched_sku': str, 'similarity': float, ...}
        """
        if self.matcher is None:
            return {'matched_sku': None, 'similarity': 0.0, 'error': 'Matcher not configured'}
        
        try:
            # Save temporary crop for embedding
            temp_crop_path = self.tmp_dir / f'temp_embed_{crop_index}.jpg'
            cv2.imwrite(str(temp_crop_path), crop_image)
            
            # Generate embedding
            crop_embedding = self.matcher.get_image_embedding(str(temp_crop_path))
            
            # Clean up temp embedding file
            try:
                temp_crop_path.unlink()
            except:
                pass
            
            if crop_embedding is None:
                return {'matched_sku': None, 'similarity': 0.0, 'error': 'Failed to generate embedding'}
            
            # Match to SKU
            if use_accuracy_mode:
                result = self.matcher.match_sku_accurate(
                    crop_embedding, 
                    sku_embeddings,
                    threshold=0.4,
                    use_exact_search=True
                )
            else:
                result = self.matcher.match_sku(
                    crop_embedding,
                    sku_embeddings,
                    threshold=0.4
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error matching crop {crop_index} to SKU: {e}")
            return {'matched_sku': None, 'similarity': 0.0, 'error': str(e)}
    
    def process_crop_complete(self, image: np.ndarray, detection: Dict, 
                             crop_index: int, sku_embeddings: Optional[Dict] = None,
                             use_accuracy_mode: bool = False) -> Dict:
        """Complete processing pipeline for one crop.
        
        Pipeline:
        1. Crop from detection box
        2. Save to temp/
        3. Extract OCR text
        4. Match to SKU via FAISS/embedding
        5. Validate: has OCR || has SKU match
        
        Args:
            image: Full image
            detection: Detection dict with box
            crop_index: Crop index for naming
            sku_embeddings: Pre-computed SKU embeddings
            use_accuracy_mode: Use exact search instead of FAISS
            
        Returns:
            Dict with complete crop processing result:
            {
                'crop_index': int,
                'crop_filename': str,
                'crop_image': np.ndarray,
                'ocr_data': Dict,
                'sku_match': Dict,
                'has_ocr_text': bool,
                'has_sku_match': bool,
                'is_valid': bool,  # has_ocr_text OR has_sku_match
                'confidence': float,  # Combined confidence score
                'error': Optional[str]
            }
        """
        result = {
            'crop_index': crop_index,
            'crop_filename': None,
            'crop_image': None,
            'ocr_data': None,
            'sku_match': None,
            'has_ocr_text': False,
            'has_sku_match': False,
            'is_valid': False,
            'confidence': 0.0,
            'error': None
        }
        
        try:
            # Step 1: Crop from detection
            crop_result = self.crop_from_detection(image, detection, crop_index)
            if crop_result is None:
                result['error'] = 'Failed to extract crop from detection'
                logger.warning(f"Crop {crop_index}: {result['error']}")
                return result
            
            crop_image, crop_filename = crop_result
            result['crop_filename'] = crop_filename
            result['crop_image'] = crop_image
            
            # Step 2: Extract OCR
            ocr_data = self.extract_crop_ocr(crop_image, crop_index)
            result['ocr_data'] = ocr_data
            
            has_ocr_text = bool(
                ocr_data.get('success') and 
                (ocr_data.get('text') or ocr_data.get('keywords'))
            )
            result['has_ocr_text'] = has_ocr_text
            
            # Step 3: Match to SKU
            sku_match = self.match_crop_to_sku(
                crop_image, 
                crop_index,
                sku_embeddings,
                use_accuracy_mode
            )
            result['sku_match'] = sku_match
            
            has_sku_match = bool(sku_match.get('matched_sku'))
            result['has_sku_match'] = has_sku_match
            
            # Step 4: Validate
            # Valid if: has OCR text OR has SKU match (or both)
            is_valid = has_ocr_text or has_sku_match
            result['is_valid'] = is_valid
            
            # Calculate confidence
            ocr_confidence = ocr_data.get('average_confidence', 0.0) if has_ocr_text else 0.0
            sku_confidence = sku_match.get('similarity', 0.0) if has_sku_match else 0.0
            
            # Combine confidences: prefer SKU match (FAISS is very accurate)
            if has_sku_match:
                result['confidence'] = sku_confidence
            elif has_ocr_text:
                result['confidence'] = ocr_confidence
            else:
                result['confidence'] = 0.0
            
            # Log result
            status = "✓ VALID" if is_valid else "✗ INVALID"
            logger.info(
                f"Crop {crop_index}: {status} | "
                f"OCR: {has_ocr_text} | SKU: {sku_match.get('matched_sku', 'None')} "
                f"({sku_confidence:.3f}) | Confidence: {result['confidence']:.3f}"
            )
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing crop {crop_index}: {e}")
            return result
    
    def process_all_detections(self, image: np.ndarray, detections: List[Dict],
                              sku_embeddings: Optional[Dict] = None,
                              use_accuracy_mode: bool = False) -> List[Dict]:
        """Process all detections through complete pipeline.
        
        Args:
            image: Full image
            detections: List of detection dicts
            sku_embeddings: Pre-computed SKU embeddings
            use_accuracy_mode: Use exact search instead of FAISS
            
        Returns:
            List of crop processing results (only valid ones by default)
        """
        results = []
        
        logger.info(f"Processing {len(detections)} detection(s)...")
        
        for i, detection in enumerate(detections):
            crop_result = self.process_crop_complete(
                image,
                detection,
                i,
                sku_embeddings,
                use_accuracy_mode
            )
            results.append(crop_result)
        
        # Filter to valid crops
        valid_results = [r for r in results if r['is_valid']]
        logger.info(f"Valid crops: {len(valid_results)}/{len(detections)}")
        
        return results
    
    def get_crop_file(self, crop_filename: str) -> Optional[Path]:
        """Get path to saved crop file."""
        crop_path = self.tmp_dir / crop_filename
        if crop_path.exists():
            return crop_path
        return None
    
    def cleanup_crop(self, crop_filename: str) -> bool:
        """Delete a saved crop file."""
        try:
            crop_path = self.tmp_dir / crop_filename
            if crop_path.exists():
                crop_path.unlink()
                logger.debug(f"Deleted crop: {crop_filename}")
                return True
        except Exception as e:
            logger.error(f"Error deleting crop {crop_filename}: {e}")
        return False
    
    def cleanup_all_crops(self) -> int:
        """Delete all temporary crop files."""
        count = 0
        try:
            for crop_file in self.tmp_dir.glob('crop_*.jpg'):
                try:
                    crop_file.unlink()
                    count += 1
                except:
                    pass
            logger.info(f"Cleaned up {count} crop files from {self.tmp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up crops: {e}")
        return count
    
    def export_processing_report(self, results: List[Dict], output_path: Optional[str] = None) -> str:
        """Export crop processing results to JSON file.
        
        Args:
            results: List of crop processing results
            output_path: Where to save report (default: PROJECT/tmp/processing_report.json)
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = str(self.tmp_dir / f'processing_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        try:
            # Prepare report data (remove numpy arrays)
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'total_crops': len(results),
                'valid_crops': sum(1 for r in results if r['is_valid']),
                'crops': []
            }
            
            for result in results:
                crop_report = {
                    'crop_index': result['crop_index'],
                    'crop_filename': result['crop_filename'],
                    'has_ocr_text': result['has_ocr_text'],
                    'has_sku_match': result['has_sku_match'],
                    'is_valid': result['is_valid'],
                    'confidence': round(result['confidence'], 3),
                    'error': result['error']
                }
                
                if result['ocr_data']:
                    crop_report['ocr_data'] = {
                        'success': result['ocr_data'].get('success'),
                        'text': result['ocr_data'].get('text'),
                        'keyword_count': result['ocr_data'].get('keyword_count'),
                        'average_confidence': round(result['ocr_data'].get('average_confidence', 0.0), 3)
                    }
                
                if result['sku_match']:
                    crop_report['sku_match'] = {
                        'matched_sku': result['sku_match'].get('matched_sku'),
                        'similarity': round(result['sku_match'].get('similarity', 0.0), 3)
                    }
                
                report_data['crops'].append(crop_report)
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"✓ Saved processing report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting processing report: {e}")
            return None
    
    # =========================================================================
    # Parallel Processing & OCR-SKU Text Matching
    # =========================================================================
    
    def match_ocr_keywords_to_sku(self, ocr_keywords: List[Dict], sku_name: str,
                                  similarity_threshold: float = 0.6) -> Dict:
        """Match OCR keywords to SKU name using text similarity.
        
        Extracts words from SKU name and finds matches in OCR keywords.
        
        Args:
            ocr_keywords: List of OCR keyword dicts [{"text": "OREO", "confidence": 0.95}]
            sku_name: SKU name to match against (e.g., "OREO MEGAPACK")
            similarity_threshold: Min similarity score (0.0-1.0)
            
        Returns:
            Dict with:
            - matched: bool (found match)
            - matched_keywords: List of matched keywords
            - similarity_scores: Dict of keyword -> similarity
            - average_similarity: float (average match score)
        """
        if not ocr_keywords or not sku_name:
            return {
                'matched': False,
                'matched_keywords': [],
                'similarity_scores': {},
                'average_similarity': 0.0
            }
        
        try:
            # Extract words from SKU name
            sku_words = sku_name.upper().split()
            
            # Extract text from OCR keywords
            ocr_texts = [kw.get('text', '').upper() for kw in ocr_keywords]
            
            matched_keywords = []
            similarity_scores = {}
            
            # Try to match each SKU word to OCR keywords
            for sku_word in sku_words:
                for ocr_kw in ocr_keywords:
                    ocr_text = ocr_kw.get('text', '').upper()
                    ocr_conf = ocr_kw.get('confidence', 0.0)
                    
                    # Calculate text similarity using SequenceMatcher
                    similarity = SequenceMatcher(None, sku_word, ocr_text).ratio()
                    
                    # Also consider OCR confidence
                    combined_score = (similarity + ocr_conf) / 2.0
                    
                    if combined_score >= similarity_threshold:
                        matched_keywords.append({
                            'sku_word': sku_word,
                            'ocr_text': ocr_text,
                            'ocr_confidence': ocr_conf,
                            'text_similarity': round(similarity, 3),
                            'combined_score': round(combined_score, 3)
                        })
                        similarity_scores[ocr_text] = combined_score
            
            # Calculate average similarity
            avg_similarity = (
                sum(similarity_scores.values()) / len(similarity_scores)
                if similarity_scores else 0.0
            )
            
            is_matched = len(matched_keywords) > 0
            
            return {
                'matched': is_matched,
                'matched_keywords': matched_keywords,
                'similarity_scores': similarity_scores,
                'average_similarity': round(avg_similarity, 3)
            }
            
        except Exception as e:
            logger.error(f"Error matching OCR to SKU '{sku_name}': {e}")
            return {
                'matched': False,
                'matched_keywords': [],
                'similarity_scores': {},
                'average_similarity': 0.0,
                'error': str(e)
            }
    
    def calculate_combined_confidence(self, faiss_similarity: float,
                                     ocr_text_match: Dict,
                                     weights: Dict = None) -> float:
        """Calculate combined confidence from FAISS + OCR matching.
        
        Formula: weighted_avg(faiss_similarity, ocr_match_score)
        - If OCR matches: confidence = 0.6*faiss + 0.4*ocr
        - If OCR missing: confidence = faiss only
        
        Args:
            faiss_similarity: FAISS similarity score (0.0-1.0)
            ocr_text_match: Result from match_ocr_keywords_to_sku()
            weights: Custom weights {'faiss': 0.6, 'ocr': 0.4}
            
        Returns:
            Combined confidence score (0.0-1.0)
        """
        if weights is None:
            weights = {'faiss': 0.60, 'ocr': 0.40}
        
        try:
            # If OCR found a match, combine scores
            if ocr_text_match.get('matched'):
                ocr_score = ocr_text_match.get('average_similarity', 0.0)
                combined = (
                    weights['faiss'] * faiss_similarity +
                    weights['ocr'] * ocr_score
                )
                return min(1.0, max(0.0, round(combined, 3)))
            else:
                # Only FAISS match, no OCR confirmation
                # Reduce confidence if OCR couldn't match
                reduced = faiss_similarity * 0.8  # 20% penalty for no OCR confirmation
                return round(reduced, 3)
                
        except Exception as e:
            logger.error(f"Error calculating combined confidence: {e}")
            return faiss_similarity  # Fallback to FAISS only
    
    def process_crops_parallel(self, image: np.ndarray, detections: List[Dict],
                              sku_embeddings: Dict = None,
                              max_workers: int = 4,
                              confidence_threshold: float = 0.70) -> List[Dict]:
        """Process multiple crops in PARALLEL using ThreadPoolExecutor.
        
        Each crop processed independently:
        1. Crop → save to tmp/
        2. Extract OCR text + keywords
        3. FAISS embedding match to SKU
        4. Match OCR keywords to SKU name
        5. Calculate combined confidence
        6. Return HIGH confidence results only
        
        Args:
            image: Full image
            detections: List of detection dicts
            sku_embeddings: Pre-computed SKU embeddings
            max_workers: Number of parallel threads
            confidence_threshold: Min confidence to include result (0.0-1.0)
            
        Returns:
            List of high-confidence crop results
        """
        results = []
        high_confidence_results = []
        
        logger.info(f"🚀 Processing {len(detections)} crops in PARALLEL (workers={max_workers})")
        
        # Define processing task for each crop
        def process_single_crop(crop_index: int, detection: Dict) -> Optional[Dict]:
            """Process one crop and return result."""
            try:
                # Step 1: Crop from detection
                crop_result = self.crop_from_detection(image, detection, crop_index)
                if crop_result is None:
                    return None
                
                crop_image, crop_filename = crop_result
                
                # Step 2: Extract OCR
                ocr_data = self.extract_crop_ocr(crop_image, crop_index)
                
                # Filter out low-confidence OCR (average_confidence 0-0.3 = unreliable)
                ocr_avg_conf = ocr_data.get('average_confidence', 0.0) if ocr_data else 0.0
                if ocr_avg_conf < 0.3:
                    logger.info(f"⚠ Crop {crop_index}: OCR confidence too low ({ocr_avg_conf:.1%}), skipping OCR data")
                    ocr_data = None
                
                # Step 3: FAISS match
                sku_match = self.match_crop_to_sku(
                    crop_image,
                    crop_index,
                    sku_embeddings,
                    use_accuracy_mode=False
                )
                
                # Step 4: Match OCR keywords to SKU
                ocr_keywords = ocr_data.get('keywords', [])
                matched_sku = sku_match.get('matched_sku')
                
                ocr_sku_match = {}
                if matched_sku and ocr_keywords:
                    ocr_sku_match = self.match_ocr_keywords_to_sku(
                        ocr_keywords,
                        matched_sku,
                        similarity_threshold=0.50
                    )
                
                # Step 5: Calculate combined confidence
                faiss_sim = sku_match.get('similarity', 0.0)
                combined_conf = self.calculate_combined_confidence(
                    faiss_sim,
                    ocr_sku_match
                )
                
                # Build result
                return {
                    'crop_index': crop_index,
                    'crop_filename': crop_filename,
                    'crop_image': crop_image,
                    'ocr_data': ocr_data,
                    'sku_match': sku_match,
                    'ocr_sku_match': ocr_sku_match,
                    'faiss_similarity': round(faiss_sim, 3),
                    'ocr_match_score': round(ocr_sku_match.get('average_similarity', 0.0), 3),
                    'confidence_combined': combined_conf,
                    'is_high_confidence': combined_conf >= confidence_threshold
                }
                
            except Exception as e:
                logger.error(f"Error processing crop {crop_index}: {e}")
                return None
        
        # Process crops in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_crop, i, det): i
                for i, det in enumerate(detections)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                crop_idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # Filter high confidence
                        if result['is_high_confidence']:
                            high_confidence_results.append(result)
                            sku = result.get('sku_match', {}).get('matched_sku', 'UNKNOWN')
                            conf = result['confidence_combined']
                            logger.info(f"✓ Crop {crop_idx}: {sku} (confidence: {conf:.1%})")
                        else:
                            sku = result.get('sku_match', {}).get('matched_sku', 'NO_MATCH')
                            conf = result['confidence_combined']
                            logger.info(f"⚠ Crop {crop_idx}: {sku} (LOW confidence: {conf:.1%} < {confidence_threshold:.1%})")
                
                except Exception as e:
                    logger.error(f"Error retrieving result for crop {crop_idx}: {e}")
        
        logger.info(f"📦 High-confidence results: {len(high_confidence_results)}/{len(detections)}")
        
        return high_confidence_results
