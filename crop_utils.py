"""
Crop Processing Utilities
Post-processing, filtering, and analysis tools for crop processing results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class CropResultAnalyzer:
    """Analyze crop processing results."""
    
    def __init__(self, results: List[Dict]):
        """Initialize with crop processing results.
        
        Args:
            results: List of detection results from /api/detect-crops
        """
        self.results = results
        self.detections = results.get('detections', [])
        self.summary = results.get('summary', {})
    
    def filter_by_confidence(self, min_confidence: float = 0.75) -> List[Dict]:
        """Get detections above confidence threshold.
        
        Args:
            min_confidence: Minimum combined confidence (0.0-1.0)
            
        Returns:
            Filtered list of detections
        """
        return [
            d for d in self.detections 
            if d.get('confidence_combined', 0.0) >= min_confidence
        ]
    
    def filter_by_ocr(self, require_ocr: bool = True) -> List[Dict]:
        """Filter detections by OCR presence.
        
        Args:
            require_ocr: If True, only detections with OCR text
            
        Returns:
            Filtered list of detections
        """
        if require_ocr:
            return [
                d for d in self.detections
                if d.get('ocr_data', {}).get('has_text', False)
            ]
        else:
            return [
                d for d in self.detections
                if not d.get('ocr_data', {}).get('has_text', False)
            ]
    
    def filter_by_sku(self, sku: str) -> List[Dict]:
        """Get detections matching specific SKU.
        
        Args:
            sku: SKU name to filter
            
        Returns:
            Detections matching SKU
        """
        return [
            d for d in self.detections
            if d.get('sku_match', {}).get('sku') == sku
        ]
    
    def group_by_sku(self) -> Dict[str, List[Dict]]:
        """Group detections by SKU.
        
        Returns:
            Dict mapping SKU -> list of detections
        """
        grouped = {}
        for detection in self.detections:
            sku = detection.get('sku_match', {}).get('sku', 'UNKNOWN')
            if sku not in grouped:
                grouped[sku] = []
            grouped[sku].append(detection)
        return grouped
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics.
        
        Returns:
            Dict with stats:
            - total_detections: int
            - average_confidence: float
            - confidence_distribution: {confidence_range: count}
            - ocr_success_rate: float (0.0-1.0)
            - sku_match_rate: float (0.0-1.0)
            - unique_skus: int
        """
        if not self.detections:
            return {
                'total_detections': 0,
                'average_confidence': 0.0,
                'confidence_distribution': {},
                'ocr_success_rate': 0.0,
                'sku_match_rate': 0.0,
                'unique_skus': 0
            }
        
        n = len(self.detections)
        confidences = [
            d.get('confidence_combined', 0.0) 
            for d in self.detections
        ]
        avg_conf = sum(confidences) / n if n > 0 else 0.0
        
        # Confidence distribution
        dist = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
        for conf in confidences:
            if conf < 0.2:
                dist['0.0-0.2'] += 1
            elif conf < 0.4:
                dist['0.2-0.4'] += 1
            elif conf < 0.6:
                dist['0.4-0.6'] += 1
            elif conf < 0.8:
                dist['0.6-0.8'] += 1
            else:
                dist['0.8-1.0'] += 1
        
        # OCR success rate
        ocr_count = sum(
            1 for d in self.detections
            if d.get('ocr_data', {}).get('has_text', False)
        )
        ocr_rate = ocr_count / n if n > 0 else 0.0
        
        # SKU match rate
        sku_count = sum(
            1 for d in self.detections
            if d.get('sku_match')
        )
        sku_rate = sku_count / n if n > 0 else 0.0
        
        # Unique SKUs
        skus = set(
            d.get('sku_match', {}).get('sku')
            for d in self.detections
            if d.get('sku_match')
        )
        unique_skus = len(skus)
        
        return {
            'total_detections': n,
            'average_confidence': round(avg_conf, 3),
            'confidence_distribution': dist,
            'ocr_success_rate': round(ocr_rate, 3),
            'sku_match_rate': round(sku_rate, 3),
            'unique_skus': unique_skus,
            'sku_list': sorted(list(skus))
        }
    
    def validate_results(self) -> Dict[str, List[str]]:
        """Validate processing results for common issues.
        
        Returns:
            Dict with issues/warnings
        """
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check total detections
        if len(self.detections) == 0:
            issues['warnings'].append('No detections found')
        
        # Check confidence levels
        low_conf = [
            d for d in self.detections
            if d.get('confidence_combined', 0.0) < 0.5
        ]
        if low_conf:
            issues['warnings'].append(
                f'{len(low_conf)} detection(s) with low confidence (<0.5)'
            )
        
        # Check OCR success
        no_ocr = [
            d for d in self.detections
            if not d.get('ocr_data', {}).get('has_text', False)
        ]
        if no_ocr:
            issues['info'].append(f'{len(no_ocr)} detection(s) without OCR text')
        
        # Check SKU matches
        no_sku = [
            d for d in self.detections
            if not d.get('sku_match')
        ]
        if no_sku:
            issues['info'].append(f'{len(no_sku)} detection(s) without SKU match')
        
        return issues
    
    def to_csv(self, output_path: str = None) -> str:
        """Export results to CSV format.
        
        Args:
            output_path: Where to save CSV (default: auto-named in current dir)
            
        Returns:
            Path to saved CSV file
        """
        import csv
        
        if output_path is None:
            output_path = f'crop_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'detection_id',
                'box_x', 'box_y', 'box_w', 'box_h',
                'ocr_text',
                'ocr_confidence',
                'sku',
                'sku_similarity',
                'is_valid',
                'confidence_combined',
                'crop_filename'
            ])
            
            writer.writeheader()
            for d in self.detections:
                box = d.get('box', [0, 0, 0, 0])
                writer.writerow({
                    'detection_id': d.get('id', ''),
                    'box_x': box[0] if len(box) > 0 else '',
                    'box_y': box[1] if len(box) > 1 else '',
                    'box_w': box[2] if len(box) > 2 else '',
                    'box_h': box[3] if len(box) > 3 else '',
                    'ocr_text': d.get('ocr_data', {}).get('text', ''),
                    'ocr_confidence': d.get('ocr_data', {}).get('confidence', ''),
                    'sku': d.get('sku_match', {}).get('sku', ''),
                    'sku_similarity': d.get('sku_match', {}).get('similarity', ''),
                    'is_valid': d.get('is_valid', 'false'),
                    'confidence_combined': d.get('confidence_combined', ''),
                    'crop_filename': d.get('crop_filename', '')
                })
        
        return output_path


class CropValidator:
    """Validate and filter crop processing results."""
    
    @staticmethod
    def is_valid_detection(detection: Dict, 
                          min_confidence: float = 0.0,
                          require_ocr: bool = False,
                          require_sku: bool = False) -> bool:
        """Check if detection meets validation criteria.
        
        Args:
            detection: Detection dict from results
            min_confidence: Minimum confidence threshold
            require_ocr: If True, OCR text must be present
            require_sku: If True, SKU match must be present
            
        Returns:
            True if detection passes all criteria
        """
        # Check confidence
        if detection.get('confidence_combined', 0.0) < min_confidence:
            return False
        
        # Check OCR requirement
        if require_ocr:
            if not detection.get('ocr_data', {}).get('has_text', False):
                return False
        
        # Check SKU requirement
        if require_sku:
            if not detection.get('sku_match'):
                return False
        
        return True
    
    @staticmethod
    def validate_batch(detections: List[Dict],
                      min_confidence: float = 0.75,
                      require_ocr: bool = False,
                      require_sku: bool = True) -> Dict:
        """Validate batch of detections with criteria.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            require_ocr: Require OCR text
            require_sku: Require SKU match
            
        Returns:
            Dict with:
            - passed: List of valid detections
            - failed: List of invalid detections
            - pass_rate: float (0.0-1.0)
        """
        passed = []
        failed = []
        
        for detection in detections:
            if CropValidator.is_valid_detection(
                detection,
                min_confidence=min_confidence,
                require_ocr=require_ocr,
                require_sku=require_sku
            ):
                passed.append(detection)
            else:
                failed.append(detection)
        
        total = len(detections)
        pass_rate = len(passed) / total if total > 0 else 0.0
        
        return {
            'passed': passed,
            'failed': failed,
            'pass_count': len(passed),
            'fail_count': len(failed),
            'pass_rate': round(pass_rate, 3),
            'total': total
        }


class CropMetricCalculator:
    """Calculate metrics and performance indicators."""
    
    @staticmethod
    def calculate_precision(predicted_skus: List[str],
                           actual_skus: List[str]) -> float:
        """Calculate precision of SKU predictions.
        
        Args:
            predicted_skus: SKUs from model
            actual_skus: Correct SKUs
            
        Returns:
            Precision: correct predictions / total predictions
        """
        if not predicted_skus:
            return 0.0
        
        correct = len(set(predicted_skus) & set(actual_skus))
        return correct / len(predicted_skus)
    
    @staticmethod
    def calculate_recall(predicted_skus: List[str],
                        actual_skus: List[str]) -> float:
        """Calculate recall of SKU predictions.
        
        Args:
            predicted_skus: SKUs from model
            actual_skus: Correct SKUs
            
        Returns:
            Recall: correct predictions / total actual
        """
        if not actual_skus:
            return 0.0
        
        correct = len(set(predicted_skus) & set(actual_skus))
        return correct / len(actual_skus)
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score.
        
        Harmonic mean of precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score (0.0-1.0)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_confusion_matrix(predictions: List[Dict],
                                   ground_truth: List[Dict]) -> Dict:
        """Calculate confusion matrix for SKU predictions.
        
        Args:
            predictions: List of detection results
            ground_truth: List of expected results
            
        Returns:
            Dict with TP, TN, FP, FN counts
        """
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        for pred, actual in zip(predictions, ground_truth):
            pred_sku = pred.get('sku_match', {}).get('sku')
            actual_sku = actual.get('sku')
            
            if pred_sku == actual_sku:
                tp += 1
            elif pred_sku is None:
                fn += 1
            else:
                fp += 1
        
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'total': tp + fn + fp
        }


# Example usage
if __name__ == '__main__':
    print(__doc__)
    
    print("\n=== Example: Analyzing Crop Results ===\n")
    
    # Mock result
    mock_result = {
        'product_count': 3,
        'detections': [
            {
                'id': 0,
                'box': [100, 200, 150, 100],
                'confidence': 0.95,
                'crop_filename': 'crop_0.jpg',
                'ocr_data': {'text': 'OREO', 'has_text': True, 'confidence': 0.97},
                'sku_match': {'sku': 'OREO MEGAPACK', 'similarity': 0.89},
                'is_valid': True,
                'confidence_combined': 0.93
            },
            {
                'id': 1,
                'box': [300, 200, 150, 100],
                'confidence': 0.85,
                'crop_filename': 'crop_1.jpg',
                'ocr_data': {'text': 'SILK', 'has_text': True, 'confidence': 0.91},
                'sku_match': {'sku': 'SILK BUBBLY', 'similarity': 0.82},
                'is_valid': True,
                'confidence_combined': 0.82
            },
            {
                'id': 2,
                'box': [500, 200, 150, 100],
                'confidence': 0.45,
                'crop_filename': 'crop_2.jpg',
                'ocr_data': {'text': '', 'has_text': False, 'confidence': 0.3},
                'sku_match': None,
                'is_valid': False,
                'confidence_combined': 0.0
            }
        ],
        'summary': {'total_crops': 3, 'valid_crops': 2, 'invalid_crops': 1}
    }
    
    # Analysis
    analyzer = CropResultAnalyzer(mock_result)
    stats = analyzer.get_statistics()
    
    print(f"Total Detections: {stats['total_detections']}")
    print(f"Average Confidence: {stats['average_confidence']}")
    print(f"OCR Success Rate: {stats['ocr_success_rate']}")
    print(f"SKU Match Rate: {stats['sku_match_rate']}")
    print(f"Unique SKUs: {stats['unique_skus']} {stats['sku_list']}")
    
    print("\n=== High Confidence Detections ===\n")
    high_conf = analyzer.filter_by_confidence(0.80)
    for d in high_conf:
        print(f"  SKU: {d.get('sku_match', {}).get('sku')} | "
              f"Confidence: {d.get('confidence_combined')}")
