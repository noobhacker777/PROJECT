"""
OpenCLIP-based SKU matching and embedding generation.
Generates embeddings for product crops and matches them against SKU dataset.
"""

import json
import os
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent

# Try to import open_clip, but handle gracefully if not available
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    logger.warning("open-clip-torch not installed. Install with: pip install open-clip-torch")

# Try to import FAISS for fast vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")


class SKUEmbeddingMatcher:
    """Generate embeddings and match product crops to SKUs using OpenCLIP."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """Initialize OpenCLIP model for embedding generation.
        
        Args:
            model_name: Model architecture (e.g., "ViT-B-32", "ViT-L-14")
            pretrained: Pretrained weights (e.g., "openai", "laion400M_e32")
        """
        if not OPENCLIP_AVAILABLE:
            raise RuntimeError("OpenCLIP not available. Install: pip install open-clip-torch")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.faiss_index = None
        self.sku_list = None
        
        try:
            print(f"[OPENCLIP] Loading OpenCLIP model: {model_name} ({pretrained})...")
            print("   (This may take a minute on first run as it downloads the model)")
            
            # create_model_and_transforms returns (model, preprocess_train, preprocess_val)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            print(f"[OK] OpenCLIP model loaded: {model_name} ({pretrained}) on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            print(f"[ERROR] Failed to load model: {e}")
            print("   Make sure you have internet connection for first-time model download")
            raise
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image using OpenCLIP.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector (numpy array)
        """
        if self.model is None:
            return None
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            return None
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenCLIP.
        
        Args:
            text: Text description
            
        Returns:
            Embedding vector (numpy array)
        """
        if self.model is None:
            return None
        
        try:
            text_input = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_text(text_input)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    def crop_detections(self, image: np.ndarray, detections: list) -> dict:
        """Crop detected regions from image.
        
        Args:
            image: Input image (numpy array)
            detections: List of detection boxes [{"box": [x, y, w, h], ...}]
            
        Returns:
            Dictionary with crop info: {"crops": [cropped_images], "boxes": detections}
        """
        crops = []
        
        for i, detection in enumerate(detections):
            box = detection.get('box', [])
            if len(box) < 4:
                continue
            
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure coordinates are within bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append({
                    'image': crop,
                    'index': i,
                    'box': box
                })
        
        return {'crops': crops, 'boxes': detections}
    
    def save_crops(self, crops: list, output_dir: str, image_name: str) -> str:
        """Save all crops to a single composite image.
        
        Args:
            crops: List of crop dictionaries with 'image' key
            output_dir: Directory to save crops
            image_name: Base name of the image (without extension)
            
        Returns:
            Path to saved composite image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not crops:
            logger.warning("No crops to save")
            return None
        
        # Create composite image with all crops arranged horizontally or in grid
        max_height = max(crop['image'].shape[0] for crop in crops)
        total_width = sum(crop['image'].shape[1] for crop in crops) + (len(crops) - 1) * 10
        
        composite = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
        
        x_offset = 0
        for crop in crops:
            crop_img = crop['image']
            h, w = crop_img.shape[:2]
            composite[0:h, x_offset:x_offset+w] = crop_img
            x_offset += w + 10
        
        output_path = os.path.join(output_dir, f"{image_name}_detected_crops.jpg")
        cv2.imwrite(output_path, composite)
        
        logger.info(f"[OK] Saved {len(crops)} crops to {output_path}")
        return output_path
    
    def build_faiss_index(self, sku_embeddings: dict) -> bool:
        """Build FAISS index for fast similarity search.
        
        Args:
            sku_embeddings: Dict of SKU -> embedding
            
        Returns:
            True if index built successfully, False otherwise
        """
        if not FAISS_AVAILABLE or not sku_embeddings:
            return False
        
        try:
            print("   [FAISS] Building index from", len(sku_embeddings), "SKU embeddings...")
            
            # Extract embeddings array and SKU list
            self.sku_list = list(sku_embeddings.keys())
            embeddings_array = np.array([sku_embeddings[sku] for sku in self.sku_list]).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Create index
            dimension = embeddings_array.shape[1]  # 768 for ViT-B-32
            
            if len(sku_embeddings) < 10000:
                # For small datasets, use exact search
                self.faiss_index = faiss.IndexFlatIP(dimension)
            else:
                # For larger datasets, use IVF (Inverted File Index)
                n_lists = min(100, len(sku_embeddings) // 100)
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, n_lists)
                self.faiss_index.train(embeddings_array)
            
            self.faiss_index.add(embeddings_array)
            print(f"   [FAISS] ✓ Index ready with {self.faiss_index.ntotal} vectors (50-100x faster search)")
            return True
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            print(f"   [WARNING] Failed to build FAISS index: {e}")
            return False
    
    def match_sku(self, crop_embedding: np.ndarray, sku_embeddings: dict = None, 
                  threshold: float = 0.5, k: int = 5) -> dict:
        """Find best matching SKU for a crop embedding using FAISS or linear search.
        
        Args:
            crop_embedding: Image embedding vector
            sku_embeddings: Dict of SKU -> embedding (optional, for linear fallback)
            threshold: Minimum similarity threshold (0-1)
            k: Number of top matches to consider
            
        Returns:
            Dict with matched SKU, similarity score, and other info
        """
        if crop_embedding is None:
            return {'matched_sku': None, 'similarity': 0.0}
        
        # Try FAISS first if index is available
        if self.faiss_index is not None and self.sku_list is not None:
            try:
                # Normalize query embedding
                query_embedding = crop_embedding.astype('float32').reshape(1, -1)
                faiss.normalize_L2(query_embedding)
                
                # Search FAISS index
                distances, indices = self.faiss_index.search(query_embedding, k=min(k, self.faiss_index.ntotal))
                
                # Convert distance to similarity (FAISS uses inner product)
                best_idx = indices[0][0]
                best_similarity = distances[0][0]
                best_match = self.sku_list[best_idx]
                
                if best_similarity >= threshold:
                    return {
                        'matched_sku': best_match,
                        'similarity': float(best_similarity),
                        'threshold_met': True,
                        'top_k': [(self.sku_list[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]
                    }
                else:
                    return {
                        'matched_sku': best_match,
                        'similarity': float(best_similarity),
                        'threshold_met': False,
                        'best_match': best_match,
                        'top_k': [(self.sku_list[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]
                    }
                    
            except Exception as e:
                logger.warning(f"FAISS search failed, falling back to linear search: {e}")
        
        # Fallback to linear search if FAISS not available or failed
        if sku_embeddings is None:
            return {'matched_sku': None, 'similarity': 0.0}
        
        best_match = None
        best_similarity = -1
        
        # Normalize embedding for cosine similarity
        crop_norm = crop_embedding / (np.linalg.norm(crop_embedding) + 1e-8)
        
        for sku_id, sku_emb in sku_embeddings.items():
            sku_norm = sku_emb / (np.linalg.norm(sku_emb) + 1e-8)
            similarity = np.dot(crop_norm, sku_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = sku_id
        
        if best_similarity >= threshold:
            return {
                'matched_sku': best_match,
                'similarity': float(best_similarity),
                'threshold_met': True
            }
        else:
            return {
                'matched_sku': best_match,
                'similarity': float(best_similarity),
                'threshold_met': False,
                'best_match': best_match
            }
    
    def get_sku_embeddings_from_dataset(self, sku_dataset_dir: str = None) -> dict:
        """Generate embeddings for all SKUs in dataset.
        
        Args:
            sku_dataset_dir: Path to openclip_dataset directory
            
        Returns:
            Dictionary of SKU -> embedding
        """
        if sku_dataset_dir is None:
            sku_dataset_dir = str(PROJECT / 'openclip_dataset')
        
        sku_embeddings = {}
        
        if not os.path.exists(sku_dataset_dir):
            logger.warning(f"SKU dataset directory not found: {sku_dataset_dir}")
            return sku_embeddings
        
        # Iterate through SKU folders
        for sku_id in os.listdir(sku_dataset_dir):
            sku_path = os.path.join(sku_dataset_dir, sku_id)
            if not os.path.isdir(sku_path):
                continue
            
            # Get first image from SKU folder as representative
            images = [f for f in os.listdir(sku_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if images:
                img_path = os.path.join(sku_path, images[0])
                embedding = self.get_image_embedding(img_path)
                if embedding is not None:
                    sku_embeddings[sku_id] = embedding
                    logger.info(f"[OK] Generated embedding for SKU: {sku_id}")
        
        return sku_embeddings
    
    def save_faiss_index(self, index_path: str) -> bool:
        """Save FAISS index and SKU list metadata to disk.
        
        Args:
            index_path: Path to save .index file
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self.faiss_index is None or self.sku_list is None:
            logger.warning("No FAISS index to save")
            return False
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, index_path)
            
            # Save SKU list metadata
            metadata_path = index_path.replace('.index', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({'sku_list': self.sku_list}, f)
            
            file_size_mb = os.path.getsize(index_path) / (1024 * 1024)
            logger.info(f"[OK] FAISS index saved to {index_path} ({file_size_mb:.2f}MB)")
            print(f"[OK] FAISS index saved to {index_path} ({file_size_mb:.2f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            print(f"[ERROR] Failed to save FAISS index: {e}")
            return False
    
    def load_faiss_index(self, index_path: str) -> bool:
        """Load FAISS index and SKU list metadata from disk.
        
        Args:
            index_path: Path to .index file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(index_path):
            logger.warning(f"Index file not found: {index_path}")
            return False
        
        try:
            # Load metadata
            metadata_path = index_path.replace('.index', '_metadata.json')
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.sku_list = metadata.get('sku_list', [])
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(index_path)
            
            file_size_mb = os.path.getsize(index_path) / (1024 * 1024)
            logger.info(f"[OK] FAISS index loaded from {index_path} ({file_size_mb:.2f}MB, {self.faiss_index.ntotal} vectors)")
            print(f"[OK] FAISS index loaded from {index_path} ({file_size_mb:.2f}MB, {self.faiss_index.ntotal} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            print(f"[ERROR] Failed to load FAISS index: {e}")
            return False


def load_sku_info(sku_json_path: str = None) -> dict:
    """Load SKU information from JSON file.
    
    Args:
        sku_json_path: Path to SKU.json
        
    Returns:
        Dictionary of SKU information
    """
    if sku_json_path is None:
        sku_json_path = str(PROJECT / 'dataset' / 'SKU.json')
    
    try:
        with open(sku_json_path, 'r') as f:
            sku_data = json.load(f)
        return sku_data
    except Exception as e:
        logger.error(f"Error loading SKU info: {e}")
        return {'skus': []}
