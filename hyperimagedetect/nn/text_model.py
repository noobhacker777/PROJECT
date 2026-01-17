
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from hyperimagedetect.utils import checks
from hyperimagedetect.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    import clip

class TextModel(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        pass

class CLIP(TextModel):

    def __init__(self, size: str, device: torch.device) -> None:
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
        return clip.tokenize(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats

class MobileCLIP(TextModel):

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        try:
            import mobileclip
        except ImportError:

            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            raise FileNotFoundError(f"Model file '{file}' not found. URL downloads disabled in offline mode.")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

class MobileCLIPTS(TextModel):

    def __init__(self, device: torch.device):
        super().__init__()
        model_file = "mobileclip_blt.ts"
        from pathlib import Path
        if not Path(model_file).exists():
            raise FileNotFoundError(f"Model file '{model_file}' not found. URL downloads disabled in offline mode.")
        self.encoder = torch.jit.load(model_file, map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        return self.tokenizer(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:

        return self.encoder(texts).to(dtype)

def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    else:
        raise ValueError(f"Unrecognized base model: '{base}'. Supported base models: 'clip', 'mobileclip'.")