
from __future__ import annotations

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from hyperimagedetect.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from hyperimagedetect.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from hyperimagedetect.utils.checks import check_requirements
from hyperimagedetect.utils.patches import imread

@dataclass
class SourceTypes:

    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False

class LoadPilAndNumpy:

    def __init__(self, im0: Image.Image | np.ndarray | list, channels: int = 3):
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, "filename", "") or f"image{i}.jpg" for i, im in enumerate(im0)]
        pil_flag = "L" if channels == 1 else "RGB"
        self.im0 = [self._single_check(im, pil_flag) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im: Image.Image | np.ndarray, flag: str = "RGB") -> np.ndarray:
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im.convert(flag))
            im = im[..., None] if flag == "L" else im[..., ::-1]
            im = np.ascontiguousarray(im)
        elif im.ndim == 2:
            im = im[..., None]
        return im

    def __len__(self) -> int:
        return len(self.im0)

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        self.count = 0
        return self

class LoadTensor:

    def __init__(self, im0: torch.Tensor) -> None:
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im: torch.Tensor, stride: int = 32) -> torch.Tensor:
        s = (
            f"torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:
            LOGGER.warning(
                f"torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. Dividing input by 255."
            )
            im = im.float() / 255.0

        return im

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], torch.Tensor, list[str]]:
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __len__(self) -> int:
        return self.bs

def autocast_list(source: list[Any]) -> list[Image.Image | np.ndarray]:
    files = []
    for im in source:
        if isinstance(im, (str, Path)):
            if str(im).startswith("http"):
                raise ValueError(f"URL loading disabled in offline mode: {im}")
            files.append(Image.open(im))
        elif isinstance(im, (Image.Image, np.ndarray)):
            files.append(im)
        else:
            raise TypeError(
                f"type {type(im).__name__} is not a supported HyperImageDetect prediction source type. \n"
                f"See https://docs.hyperimagedetect.com/modes/predict for supported source types."
            )

    return files

def get_best_youtube_url(url: str, method: str = "pytube") -> str | None:
    if method == "pytube":
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:
                return stream.url

    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy

        return pafy.new(url).getbestvideo(preftype="mp4").url

    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
        for f in reversed(info_dict.get("formats", [])):
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")

LOADERS = (LoadPilAndNumpy, LoadTensor)