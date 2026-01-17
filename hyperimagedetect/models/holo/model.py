
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hyperimagedetect.data.build import load_inference_source
from hyperimagedetect.engine.model import Model
from hyperimagedetect.models import holo
from hyperimagedetect.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
)
from hyperimagedetect.utils import ROOT, YAML

class HOLO(Model):

    def __init__(self, model: str | Path = "holo11n.pt", task: str | None = None, verbose: bool = False):
        path = Path(model if isinstance(model, (str, Path)) else "")

        if not path.is_absolute() and not path.exists():
            local_paths = [
                Path("hyperimagedetect/cfg/models") / path,
                Path("hyperimagedetect/cfg/models/11") / path,
                Path(".") / path,
            ]
            for local_path in local_paths:
                if local_path.exists():
                    path = local_path
                    model = str(path)
                    if verbose:
                        from hyperimagedetect.utils import LOGGER
                        LOGGER.info(f"Auto-detected model: {path}")
                    break

        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": holo.classify.ClassificationTrainer,
                "validator": holo.classify.ClassificationValidator,
                "predictor": holo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": holo.detect.DetectionTrainer,
                "validator": holo.detect.DetectionValidator,
                "predictor": holo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": holo.segment.SegmentationTrainer,
                "validator": holo.segment.SegmentationValidator,
                "predictor": holo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": holo.pose.PoseTrainer,
                "validator": holo.pose.PoseValidator,
                "predictor": holo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": holo.obb.OBBTrainer,
                "validator": holo.obb.OBBValidator,
                "predictor": holo.obb.OBBPredictor,
            },
        }
