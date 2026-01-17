
from __future__ import annotations

from copy import copy
from pathlib import Path

from hyperimagedetect.models import holo
from hyperimagedetect.nn.tasks import SegmentationModel
from hyperimagedetect.utils import DEFAULT_CFG, RANK

class SegmentationTrainer(holo.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        model = SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return holo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )