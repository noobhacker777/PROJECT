
from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from hyperimagedetect.models import holo
from hyperimagedetect.nn.tasks import OBBModel
from hyperimagedetect.utils import DEFAULT_CFG, RANK

class OBBTrainer(holo.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> OBBModel:
        model = OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return holo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )