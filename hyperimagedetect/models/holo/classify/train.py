
from __future__ import annotations

from copy import copy
from typing import Any

import torch

from hyperimagedetect.data import ClassificationDataset, build_dataloader
from hyperimagedetect.engine.trainer import BaseTrainer
from hyperimagedetect.models import holo
from hyperimagedetect.nn.tasks import ClassificationModel
from hyperimagedetect.utils import DEFAULT_CFG, RANK
from hyperimagedetect.utils.plotting import plot_images
from hyperimagedetect.utils.torch_utils import is_parallel, torch_distributed_zero_first

class ClassificationTrainer(BaseTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = ClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        return model

    def setup_model(self):
        import torchvision

        if str(self.model) in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank, drop_last=self.args.compile)

        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def progress_string(self) -> str:
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        self.loss_names = ["loss"]
        return holo.classify.ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int):
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])
        plot_images(
            labels=batch,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )