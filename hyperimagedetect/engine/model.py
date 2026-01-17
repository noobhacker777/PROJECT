
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from hyperimagedetect.cfg import TASK2DATA, get_cfg, get_save_dir
from hyperimagedetect.engine.results import Results
from hyperimagedetect.nn.tasks import guess_model_task, load_checkpoint, yaml_model_load
from hyperimagedetect.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    YAML,
    callbacks,
    checks,
)

class Model(torch.nn.Module):

    def __init__(
        self,
        model: str | Path | Model = "holo11n.pt",
        task: str | None = None,
        verbose: bool = False,
    ) -> None:
        if isinstance(model, Model):
            self.__dict__ = model.__dict__
            return
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.ckpt = {}
        self.cfg = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None
        self.task = task
        self.model_name = None
        model = str(model).strip()

        if self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"
            return

        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if str(model).endswith((".yaml", ".yml")):
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        del self.training

    def __call__(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            raise ValueError(
                f"❌ OFFLINE MODE ACTIVE: Cannot download from URL\n"
                f"   URL: {weights}\n"
                f"   Solution: Use local file path instead\n"
                f"   Examples:\n"
                f"      - HOLO('hyperimagedetect/cfg/models/11/holo11.yaml')\n"
                f"      - HOLO('path/to/local/model.pt')\n"
                f"   No internet downloads are permitted."
            )
        weights = checks.check_model_file_from_stem(weights)

        if str(weights).rpartition(".")[-1] == "pt":
            self.model, self.ckpt = load_checkpoint(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        pt_str = isinstance(self.model, (str, Path)) and str(self.model).rpartition(".")[-1] == "pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'holo predict model=holo11n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> Model:
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: str | Path = "holo11n.pt") -> Model:
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights
            weights, self.ckpt = load_checkpoint(weights)
        self.model.load(weights)
        return self

    def save(self, filename: str | Path = "saved_model.pt") -> None:
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from hyperimagedetect import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,

        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self) -> None:
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> list[Results]:
        if source is None:
            source = ASSETS
            LOGGER.warning(f"'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("holo") or ARGV[0].endswith("hyperimagedetect")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True}
        args = {**self.overrides, **custom, **kwargs}
        prompts = args.pop("prompts", None)

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> list[Results]:
        if not hasattr(self.predictor, "trackers"):
            from hyperimagedetect.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1
        kwargs["batch"] = kwargs.get("batch") or 1
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        custom = {"rect": True}
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, data=None, format="", verbose=False, **kwargs: Any):
        self._check_is_pytorch_model()
        from hyperimagedetect.utils.benchmarks import benchmark

        from .exporter import export_formats

        custom = {"verbose": False}
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        fmts = export_formats()
        export_args = set(dict(zip(fmts["Argument"], fmts["Arguments"])).get(format, [])) - {"batch"}
        export_kwargs = {k: v for k, v in args.items() if k in export_args}
        return benchmark(
            model=self,
            data=data,
            imgsz=args["imgsz"],
            device=args["device"],
            verbose=verbose,
            format=format,
            **export_kwargs,
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,
            "verbose": False,
        }
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        self._check_is_pytorch_model()

        if isinstance(kwargs.get("pretrained", None), (str, Path)):
            self.load(kwargs["pretrained"])
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.train()
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = load_checkpoint(ckpt)
            self.overrides = self._reset_ckpt_args(self.model.args)
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        self._check_is_pytorch_model()
        if use_ray:
            from hyperimagedetect.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> Model:
        self._check_is_pytorch_model()
        self = super()._apply(fn)
        self.predictor = None
        self.overrides["device"] = self.device
        return self

    @property
    def names(self) -> dict[int, str]:
        from hyperimagedetect.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:
            predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            predictor.setup_model(model=self.model, verbose=False)
            return predictor.model.names
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        include = {"imgsz", "data", "task", "single_cls"}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]
            raise NotImplementedError(f"'{name}' model does not support '{mode}' mode for '{self.task}' task.") from e

    @property
    def task_map(self) -> dict:
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name):
        return self._modules["model"] if name == "model" else getattr(self.model, name)