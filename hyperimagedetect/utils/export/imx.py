
from __future__ import annotations

import subprocess
import types
from pathlib import Path

import numpy as np
import torch

from hyperimagedetect.nn.modules import Detect, Pose, Segment
from hyperimagedetect.utils import LOGGER
from hyperimagedetect.utils.patches import onnx_export_patch
from hyperimagedetect.utils.tal import make_anchors
from hyperimagedetect.utils.torch_utils import copy_attr

MCT_CONFIG = {
    "holo11": {
        "detect": {
            "layer_names": ["sub", "mul_2", "add_14", "cat_21"],
            "weights_memory": 2585350.2439,
            "n_layers": 238,
        },
        "pose": {
            "layer_names": ["sub", "mul_2", "add_14", "cat_22", "cat_23", "mul_4", "add_15"],
            "weights_memory": 2437771.67,
            "n_layers": 257,
        },
        "classify": {"layer_names": [], "weights_memory": np.inf, "n_layers": 112},
        "segment": {"layer_names": ["sub", "mul_2", "add_14", "cat_22"], "weights_memory": 2466604.8, "n_layers": 265},
    },
    "": {
        "detect": {"layer_names": ["sub", "mul", "add_6", "cat_17"], "weights_memory": 2550540.8, "n_layers": 168},
        "pose": {
            "layer_names": ["add_7", "mul_2", "cat_19", "mul", "sub", "add_6", "cat_18"],
            "weights_memory": 2482451.85,
            "n_layers": 187,
        },
        "classify": {"layer_names": [], "weights_memory": np.inf, "n_layers": 73},
        "segment": {"layer_names": ["sub", "mul", "add_6", "cat_18"], "weights_memory": 2580060.0, "n_layers": 195},
    },
}

class FXModel(torch.nn.Module):

    def __init__(self, model, imgsz=(640, 640)):
        super().__init__()
        copy_attr(self, model)
        self.model = model.model
        self.imgsz = imgsz

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, Detect):
                m._inference = types.MethodType(_inference, m)
                m.anchors, m.strides = (
                    x.transpose(0, 1)
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )
            if type(m) is Pose:
                m.forward = types.MethodType(pose_forward, m)
            if type(m) is Segment:
                m.forward = types.MethodType(segment_forward, m)
            x = m(x)
            y.append(x)
        return x

def _inference(self, x: list[torch.Tensor]) -> tuple[torch.Tensor]:
    x_cat = torch.cat([xi.view(x[0].shape[0], self.no, -1) for xi in x], 2)
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
    return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)

def pose_forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = x[0].shape[0]
    kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
    x = Detect.forward(self, x)
    pred_kpt = self.kpts_decode(bs, kpt)
    return (*x, pred_kpt.permute(0, 2, 1))

def segment_forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = self.proto(x[0])
    bs = p.shape[0]
    mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
    x = Detect.forward(self, x)
    return (*x, mc.transpose(1, 2), p)

class NMSWrapper(torch.nn.Module):

    def __init__(
        self,
        model: torch.nn.Module,
        score_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        task: str = "detect",
    ):
        super().__init__()
        self.model = model
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.task = task

    def forward(self, images):
        from edgemdt_cl.pytorch.nms.nms_with_indices import multiclass_nms_with_indices

        outputs = self.model(images)
        boxes, scores = outputs[0], outputs[1]
        nms_outputs = multiclass_nms_with_indices(
            boxes=boxes,
            scores=scores,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )
        if self.task == "pose":
            kpts = outputs[2]
            out_kpts = torch.gather(kpts, 1, nms_outputs.indices.unsqueeze(-1).expand(-1, -1, kpts.size(-1)))
            return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, out_kpts
        if self.task == "segment":
            mc, proto = outputs[2], outputs[3]
            out_mc = torch.gather(mc, 1, nms_outputs.indices.unsqueeze(-1).expand(-1, -1, mc.size(-1)))
            return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, out_mc, proto
        return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, nms_outputs.n_valid

def torch2imx(
    model: torch.nn.Module,
    file: Path | str,
    conf: float,
    iou: float,
    max_det: int,
    metadata: dict | None = None,
    gptq: bool = False,
    dataset=None,
    prefix: str = "",
):
    import model_compression_toolkit as mct
    import onnx
    from edgemdt_tpc import get_target_platform_capabilities

    LOGGER.info(f"\n{prefix} starting export with model_compression_toolkit {mct.__version__}...")

    def representative_dataset_gen(dataloader=dataset):
        for batch in dataloader:
            img = batch["img"]
            img = img / 255.0
            yield [img]

    tpc = get_target_platform_capabilities(tpc_version="4.0", device_type="imx500")

    bit_cfg = mct.core.BitWidthConfig()
    mct_config = MCT_CONFIG["holo11" if "C2PSA" in model.__str__() else ""][model.task]

    if len(list(model.modules())) != mct_config["n_layers"]:
        raise ValueError("IMX export only supported for holo11n models.")

    for layer_name in mct_config["layer_names"]:
        bit_cfg.set_manual_activation_bit_width([mct.core.common.network_editors.NodeNameFilter(layer_name)], 16)

    config = mct.core.CoreConfig(
        mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=10),
        quantization_config=mct.core.QuantizationConfig(concat_threshold_update=True),
        bit_width_config=bit_cfg,
    )

    resource_utilization = mct.core.ResourceUtilization(weights_memory=mct_config["weights_memory"])

    quant_model = (
        mct.gptq.pytorch_gradient_post_training_quantization(
            model=model,
            representative_data_gen=representative_dataset_gen,
            target_resource_utilization=resource_utilization,
            gptq_config=mct.gptq.get_pytorch_gptq_config(
                n_epochs=1000, use_hessian_based_weights=False, use_hessian_sample_attention=False
            ),
            core_config=config,
            target_platform_capabilities=tpc,
        )[0]
        if gptq
        else mct.ptq.pytorch_post_training_quantization(
            in_module=model,
            representative_data_gen=representative_dataset_gen,
            target_resource_utilization=resource_utilization,
            core_config=config,
            target_platform_capabilities=tpc,
        )[0]
    )

    if model.task != "classify":
        quant_model = NMSWrapper(
            model=quant_model,
            score_threshold=conf or 0.001,
            iou_threshold=iou,
            max_detections=max_det,
            task=model.task,
        )

    f = Path(str(file).replace(file.suffix, "_imx_model"))
    f.mkdir(exist_ok=True)
    onnx_model = f / Path(str(file.name).replace(file.suffix, "_imx.onnx"))

    with onnx_export_patch():
        mct.exporter.pytorch_export_model(
            model=quant_model, save_model_path=onnx_model, repr_dataset=representative_dataset_gen
        )

    model_onnx = onnx.load(onnx_model)
    for k, v in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    onnx.save(model_onnx, onnx_model)

    subprocess.run(
        ["imxconv-pt", "-i", str(onnx_model), "-o", str(f), "--no-input-persistency", "--overwrite-output"],
        check=True,
    )

    with open(f / "labels.txt", "w", encoding="utf-8") as file:
        file.writelines([f"{name}\n" for _, name in model.names.items()])

    return f