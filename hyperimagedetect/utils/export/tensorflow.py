
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hyperimagedetect.nn.modules import Detect, Pose
from hyperimagedetect.utils import LOGGER
from hyperimagedetect.utils.files import spaces_in_path
from hyperimagedetect.utils.tal import make_anchors

def attempt_download_asset(name: str, unzip: bool = False, delete: bool = False) -> None:
    pass

def tf_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    for m in model.modules():
        if not isinstance(m, Detect):
            continue
        import types

        m._inference = types.MethodType(_tf_inference, m)
        if type(m) is Pose:
            m.kpts_decode = types.MethodType(tf_kpts_decode, m)
    return model

def _tf_inference(self, x: list[torch.Tensor]) -> tuple[torch.Tensor]:
    shape = x[0].shape
    x_cat = torch.cat([xi.view(x[0].shape[0], self.no, -1) for xi in x], 2)
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape
    grid_h, grid_w = shape[2], shape[3]
    grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
    return torch.cat((dbox, cls.sigmoid()), 1)

def tf_kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
    ndim = self.kpt_shape[1]
    y = kpts.view(bs, *self.kpt_shape, -1)
    grid_h, grid_w = self.shape[2], self.shape[3]
    grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)

def onnx2saved_model(
    onnx_file: str,
    output_dir: Path,
    int8: bool = False,
    images: np.ndarray = None,
    disable_group_convolution: bool = False,
    prefix="",
):
    onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
    if not onnx2tf_file.exists():
        attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)
    np_data = None
    if int8:
        tmp_file = output_dir / "tmp_tflite_int8_calibration_images.npy"
        if images is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(tmp_file), images)
            np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

    import onnx.helper

    if not hasattr(onnx.helper, "float32_to_bfloat16"):
        import struct

        def float32_to_bfloat16(fval):
            ival = struct.unpack("=I", struct.pack("=f", fval))[0]
            return ival >> 16

        onnx.helper.float32_to_bfloat16 = float32_to_bfloat16

    import onnx2tf

    LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
    keras_model = onnx2tf.convert(
        input_onnx_file_path=onnx_file,
        output_folder_path=str(output_dir),
        not_use_onnxsim=True,
        verbosity="error",
        output_integer_quantized_tflite=int8,
        custom_input_op_name_np_data_path=np_data,
        enable_batchmatmul_unfold=True and not int8,
        output_signaturedefs=True,
        disable_group_convolution=disable_group_convolution,
    )

    if int8:
        tmp_file.unlink(missing_ok=True)
        for file in output_dir.rglob("*_dynamic_range_quant.tflite"):
            file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
        for file in output_dir.rglob("*_integer_quant_with_int16_act.tflite"):
            file.unlink()
    return keras_model

def keras2pb(keras_model, file: Path, prefix=""):
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    m = tf.function(lambda x: keras_model(x))
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(file.parent), name=file.name, as_text=False)

def tflite2edgetpu(tflite_file: str | Path, output_dir: str | Path, prefix: str = ""):
    import subprocess

    cmd = (
        "edgetpu_compiler "
        f'--out_dir "{output_dir}" '
        "--show_operations "
        "--search_delegate "
        "--delegate_search_step 30 "
        "--timeout_sec 180 "
        f'"{tflite_file}"'
    )
    LOGGER.info(f"{prefix} running '{cmd}'")
    subprocess.run(cmd, shell=True)

def pb2tfjs(pb_file: str, output_dir: str, half: bool = False, int8: bool = False, prefix: str = ""):
    import subprocess

    import tensorflow as tf
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")

    gd = tf.Graph().as_graph_def()
    with open(pb_file, "rb") as file:
        gd.ParseFromString(file.read())
    outputs = ",".join(gd_outputs(gd))
    LOGGER.info(f"\n{prefix} output node names: {outputs}")

    quantization = "--quantize_float16" if half else "--quantize_uint8" if int8 else ""
    with spaces_in_path(pb_file) as fpb_, spaces_in_path(output_dir) as f_:
        cmd = (
            "tensorflowjs_converter "
            f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
        )
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)

    if " " in output_dir:
        LOGGER.warning(f"{prefix} your model may not work correctly with spaces in path '{output_dir}'.")

def gd_outputs(gd):
    name_list, input_list = [], []
    for node in gd.node:
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))