
import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from hyperimagedetect.nn.autobackend import check_class_names
from hyperimagedetect.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
)
from hyperimagedetect.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
from hyperimagedetect.utils.checks import check_requirements, check_suffix, check_yaml
from hyperimagedetect.utils.loss import (
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from hyperimagedetect.utils.ops import make_divisible
from hyperimagedetect.utils.patches import torch_load
from hyperimagedetect.utils.plotting import feature_visualization
from hyperimagedetect.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)

class BaseModel(torch.nn.Module):

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        try:
            import thop
        except ImportError:
            thop = None

        c = m == self.model[-1] and isinstance(x, list)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.fuse_forward = m.forward_fuse

        return self

    def is_fused(self, thresh=10):
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(
            m, Detect
        ):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        updated_csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(updated_csd, strict=False)
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"

        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")

class DetectionModel(BaseModel):

    def __init__(self, cfg="holo11n.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if self.yaml["backbone"][0][2] == "Silence":
            self.yaml["backbone"][0][2] = "nn.Identity"

        self.yaml["channels"] = ch
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            self.model.eval()
            m.training = True
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)
        if verbose:
            LOGGER.info("")

    def _predict_augment(self, x):
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))
        y[0] = y[0][..., :-i]
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y

    def init_criterion(self):
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)

class OBBModel(DetectionModel):

    def __init__(self, cfg="holo11n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8OBBLoss(self)

class SegmentationModel(DetectionModel):

    def __init__(self, cfg="holo11n-seg.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8SegmentationLoss(self)

class PoseModel(DetectionModel):

    def __init__(self, cfg="holo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8PoseLoss(self)

class ClassificationModel(BaseModel):

    def __init__(self, cfg="holo11n-cls.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        ch = self.yaml["channels"] = self.yaml.get("channels", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.stride = torch.Tensor([1])
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}

    @staticmethod
    def reshape_outputs(model, nc):
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]
        if isinstance(m, Classify):
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        return v8ClassificationLoss()

class Ensemble(torch.nn.ModuleList):

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)
        return y, None

@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        for old, new in modules.items():
            try:
                sys.modules[old] = import_module(new)
            except ModuleNotFoundError:
                import types
                sys.modules[old] = types.ModuleType(old)

        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            try:
                old_mod = import_module(old_module) if old_module in sys.modules else None
                new_mod = import_module(new_module)
                if old_mod is not None:
                    setattr(old_mod, old_attr, getattr(new_mod, new_attr))
            except (ModuleNotFoundError, AttributeError):
                pass

        yield
    finally:
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]

class SafeClass:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return self

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __contains__(self, item):
        return False

    def __bool__(self):
        return False

class SafeUnpickler(pickle.Unpickler):

    def __init__(self, *args, module_mapping=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_mapping = module_mapping or {}

    def find_class(self, module, name):
        original_module = module
        if module in self.module_mapping:
            module = self.module_mapping[module]

        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            "hyperimagedetect",
        )

        try:
            if module in safe_modules or module.startswith(("torch.", "numpy.", "hyperimagedetect.")):
                return super().find_class(module, name)
            import sys
            from importlib import import_module

            if module not in sys.modules:
                import_module(module)
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return SafeClass

def torch_safe_load(weight, safe_only=False):
    from pathlib import Path

    check_suffix(file=weight, suffix=".pt")
    file = Path(weight)

    if not file.exists():
        raise FileNotFoundError(f"❌ OFFLINE MODE: Model file not found: {weight}\nUse local files only.")

    try:
        safe_pickle = types.ModuleType("safe_pickle")
        safe_pickle.Unpickler = SafeUnpickler
        safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()

        with open(file, "rb") as f:
            ckpt = torch_load(f, pickle_module=safe_pickle)

    except ModuleNotFoundError as e:
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an HyperImageDetect  model originally trained "
                    f"with HOLO.\nThis model is NOT forwards compatible with "
                    f". Please use models trained with this version."
                    f"\nRecommend fixes are to train a new model using the latest 'hyperimagedetect' package or to "
                    f"run a command with an official HyperImageDetect model, i.e. 'holo predict model=holo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in HyperImageDetect requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'hyperimagedetect' package or to "
            f"run a command with an official HyperImageDetect model, i.e. 'holo predict model=holo11n.pt'"
        )
        check_requirements(e.name)
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save HOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file

def _infer_model_scale_from_checkpoint(state_dict, task="detect"):
    """Infer the model scale (n, s, m, l, x) from checkpoint state_dict shapes."""
    if not state_dict:
        return "m"  # Default to medium
    
    # Look for distinguishing features in layer 23 (detection head)
    # Different scales have different channel counts
    sample_keys = [
        "model.23.cv3.2.1.1.conv.weight",  # Shape will be [channels, channels, 1, 1]
        "model.24.cv2.2.1.1.conv.weight",  # Backup key
    ]
    
    for key in sample_keys:
        if key in state_dict:
            shape = state_dict[key].shape
            # shape[0] is output channels, shape[1] is input channels
            output_ch = shape[0]
            
            # Map channel counts to model scales based on holo11.yaml scales
            # n: [0.50, 0.25] -> base*0.25 = 64*0.25 = 16, 256*0.25 = 64
            # s: [0.50, 0.50] -> base*0.50 = 64*0.50 = 32, 256*0.50 = 128
            # m: [0.50, 1.00] -> base*1.00 = 64, 256*1.00 = 256
            # l: [1.00, 1.00] -> base*1.00 = 64, 256*1.00 = 256
            # x: [1.00, 1.50] -> base*1.50 = 96, 256*1.50 = 384
            
            scale_mapping = {
                16: "n", 32: "n", 64: "n",    # nano range
                64: "s", 80: "s", 128: "s",   # small range  
                128: "m", 160: "m", 192: "m", 256: "m",  # medium range
                256: "l", 320: "l",           # large range
                384: "x", 400: "x", 512: "x", # xlarge range
            }
            
            # Find exact or closest match
            if output_ch in scale_mapping:
                scale = scale_mapping[output_ch]
                LOGGER.info(f"Inferred model scale '{scale}' from checkpoint (output channels: {output_ch})")
                return scale
            
            # Find closest match
            closest_scale = "m"
            closest_diff = float("inf")
            for ch, scale in scale_mapping.items():
                diff = abs(output_ch - ch)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_scale = scale
            
            LOGGER.info(f"Inferred model scale '{closest_scale}' from checkpoint (output channels: {output_ch}, nearest: {closest_diff})")
            return closest_scale
    
    LOGGER.debug("Could not infer model scale from checkpoint, using default 'm'")
    return "m"

def _try_load_with_scale(model_class, cfg_file, nc, state_dict, task="detect", inferred_scale="m"):
    """Try loading state_dict with different model scales until shapes match."""
    from hyperimagedetect.utils import ROOT, YAML
    
    # Load the base yaml
    if isinstance(cfg_file, str):
        cfg_file = Path(cfg_file)
    
    if not cfg_file.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_file}")
    
    # Load yaml
    yaml_dict = YAML.load(str(cfg_file))
    
    scales = yaml_dict.get("scales", {})
    if not scales:
        # No scales defined, just create model with base yaml
        LOGGER.warning("No model scales defined in yaml, using base configuration")
        model = model_class(cfg=str(cfg_file), nc=nc, verbose=False)
        return model
    
    # Try scales in order: inferred first, then largest to smallest
    # (larger models are more likely to fit small weights, not vice versa)
    scale_order = [inferred_scale]
    for scale in ["x", "l", "m", "s", "n"]:
        if scale not in scale_order and scale in scales:
            scale_order.append(scale)
    
    errors = {}
    best_model = None
    best_match_count = -1
    best_scale_used = None
    
    for scale in scale_order:
        if scale not in scales:
            continue
            
        try:
            # Create model with this scale
            scale_yaml = yaml_dict.copy()
            scale_yaml["scale"] = scale
            
            model = model_class(cfg=scale_yaml, nc=nc, verbose=False)
            
            # Count matching parameter shapes
            model_state = model.state_dict()
            matches = 0
            mismatches = 0
            
            for key in state_dict.keys():
                if key in model_state:
                    if state_dict[key].shape == model_state[key].shape:
                        matches += 1
                    else:
                        mismatches += 1
            
            total = matches + mismatches
            if total == 0:
                continue
                
            match_ratio = matches / total
            LOGGER.debug(f"Scale '{scale}': {matches}/{total} params match ({match_ratio:.1%})")
            
            # If this is the best match so far, save it
            if matches > best_match_count:
                best_match_count = matches
                best_model = model
                best_scale_used = scale
                
                # If we have perfect/near-perfect match (>95%), stop searching
                if match_ratio > 0.95:
                    LOGGER.info(f"Excellent match found with scale '{scale}' ({match_ratio:.1%})")
                    break
                    
        except Exception as e:
            errors[scale] = str(e)
            LOGGER.debug(f"Failed to load scale '{scale}': {e}")
            continue
    
    if best_model is None:
        available_scales = [s for s in scale_order if s in scales]
        raise RuntimeError(
            f"Could not find matching model scale for checkpoint. "
            f"Tried scales: {', '.join(available_scales)}. "
            f"Errors: {errors}"
        )
    
    match_ratio = best_match_count / (best_match_count + sum(
        1 for key in state_dict.keys() 
        if key in best_model.state_dict() and state_dict[key].shape != best_model.state_dict()[key].shape
    ))
    LOGGER.info(f"Using model scale '{best_scale_used}' with {best_match_count} matching parameters ({match_ratio:.1%})")
    return best_model

def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    ckpt, weight = torch_safe_load(weight)
    # Get training args with defaults
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}
    # If no train_args, try to populate from meta
    if not ckpt.get("train_args"):
        meta = ckpt.get("meta", {})
        if meta:
            args.update({
                "task": meta.get("task", args.get("task", "detect")),
                "nc": meta.get("nc", args.get("nc", 80)),
                "imgsz": meta.get("imgsz", args.get("imgsz", 640)),
            })
    
    ema_or_model = ckpt.get("ema") or ckpt["model"]
    
    # Handle case where checkpoint contains state_dict instead of model object
    if isinstance(ema_or_model, dict):
        LOGGER.warning(
            f"Model in checkpoint '{weight}' is stored as state_dict (weights only). "
            f"Attempting to reconstruct model architecture from metadata..."
        )
        # Extract architecture information from args/metadata
        task = args.get("task", "detect")
        nc = args.get("nc", 80)
        
        # Select appropriate model class based on task
        model_map = {
            "detect": DetectionModel,
            "segment": SegmentationModel,
            "pose": PoseModel,
            "obb": OBBModel,
            "classify": ClassificationModel,
        }
        model_class = model_map.get(task)
        if not model_class:
            raise ValueError(
                f"Unknown task '{task}'. Cannot reconstruct model. "
                f"Supported tasks: {', '.join(model_map.keys())}"
            )
        
        try:
            # Try to create model from args/yaml if available
            cfg = args.get("cfg") or args.get("model")
            if cfg:
                model = model_class(cfg=cfg, nc=nc, verbose=False)
                LOGGER.info(f"Using model configuration from checkpoint args: {cfg}")
            else:
                # Try to infer model scale from checkpoint weights
                inferred_scale = _infer_model_scale_from_checkpoint(ema_or_model, task)
                
                # Try different model scales to find matching architecture
                from hyperimagedetect.utils import ROOT
                cfg_file = ROOT / "cfg" / "models" / "11" / "holo11.yaml"
                if not cfg_file.exists():
                    raise FileNotFoundError(f"Model config not found: {cfg_file}")
                
                LOGGER.info(f"Checkpoint config not found, attempting to match architecture with scale '{inferred_scale}'...")
                model = _try_load_with_scale(
                    model_class, 
                    cfg_file, 
                    nc, 
                    ema_or_model,
                    task=task,
                    inferred_scale=inferred_scale
                )
            
            # Load the state_dict into the reconstructed model
            LOGGER.info(f"Loading state_dict into {task} model with {nc} classes")
            incompatible_keys = model.load_state_dict(ema_or_model, strict=False)
            
            # Log any significant mismatches
            if incompatible_keys.missing_keys:
                missing_count = len(incompatible_keys.missing_keys)
                if missing_count > 10:
                    LOGGER.warning(f"Model has {missing_count} missing keys from checkpoint (may indicate model architecture mismatch)")
            
            if incompatible_keys.unexpected_keys:
                unexpected_count = len(incompatible_keys.unexpected_keys)
                if unexpected_count > 10:
                    LOGGER.debug(f"Checkpoint has {unexpected_count} unexpected keys")
            
            model = model.float()
        except Exception as e:
            raise RuntimeError(
                f"Failed to reconstruct model from state_dict checkpoint. "
                f"Error: {str(e)}\n"
                f"Solution: Please save the model using model.save('filename.pt') "
                f"to create a properly formatted checkpoint with the full model object."
            ) from e
    else:
        model = ema_or_model.float()

    model.args = args
    model.pt_path = weight
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)

    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None

    return model, ckpt

def parse_model(d, ch, verbose=True):
    import ast

    legacy = True
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {Detect, Segment, Pose, OBB, ImagePoolingAttn}
        ):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)

def yaml_model_load(path):
    path = Path(path)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d

def guess_model_scale(model_path):
    try:
        return re.search(r"holo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""

def guess_model_task(model):

    def cfg2task(cfg):
        m = cfg["head"][-1][-2].lower()
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    if isinstance(model, torch.nn.Module):
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, (Segment,)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect,)):
                return "detect"

    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"