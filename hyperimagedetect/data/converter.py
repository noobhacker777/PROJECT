
from __future__ import annotations

import asyncio
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hyperimagedetect.utils import ASSETS_URL, DATASETS_DIR, LOGGER, NUM_THREADS, TQDM, YAML
from hyperimagedetect.utils.checks import check_file, check_requirements
from hyperimagedetect.utils.files import increment_path

def download(*args, **kwargs):
    raise RuntimeError("❌ OFFLINE MODE: Cannot download. Use local files only.")

def zip_directory(*args, **kwargs):
    raise RuntimeError("❌ OFFLINE MODE: Cannot download. Use local files only.")

def coco91_to_coco80_class() -> list[int]:
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]

def coco80_to_coco91_class() -> list[int]:
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

def convert_coco(
    labels_dir: str = "../coco/annotations/",
    save_dir: str = "coco_converted/",
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
    lvis: bool = False,
):
    save_dir = increment_path(save_dir)
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)

    coco80 = coco91_to_coco80_class()

    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        images = {f"{x['id']:d}": x for x in data["images"]}
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        image_txt = []
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else img["file_name"]
            if lvis:
                image_txt.append(str(Path("./images") / f))

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls, *s]
                        segments.append(s)
                    if use_keypoints and ann.get("keypoints") is not None:
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )

            with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                        )
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        if lvis:
            filename = Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")
            with open(filename, "a", encoding="utf-8") as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")

def min_index(arr1: np.ndarray, arr2: np.ndarray):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments: list[list]):
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    for k in range(2):

        if k == 0:
            for i, idx in enumerate(idx_list):

                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])

                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def holo_bbox2segment(im_dir: str | Path, save_dir: str | Path | None = None, sam_model: str = "sam_b.pt", device=None):
    from hyperimagedetect.data import HOLODataset
    from hyperimagedetect.utils.ops import xywh2xyxy

    dataset = HOLODataset(im_dir, data=dict(names=list(range(1000)), channels=3))
    if len(dataset.labels[0]["segments"]) > 0:
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label["im_file"])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        label["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            if len(s) == 0:
                continue
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"Generated segment labels saved in {save_dir}")

def create_synthetic_coco_dataset():

    def create_synthetic_image(image_file: Path):
        if not image_file.exists():
            size = (random.randint(480, 640), random.randint(480, 640))
            Image.new(
                "RGB",
                size=size,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ).save(image_file)

    dir = DATASETS_DIR / "coco"

    shutil.rmtree(dir / "labels" / "test2017", ignore_errors=True)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for subset in {"train2017", "val2017"}:
            subset_dir = dir / "images" / subset
            subset_dir.mkdir(parents=True, exist_ok=True)

            label_list_file = dir / f"{subset}.txt"
            if label_list_file.exists():
                with open(label_list_file, encoding="utf-8") as f:
                    image_files = [dir / line.strip() for line in f]

                futures = [executor.submit(create_synthetic_image, image_file) for image_file in image_files]
                for _ in TQDM(as_completed(futures), total=len(futures), desc=f"Generating images for {subset}"):
                    pass
            else:
                LOGGER.warning(f"Labels file {label_list_file} does not exist. Skipping image creation for {subset}.")

    LOGGER.info("Synthetic COCO dataset created successfully.")

def convert_to_multispectral(path: str | Path, n_channels: int = 10, replace: bool = False, zip: bool = False):
    from scipy.interpolate import interp1d

    from hyperimagedetect.data.utils import IMG_FORMATS

    path = Path(path)
    if path.is_dir():
        im_files = [f for ext in (IMG_FORMATS - {"tif", "tiff"}) for f in path.rglob(f"*.{ext}")]
        for im_path in im_files:
            try:
                convert_to_multispectral(im_path, n_channels)
                if replace:
                    im_path.unlink()
            except Exception as e:
                LOGGER.info(f"Error converting {im_path}: {e}")

        if zip:
            zip_directory(path)
    else:
        output_path = path.with_suffix(".tiff")
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        rgb_wavelengths = np.array([650, 510, 475])
        target_wavelengths = np.linspace(450, 700, n_channels)
        f = interp1d(rgb_wavelengths.T, img, kind="linear", bounds_error=False, fill_value="extrapolate")
        multispectral = f(target_wavelengths)
        cv2.imwritemulti(str(output_path), np.clip(multispectral, 0, 255).astype(np.uint8).transpose(2, 0, 1))
        LOGGER.info(f"Converted {output_path}")