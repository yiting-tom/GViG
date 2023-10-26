import base64
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image


def url_to_base64(url: str, file_dir: Path):
    filename = url.split("/")[-1]
    filepath = file_dir / filename
    if filepath.exists():
        return filepath_to_base64(filepath)
    else:
        raise FileNotFoundError(f"{filepath} does not exist")


def filepath_to_base64(filepath: Union[str, Path]) -> str:
    img = Image.open(filepath)  # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str


def pil_image_to_base64(img: Image) -> str:
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str


def np_to_base64(img: np.ndarray) -> str:
    img = Image.fromarray(img)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_pil_image(base64_str: str) -> Image:
    img_bytes = base64.b64decode(base64_str)
    img_buffer = BytesIO(img_bytes)
    img = Image.open(img_buffer)
    return img


def base64_to_np(base64_str: str) -> np.ndarray:
    img_data = base64.b64decode(base64_str)
    nparr = np.fromstring(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1]


def wsdm_bbox_ratio_to_abs(data: dict):
    whwh = torch.tensor([data["width"], data["height"], data["width"], data["height"]])
    return data["bbox"].tensor / whwh


def wsdm_bbox_abs_to_ratio(data: dict):
    whwh = torch.tensor([data["width"], data["height"], data["width"], data["height"]])
    return (data["bbox"].tensor * whwh).int()


def bbox_abs_to_ratio(bbox: torch.Tensor, whwh: torch.Tensor):
    return bbox / whwh


def bbox_ratio_to_abs(bbox: torch.Tensor, whwh: torch.Tensor):
    return bbox * whwh
