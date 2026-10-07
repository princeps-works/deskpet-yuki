"""Screenshot-anchored OCR exclusion for the verified Sakura dialogue layout."""

from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


def _edges(image):
    gray = np.asarray(image.convert("L"), dtype=np.int16)
    result = np.zeros(gray.shape, dtype=bool)
    result[:, 1:] |= np.abs(gray[:, 1:] - gray[:, :-1]) > 25
    result[1:, :] |= np.abs(gray[1:, :] - gray[:-1, :]) > 25
    return result


def _nearby(edges):
    padded = np.pad(edges, 1)
    height, width = edges.shape
    result = np.zeros_like(edges)
    for y in range(3):
        for x in range(3):
            result |= padded[y:y + height, x:x + width]
    return result


@lru_cache(maxsize=1)
def _profile():
    directory = Path(__file__).resolve().parent.parent / "config"
    profile = json.loads((directory / "vn_ui_regions.json").read_text(encoding="utf-8"))
    with Image.open(directory / profile["anchor_image"]) as image:
        reference = image.convert("RGB")
    return profile, reference


def mask_visual_novel_ui(image, *, title="", process_name=""):
    """Keep the original image unless game identity, shape and anchor agree.

    Mask before cropping/resizing so the same relative coordinates serve every
    OCR variant. The unmodified screenshot remains available to visual context.
    """
    started_at = time.perf_counter()
    profile, reference = _profile()
    if process_name.casefold() != profile["process_name"].casefold() or not title.startswith(profile["title_prefix"]):
        return image
    width, height = image.size
    if abs(width / max(1, height) / profile["aspect_ratio"] - 1) > 0.02:
        return image
    anchor_box = tuple(round(value * size) for value, size in zip(profile["anchor_box"], (width, height, width, height)))
    patch = image.crop(anchor_box)
    if min(patch.size) < 16:
        return image
    compare_size = tuple(min(a, b) for a, b in zip(reference.size, patch.size))
    expected = _edges(reference.resize(compare_size, Image.Resampling.BILINEAR))
    observed = _edges(patch.resize(compare_size, Image.Resampling.BILINEAR))
    if not expected.any() or not observed.any():
        return image
    recall = float((expected & _nearby(observed)).sum()) / int(expected.sum())
    precision = float((observed & _nearby(expected)).sum()) / int(observed.sum())
    if recall < 0.80 or precision < 0.65:
        return image
    box = tuple(round(value * size) for value, size in zip(profile["exclude_box"], (width, height, width, height)))
    masked = image.copy()
    ImageDraw.Draw(masked).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=0)
    masked.info["vn_ui_mask"] = "sakura"
    masked.info["vn_ui_mask_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    return masked
