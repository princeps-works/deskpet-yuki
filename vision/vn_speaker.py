"""Name-box correction for the verified 16:9 Camus dialogue layout."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from desktop_pet.vision.ocr import OCRLine, _result_from_lines


NAME_BOX = (0.208, 0.750, 0.258, 0.807)


def _edges(image):
    gray = np.asarray(image.convert('L').resize((70, 45)))
    return cv2.Canny(gray, 100, 200)


@lru_cache(maxsize=1)
def _rin_templates():
    templates = []
    for name in ('camus_rin_name.png', 'camus_rin_name_large.png'):
        with Image.open(Path(__file__).resolve().parent.parent / 'config' / name) as image:
            templates.append(_edges(image)[3:36, 12:45])
    return templates


def correct_camus_speaker(result, frame, *, title, process_name, text_ratio, bottom_margin):
    """Match the confirmed 燐 glyph, never substitute names in dialogue text.

    The stock recognizer has no 燐 in its alphabet. Keep its name reading in
    the returned diagnostic note; unfamiliar glyphs stay untouched.
    """
    if (frame is None or not title.startswith('青空下的加缪')
            or process_name.casefold() != '青い空のカミュ.exe'.casefold()):
        return result, ''
    width, height = frame.size
    if abs(width / height / (16 / 9) - 1) > .02:
        return result, ''
    top = round(height * max(.2, min(.95, text_ratio)))
    bottom = round(height * (1 - max(0, min(.4, bottom_margin))))
    left, y1, right, y2 = [round(v * s) for v, s in zip(NAME_BOX, (width, height, width, height))]
    if top > y1 or bottom < y2 or bottom - top < 24:
        return result, ''
    patch = frame.crop((left, y1, right, y2))
    observed = _edges(patch)
    score = max(float(cv2.minMaxLoc(cv2.matchTemplate(observed, template, cv2.TM_CCOEFF_NORMED))[1])
                for template in _rin_templates())
    if score < .40:
        return result, ''
    box = (left / width, (y1 - top) / (bottom - top), right / width, (y2 - top) / (bottom - top))
    def is_name(line):
        if line.box is None or len(line.text.strip()) > 3:
            return False
        x1, t, x2, b = line.box
        return box[0] <= (x1 + x2) / 2 <= box[2] and box[1] <= (t + b) / 2 <= box[3]
    names = [line for line in result.lines if is_name(line)]
    dialogue = [line for line in result.lines if not is_name(line)]
    if not any(len(line.text.strip()) >= 4 for line in dialogue):
        return result, ''
    raw = '|'.join(line.text for line in names)
    corrected = _result_from_lines([OCRLine('燐', score, box), *dialogue])
    return corrected, f'name_roi=燐;raw_name={raw or "<empty>"};glyph_match={score:.3f}'
