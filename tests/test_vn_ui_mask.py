"""Anchored masking must leave names/dialogue and uncertain layouts intact."""

from pathlib import Path

from PIL import Image

from desktop_pet.vision.vn_ui_mask import _profile, mask_visual_novel_ui


_TITLE = "樱花、萌放 -as the Night's, Reincarnation- 文本版本-test"


def _frame(color=(30, 40, 60), size=(1280, 720)):
    profile, reference = _profile()
    frame = Image.new("RGB", size, color)
    box = tuple(round(value * length) for value, length in zip(profile["anchor_box"], (*size, *size)))
    frame.paste(reference.resize((box[2] - box[0], box[3] - box[1])), box[:2])
    return frame


def test_ui_mask_preserves_names_dialogue_and_original_screenshot():
    frame = _frame()
    before = frame.tobytes()
    masked = mask_visual_novel_ui(frame, title=_TITLE, process_name="Sakura.exe")
    assert masked.info.get("vn_ui_mask") == "sakura"
    assert masked.getpixel((1200, 600)) == (0, 0, 0)
    assert masked.getpixel((150, 600)) == frame.getpixel((150, 600))  # Speaker.
    assert masked.getpixel((1000, 640)) == frame.getpixel((1000, 640))  # Dialogue.
    assert frame.tobytes() == before


def test_ui_mask_reuses_anchor_on_different_backgrounds_and_scales():
    for color in ((0, 0, 0), (250, 250, 250), (170, 80, 120)):
        for size in ((1280, 720), (1920, 1080), (2560, 1440)):
            masked = mask_visual_novel_ui(_frame(color, size), title=_TITLE, process_name="Sakura.exe")
            assert masked.info.get("vn_ui_mask") == "sakura"


def test_ui_mask_keeps_unknown_games_menus_and_changed_aspect_ratio():
    frame = _frame()
    assert mask_visual_novel_ui(frame, title="另一个游戏", process_name="Sakura.exe") is frame
    assert mask_visual_novel_ui(frame, title=_TITLE, process_name="Other.exe") is frame
    menu = Image.new("RGB", frame.size, (250, 250, 250))
    assert mask_visual_novel_ui(menu, title=_TITLE, process_name="Sakura.exe") is menu
    resized = frame.resize((1280, 800))
    assert mask_visual_novel_ui(resized, title=_TITLE, process_name="Sakura.exe") is resized


def test_ui_mask_matches_button_with_reduced_contrast_on_bright_cg():
    profile, reference = _profile()
    frame = _frame((235, 220, 240), (2560, 1440))
    box = tuple(round(value * length) for value, length in zip(profile["anchor_box"], (*frame.size, *frame.size)))
    bright = Image.blend(reference, Image.new("RGB", reference.size, "white"), 0.55)
    frame.paste(bright, box[:2])
    assert mask_visual_novel_ui(frame, title=_TITLE, process_name="Sakura.exe").info.get("vn_ui_mask") == "sakura"


def test_ui_mask_coordinates_survive_crop_resize_and_padding():
    from desktop_pet.main import _crop_text_band, _ocr_text_variants, _resize_for_ocr

    frame = _frame()
    masked = mask_visual_novel_ui(frame, title=_TITLE, process_name="Sakura.exe")
    crop = _crop_text_band(masked, 0.58, bottom_margin=0.08)
    small = _resize_for_ocr(crop, 800)
    assert small.getpixel((750, 100)) == (0, 0, 0)
    assert crop.info["vn_ui_mask"] == "sakura"
    assert all(variant.size[0] >= small.size[0] for variant in _ocr_text_variants(small))


def test_both_capture_paths_share_mask_and_fallback_uses_it():
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    assert "_crop_region(ocr_frame, interior_region)" in source
    assert "_crop_text_band(ocr_frame, text_ratio" in source
    assert "fallback_frame = mask_visual_novel_ui(" in source
