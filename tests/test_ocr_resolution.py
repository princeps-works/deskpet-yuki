"""Exercise the worker's OCR input sizing without capturing a desktop window."""

from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from desktop_pet import main
from desktop_pet.vision import ocr


def test_worker_honors_native_and_bounded_ocr_resolution():
    frame = Image.new("RGB", (1362, 260), "black")

    class InputObserved(Exception):
        pass

    for edge, expected in [(0, frame.size), (800, (800, 153)), (1600, frame.size)]:
        observed = []

        def extract(image):
            observed.append(image.size)
            raise InputObserved

        with patch.object(main, "_init_subprocess_scan_runtime"), \
             patch.object(main, "_SUBPROC_MM_RUNTIME", {"settings": SimpleNamespace(ocr_text_max_edge=edge)}), \
             patch.object(main, "_SUBPROC_SCAN_CACHE", {}), \
             patch.object(main, "_subprocess_window_resolver", return_value=None), \
             patch.object(main, "_resolve_vn_capture", return_value=(frame, None, frame, "fixture")), \
             patch.object(main, "_vn_text_is_settling", return_value=False), \
             patch.object(ocr, "get_ocr_runtime_status", return_value=(True, "fixture")), \
             patch.object(ocr, "extract_ocr_result", side_effect=extract):
            try:
                main.run_scan_pipeline_subprocess(0, None, visual_novel_mode=True)
            except InputObserved:
                pass
        assert observed == [expected], (edge, observed, expected)
