from desktop_pet.vision.scene_analyzer import analyze_scene
from desktop_pet.vision.multimodal import describe_screen_image_compat
from PIL import Image


def test_analyze_scene_non_empty():
    out = analyze_scene("hello world")
    assert out.should_comment
    assert "屏幕内容摘要" in out.summary


def test_analyze_scene_ignores_tiny_context():
    out = analyze_scene("hello")
    assert not out.should_comment


def test_uploaded_image_uses_ocr_plus_vision_prompt():
    class _FakeVisionClient:
        def __init__(self):
            self.prompt = ""

        def describe_image(self, _image, user_text: str, system_prompt: str):
            self.prompt = user_text
            assert "视觉理解助手" in system_prompt
            return "检测到两个人物"

    client = _FakeVisionClient()
    result = describe_screen_image_compat(
        client,
        Image.new("RGB", (64, 64), "white"),
        timeout_sec=1.0,
        max_edge=1024,
        ocr_context="示例文字",
        route="vision+ocr",
        focus_query="图里有几个人？",
        content_kind="uploaded_image",
    )
    assert result.reason == "ok"
    assert result.summary == "检测到两个人物"
    assert "用户上传图片" in client.prompt
    assert "示例文字" in client.prompt
    assert "人物或物体数量" in client.prompt
