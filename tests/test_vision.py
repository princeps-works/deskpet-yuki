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


def test_vision_failure_records_type_and_status_without_error_body():
    class Failure(RuntimeError):
        status_code = 429
    class Client:
        def describe_image(self, *_args):
            raise Failure("sensitive provider response")
    result = describe_screen_image_compat(Client(), Image.new("RGB", (64, 64)), timeout_sec=1, max_edge=320)
    assert result.reason == "error:Failure:status=429"
    assert "sensitive" not in result.reason
    assert result.summary == ""


def test_vision_timeout_and_empty_response_remain_distinct():
    import threading
    event = threading.Event()
    class Client:
        def describe_image(self, *_args):
            event.wait(2)
            return ""
    try:
        result = describe_screen_image_compat(Client(), Image.new("RGB", (64, 64)), timeout_sec=0.5, max_edge=320)
        assert result.reason == "timeout"
    finally:
        event.set()
    result = describe_screen_image_compat(Client(), Image.new("RGB", (64, 64)), timeout_sec=1, max_edge=320)
    assert result.reason == "empty"


def test_fast_vision_only_applies_to_deepseek_visual_novel_requests():
    from types import SimpleNamespace
    class Client:
        def __init__(self, model):
            self.settings = SimpleNamespace(vision_model_name=model)
            self.options = []
        def describe_image(self, *_args, **kwargs):
            self.options.append(kwargs)
            return "观察结果"
    for model, kind, expected in (("deepseek-v4-flash-vision-exp", "visual_novel", {"disable_thinking": True}),
                                  ("deepseek-v4-flash-vision-exp", "uploaded_image", {}),
                                  ("other-model", "visual_novel", {})):
        client = Client(model)
        result = describe_screen_image_compat(client, Image.new("RGB", (64, 64)),
                                             timeout_sec=1, max_edge=320, content_kind=kind)
        assert result.reason == "ok" and client.options == [expected]


def test_multimodal_client_forwards_fast_reasoning_only_when_requested():
    from types import SimpleNamespace
    from desktop_pet.llm.client import LLMClient
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="观察结果"))])
    client = LLMClient.__new__(LLMClient)
    client.settings = SimpleNamespace(vision_model_name="deepseek-v4-flash-vision-exp", mm_output_max_tokens=1024, mm_timeout_sec=20)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.describe_image(Image.new("RGB", (64, 64)), "测试", "视觉规则")
    client.describe_image(Image.new("RGB", (64, 64)), "测试", "视觉规则", disable_thinking=True)
    assert "extra_body" not in requests[0]
    assert requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}


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
