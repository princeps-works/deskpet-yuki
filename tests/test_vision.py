from desktop_pet.vision.scene_analyzer import analyze_scene


def test_analyze_scene_non_empty():
    out = analyze_scene("hello")
    assert out.should_comment
    assert "屏幕内容摘要" in out.summary
