from types import SimpleNamespace
from pathlib import Path
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ast
import ctypes
from ctypes import wintypes
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtWidgets import QApplication, QPushButton
from desktop_pet.ui.vn_sidebar import VnSidebar
from desktop_pet.ui.pet_window import DesktopPet
import desktop_pet.main as main_module

_app = None

def _check_menu():
    global _app
    _app = QApplication.instance() or QApplication([])
    settings = SimpleNamespace(vision_capture_mode="window", vision_shots_per_minute=3,
        vision_change_threshold=.1, visual_novel_text_ratio=.58, vision_window_margin_px=0)
    panel = VnSidebar(settings)
    try:
        assert [b.text() for b in panel.pages.widget(0).findChildren(QPushButton)] == ["扫描设置", "记忆设置", "剧情缓存"]
        events = []
        panel.select_story_requested.connect(lambda: events.append("story"))
        panel.add_memory_requested.connect(lambda value: events.append(value))
        panel.story_button.click()
        assert events == ["story"]
        panel.memory_button.click()
        assert panel.pages.currentIndex() == 2
        texts = {b.text() for b in panel.pages.currentWidget().findChildren(QPushButton)}
        assert {"新增记忆", "清空记忆缓存", "查看详细记忆文件", "重新载入"} <= texts
        panel.memory_input.setText("  惠麻想学吉他  ")
        panel._on_add_memory()
        assert events[-1] == "惠麻想学吉他"
        panel._show_page(0)
        panel.scan_button.click()
        assert panel.pages.currentIndex() == 1
        texts = {b.text() for b in panel.pages.currentWidget().findChildren(QPushButton)}
        assert {"锁定此窗口", "框选文本区", "框选画面区域", "试读一次文本", "清区域", "选区域"} <= texts
    finally:
        panel.close()
        panel.deleteLater()


def test_hover_anchor_only_covers_visual_novel_button_on_negative_origin_screen():
    button = SimpleNamespace(isVisible=lambda: True, mapToGlobal=lambda p: QPoint(-1850, 450), size=lambda: QRect(0,0,88,30).size())
    host = SimpleNamespace(btn_visual_novel=button)
    anchor = DesktopPet._vn_panel_anchor_rect(host)
    assert anchor.contains(QPoint(-1810,465))
    assert not anchor.contains(QPoint(-1810,510))


def test_native_model_rect_preserves_negative_origin_and_dpi_scale():
    source = Path(main_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_live2d_target_geometry_native")
    for scale in [1,1.5]:
        host = SimpleNamespace(width=lambda: 380,height=lambda:430,get_live2d_py_target_geometry=lambda:(-1800,300,280,430))
        class User32:
            def GetWindowRect(self, hwnd, ptr):
                r = ctypes.cast(ptr,ctypes.POINTER(wintypes.RECT)).contents
                r.left,r.top = -2560,200
                r.right,r.bottom = -2560+round(380*scale),200+round(430*scale)
                return 1
        env = {"pet":host,"_WinRect":wintypes.RECT,"ctypes":ctypes}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"native_geometry","exec"),env)
        assert env[fn.name](User32(),123) == (-2560,200,round(280*scale),round(430*scale))


def test_menu_has_three_entries_and_memory_and_scan_operations_survive_navigation():
    result = subprocess.run([sys.executable, __file__, "--gui"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hover_does_not_resurrect_removed_cache_shortcut_over_chat_button():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QWidget
    state = {}
    button = lambda name: SimpleNamespace(setVisible=lambda visible: state.update({name:visible}))
    panel = SimpleNamespace(control_bar=SimpleNamespace(setVisible=lambda v:None,setStyleSheet=lambda v:None),
        control_border=None,btn_chat=button("chat"),btn_comment=None,btn_toggle_scan=None,
        btn_visual_novel=None,btn_story_cache=button("story"),btn_select_region=button("select"),
        btn_clear_region=button("clear"),btn_tts=None,btn_gaze=None,btn_quit=None,
        visual_novel_mode_enabled=True,_controls_style_visible="",_controls_style_hidden="",
        _controls_hide_timer=SimpleNamespace(stop=lambda:None))
    DesktopPet._set_controls_visible(panel,True)
    assert state == {"chat":True,"story":False,"select":False,"clear":False}
    panel.visual_novel_mode_enabled=False
    DesktopPet._set_controls_visible(panel,True)
    assert state["select"] and state["clear"]

if __name__ == "__main__":
    _check_menu()
