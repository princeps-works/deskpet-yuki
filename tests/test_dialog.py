from desktop_pet.llm.dialog_manager import DialogManager


class _FakeClient:
    def chat(self, user_text: str, system_prompt: str) -> str:
        return f"ok:{user_text[:10]}"


def test_dialog_reply():
    mgr = DialogManager(_FakeClient())
    assert mgr.reply("hello").startswith("ok:")
