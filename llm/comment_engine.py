from __future__ import annotations

from desktop_pet.config.prompts import get_system_screen_comment_prompt
from desktop_pet.llm.client import LLMClient


class CommentEngine:
    def __init__(self, llm_client: LLMClient, *, tutor_enabled: bool = False) -> None:
        self._client = llm_client
        self._tutor_enabled = bool(tutor_enabled)

    def set_tutor_enabled(self, enabled: bool) -> None:
        self._tutor_enabled = bool(enabled)

    def comment_on_summary(
        self,
        screen_summary: str,
        long_memory_hint: str = "",
        memory_weight: float = 0.2,
    ) -> str:
        weight = max(0.0, min(1.0, memory_weight))
        parts = [
            f"屏幕摘要: {screen_summary}",
            f"说明：下面的长期记忆仅作低权重参考（建议权重{weight:.2f}），优先依据当前屏幕摘要。",
        ]
        if long_memory_hint.strip():
            parts.append(long_memory_hint.strip())
        parts.append("请你以妹妹对哥哥说话的方式，给一句不超过25字的评论。")
        prompt = "\n".join(parts)
        system_prompt = get_system_screen_comment_prompt(tutor_enabled=self._tutor_enabled)
        return self._client.chat(user_text=prompt, system_prompt=system_prompt)
