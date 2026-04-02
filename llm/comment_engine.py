from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, Tuple

from desktop_pet.config.prompts import get_system_screen_comment_prompt
from desktop_pet.llm.client import LLMClient


class CommentEngine:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        tutor_enabled: bool = False,
        style_weights_text: str = "",
        stressed_keywords_text: str = "",
        positive_keywords_text: str = "",
        focused_keywords_text: str = "",
    ) -> None:
        self._client = llm_client
        self._tutor_enabled = bool(tutor_enabled)
        self._style_rng = random.Random()
        self._last_style_name = ""
        self._style_base_weights = self._parse_style_weights(style_weights_text)
        self._stressed_words = self._parse_keywords(
            stressed_keywords_text,
            [
                "烦", "崩溃", "压力", "焦虑", "累", "卡住", "不会", "好难", "deadline", "bug",
                "报错", "错误", "失败", "加班", "熬夜", "头疼", "麻了",
            ],
        )
        self._positive_words = self._parse_keywords(
            positive_keywords_text,
            [
                "哈哈", "开心", "搞定", "完成", "顺利", "不错", "太好了", "舒服", "进步", "通过",
                "成功", "耶", "轻松", "满意",
            ],
        )
        self._focused_words = self._parse_keywords(
            focused_keywords_text,
            [
                "学习", "复习", "写作业", "刷题", "阅读", "写代码", "调试", "文档", "论文", "做题",
                "专注", "计划", "总结", "记笔记",
            ],
        )

    def set_tutor_enabled(self, enabled: bool) -> None:
        self._tutor_enabled = bool(enabled)

    def _parse_keywords(self, text: str, default_words: list[str]) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return list(default_words)
        words = [item.strip().lower() for item in raw.split(",") if item.strip()]
        return words or list(default_words)

    def _parse_style_weights(self, text: str) -> Dict[str, float]:
        defaults = {
            "陪伴评论": 1.0,
            "轻松提问": 1.2,
            "俏皮打趣": 1.0,
            "温柔锐评": 0.8,
            "行动建议": 1.0,
        }
        raw = str(text or "").strip()
        if not raw:
            return defaults

        parsed = dict(defaults)
        for pair in raw.split(","):
            item = pair.strip()
            if not item or ":" not in item:
                continue
            key, value = item.split(":", 1)
            name = key.strip()
            if name not in parsed:
                continue
            try:
                weight = float(value.strip())
            except Exception:
                continue
            parsed[name] = max(0.0, min(5.0, weight))
        return parsed

    def _infer_emotion_context(self, recent_dialog_hint: str, screen_summary: str) -> str:
        text = (recent_dialog_hint + "\n" + screen_summary).lower()

        stressed_score = sum(1 for w in self._stressed_words if w in text)
        positive_score = sum(1 for w in self._positive_words if w in text)
        focused_score = sum(1 for w in self._focused_words if w in text)

        if stressed_score >= max(2, positive_score + 1):
            return "stressed"
        if positive_score >= max(2, stressed_score + 1):
            return "positive"
        if focused_score >= 2:
            return "focused"
        return "neutral"

    def _next_style(self, emotion: str) -> Tuple[str, str]:
        styles = [
            (
                "陪伴评论",
                "像妹妹在旁边陪着哥哥，给有温度的观察。",
                float(self._style_base_weights.get("陪伴评论", 1.0)),
            ),
            (
                "轻松提问",
                "提出一个轻量问题，引导哥哥继续表达或思考。",
                float(self._style_base_weights.get("轻松提问", 1.2)),
            ),
            (
                "俏皮打趣",
                "用不刻薄的玩笑语气，轻松调侃当前状态。",
                float(self._style_base_weights.get("俏皮打趣", 1.0)),
            ),
            (
                "温柔锐评",
                "点出一个明显问题或习惯，但保持关心和分寸。",
                float(self._style_base_weights.get("温柔锐评", 0.8)),
            ),
            (
                "行动建议",
                "给一个可以马上执行的小建议，避免说教。",
                float(self._style_base_weights.get("行动建议", 1.0)),
            ),
        ]

        emotion_multipliers: dict[str, dict[str, float]] = {
            "stressed": {
                "陪伴评论": 1.5,
                "轻松提问": 0.9,
                "俏皮打趣": 0.6,
                "温柔锐评": 0.55,
                "行动建议": 1.45,
            },
            "positive": {
                "陪伴评论": 0.95,
                "轻松提问": 1.15,
                "俏皮打趣": 1.55,
                "温柔锐评": 0.75,
                "行动建议": 1.0,
            },
            "focused": {
                "陪伴评论": 0.95,
                "轻松提问": 1.25,
                "俏皮打趣": 0.75,
                "温柔锐评": 0.95,
                "行动建议": 1.4,
            },
            "neutral": {
                "陪伴评论": 1.0,
                "轻松提问": 1.0,
                "俏皮打趣": 1.0,
                "温柔锐评": 1.0,
                "行动建议": 1.0,
            },
        }
        multiplier = emotion_multipliers.get(emotion, emotion_multipliers["neutral"])
        adjusted_styles = [
            (name, instruction, float(base_weight) * float(multiplier.get(name, 1.0)))
            for name, instruction, base_weight in styles
        ]

        # Randomized style selection with anti-repeat to avoid mechanical cycling.
        candidates = [item for item in adjusted_styles if item[0] != self._last_style_name]
        if not candidates:
            candidates = adjusted_styles

        total = sum(float(item[2]) for item in candidates)
        if total <= 0:
            picked = candidates[0]
        else:
            ticket = self._style_rng.random() * total
            acc = 0.0
            picked = candidates[-1]
            for item in candidates:
                acc += float(item[2])
                if ticket <= acc:
                    picked = item
                    break

        self._last_style_name = picked[0]
        return picked[0], picked[1]

    def comment_on_summary(
        self,
        screen_summary: str,
        long_memory_hint: str = "",
        memory_weight: float = 0.2,
        recent_dialog_hint: str = "",
    ) -> str:
        weight = max(0.0, min(1.0, memory_weight))
        emotion = self._infer_emotion_context(recent_dialog_hint, screen_summary)
        style_name, style_instruction = self._next_style(emotion)
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [
            "你正在生成一条自动互动话术。",
            f"当前时间: {now_text}",
            f"当前情绪上下文: {emotion}",
            f"本次风格: {style_name}",
            f"风格要求: {style_instruction}",
            f"近期扫屏信息: {screen_summary}",
            f"说明：下面的长期记忆仅作低权重参考（建议权重{weight:.2f}），优先依据当前屏幕摘要。",
        ]
        if recent_dialog_hint.strip():
            parts.append(recent_dialog_hint.strip())
        if long_memory_hint.strip():
            parts.append(long_memory_hint.strip())
        parts.append(
            "输出要求：\n"
            "1) 以妹妹对哥哥说话的方式，输出1到2句。\n"
            "2) 可以是评论、提问、打趣、温柔锐评或小建议，不要每次都用同一种句式。\n"
            "3) 结合扫屏与历史对话，不要只复述‘哥哥在做什么’。\n"
            "4) 不要太长，尽量控制在60字以内。\n"
            "5) 不输出解释、标签或括号备注。"
        )
        prompt = "\n".join(parts)
        system_prompt = get_system_screen_comment_prompt(tutor_enabled=self._tutor_enabled)
        return self._client.chat(user_text=prompt, system_prompt=system_prompt)
