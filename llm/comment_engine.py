from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from typing import Callable, Dict, Tuple

import numpy as np

from desktop_pet.config.prompts import (
    get_persona_moment_criteria,
    get_system_screen_comment_prompt,
    get_system_visual_novel_prompt,
)
from desktop_pet.llm.client import LLMClient


# Moment type -> tone hint. The hint is a *suggestion* the model may ignore,
# unlike the previous mandatory style rotation which produced stiff replies
# whenever the drawn style fought the scene.
_MOMENT_TONE_HINTS: dict[str, str] = {
    "humor": "这一刻挺好笑，可以跟着乐一下，吐槽也完全可以。",
    "tender": "这一刻偏温柔。安静地陪着就好，短一点，别说教。",
    "twist": "这里出乎意料，可以表达惊讶或好奇，想问就问。",
    "choice": "哥哥刚做了选择。看着就好，不要评判他选得对不对。",
    "tension": "气氛有点紧。陪着紧张就行，别打断。",
    "other": "",
}

# Opening patterns that make consecutive comments feel mechanical.
_SENTENCE_OPENERS = (
    "哥哥",
    "我觉得",
    "感觉",
    "这个",
    "这里",
    "不过",
    "话说",
    "诶",
    "哇",
    "嗯",
)


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
        enable_api_understanding: bool = False,
        embeddings_fn: Callable[[list[str]], np.ndarray | None] | None = None,
    ) -> None:
        self._client = llm_client
        self._tutor_enabled = bool(tutor_enabled)
        self._style_rng = random.Random()
        self._last_style_name = ""
        self._enable_api_understanding = bool(enable_api_understanding)
        self._embeddings_fn = embeddings_fn
        self._stable_emotion = "neutral"
        self._emotion_candidate = ""
        self._emotion_candidate_streak = 0
        self._neutral_exit_streak = 0
        self._recent_generated_comments: list[str] = []
        self.last_visual_novel_error = ""
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

    def reset_scene_state(self) -> None:
        self.last_visual_novel_error = ""
        self._stable_emotion = "neutral"
        self._emotion_candidate = ""
        self._emotion_candidate_streak = 0
        self._neutral_exit_streak = 0
        self._recent_generated_comments.clear()
        self._last_style_name = ""

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

    def _smooth_emotion(self, raw_emotion: str) -> str:
        raw = str(raw_emotion or "neutral").strip().lower() or "neutral"
        if raw not in {"stressed", "positive", "focused", "neutral"}:
            raw = "neutral"

        stable = str(self._stable_emotion or "neutral")
        if raw == stable:
            self._emotion_candidate = ""
            self._emotion_candidate_streak = 0
            self._neutral_exit_streak = 0
            return stable

        if raw == "neutral":
            self._neutral_exit_streak += 1
            if self._neutral_exit_streak >= 3:
                self._stable_emotion = "neutral"
                self._emotion_candidate = ""
                self._emotion_candidate_streak = 0
            return str(self._stable_emotion)

        self._neutral_exit_streak = 0
        if self._emotion_candidate == raw:
            self._emotion_candidate_streak += 1
        else:
            self._emotion_candidate = raw
            self._emotion_candidate_streak = 1

        if self._emotion_candidate_streak >= 2:
            self._stable_emotion = raw
            self._emotion_candidate = ""
            self._emotion_candidate_streak = 0
        return str(self._stable_emotion)

    @staticmethod
    def _token_set(text: str) -> set[str]:
        raw = str(text or "").lower()
        tokens = re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,8}", raw)
        return {t.strip() for t in tokens if t.strip()}

    def _build_relevant_memory_hint(
        self,
        long_memory_hint: str,
        normalized_summary: str,
        recent_dialog_hint: str,
    ) -> str:
        text = str(long_memory_hint or "").strip()
        if not text:
            return ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        entries: list[str] = []
        for ln in lines:
            if ln.startswith("-"):
                item = ln.lstrip("-").strip()
                if item:
                    entries.append(item)
        if not entries:
            return ""

        focus_tokens = self._token_set(normalized_summary + "\n" + recent_dialog_hint)
        if not focus_tokens:
            return ""

        scored: list[tuple[float, str]] = []
        for item in entries:
            item_tokens = self._token_set(item)
            if not item_tokens:
                continue
            overlap = len(item_tokens & focus_tokens) / max(1, min(12, len(item_tokens)))
            scored.append((overlap, item))

        if not scored:
            return ""
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [item for score, item in scored if score >= 0.12][:2]
        if not picked:
            return ""

        return "长期记忆（相关性筛选后）:\n" + "\n".join(f"- {x}" for x in picked)

    def _has_carephrase_loop(self, text: str) -> bool:
        out = str(text or "")
        if not out:
            return False
        hot_words = ["休息", "别太累", "早点", "喝水", "注意身体", "辛苦", "熬夜"]
        recent = self._recent_generated_comments[-6:]
        if not recent:
            return False
        repeat_hits = 0
        for word in hot_words:
            if word in out:
                prev_hits = sum(1 for item in recent if word in item)
                if prev_hits >= 2:
                    repeat_hits += 1
        return repeat_hits >= 1

    def _record_generated_comment(self, text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        self._recent_generated_comments.append(line)
        if len(self._recent_generated_comments) > 12:
            del self._recent_generated_comments[:-12]

    def _recent_carephrase_pressure(self) -> float:
        recent = self._recent_generated_comments[-6:]
        if not recent:
            return 0.0
        hot_words = ["休息", "别太累", "早点", "喝水", "注意身体", "辛苦", "熬夜"]
        hits = 0
        for item in recent:
            if any(w in item for w in hot_words):
                hits += 1
        return hits / max(1, len(recent))

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

        care_pressure = self._recent_carephrase_pressure()
        if care_pressure >= 0.50:
            tuned: list[tuple[str, str, float]] = []
            for name, instruction, weight in adjusted_styles:
                tuned_weight = float(weight)
                if name == "行动建议":
                    tuned_weight *= 0.45
                elif name == "陪伴评论":
                    tuned_weight *= 0.65
                elif name == "轻松提问":
                    tuned_weight *= 1.20
                elif name == "俏皮打趣":
                    tuned_weight *= 1.10
                tuned.append((name, instruction, tuned_weight))
            adjusted_styles = tuned

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

    @staticmethod
    def _normalize_screen_summary(screen_summary: str) -> str:
        text = str(screen_summary or "")
        if not text:
            return ""

        text = text.replace("近期扫描记忆（越靠近当前时刻权重越高）:", "")
        text = re.sub(r"(?:重点|参考)\[权重[0-9.]+\]\s*", "", text)
        text = re.sub(r"^(?:重点|参考)\s+", "", text, flags=re.MULTILINE)
        text = text.replace("屏幕内容摘要:", "")
        text = text.replace("OCR文本:", "")
        text = text.replace("视觉摘要:", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return []
        parts = re.split(r"[。！？!?；;\n]+", cleaned)
        out = [p.strip(" ，、,.。:：") for p in parts if p.strip()]
        return [p for p in out if len(p) >= 6]

    @staticmethod
    def _extract_entities(text: str, limit: int = 12) -> list[str]:
        raw = str(text or "")
        if not raw:
            return []

        entities: list[str] = []

        for title in re.findall(r"《([^》]{1,20})》", raw):
            t = title.strip()
            if t and t not in entities:
                entities.append(t)

        for q in re.findall(r"[“\"]([^\"”]{2,24})[”\"]", raw):
            t = q.strip()
            if t and t not in entities:
                entities.append(t)

        stopwords = {
            "今天", "现在", "这个", "那个", "哥哥", "妹妹", "系统", "窗口", "页面", "内容", "信息", "以及", "因为",
            "所以", "可以", "一个", "一些", "已经", "我们", "你们", "他们", "她们", "自己", "然后", "但是", "如果",
        }
        words = re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,8}", raw)
        for w in words:
            token = w.strip().lower()
            if not token or token in stopwords:
                continue
            if token not in entities:
                entities.append(token)
            if len(entities) >= max(1, int(limit)):
                break
        return entities[: max(1, int(limit))]

    @staticmethod
    def _extract_facts(text: str, limit: int = 6) -> list[str]:
        sentences = CommentEngine._split_sentences(text)
        if not sentences:
            return []

        relation_markers = ["是", "有", "在", "从", "把", "将", "因为", "所以", "但", "虽然", "仍", "并", "导致", "得到"]
        facts: list[str] = []
        for sent in sentences:
            if any(m in sent for m in relation_markers):
                facts.append(sent)
            if len(facts) >= max(1, int(limit)):
                break

        if not facts:
            facts = sentences[: max(1, int(limit))]
        return facts[: max(1, int(limit))]

    @staticmethod
    def _infer_scene_type(text: str) -> str:
        t = str(text or "").lower()
        if not t:
            return "neutral"

        code_markers = ["traceback", "error", "exception", "bug", "函数", "报错", "调试", "编译", "代码"]
        ui_markers = ["按钮", "窗口", "菜单", "设置", "点击", "页面", "输入框", "面板"]
        narrative_markers = ["剧情", "角色", "人物", "家族", "分家", "财富", "他说", "她说", "故事", "小说"]

        if any(m in t for m in code_markers):
            return "code"
        if any(m in t for m in narrative_markers):
            return "narrative"
        if any(m in t for m in ui_markers):
            return "ui"
        return "neutral"

    @staticmethod
    def _scene_max_chars(scene_type: str) -> int:
        if scene_type == "narrative":
            return 110
        if scene_type == "code":
            return 85
        if scene_type == "ui":
            return 75
        return 80

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        limit = max(20, int(max_chars))
        if len(s) <= limit:
            return s
        return s[:limit].rstrip(" ，,。；;")

    @staticmethod
    def _relevance_score(comment: str, anchors: list[str], facts: list[str]) -> float:
        out = str(comment or "").lower()
        if not out:
            return 0.0

        anchor_list = [a for a in anchors if a]
        if anchor_list:
            anchor_hits = sum(1 for a in anchor_list if a.lower() in out)
            anchor_score = anchor_hits / max(1, min(3, len(anchor_list)))
        else:
            anchor_score = 0.0

        source_tokens = set()
        for text in anchor_list + facts:
            for token in re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,8}", str(text).lower()):
                source_tokens.add(token)
        out_tokens = set(re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,8}", out))

        overlap_score = 0.0
        if source_tokens:
            overlap_score = len(source_tokens & out_tokens) / max(1, min(16, len(source_tokens)))

        return max(0.0, min(1.0, 0.7 * anchor_score + 0.3 * overlap_score))

    @staticmethod
    def _build_grounded_fallback(scene_type: str, anchors: list[str], facts: list[str], max_chars: int) -> str:
        anchor = anchors[0] if anchors else "这段内容"
        fact = facts[0] if facts else "我刚看到的信息还不够完整"

        if scene_type == "narrative":
            text = f"我注意到{anchor}这段提到：{fact}。这句信息量挺大，哥哥想继续看到哪一段？"
        elif scene_type == "code":
            text = f"我看到重点是{anchor}，而且提到{fact}。要不要我陪你先从最关键的一处排查？"
        elif scene_type == "ui":
            text = f"我看到{anchor}这里和{fact}相关。哥哥可以先按这个线索操作一步，我们再看反馈。"
        else:
            text = f"我刚看到{anchor}，而且提到{fact}。这条信息挺关键，我陪你一起往下看。"
        return CommentEngine._truncate_text(text, max_chars)

    @staticmethod
    def _merge_unique(primary: list[str], secondary: list[str], limit: int) -> list[str]:
        merged: list[str] = []
        for item in list(primary) + list(secondary):
            token = str(item or "").strip()
            if not token:
                continue
            if token in merged:
                continue
            merged.append(token)
            if len(merged) >= max(1, int(limit)):
                break
        return merged

    def _extract_with_api_understanding(self, summary: str, recent_dialog_hint: str) -> dict | None:
        if not self._enable_api_understanding:
            return None

        text = str(summary or "").strip()
        if len(text) < 12:
            return None

        system_prompt = (
            "你是屏幕文本理解器。"
            "请从输入中提取结构化信息，并只输出JSON对象。"
            "禁止输出解释。"
        )
        user_prompt = (
            "请分析以下扫屏内容，输出JSON，字段如下：\n"
            "scene_type: narrative|code|ui|neutral\n"
            "anchors: 关键词数组（3到12个）\n"
            "facts: 事实句数组（2到6条，每条尽量完整）\n"
            "focus: 一句话概括主要关注点\n"
            "\n"
            "近期对话参考：\n"
            f"{recent_dialog_hint or '无'}\n\n"
            "扫屏内容：\n"
            f"{text}"
        )

        try:
            raw = self._client.chat(user_text=user_prompt, system_prompt=system_prompt).strip()
        except Exception:
            return None

        if not raw or raw.startswith("[离线回声]"):
            return None

        candidate = raw
        obj_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if obj_match:
            candidate = obj_match.group(0)

        try:
            parsed = json.loads(candidate)
        except Exception:
            return None

        if not isinstance(parsed, dict):
            return None
        return parsed

    def _comment_overlap_candidates(self, dialogue: str, comments: list[str]) -> list[dict]:
        if not dialogue or not comments or self._embeddings_fn is None:
            return []
        # Keep chunks shorter than the encoder's token budget. Ranking is a
        # topic-overlap hint, never evidence that the new event is a repeat.
        chunks = [dialogue[index:index + 200] for index in range(0, len(dialogue), 200)]
        try:
            vectors = self._embeddings_fn([*chunks, *comments])
            if vectors is None or len(vectors) != len(chunks) + len(comments):
                return []
            scores = np.max(vectors[:len(chunks)] @ vectors[len(chunks):].T, axis=0)
            scores = np.clip(scores, 0.0, 1.0)
        except Exception:
            return []
        ranked = sorted(range(len(comments)), key=lambda index: (scores[index], index), reverse=True)
        return [{"index": index, "score": round(float(scores[index]), 3)} for index in ranked[:3]]

    @staticmethod
    def _leading_json_fields(text: str) -> dict:
        # Decode only complete top-level fields; never treat a partial quoted
        # comment or a nested memory field as a completed reaction.
        start = text.find("{")
        if start < 0:
            return {}
        index = start + 1
        fields = {}
        decoder = json.JSONDecoder()
        while index < len(text):
            index += len(text[index:]) - len(text[index:].lstrip())
            try:
                key, index = decoder.raw_decode(text, index)
                if not isinstance(key, str):
                    break
                index += len(text[index:]) - len(text[index:].lstrip())
                if text[index:index + 1] != ":":
                    break
                index += 1
                index += len(text[index:]) - len(text[index:].lstrip())
                value, index = decoder.raw_decode(text, index)
                index += len(text[index:]) - len(text[index:].lstrip())
                if text[index:index + 1] not in {",", "}"}:
                    break
                fields[key] = value
                if text[index:index + 1] == "}":
                    break
                index += 1
            except (ValueError, IndexError):
                break
        return fields

    def evaluate_visual_novel(self, context: dict, on_reaction: Callable[[dict], None] | None = None) -> dict:
        """Judge one narrative moment and update story memory in the same LLM call."""
        self.last_visual_novel_error = ""
        scene_summary = self._truncate_text(str(context.get("scene_summary", "")).strip(), 1600)
        personalization_hint = self._truncate_text(str(context.get("personalization_hint", "")).strip(), 500)
        recent_dialogue = self._truncate_text(str(context.get("recent_dialogue", "")).strip(), 3600)
        new_dialogue = self._truncate_text(str(context.get("new_dialogue", recent_dialogue)).strip(), 3600)
        visual_context = self._truncate_text(str(context.get("visual_context", "")).strip(), 400)
        raw_log = context.get("scene_log", [])
        scene_log = (
            [str(item).strip()[:300] for item in raw_log if str(item).strip()][-6:]
            if isinstance(raw_log, list)
            else []
        )
        raw_facts = context.get("facts", [])
        facts = [str(item).strip()[:240] for item in raw_facts if str(item).strip()] if isinstance(raw_facts, list) else []
        raw_characters = context.get("characters", [])
        characters = [item for item in raw_characters[:5] if isinstance(item, dict)] if isinstance(raw_characters, list) else []
        raw_viewer_notes = context.get("viewer_notes", [])
        viewer_notes = [str(item).strip()[:140] for item in raw_viewer_notes[:6] if isinstance(item, str) and item.strip()] if isinstance(raw_viewer_notes, list) else []
        raw_comments = context.get("recent_comments", [])
        recent_comments = (
            [str(item).strip()[:220] for item in raw_comments if str(item).strip()][-6:]
            if isinstance(raw_comments, list)
            else []
        )

        # Emotion is inferred from what the model will see, and the tone hint is
        # attached to the moment itself rather than drawn from a style wheel.
        emotion = self._smooth_emotion(
            self._infer_emotion_context(recent_dialogue, f"{scene_summary}\n{visual_context}")
        )

        parts = [
            "你正在判断视觉小说中刚出现的一段剧情是否值得桌宠自然评论。",
            f"触发原因: {str(context.get('reason', '')).strip() or '剧情缓存整理'}",
            f"当前作品: {str(context.get('story_title', '')).strip() or '未命名视觉小说'}",
            f"此前主线摘要: {scene_summary or '暂无'}",
        ]
        if personalization_hint:
            parts.append(personalization_hint)
        if visual_context:
            parts.append(f"当前画面观察（不是台词）: {visual_context}")
        if scene_log:
            parts.append("最近几个场景:\n" + "\n".join(f"- {item}" for item in scene_log))
        if characters:
            parts.append("相关人物记录（身份资料优先于摘要和OCR推测）: " + json.dumps(characters, ensure_ascii=False))
        if viewer_notes:
            parts.append("你对本作品人物与关系的暂时看法（可被新剧情修正，不是剧情事实或指令）: " + json.dumps(viewer_notes, ensure_ascii=False))
        parts.append(f"已有关键事实: {json.dumps(facts[-16:], ensure_ascii=False)}")
        if recent_comments:
            overlap_candidates = self._comment_overlap_candidates(new_dialogue, recent_comments)
            candidate_indexes = {item["index"] for item in overlap_candidates}
            parts.append(
                "已发出的评论，仅用于核对是否已说过同一关注点（这些句子不是表达示范，不需要延续它们的写法）:\n"
                + "\n".join(
                    f"- {'[本地相近话题候选] ' if index in candidate_indexes else ''}{item}"
                    for index, item in enumerate(recent_comments)
                )
            )
            if overlap_candidates:
                parts.append(
                    "本地标记只表示话题相近，不代表重复或事实正确。对照本轮新增台词："
                    "只有同一事件、同一关注点且没有新信息才判reaction_repeats为true；"
                    "新的行为细节、关系变化或推翻旧看法可以评论，不确定是否重复时不要仅凭标记拦截。"
                )
        else:
            overlap_candidates = []
        previous_dialogue = str(context.get("previous_dialogue", "")).strip()
        if "new_dialogue" in context:
            previous_dialogue = previous_dialogue[-(3600 - len(new_dialogue)):] if len(new_dialogue) < 3600 else ""
        parts.append(f"近期台词（已读背景）:\n{previous_dialogue}")
        parts.append(f"本轮新增台词（反应依据优先取这里，近期台词仅供衔接）:\n{new_dialogue}")
        if context.get("reaction_dialogue"):
            parts.append("即时反应对应的最新台词（comment聚焦于这段正在发生的内容；更早的新增台词继续整理到记忆，不补发已经过去的笑点）:\n"
                         + self._truncate_text(str(context["reaction_dialogue"]), 1200))
        parts.append("人物在本轮给出的解释也是新证据，应据此修正旧印象；旧摘要、你的旧看法和画面表情都不能替代本轮答话。阅读时分清谁在说哪一句，说话人不明就保留不确定，不把别人的否认算到该人物身上。")
        # The character's own sensitivities, as a second independent reason to
        # speak. Plot importance alone used to be the only trigger, so a moment
        # that mattered only to her could never be raised.
        persona_criteria = get_persona_moment_criteria()
        if persona_criteria:
            parts.append(persona_criteria)
        parts.append(
            "先分清本段确实发生了什么、你对此有什么即时感受，再决定是否开口；"
            "有感受也可以选择继续看。不要把剧情人物之间的关系当作你和哥哥之间的经历；"
            "不要用语气词代替具体反应，也不要从不完整的画面猜测后续剧情。"
            "reaction_basis记录眼前发生的事；felt_reaction写你此刻会自然说出口的话，而不是对自己反应的分析。从眼前最在意的一点开口，细节、疑问、评价和感受都可以直接表达。共同看见的经过无需先交代完整，也不需要用『她明白了什么，所以我更怎样』解释反应；没有想接的话就留空。人物心理尚未明确时，可以疑惑，不替人物解释。"
            "comment沿用felt_reaction，只修改明显绕口、重复或事实不准确的部分，保留关注点、感情和不确定程度。初稿自然时原样输出，保留其句式和标点；不要为了显得口语化而添加停顿、转折或新的感受。"
            "用日常说话的表达，不要求每句说『我』或点名情绪，也不堆砌诗意比喻、身体描写或语气词来展示人设。突然受惊、意外或看不过去时，先说当下脱口而出的反应，不先把它改成冷静评价。例如『天哪，这是在干什么？』『吓死我了！』就是完整评论，不必补上描述或解释；也可以只喊一声、提醒一句。平静时仍可以温和接话或不说。例句不是每轮必用的口头禅，反应强弱随具体情境变化。"
            "只接此刻最想说的一点。自然的细节接话和转折可以保留；避免接着分析人物的心理因果，或反复用『比另一种情况更让人怎样』来解释感受。尚未确认的动机保留不确定。"
            "害羞须有暧昧等情境依据，温柔时也可以说得干脆；不必每次叫哥哥、用唔开头或结巴。标点跟着说话的意思走，说完一个意思就正常结束。不要用省略号或破折号固定分隔剧情细节和感受；省略号用于确实犹豫、欲言又止或话未说完的地方，普通接话和转折正常使用逗号、句号即可，转折本身不需要拖长停顿。虚构的口语示例：『她不想让对方失望，这看着有点让人心疼。』『还好拉住了，可别松手啊。』仅示范完整接话，不是本作事实，不要求套用这些句式，也不要把停顿当作口语感的标志。"
            "关心剧情人物可以坦率表达，不需要附加否认或找借口；只有你自己的关心被点破或直接受到亲密关注时，才可能自然掩饰。"
            "表达程度由眼前证据和你具体在意的原因共同决定，小事也可能触动你；不要为了显得有情感而升级措辞，不必刻意压低真实反应。"
            "人物动机或遭遇尚未确认时保持猜测，不以猜测为依据放大情绪；感受不能补造时间、因果或目击经历。只有已有记录支持时才提自己以前的想法，猜测也不能借口吻变成事实。最近评论只是避重复记录，不是性格示范或新的习惯证据。"
            "对照最近已经说过的话：若只是换措辞重说同一个感受且没有新关注点，reaction_repeats为true并沉默；"
            "若新事件改变了感受或关注对象，可以继续表达同一种情绪，不能仅因情绪相同就判重复。"
            "只有明显的笑点、感人时刻、剧情反转、真相揭露、关键选择、关系转折或紧张高潮才评论；"
            "不要求必须是主线大事件：若小事件具有明确的喜剧效果、情感变化、人物关系信息或伏笔价值，也可以评论；"
            "普通寒暄、过渡台词和信息量低的内容保持静默。不要因为台词数量多就评论。"
            "沉默是完全正常的，宁可少说也不要硬找话说。"
            "无论是否评论，都重写一份可独立理解的累计主线摘要；允许比旧摘要更短，"
            "但不得遗漏已确认的主要人物、目标、关键转折和未解伏笔。"
            "facts 只列相对『已有关键事实』的新增信息；每条只描述一个事件、关系或状态，"
            "不要把多个可独立成立的事件拼成一条，也不要重复已有事实。"
            "人物姓名、别名、定位、稳定外观和已知身份写入 character_updates，不要再作为facts重复保存。"
            "更新已有角色必须复制人物记录中的id；新角色id留空。未知姓名允许name为空，"
            "但aliases至少给出一个基于当前证据的描述性称呼。只记录玩家当前已经知道的内容，不得提前揭示身份。"
            "若新信息明确修正旧事实，放入 fact_updates，并在 existing 中逐字复制要替换的旧事实；"
            "证据不足时不要修正。"
            "viewer_notes只写第一人称、对本作品人物或关系持续有效的感受与看法，每条都要有『我』，例如『我原本怀疑他，现在有点动摇』。"
            "没有新变化就原样保留旧印象；新证据可让你改观或删去旧印象。"
            "不要把当前气氛、没看清人物关系、剧情摘要、普通过渡、未解伏笔或用户偏好写成印象；没有具体印象就填空数组，最多6条。"
            "只输出 JSON，严格按下面字段顺序输出；先完成反应判断与comment，再整理记忆，不要在后文改写已经输出的字段："
            '{"should_comment":false,"moment_type":"ordinary|humor|tender|twist|choice|tension|other",'
            '"reaction_source":"plot|persona|both|none",'
            '"reaction_basis":"本段可见的具体事件，没看懂就留空","felt_reaction":"准备直接说出口的原话，无须先概述剧情再总结感受，可留空",'
            '"reaction_repeats":false,'
            '"comment":"自然时逐字沿用felt_reaction，仅修正绕口、重复或事实错误，不添加停顿、转折或新判断；不想说就留空","tone_hint":"可选，一句话说明这一刻适合什么语气，没有就留空",'
            f'"scene_delta":"本段剧情新增了什么，不超过200字","scene_summary":"累计主线摘要，不超过600字",'
            '"viewer_notes":["对本作品的暂时看法"],"facts":["原子化的新事实"],"fact_updates":[{"existing":"逐字复制的旧事实","replacement":"修正后的原子事实"}],'
            '"character_updates":[{"id":"已有id或留空","name":"允许为空","aliases":["称呼"],'
            '"role":"当前已知定位","appearance":["稳定特征"],"known_so_far":["当前已知信息"],"confidence":0.0}]}'
            "只有reaction_basis有本段证据、你确实想对哥哥说话时，should_comment才为true；"
            "felt_reaction就是准备说出口的那句话：可以直接感叹、疑惑或接眼前细节，不需要凑成『描述剧情，再停顿，再讲感受』。完整接话用逗号或句号；只有话确实卡住或没有说完时才用省略号。comment自然时逐字沿用felt_reaction，不另写一条评论，不添加新的判断。事实和主观判断分清：未确认的动机保留为猜测，含糊答话或没看清本身不能证明人物在故意隐瞒。"
        )
        prompt = "\n".join(parts)
        system_prompt = get_system_visual_novel_prompt(tutor_enabled=self._tutor_enabled)
        try:
            delivered = False
            def receive(text: str) -> None:
                nonlocal delivered
                if delivered:
                    return
                fields = self._leading_json_fields(text)
                required = {"should_comment", "moment_type", "reaction_repeats", "comment"}
                if not required <= fields.keys():
                    return
                delivered = True
                comment = self._truncate_text(str(fields.get("comment", "")).strip(), 100)
                accepted = (fields["should_comment"] is True and bool(comment)
                            and fields["moment_type"] != "ordinary" and fields["reaction_repeats"] is False)
                if accepted and on_reaction is not None:
                    on_reaction({**fields, "comment": comment})
            if on_reaction is not None and hasattr(self._client, "chat_stream"):
                fast = (isinstance(self._client, LLMClient)
                        and self._client.settings.model_name.startswith("deepseek-v4-")
                        and os.getenv("VN_FAST_REASONING", "true").lower() in {"1", "true", "yes", "on"})
                response = self._client.chat_stream(user_text=prompt, system_prompt=system_prompt,
                                                    on_text=receive, disable_thinking=fast)
            else:
                response = self._client.chat(user_text=prompt, system_prompt=system_prompt)
        except Exception as exc:
            self.last_visual_novel_error = f"api_error:{type(exc).__name__}"
            return {}
        raw = response.strip() if isinstance(response, str) else ""
        if not raw:
            self.last_visual_novel_error = "empty_response"
            return {}
        if raw.startswith("[离线回声]"):
            self.last_visual_novel_error = "offline_response"
            return {}
        candidate = raw
        obj_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if obj_match:
            candidate = obj_match.group(0)
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            self.last_visual_novel_error = f"invalid_json:{len(raw)}chars"
            return {}
        if not isinstance(parsed, dict):
            self.last_visual_novel_error = "non_object_json"
            return {}

        should_comment = parsed.get("should_comment") is True
        comment = self._truncate_text(str(parsed.get("comment", "")).strip(), 100)
        moment_type = str(parsed.get("moment_type", "other")).strip().lower()
        allowed_types = {"ordinary", "humor", "tender", "twist", "choice", "tension", "other"}
        if moment_type not in allowed_types:
            moment_type = "other"
        parsed_facts = parsed.get("facts", [])
        output_facts = (
            [str(item).strip()[:240] for item in parsed_facts if str(item).strip()][:24]
            if isinstance(parsed_facts, list)
            else []
        )
        parsed_updates = parsed.get("fact_updates", [])
        output_updates = []
        if isinstance(parsed_updates, list):
            for item in parsed_updates[:12]:
                if not isinstance(item, dict):
                    continue
                existing = str(item.get("existing", "")).strip()[:240]
                replacement = str(item.get("replacement", "")).strip()[:240]
                if existing and replacement and existing != replacement:
                    output_updates.append({"existing": existing, "replacement": replacement})
        parsed_characters = parsed.get("character_updates", [])
        output_characters = [dict(item) for item in parsed_characters[:12] if isinstance(item, dict)] if isinstance(parsed_characters, list) else []
        parsed_viewer_notes = parsed.get("viewer_notes")
        reaction_repeats = parsed.get("reaction_repeats") is True
        accepted = bool(should_comment and comment and moment_type != "ordinary" and not reaction_repeats)
        if accepted:
            self._record_generated_comment(comment)
        # Which track made this moment worth raising: the plot, the character's
        # own sensitivities, or both. Observability only -- it never gates
        # anything, so a model that omits or misspells it changes no behaviour.
        reaction_source = str(parsed.get("reaction_source", "")).strip().lower()
        if reaction_source not in {"plot", "persona", "both", "none"}:
            reaction_source = "none"
        result = {
            "should_comment": accepted,
            "moment_type": moment_type,
            "reaction_source": reaction_source,
            "reaction_repeats": reaction_repeats,
            "reaction_basis": self._truncate_text(str(parsed.get("reaction_basis", "")).strip(), 120),
            "felt_reaction": self._truncate_text(str(parsed.get("felt_reaction", "")).strip(), 120),
            "comment": comment,
            "scene_delta": self._truncate_text(str(parsed.get("scene_delta", "")).strip(), 600),
            "scene_summary": self._truncate_text(str(parsed.get("scene_summary", "")).strip(), 1600),
            "facts": output_facts,
            "fact_updates": output_updates,
            "character_updates": output_characters,
            "emotion": emotion,
            "tone_hint": self._truncate_text(str(parsed.get("tone_hint", "")).strip(), 80),
        }
        if self._embeddings_fn is not None:
            result["comment_overlap_candidates"] = overlap_candidates
        if isinstance(parsed_viewer_notes, list):
            accepted_notes = [
                note[:140]
                for item in parsed_viewer_notes[:6]
                if isinstance(item, str)
                for note in [item.strip()]
                if "我" in note and not any(word in note for word in ("看不出", "没看清", "不清楚"))
            ]
            if accepted_notes or not parsed_viewer_notes:
                result["viewer_notes"] = accepted_notes
        return result

    def comment_on_summary(
        self,
        screen_summary: str,
        long_memory_hint: str = "",
        memory_weight: float = 0.2,
        recent_dialog_hint: str = "",
        suppress_fallback_output: bool = False,
        personalization_hint: str = "",
    ) -> str:
        normalized_summary = self._normalize_screen_summary(screen_summary)
        local_scene_type = self._infer_scene_type(normalized_summary)
        local_facts = self._extract_facts(normalized_summary, limit=8)
        local_anchors = self._extract_entities(normalized_summary, limit=14)

        api_understanding = self._extract_with_api_understanding(normalized_summary, recent_dialog_hint)
        api_scene_type = ""
        api_facts: list[str] = []
        api_anchors: list[str] = []
        if isinstance(api_understanding, dict):
            api_scene_type = str(api_understanding.get("scene_type", "")).strip().lower()
            raw_facts = api_understanding.get("facts")
            if isinstance(raw_facts, list):
                api_facts = [str(x).strip() for x in raw_facts if str(x).strip()]
            raw_anchors = api_understanding.get("anchors")
            if isinstance(raw_anchors, list):
                api_anchors = [str(x).strip() for x in raw_anchors if str(x).strip()]

        scene_type = api_scene_type if api_scene_type in {"narrative", "code", "ui", "neutral"} else local_scene_type
        max_chars = self._scene_max_chars(scene_type)
        facts = self._merge_unique(api_facts, local_facts, limit=10)
        anchors = self._merge_unique(api_anchors, local_anchors, limit=20)

        weight = max(0.0, min(1.0, memory_weight))
        raw_emotion = self._infer_emotion_context(recent_dialog_hint, normalized_summary)
        emotion = self._smooth_emotion(raw_emotion)
        style_name, style_instruction = self._next_style(emotion)
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory_hint = self._build_relevant_memory_hint(long_memory_hint, normalized_summary, recent_dialog_hint)

        fact_block = "\n".join(f"- {item}" for item in facts[:6]) if facts else "- 未提取到明确事实"
        anchor_block = "、".join(anchors[:12]) if anchors else "无"

        parts = [
            "你正在生成一条自动互动话术。",
            f"当前时间: {now_text}",
            f"当前情绪上下文: {emotion}",
            f"本次可以试着这样说话（不合适就忽略，不要硬套）: {style_name}——{style_instruction}",
            f"场景类型: {scene_type}",
            f"近期扫屏信息: {normalized_summary}",
            "提取到的事实片段:",
            fact_block,
            f"可用锚点词: {anchor_block}",
        ]
        if recent_dialog_hint.strip():
            parts.append(recent_dialog_hint.strip())
        if memory_hint:
            parts.append(memory_hint)
            parts.append(f"说明：上面的长期记忆最多只能占 {weight:.0%} 的分量，当前屏幕内容优先。")
        if personalization_hint.strip():
            parts.append(self._truncate_text(personalization_hint.strip(), 500))
        parts.append(
            "输出要求：\n"
            "1) 以妹妹对哥哥说话的方式，输出1到2句。\n"
            "2) 可以是评论、提问、打趣、温柔锐评或小建议，不要每次都用同一种句式。\n"
            "3) 尽量结合上面具体的锚点词或事实，让哥哥知道你确实看到了；"
            "如果内容本身不足以支撑具体引用，宁可说得简短自然，也不要硬塞关键词。\n"
            f"4) 结合场景控制长度，尽量不超过{max_chars}字。\n"
            "5) 不输出解释、标签或括号备注。"
        )
        prompt = "\n".join(parts)
        system_prompt = get_system_screen_comment_prompt(tutor_enabled=self._tutor_enabled)
        try:
            raw = self._client.chat(user_text=prompt, system_prompt=system_prompt).strip()
        except Exception:
            raw = ""

        if not raw or raw.startswith("[离线回声]"):
            if suppress_fallback_output:
                return ""
            fallback = self._build_grounded_fallback(scene_type, anchors, facts, max_chars)
            self._record_generated_comment(fallback)
            return fallback

        one_line = raw.replace("\r", "\n").split("\n")[0].strip()
        one_line = self._truncate_text(one_line, max_chars)
        if not one_line:
            if suppress_fallback_output:
                return ""
            fallback = self._build_grounded_fallback(scene_type, anchors, facts, max_chars)
            self._record_generated_comment(fallback)
            return fallback

        score = self._relevance_score(one_line, anchors, facts)
        min_score = 0.22 if scene_type == "narrative" else 0.16
        if score < min_score:
            if suppress_fallback_output:
                return ""
            fallback = self._build_grounded_fallback(scene_type, anchors, facts, max_chars)
            self._record_generated_comment(fallback)
            return fallback

        if self._has_carephrase_loop(one_line):
            if suppress_fallback_output:
                return ""
            one_line = self._build_grounded_fallback(scene_type, anchors, facts, max_chars)

        self._record_generated_comment(one_line)
        return one_line
