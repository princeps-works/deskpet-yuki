from __future__ import annotations

import json
import random
import re
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
        enable_api_understanding: bool = False,
    ) -> None:
        self._client = llm_client
        self._tutor_enabled = bool(tutor_enabled)
        self._style_rng = random.Random()
        self._last_style_name = ""
        self._enable_api_understanding = bool(enable_api_understanding)
        self._stable_emotion = "neutral"
        self._emotion_candidate = ""
        self._emotion_candidate_streak = 0
        self._neutral_exit_streak = 0
        self._recent_generated_comments: list[str] = []
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

    def evaluate_visual_novel(self, context: dict) -> dict:
        """Judge one narrative moment and update story memory in the same LLM call."""
        scene_summary = self._truncate_text(str(context.get("scene_summary", "")).strip(), 1600)
        recent_dialogue = self._truncate_text(str(context.get("recent_dialogue", "")).strip(), 3600)
        raw_facts = context.get("facts", [])
        facts = [str(item).strip()[:240] for item in raw_facts if str(item).strip()] if isinstance(raw_facts, list) else []
        prompt = (
            "你正在判断视觉小说中刚出现的一段剧情是否值得桌宠自然评论。\n"
            f"触发原因: {str(context.get('reason', '')).strip() or '剧情缓存整理'}\n"
            f"此前场景摘要: {scene_summary or '暂无'}\n"
            f"已有关键事实: {json.dumps(facts[-16:], ensure_ascii=False)}\n"
            f"近期台词:\n{recent_dialogue}\n\n"
            "只有明显的笑点、感人时刻、剧情反转、真相揭露、关键选择、关系转折或紧张高潮才评论；"
            "不要求必须是主线大事件：若小事件具有明确的喜剧效果、情感变化、人物关系信息或伏笔价值，也可以评论；"
            "普通寒暄、过渡台词和信息量低的内容保持静默。不要因为台词数量多就评论。\n"
            "无论是否评论，都压缩更新场景摘要并保留后续理解剧情所需的关键事实。\n"
            "只输出 JSON："
            '{"should_comment":false,"moment_type":"ordinary|humor|tender|twist|choice|tension|other",'
            '"comment":"","scene_summary":"不超过600字的累计场景摘要","facts":["关键事实"]}'
        )
        system_prompt = get_system_screen_comment_prompt(tutor_enabled=self._tutor_enabled)
        try:
            raw = self._client.chat(user_text=prompt, system_prompt=system_prompt).strip()
        except Exception:
            return {}
        if not raw or raw.startswith("[离线回声]"):
            return {}
        candidate = raw
        obj_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if obj_match:
            candidate = obj_match.group(0)
        try:
            parsed = json.loads(candidate)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
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
        return {
            "should_comment": bool(should_comment and comment and moment_type != "ordinary"),
            "moment_type": moment_type,
            "comment": comment,
            "scene_summary": self._truncate_text(str(parsed.get("scene_summary", "")).strip(), 1600),
            "facts": output_facts,
        }

    def comment_on_summary(
        self,
        screen_summary: str,
        long_memory_hint: str = "",
        memory_weight: float = 0.2,
        recent_dialog_hint: str = "",
        suppress_fallback_output: bool = False,
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
            f"本次风格: {style_name}",
            f"风格要求: {style_instruction}",
            f"场景类型: {scene_type}",
            f"近期扫屏信息: {normalized_summary}",
            "提取到的事实片段:",
            fact_block,
            f"可用锚点词: {anchor_block}",
            f"说明：下面的长期记忆仅作低权重参考（建议权重{weight:.2f}），优先依据当前屏幕摘要。",
        ]
        if recent_dialog_hint.strip():
            parts.append(recent_dialog_hint.strip())
        if memory_hint:
            parts.append(memory_hint)
        parts.append(
            "输出要求：\n"
            "1) 以妹妹对哥哥说话的方式，输出1到2句。\n"
            "2) 可以是评论、提问、打趣、温柔锐评或小建议，不要每次都用同一种句式。\n"
            "3) 必须引用至少1个锚点词或事实要素，不要只复述‘哥哥在做什么’。\n"
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
