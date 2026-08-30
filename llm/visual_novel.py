from __future__ import annotations

import json
import re
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class NovelObservation:
    accepted: bool
    should_evaluate: bool
    reason: str
    semantic_score: float


class VisualNovelStoryLibrary:
    """Manage isolated story JSON files and remember the active cache."""

    _INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

    def __init__(self, directory: Path, *, legacy_path: Path | None = None) -> None:
        self.directory = Path(directory)
        self.legacy_path = Path(legacy_path) if legacy_path is not None else None
        self.active_marker = self.directory / "active_story.txt"
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.list_stories() and self.legacy_path is not None and self.legacy_path.is_file():
            shutil.copy2(self.legacy_path, self.directory / "default.json")
        if not self.list_stories():
            self._write_empty(self.directory / "default.json")

        active = self.active_marker.read_text(encoding="utf-8").strip() if self.active_marker.is_file() else ""
        if active not in self.list_stories():
            active = self.list_stories()[0]
        self.set_active(active)

    @staticmethod
    def _write_empty(path: Path) -> None:
        payload = {
            "version": 1,
            "total_lines": 0,
            "summarized_line_count": 0,
            "scene_summary": "",
            "facts": [],
            "lines": [],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def _filename(cls, name: str) -> str:
        value = cls._INVALID_FILENAME.sub("_", str(name or "").strip()).strip(" .")
        if value.lower().endswith(".json"):
            value = value[:-5].strip(" .")
        value = value[:80].strip(" .")
        if not value:
            raise ValueError("缓存名称不能为空")
        return value + ".json"

    @staticmethod
    def _is_valid_story(path: Path) -> bool:
        try:
            return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
        except Exception:
            return False

    def list_stories(self) -> list[str]:
        return sorted(
            (
                path.name
                for path in self.directory.glob("*.json")
                if path.is_file() and self._is_valid_story(path)
            ),
            key=str.casefold,
        )

    @property
    def active_name(self) -> str:
        value = self.active_marker.read_text(encoding="utf-8").strip() if self.active_marker.is_file() else ""
        stories = self.list_stories()
        return value if value in stories else stories[0]

    @property
    def active_path(self) -> Path:
        return self.directory / self.active_name

    def create(self, name: str) -> Path:
        filename = self._filename(name)
        path = self.directory / filename
        if path.exists():
            raise FileExistsError(f"缓存已存在：{filename}")
        self._write_empty(path)
        return path

    def set_active(self, name: str) -> Path:
        filename = str(name or "").strip()
        if filename not in self.list_stories():
            raise FileNotFoundError(f"缓存不存在或格式无效：{filename}")
        temp_marker = self.active_marker.with_suffix(".tmp")
        temp_marker.write_text(filename, encoding="utf-8")
        temp_marker.replace(self.active_marker)
        return self.directory / filename

    def delete(self, name: str) -> None:
        filename = str(name or "").strip()
        stories = self.list_stories()
        if filename not in stories:
            raise FileNotFoundError(f"缓存不存在：{filename}")
        if len(stories) <= 1:
            raise ValueError("至少需要保留一个剧情缓存")
        if filename == self.active_name:
            raise ValueError("请先切换到其他缓存，再删除当前缓存")
        (self.directory / filename).unlink()


class VisualNovelTracker:
    """Keep visual-novel dialogue and nominate semantic moments for the LLM."""

    _MOMENT_ANCHORS = (
        "这段剧情出现了重要反转或真相揭露",
        "这段对话有明显的笑点或幽默冲突",
        "这段剧情令人感动或人物感情发生重要变化",
        "角色面临关键选择或关系发生转折",
        "剧情出现紧张高潮、危机或重大事件",
    )
    _RECALL_ANCHORS = (
        "用户正在询问刚才视觉小说或游戏的剧情、角色、台词和事件",
        "用户想回忆此前屏幕上识别到的内容",
    )
    _RUNTIME_NOISE = re.compile(
        r"(?i)(?:heartbeat|context_bui(?:lt)?|scan_(?:wait|busy|submit)|vn_(?:buffer|skip|wait|evaluate)|"
        r"mm_(?:result|skip|fail|cooldown)|vision_route|resource_policy|ocr_only)"
    )
    _LOG_PREFIX = re.compile(
        r"(?i)^\s*(?:\[[^\]]*\b(?:debug|info|warn(?:ing)?|error|trace|heartbeat)\b[^\]]*\]|"
        r"(?:debug|info|warn(?:ing)?|error|trace)\s*[:：\-])"
    )
    _LOG_ASSIGNMENT = re.compile(r"(?i)\b[a-z_][\w.-]*\s*=\s*[^\s,;]+")
    _STACK_OR_PATH = re.compile(
        r"(?i)(?:traceback \(most recent call last\)|\bat .+\([^()]+:\d+\)|"
        r"[a-z]:\\[^\r\n]+|(?:^|\s)(?:/[^/\s]+){2,})"
    )
    _REPEATED_PUNCTUATION = re.compile(r"([^\w\s\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af])\1{3,}")
    _MIXED_CODE_TOKEN = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z\d_-]{7,}$", re.IGNORECASE)

    def __init__(
        self,
        path: Path,
        *,
        similarity_fn: Callable[[str, str], float | None] | None = None,
        similarities_fn: Callable[[str, list[str]], list[float] | None] | None = None,
        max_lines: int = 600,
        recent_limit: int = 20,
        summary_batch_size: int = 24,
        min_context_similarity: float = 0.30,
    ) -> None:
        self.path = Path(path)
        self._similarity_fn = similarity_fn
        self._similarities_fn = similarities_fn
        self._max_lines = max(60, int(max_lines))
        self._recent_limit = max(6, int(recent_limit))
        self._summary_batch_size = max(8, int(summary_batch_size))
        self._min_context_similarity = max(0.0, min(1.0, float(min_context_similarity)))
        self._reset_story_state()
        self._load()

    def _reset_story_state(self) -> None:
        self._lines: list[dict] = []
        self._scene_summary = ""
        self._facts: list[str] = []
        self._total_lines = 0
        self._summarized_line_count = 0
        self._last_requested_line_count = 0
        self._pending_candidate = False
        self._score_history: list[tuple[int, float]] = []
        self._last_peak_line_id = 0
        self._dirty_count = 0

    def switch_story(self, path: Path) -> None:
        next_path = Path(path)
        if next_path == self.path:
            return
        self.flush()
        self.path = next_path
        self._reset_story_state()
        self._load()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _clamp_score(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _similarity(self, left: str, right: str) -> float | None:
        if not callable(self._similarity_fn):
            return None
        try:
            return self._clamp_score(self._similarity_fn(left, right))
        except Exception:
            return None

    def _similarities(self, query: str, texts: list[str]) -> list[float] | None:
        if callable(self._similarities_fn):
            try:
                scores = self._similarities_fn(query, texts)
                if scores is not None and len(scores) == len(texts):
                    return [self._clamp_score(score) or 0.0 for score in scores]
            except Exception:
                return None
        scores = [self._similarity(query, text) for text in texts]
        if not any(score is not None for score in scores):
            return None
        return [score or 0.0 for score in scores]

    @classmethod
    def _is_runtime_noise(cls, text: str) -> bool:
        return bool(cls._RUNTIME_NOISE.search(str(text or "")))

    @classmethod
    def _looks_like_log(cls, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        return bool(
            cls._LOG_PREFIX.search(value)
            or cls._STACK_OR_PATH.search(value)
            or len(cls._LOG_ASSIGNMENT.findall(value)) >= 2
        )

    @classmethod
    def _looks_like_gibberish(cls, text: str) -> bool:
        value = re.sub(r"\s+", "", str(text or ""))
        if not value:
            return True
        if cls._REPEATED_PUNCTUATION.search(value) or cls._MIXED_CODE_TOKEN.fullmatch(value):
            return True
        meaningful = sum(
            char.isalnum() or "\u3400" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff"
            for char in value
        )
        punctuation = sum(not char.isalnum() and not char.isspace() for char in value)
        return len(value) >= 4 and meaningful <= 1 and punctuation / len(value) >= 0.5

    @staticmethod
    def _looks_sentence_like(text: str) -> bool:
        value = str(text or "").strip()
        east_asian_chars = sum(
            "\u3400" <= char <= "\u9fff"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7af"
            for char in value
        )
        if east_asian_chars >= 2:
            return True
        return len(re.findall(r"[A-Za-z]{2,}", value)) >= 2

    def prepare_ocr_text(self, text: str, *, confidence: float = 0.0) -> tuple[str, str]:
        """Filter OCR noise before observe; semantic continuity only rescues weak fragments."""
        raw_lines = [line.strip() for line in re.split(r"[\r\n]+", str(text or "")) if line.strip()]
        if not raw_lines:
            return "", "quality_empty"

        try:
            score = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            score = 0.0
        previous_lines: list[str] = []
        for item in reversed(self._lines):
            previous = self._normalize(item.get("text", ""))
            if (
                not previous
                or self._is_runtime_noise(previous)
                or self._looks_like_log(previous)
                or self._looks_like_gibberish(previous)
            ):
                continue
            previous_lines.append(previous)
            if len(previous_lines) >= 3:
                break
        previous_lines.reverse()
        previous_context = "\n".join(previous_lines)
        accepted: list[str] = []
        rejected_log = 0
        rejected_gibberish = 0
        rejected_low_confidence = 0
        for line in raw_lines:
            normalized = self._normalize(line)
            if self._is_runtime_noise(normalized) or self._looks_like_log(normalized):
                rejected_log += 1
                continue
            if self._looks_like_gibberish(normalized):
                rejected_gibberish += 1
                continue
            if score < 0.35 and not self._looks_sentence_like(normalized):
                continuity = self._similarity(normalized, previous_context) if previous_context else None
                if continuity is None or continuity < 0.42:
                    rejected_low_confidence += 1
                    continue
            accepted.append(normalized)

        reason_parts: list[str] = []
        if rejected_log:
            reason_parts.append(f"log={rejected_log}")
        if rejected_gibberish:
            reason_parts.append(f"gibberish={rejected_gibberish}")
        if rejected_low_confidence:
            reason_parts.append(f"low_confidence={rejected_low_confidence}")
        reason = "quality_ok" if not reason_parts else "quality_filtered:" + ",".join(reason_parts)
        prepared = "\n".join(accepted).strip()
        if prepared and self._min_context_similarity > 0.0 and len(previous_lines) >= 3:
            context_similarity = self._similarity(prepared, previous_context)
            if context_similarity is not None and context_similarity < self._min_context_similarity:
                return "", (
                    f"context_similarity={context_similarity:.2f}"
                    f"<{self._min_context_similarity:.2f}"
                )
        return prepared, reason

    def retrieve_for_query(self, query: str, *, limit: int = 6, max_chars: int = 1200) -> str:
        """Return relevant story-cache excerpts without modifying either memory store."""
        normalized_query = self._normalize(query)
        if not normalized_query:
            return ""

        candidates: list[tuple[str, str]] = []
        if self._scene_summary and not self._is_runtime_noise(self._scene_summary):
            candidates.append(("剧情摘要", self._scene_summary))
        candidates.extend(
            ("关键事实", fact)
            for fact in self._facts[-24:]
            if fact and not self._is_runtime_noise(fact)
        )
        candidates.extend(
            ("近期台词", text)
            for item in self._lines[-80:]
            if (text := self._normalize(item.get("text", ""))) and not self._is_runtime_noise(text)
        )

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for kind, text in candidates:
            if text in seen:
                continue
            seen.add(text)
            deduped.append((kind, text))
        if not deduped:
            return ""

        candidate_scores = self._similarities(normalized_query, [text for _, text in deduped])
        intent_scores = self._similarities(normalized_query, list(self._RECALL_ANCHORS))
        if candidate_scores is None or intent_scores is None:
            return ""
        intent_score = max(intent_scores, default=0.0)
        best_candidate_score = max(candidate_scores, default=0.0)
        if intent_score < 0.52 and best_candidate_score < 0.62:
            return ""

        minimum_score = 0.34 if intent_score >= 0.52 else 0.62
        ranked = sorted(
            (
                (score, index, kind, text)
                for index, ((kind, text), score) in enumerate(zip(deduped, candidate_scores))
                if score >= minimum_score
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if intent_score >= 0.52 and self._scene_summary:
            summary_row = next((row for row in ranked if row[2] == "剧情摘要"), None)
            if summary_row is None:
                summary_index = next((i for i, row in enumerate(deduped) if row[0] == "剧情摘要"), -1)
                if summary_index >= 0:
                    ranked.insert(0, (candidate_scores[summary_index], summary_index, *deduped[summary_index]))

        output: list[str] = []
        used = 0
        for _, _, kind, text in ranked[: max(1, int(limit))]:
            line = f"{kind}: {text}"
            remaining = max(0, int(max_chars) - used)
            if remaining <= 0:
                break
            output.append(line[:remaining])
            used += len(output[-1]) + 1
        return "\n".join(output).strip()

    def build_planner_hint(self, *, max_chars: int = 600) -> str:
        """Expose only compressed story memory for search-query disambiguation."""
        parts: list[str] = []
        if self._scene_summary and not self._is_runtime_noise(self._scene_summary):
            parts.append(f"剧情摘要: {self._scene_summary}")
        clean_facts = [fact for fact in self._facts[-24:] if fact and not self._is_runtime_noise(fact)]
        if clean_facts:
            parts.append("关键事实: " + "；".join(clean_facts))
        return "\n".join(parts)[: max(0, int(max_chars))].strip()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        raw_lines = data.get("lines", [])
        if isinstance(raw_lines, list):
            self._lines = [item for item in raw_lines if isinstance(item, dict)][-self._max_lines :]
        self._scene_summary = self._normalize(data.get("scene_summary", ""))[:1600]
        raw_facts = data.get("facts", [])
        if isinstance(raw_facts, list):
            self._facts = [self._normalize(item) for item in raw_facts if self._normalize(item)][-24:]
        self._total_lines = max(int(data.get("total_lines", 0) or 0), len(self._lines))
        self._summarized_line_count = max(0, int(data.get("summarized_line_count", 0) or 0))
        self._last_requested_line_count = self._summarized_line_count

    def flush(self) -> None:
        if self._dirty_count <= 0:
            return
        payload = {
            "version": 1,
            "total_lines": self._total_lines,
            "summarized_line_count": self._summarized_line_count,
            "scene_summary": self._scene_summary,
            "facts": self._facts[-24:],
            "lines": self._lines[-self._max_lines :],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)
        self._dirty_count = 0

    def _is_duplicate(self, text: str) -> bool:
        for item in reversed(self._lines[-3:]):
            previous = self._normalize(item.get("text", ""))
            if text == previous:
                return True
            if previous and SequenceMatcher(None, text, previous).ratio() >= 0.94:
                return True
        return False

    def _semantic_window_score(self, text: str) -> float | None:
        current_window = "\n".join(
            [
                *[self._normalize(item.get("text", "")) for item in self._lines[-3:]],
                text,
            ]
        )
        anchor_scores = [
            score
            for anchor in self._MOMENT_ANCHORS
            if (score := self._similarity(current_window, anchor)) is not None
        ]
        if not anchor_scores:
            return None
        anchor_score = max(anchor_scores)
        previous_window = "\n".join(
            self._normalize(item.get("text", "")) for item in self._lines[-7:-3]
        )
        previous_similarity = self._similarity(current_window, previous_window)
        semantic_change = 0.0 if previous_similarity is None else 1.0 - previous_similarity
        return (0.60 * anchor_score) + (0.40 * semantic_change)

    def _record_window_score(self, line_id: int, score: float) -> tuple[bool, float]:
        self._score_history.append((line_id, score))
        self._score_history = self._score_history[-24:]
        if len(self._score_history) < 7:
            return False, score

        before = self._score_history[-3]
        candidate = self._score_history[-2]
        after = self._score_history[-1]
        baseline_values = [value for _, value in self._score_history[:-2]][-20:]
        baseline = statistics.median(baseline_values)
        deviations = [abs(value - baseline) for value in baseline_values]
        mad = statistics.median(deviations)
        scale = max(0.015, 1.4826 * mad)
        relative_peak = (candidate[1] - baseline) / scale
        is_peak = (
            candidate[1] > before[1]
            and candidate[1] >= after[1]
            and relative_peak >= 1.5
            and candidate[0] - self._last_peak_line_id > 4
        )
        if is_peak:
            self._last_peak_line_id = candidate[0]
            return True, candidate[1]
        return False, score

    def observe(self, text: str, *, confidence: float = 0.0) -> NovelObservation:
        normalized = self._normalize(text)
        if len(normalized) < 2 or self._is_duplicate(normalized):
            return NovelObservation(False, False, "duplicate_or_empty", 0.0)

        self._total_lines += 1
        semantic_score = self._semantic_window_score(normalized)
        self._lines.append(
            {
                "id": self._total_lines,
                "captured_at": datetime.now().isoformat(timespec="seconds"),
                "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
                "text": normalized,
            }
        )
        if len(self._lines) > self._max_lines:
            del self._lines[:-self._max_lines]

        peak_detected = False
        reported_score = 0.0 if semantic_score is None else semantic_score
        if semantic_score is not None:
            peak_detected, reported_score = self._record_window_score(self._total_lines, semantic_score)
        if peak_detected:
            self._pending_candidate = True
        summary_due = self._total_lines - self._summarized_line_count >= self._summary_batch_size
        should_evaluate = self._pending_candidate or summary_due
        reason = "semantic_candidate" if self._pending_candidate else "summary_due" if summary_due else "buffered"
        self._dirty_count += 1
        if self._dirty_count >= 5:
            self.flush()
        return NovelObservation(True, should_evaluate, reason, reported_score)

    def build_evaluation_payload(self) -> dict | None:
        if self._total_lines <= self._last_requested_line_count:
            return None
        summary_due = self._total_lines - self._summarized_line_count >= self._summary_batch_size
        if not self._pending_candidate and not summary_due:
            return None
        self._last_requested_line_count = self._total_lines
        reason = "semantic_candidate" if self._pending_candidate else "summary_due"
        self._pending_candidate = False
        recent = self._lines[-self._recent_limit :]
        return {
            "reason": reason,
            "line_count": self._last_requested_line_count,
            "scene_summary": self._scene_summary,
            "facts": list(self._facts[-16:]),
            "recent_dialogue": "\n".join(str(item.get("text", "")) for item in recent),
        }

    @property
    def has_pending_evaluation(self) -> bool:
        if self._total_lines <= self._last_requested_line_count:
            return False
        summary_due = self._total_lines - self._summarized_line_count >= self._summary_batch_size
        return self._pending_candidate or summary_due

    def apply_evaluation(self, result: dict) -> None:
        if not isinstance(result, dict):
            return
        summary = self._normalize(result.get("scene_summary", ""))
        if summary:
            self._scene_summary = summary[:1600]
        raw_facts = result.get("facts", [])
        if isinstance(raw_facts, list):
            for value in raw_facts:
                fact = self._normalize(value)
                if fact and fact not in self._facts:
                    self._facts.append(fact[:240])
            self._facts = self._facts[-24:]
        self._summarized_line_count = max(
            self._summarized_line_count,
            min(self._last_requested_line_count, self._total_lines),
        )
        self._dirty_count += 1
        self.flush()
