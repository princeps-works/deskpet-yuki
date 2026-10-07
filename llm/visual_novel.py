from __future__ import annotations

import json
import hashlib
import re
import shutil
import statistics
import threading
import time
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


@dataclass(frozen=True)
class DedupVerdict:
    """Outcome of one fact de-duplication decision (kept for calibration)."""

    duplicate: bool
    index: int
    ngram: float
    cosine: float | None
    char_ratio: float
    containment: float
    rule: str
    top_matches: tuple[dict[str, object], ...] = ()


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    """Character n-grams over CJK text plus lowercase latin words."""
    raw = str(text or "")
    compact = re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】…—\-]+", "", raw)
    grams: set[str] = set()
    for segment in re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+", compact):
        if len(segment) < size:
            grams.add(segment)
            continue
        grams.update(segment[index : index + size] for index in range(len(segment) - size + 1))
    grams.update(word.lower() for word in re.findall(r"[A-Za-z0-9]{2,}", raw))
    return grams


def _ngram_jaccard(left: str, right: str, size: int = 2) -> float:
    a = _char_ngrams(left, size)
    b = _char_ngrams(right, size)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _ngram_containment(left: str, right: str, size: int = 2) -> float:
    """Return how much of the shorter fact is covered by the longer one."""
    a = _char_ngrams(left, size)
    b = _char_ngrams(right, size)
    if not a or not b:
        return 0.0
    return len(a & b) / float(min(len(a), len(b)))


class DedupCalibrator:
    """Append-only log of de-duplication decisions.

    The semantic threshold cannot be picked from first principles: BGE-style
    Chinese embeddings compress unrelated text into the 0.3-0.7 band, so the
    only trustworthy way to tune it is to record the observed distribution.
    """

    def __init__(self, path: Path, *, enabled: bool = True, max_bytes: int = 4_000_000) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.max_bytes = max(100_000, int(max_bytes))
        self._lock = threading.Lock()
        self._disabled_reason = ""

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        return self._disabled_reason or "ready"

    def record(self, entry: dict) -> None:
        if not self.enabled or self._disabled_reason:
            return
        try:
            payload = dict(entry)
            payload.setdefault("ts", datetime.now().astimezone().isoformat(timespec="seconds"))
            line = json.dumps(payload, ensure_ascii=False)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.is_file() and self.path.stat().st_size > self.max_bytes:
                    self._rotate()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as exc:  # pragma: no cover - calibration must never break runtime
            self._disabled_reason = f"write_failed:{type(exc).__name__}"

    def _rotate(self) -> None:
        archive = self.path.with_suffix(self.path.suffix + ".1")
        try:
            if archive.is_file():
                archive.unlink()
            self.path.replace(archive)
        except Exception:
            try:
                self.path.unlink()
            except Exception:
                pass


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
            "version": 3,
            "total_lines": 0,
            "summarized_line_count": 0,
            "scene_summary": "",
            "scene_log": [],
            "facts": [],
            "characters": {},
            "viewer_notes": [],
            "game_title": "",
            "recent_comments": [],
            "lines": [],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset_story(self, name: str | None = None) -> Path:
        """Empty the active (or named) story cache in place."""
        target = self.directory / str(name) if name else self.active_path
        if not target.is_file():
            raise FileNotFoundError(f"缓存不存在：{target.name}")
        self._write_empty(target)
        return target

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


_PERSONA_ANCHORS_CACHE: tuple[str, ...] | None = None


def _persona_moment_anchors() -> tuple[str, ...]:
    """Persona-specific moment anchors, taken from the character sheet.

    Lazily imported and cached: the tracker must keep working when constructed
    standalone (tests build it directly) and a missing or malformed persona file
    must never break story tracking. An empty tuple simply means the persona
    track contributes nothing and the plot track decides alone, which is exactly
    the previous behaviour.
    """
    global _PERSONA_ANCHORS_CACHE
    if _PERSONA_ANCHORS_CACHE is None:
        try:
            from desktop_pet.config.prompts import get_persona_moment_anchors

            _PERSONA_ANCHORS_CACHE = tuple(get_persona_moment_anchors())
        except Exception:
            _PERSONA_ANCHORS_CACHE = ()
    return _PERSONA_ANCHORS_CACHE


class VisualNovelTracker:
    """Keep visual-novel dialogue and nominate semantic moments for the LLM."""

    # Two independent nomination tracks. Plot anchors are the original
    # "is this dramatically interesting" set; persona anchors come from the
    # character sheet and describe what *she* reacts to regardless of plot
    # importance (another girl, being lied to, being praised...). They are
    # combined with OR, so either track alone can nominate a moment.
    _PLOT_ANCHORS = (
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
        max_facts: int = 60,
        recent_comment_limit: int = 6,
        scene_log_limit: int = 200,
        dedup_ngram_merge: float = 0.75,
        dedup_ngram_guard: float = 0.12,
        dedup_semantic_merge: float = 0.82,
        calibrator: DedupCalibrator | None = None,
    ) -> None:
        self.path = Path(path)
        self._similarity_fn = similarity_fn
        self._similarities_fn = similarities_fn
        self._max_lines = max(60, int(max_lines))
        self._recent_limit = max(6, int(recent_limit))
        self._summary_batch_size = max(8, int(summary_batch_size))
        self._min_context_similarity = max(0.0, min(1.0, float(min_context_similarity)))
        self._max_facts = max(8, int(max_facts))
        self._recent_comment_limit = max(0, min(24, int(recent_comment_limit)))
        self._scene_log_limit = max(2, min(1000, int(scene_log_limit)))
        self._dedup_ngram_merge = max(0.3, min(1.0, float(dedup_ngram_merge)))
        self._dedup_ngram_guard = max(0.05, min(1.0, float(dedup_ngram_guard)))
        self._dedup_semantic_merge = max(0.5, min(1.0, float(dedup_semantic_merge)))
        self._calibrator = calibrator
        self._reset_story_state()
        self._load()

    def _reset_story_state(self) -> None:
        self._lines: list[dict] = []
        self._scene_summary = ""
        self._scene_log: list[str] = []
        self._facts: list[str] = []
        self._characters: dict[str, dict] = {}
        self._viewer_notes: list[str] = []
        self._game_title = ""
        self._recent_comments: list[str] = []
        self._total_lines = 0
        self._summarized_line_count = 0
        self._last_requested_line_count = 0
        self._pending_candidate = False
        self._score_history: list[tuple[int, float]] = []
        self._last_peak_line_id = 0
        self._dirty_count = 0
        self._last_observation_accepted = False
        self._pending_ocr: tuple[str, float, float] | None = None

    def switch_story(self, path: Path) -> None:
        next_path = Path(path)
        if next_path == self.path:
            self._reset_story_state()
            self._load()
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
        """Return relevant story-cache excerpts without modifying either memory store.

        Three confidence bands instead of a single accept/reject gate: a hard
        cutoff made the recall path silently useless whenever the player phrased
        a question slightly differently from the stored text.
        """
        normalized_query = self._normalize(query)
        if not normalized_query:
            return ""

        candidates: list[tuple[str, str]] = []
        if self._scene_summary and not self._is_runtime_noise(self._scene_summary):
            candidates.append(("剧情摘要", self._scene_summary))
        candidates.extend(
            ("场景", item)
            for item in self._scene_log[-self._scene_log_limit :]
            if item and not self._is_runtime_noise(item)
        )
        candidates.extend(
            ("人物", self._character_text(item))
            for item in self._characters.values()
            if self._character_text(item)
        )
        candidates.extend(
            ("关键事实", fact)
            for fact in self._facts
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
            return self._fallback_excerpt(max_chars=max_chars, query=normalized_query)

        intent_score = max(intent_scores, default=0.0)
        story_title = self.story_title.strip()
        if story_title and story_title.lower() != "default" and story_title in normalized_query:
            intent_score = max(intent_score, 0.90)
        best_candidate_score = max(candidate_scores, default=0.0)
        confidence = max(intent_score, best_candidate_score)

        if confidence < 0.42:
            return ""

        if confidence < 0.52:
            # Weak match: hand over only high-level plot context so a casual
            # question cannot drag unrelated story details into the reply.
            return self._fallback_excerpt(max_chars=max_chars, query=normalized_query)

        minimum_score = 0.34 if intent_score >= 0.52 else 0.52
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

        if not ranked:
            return self._fallback_excerpt(max_chars=max_chars, query=normalized_query)

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

    def _fallback_excerpt(self, *, max_chars: int, query: str = "") -> str:
        """Low-confidence excerpt:主线摘要 plus the newest facts only."""
        parts: list[str] = []
        normalized_query = self._normalize(query).casefold()
        matching_characters = []
        for item in self._characters.values():
            names = [str(item.get("name", "")).strip()]
            names.extend(str(value).strip() for value in item.get("aliases", []) if str(value).strip())
            if normalized_query and any(value.casefold() in normalized_query for value in names if value):
                text = self._character_text(item)
                if text:
                    matching_characters.append(text)
        if matching_characters:
            parts.append("人物: " + "；".join(matching_characters[:3]))
        if self._scene_summary and not self._is_runtime_noise(self._scene_summary):
            parts.append(f"剧情摘要: {self._scene_summary[:280]}")
        clean_facts = [fact for fact in self._facts[-4:] if fact and not self._is_runtime_noise(fact)]
        if clean_facts:
            budget = max(0, int(max_chars) - sum(len(part) + 1 for part in parts))
            if budget > 0:
                parts.append("关键事实: " + "；".join(clean_facts)[:budget])
        return "\n".join(parts).strip()[: max(0, int(max_chars))]

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
        raw_log = data.get("scene_log", [])
        if isinstance(raw_log, list):
            self._scene_log = [
                self._normalize(item)[:600] for item in raw_log if self._normalize(item)
            ][-self._scene_log_limit :]
        raw_facts = data.get("facts", [])
        if isinstance(raw_facts, list):
            self._facts = [self._normalize(item) for item in raw_facts if self._normalize(item)][
                -self._max_facts :
            ]
        raw_characters = data.get("characters", {})
        if isinstance(raw_characters, dict):
            self._characters = {
                str(character_id): item
                for character_id, item in raw_characters.items()
                if str(character_id).strip() and isinstance(item, dict)
            }
        raw_viewer_notes = data.get("viewer_notes", [])
        if isinstance(raw_viewer_notes, list):
            self._viewer_notes = [
                self._normalize(item)[:140]
                for item in raw_viewer_notes
                if isinstance(item, str) and self._normalize(item)
            ][:6]
        self._game_title = self._normalize(data.get("game_title", ""))[:200]
        raw_comments = data.get("recent_comments", [])
        if isinstance(raw_comments, list):
            self._recent_comments = [
                self._normalize(item)[:220] for item in raw_comments if self._normalize(item)
            ][-self._recent_comment_limit :] if self._recent_comment_limit else []
        self._total_lines = max(int(data.get("total_lines", 0) or 0), len(self._lines))
        self._summarized_line_count = max(0, int(data.get("summarized_line_count", 0) or 0))
        self._last_requested_line_count = self._summarized_line_count

    def flush(self) -> None:
        if self._dirty_count <= 0:
            return
        payload = {
            "version": 3,
            "total_lines": self._total_lines,
            "summarized_line_count": self._summarized_line_count,
            "scene_summary": self._scene_summary,
            "scene_log": self._scene_log[-self._scene_log_limit :],
            "facts": self._facts[-self._max_facts :],
            "characters": self._characters,
            "viewer_notes": self._viewer_notes[:6],
            "game_title": self._game_title,
            "recent_comments": self._recent_comments[-self._recent_comment_limit :]
            if self._recent_comment_limit
            else [],
            "lines": self._lines[-self._max_lines :],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)
        self._dirty_count = 0

    # -- fact de-duplication ------------------------------------------------

    def _fact_verdict(self, candidate: str) -> DedupVerdict:
        """Decide whether a candidate fact duplicates one already stored.

        Two independent evidence sources are combined, because a pure embedding
        threshold is unreliable on Chinese short text: BGE-style models put
        unrelated sentences well above zero, so a naive cut merges facts that
        merely share a topic. Character n-gram overlap is exact and catches the
        common case (same fact, different phrasing), while the cosine score
        breaks ties.
        """
        if not self._facts:
            return DedupVerdict(False, -1, 0.0, None, 0.0, 0.0, "no_existing")

        metrics = [
            {
                "index": index,
                "ngram": _ngram_jaccard(candidate, existing),
                "containment": _ngram_containment(candidate, existing),
                "char_ratio": SequenceMatcher(None, candidate, existing).ratio(),
                "cosine": None,
            }
            for index, existing in enumerate(self._facts)
        ]
        scores = self._similarities(candidate, self._facts)
        if scores is not None:
            for item, score in zip(metrics, scores):
                item["cosine"] = float(score)

        semantic_ranked = sorted(
            metrics,
            key=lambda item: (
                -1.0 if item["cosine"] is None else float(item["cosine"]),
                float(item["containment"]),
                float(item["ngram"]),
            ),
            reverse=True,
        )
        lexical_best = max(
            metrics,
            key=lambda item: (
                float(item["ngram"]),
                float(item["char_ratio"]),
                float(item["containment"]),
            ),
        )

        chosen = lexical_best
        rule = "kept"
        duplicate = False
        if float(lexical_best["ngram"]) >= self._dedup_ngram_merge:
            duplicate = True
            rule = "ngram"
        elif float(lexical_best["char_ratio"]) >= 0.88:
            duplicate = True
            rule = "char_fallback"
        elif scores is not None:
            for item in semantic_ranked[:3]:
                cosine = float(item["cosine"])
                lexical_support = (
                    float(item["ngram"]) >= self._dedup_ngram_guard
                    or float(item["containment"]) >= 0.32
                )
                if cosine >= self._dedup_semantic_merge and lexical_support:
                    chosen = item
                    duplicate = True
                    rule = "semantic+lexical"
                    break

        top_matches = tuple(
            {
                "index": int(item["index"]),
                "matched": self._facts[int(item["index"])][:160],
                "cosine": item["cosine"],
                "ngram": round(float(item["ngram"]), 4),
                "containment": round(float(item["containment"]), 4),
                "char_ratio": round(float(item["char_ratio"]), 4),
            }
            for item in semantic_ranked[:3]
        )
        return DedupVerdict(
            duplicate=duplicate,
            index=int(chosen["index"]),
            ngram=float(chosen["ngram"]),
            cosine=None if chosen["cosine"] is None else float(chosen["cosine"]),
            char_ratio=float(chosen["char_ratio"]),
            containment=float(chosen["containment"]),
            rule=rule,
            top_matches=top_matches,
        )

    def _merge_facts(self, incoming: list[str], *, keep_longest: bool = True) -> tuple[list[str], int]:
        """Merge new facts into the store, returning (added, duplicate_count)."""
        added: list[str] = []
        duplicates = 0
        for raw in incoming:
            fact = self._normalize(raw)[:240]
            if not fact:
                continue
            verdict = self._fact_verdict(fact)
            if self._calibrator is not None:
                self._calibrator.record(
                    {
                        "story": self.path.name,
                        "candidate": fact[:160],
                        "matched": self._facts[verdict.index][:160] if verdict.index >= 0 else "",
                        "ngram": round(verdict.ngram, 4),
                        "cosine": None if verdict.cosine is None else round(verdict.cosine, 4),
                        "char_ratio": round(verdict.char_ratio, 4),
                        "containment": round(verdict.containment, 4),
                        "top_matches": list(verdict.top_matches),
                        "rule": verdict.rule,
                        "decision": "merge" if verdict.duplicate else "keep",
                    }
                )
            if verdict.duplicate and verdict.index >= 0:
                duplicates += 1
                existing = self._facts[verdict.index]
                if keep_longest and len(fact) > len(existing):
                    self._facts[verdict.index] = fact
                continue
            if fact not in self._facts:
                self._facts.append(fact)
                added.append(fact)
        if len(self._facts) > self._max_facts:
            # Drop the oldest entries first: recent plot context matters most.
            self._facts = self._facts[-self._max_facts :]
        return added, duplicates

    def _select_facts_for_context(
        self,
        query: str,
        *,
        limit: int = 16,
        max_chars: int = 760,
        recent_count: int = 4,
    ) -> list[str]:
        """Select relevant facts while always retaining a small recent tail."""
        if not self._facts:
            return []
        limit = max(1, int(limit))
        max_chars = max(120, int(max_chars))
        recent_count = max(1, min(limit, int(recent_count)))
        recent_indexes = list(range(max(0, len(self._facts) - recent_count), len(self._facts)))
        selected: set[int] = set()
        used = 0
        for index in recent_indexes:
            cost = len(self._facts[index]) + (1 if selected else 0)
            if selected and used + cost > max_chars:
                continue
            selected.add(index)
            used += cost

        scores = self._similarities(self._normalize(query), self._facts)
        if scores is not None:
            ranked = sorted(range(len(self._facts)), key=lambda index: scores[index], reverse=True)
        else:
            ranked = list(range(len(self._facts) - 1, -1, -1))

        for index in ranked:
            if index in selected:
                continue
            cost = len(self._facts[index]) + (1 if selected else 0)
            if used + cost > max_chars:
                continue
            selected.add(index)
            used += cost
            if len(selected) >= limit:
                break

        output: list[str] = []
        for index in sorted(selected):
            fact = self._facts[index]
            output.append(fact)
        return output

    @staticmethod
    def _character_text(item: dict) -> str:
        name = str(item.get("name", "")).strip()
        aliases = [str(value).strip() for value in item.get("aliases", []) if str(value).strip()]
        label = name or (aliases[0] if aliases else "")
        if not label:
            return ""
        parts = [label]
        role = str(item.get("role", "")).strip()
        if role:
            parts.append(role)
        appearance = [str(value).strip() for value in item.get("appearance", []) if str(value).strip()]
        known = [str(value).strip() for value in item.get("known_so_far", []) if str(value).strip()]
        if appearance:
            parts.append("、".join(appearance[-2:]))
        if known:
            parts.append("；".join(known[-3:]))
        return "：".join(parts)

    def _select_characters_for_context(
        self,
        query: str,
        *,
        limit: int = 5,
        max_chars: int = 420,
    ) -> list[dict]:
        if not self._characters:
            return []
        records = list(self._characters.items())
        descriptions = [self._character_text(item) for _, item in records]
        semantic = self._similarities(self._normalize(query), descriptions)
        normalized_query = self._normalize(query).casefold()

        ranked: list[tuple[float, int, str, dict]] = []
        for index, (character_id, item) in enumerate(records):
            names = [str(item.get("name", "")).strip()]
            names.extend(str(value).strip() for value in item.get("aliases", []) if str(value).strip())
            mentioned = any(value.casefold() in normalized_query for value in names if value)
            score = semantic[index] if semantic is not None else 0.0
            last_seen = int(item.get("last_seen_line", 0) or 0)
            ranked.append((score + (1.0 if mentioned else 0.0), last_seen, character_id, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)

        output: list[dict] = []
        used = 0
        for _, _, character_id, item in ranked[: max(1, int(limit))]:
            compact = {
                "id": character_id,
                "name": str(item.get("name", "")).strip(),
                "aliases": list(item.get("aliases", []))[-3:],
                "role": str(item.get("role", "")).strip(),
                "appearance": list(item.get("appearance", []))[-2:],
                "known_so_far": list(item.get("known_so_far", []))[-4:],
                "status": str(item.get("status", "provisional")),
                "confidence": item.get("confidence", 0.0),
            }
            encoded = json.dumps(compact, ensure_ascii=False)
            while len(encoded) > max_chars - used and compact["known_so_far"]:
                compact["known_so_far"].pop(0)
                encoded = json.dumps(compact, ensure_ascii=False)
            while len(encoded) > max_chars - used and compact["appearance"]:
                compact["appearance"].pop(0)
                encoded = json.dumps(compact, ensure_ascii=False)
            if len(encoded) > max_chars - used:
                continue
            output.append(compact)
            used += len(encoded) + 1
        return output

    def _apply_character_updates(self, raw_updates: object) -> list[str]:
        if not isinstance(raw_updates, list):
            return []

        def clean_list(value: object, *, limit: int, chars: int) -> list[str]:
            if not isinstance(value, list):
                return []
            output: list[str] = []
            for raw in value:
                text = self._normalize(raw)[:chars]
                if text and text not in output:
                    output.append(text)
                if len(output) >= limit:
                    break
            return output

        changed: list[str] = []
        evidence_ids = [int(item.get("id", 0) or 0) for item in self._lines[-6:] if item.get("id")]
        for raw in raw_updates[:12]:
            if not isinstance(raw, dict):
                continue
            requested_id = self._normalize(raw.get("id", ""))
            name = self._normalize(raw.get("name", ""))[:60]
            aliases = clean_list(raw.get("aliases", []), limit=8, chars=60)
            if not name and not aliases:
                continue

            character_id = requested_id if requested_id in self._characters else ""
            if not character_id and name:
                for existing_id, existing in self._characters.items():
                    existing_names = [self._normalize(existing.get("name", ""))]
                    existing_names.extend(clean_list(existing.get("aliases", []), limit=8, chars=60))
                    if name in existing_names:
                        character_id = existing_id
                        break
            if not character_id:
                seed = name or aliases[0]
                digest = hashlib.sha1(
                    f"{seed}|{self._total_lines}|{len(self._characters)}".encode("utf-8")
                ).hexdigest()[:10]
                character_id = f"char_{digest}"

            existing = dict(self._characters.get(character_id, {}))
            old_name = self._normalize(existing.get("name", ""))
            merged_aliases = clean_list(existing.get("aliases", []), limit=8, chars=60)
            if old_name and name and old_name != name and old_name not in merged_aliases:
                merged_aliases.append(old_name)
            for alias in aliases:
                if alias != name and alias not in merged_aliases:
                    merged_aliases.append(alias)

            appearance = clean_list(existing.get("appearance", []), limit=8, chars=80)
            for value in clean_list(raw.get("appearance", []), limit=8, chars=80):
                if value not in appearance:
                    appearance.append(value)
            known = clean_list(existing.get("known_so_far", []), limit=16, chars=120)
            for value in clean_list(raw.get("known_so_far", []), limit=16, chars=120):
                if value not in known:
                    known.append(value)
            prior_evidence = [int(value) for value in existing.get("evidence_line_ids", []) if str(value).isdigit()]
            merged_evidence = list(dict.fromkeys([*prior_evidence, *evidence_ids]))[-24:]
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(float(existing.get("confidence", 0.0) or 0.0), confidence)
            final_name = name or old_name
            self._characters[character_id] = {
                "id": character_id,
                "name": final_name,
                "aliases": merged_aliases[-8:],
                "role": self._normalize(raw.get("role", ""))[:100]
                or self._normalize(existing.get("role", ""))[:100],
                "appearance": appearance[-8:],
                "known_so_far": known[-16:],
                "status": "confirmed" if final_name and confidence >= 0.75 else "provisional",
                "confidence": round(confidence, 4),
                "first_seen_line": int(existing.get("first_seen_line", self._total_lines) or 0),
                "last_seen_line": self._total_lines,
                "evidence_line_ids": merged_evidence,
            }
            changed.append(character_id)

        if len(self._characters) > 120:
            newest = sorted(
                self._characters.items(),
                key=lambda row: int(row[1].get("last_seen_line", 0) or 0),
                reverse=True,
            )[:120]
            self._characters = dict(reversed(newest))
        return changed

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
        def best_anchor_score(anchors) -> float | None:
            scores = [
                score
                for anchor in anchors
                if (score := self._similarity(current_window, anchor)) is not None
            ]
            return max(scores) if scores else None

        # OR over the two tracks: whichever the window resembles more decides the
        # anchor term. Before this, only the plot track existed, so a moment that
        # mattered to the character but not to the plot (a rival appearing, a
        # promise broken) could never be nominated.
        plot_score = best_anchor_score(self._PLOT_ANCHORS)
        persona_score = best_anchor_score(_persona_moment_anchors())
        available = [score for score in (plot_score, persona_score) if score is not None]
        if not available:
            return None
        anchor_score = max(available)
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
            self._last_observation_accepted = False
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
        self._last_observation_accepted = True
        if self._dirty_count >= 5:
            self.flush()
        return NovelObservation(True, should_evaluate, reason, reported_score)

    @staticmethod
    def _ocr_extends(shorter: str, longer: str) -> bool:
        short = "".join(char for char in shorter if char.isalnum())
        long = "".join(char for char in longer if char.isalnum())
        if len(short) < 10 or len(long) < len(short) + 2:
            return False
        return SequenceMatcher(None, short, long[: len(short)]).ratio() >= 0.85

    def _commit_ocr(self, text: str, confidence: float) -> NovelObservation:
        if self._lines and self._total_lines > max(self._summarized_line_count, self._last_requested_line_count):
            last = self._lines[-1]
            if self._ocr_extends(str(last.get("text", "")), text):
                last.update(text=text, confidence=round(confidence, 4), captured_at=datetime.now().isoformat(timespec="seconds"))
                self._dirty_count += 1
                self._last_observation_accepted = True
                if self._dirty_count >= 5:
                    self.flush()
                return NovelObservation(True, self.has_pending_evaluation, "extended_previous", 0.0)
        return self.observe(text, confidence=confidence)

    def observe_ocr(self, text: str, *, confidence: float = 0.0, now: float | None = None, frame_stable: bool = True) -> NovelObservation:
        """Wait briefly for typewriter text; keep only the latest version of one line."""
        value = self._normalize(text)
        current = time.monotonic() if now is None else float(now)
        self._last_observation_accepted = False
        pending = self._pending_ocr
        if pending is None:
            if value:
                if self._is_duplicate(value) and not (
                    self._lines and self._ocr_extends(str(self._lines[-1].get("text", "")), value)
                ):
                    return NovelObservation(False, False, "duplicate_or_empty", 0.0)
                self._pending_ocr = (value, confidence, current)
            return NovelObservation(False, False, "stabilizing", 0.0)

        previous, previous_confidence, since = pending
        if value == previous or not value:
            if not value or not frame_stable or current - since < 0.8:
                return NovelObservation(False, False, "stabilizing", 0.0)
            self._pending_ocr = None
            return self._commit_ocr(previous, previous_confidence)
        if self._ocr_extends(previous, value):
            self._pending_ocr = (value, confidence, current)
            return NovelObservation(False, False, "stabilizing", 0.0)
        if self._ocr_extends(value, previous):
            return NovelObservation(False, False, "stabilizing", 0.0)

        self._pending_ocr = (value, confidence, current)
        if len(previous) < 8:
            return NovelObservation(False, False, "stabilizing", 0.0)
        return self._commit_ocr(previous, previous_confidence)

    @property
    def has_pending_ocr(self) -> bool:
        return self._pending_ocr is not None

    @property
    def last_observation_accepted(self) -> bool:
        """True when the most recent observe() call stored new dialogue.

        Used by the vision gate: a changing story is the only situation where
        spending an image request is worthwhile.
        """
        return bool(self._last_observation_accepted)

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
            "story_title": self.story_title,
            "line_count": self._last_requested_line_count,
            "scene_summary": self._scene_summary,
            "scene_log": list(self._scene_log[-self._scene_log_limit :]),
            "facts": self._select_facts_for_context(
                "\n".join(str(item.get("text", "")) for item in recent[-6:])
            ),
            "characters": self._select_characters_for_context(
                "\n".join(str(item.get("text", "")) for item in recent[-6:])
            ),
            "viewer_notes": list(self._viewer_notes),
            "recent_dialogue": "\n".join(str(item.get("text", "")) for item in recent),
            "reaction_dialogue": "\n".join(str(item.get("text", "")) for item in recent[-3:]),
            "new_dialogue": "\n".join(
                str(item.get("text", "")) for item in recent
                if int(item.get("id", 0)) > self._summarized_line_count
            ),
            "previous_dialogue": "\n".join(
                str(item.get("text", "")) for item in recent
                if int(item.get("id", 0)) <= self._summarized_line_count
            ),
            "recent_comments": list(self._recent_comments[-self._recent_comment_limit :])
            if self._recent_comment_limit
            else [],
        }

    @property
    def has_pending_evaluation(self) -> bool:
        if self._total_lines <= self._last_requested_line_count:
            return False
        summary_due = self._total_lines - self._summarized_line_count >= self._summary_batch_size
        return self._pending_candidate or summary_due

    def restore_evaluation_cursor(self) -> None:
        """Give back an evaluation slot after a failed generation.

        build_evaluation_payload() consumes the pending peak immediately, so a
        malformed LLM response would otherwise discard that story moment for
        good: the peak is cleared, the cursor advanced, and nothing is retried.
        """
        self._last_requested_line_count = min(
            self._last_requested_line_count,
            max(0, self._total_lines - 1),
        )
        self._pending_candidate = True

    def remember_comment(self, comment: str) -> None:
        text = self._normalize(comment)[:220]
        if not text:
            return
        if text in self._recent_comments:
            self._recent_comments.remove(text)
        self._recent_comments.append(text)
        if self._recent_comment_limit:
            self._recent_comments = self._recent_comments[-self._recent_comment_limit :]
        else:
            self._recent_comments = []
        self._dirty_count += 1

    @property
    def recent_comments(self) -> list[str]:
        return list(self._recent_comments)

    @property
    def story_title(self) -> str:
        return self.path.stem

    @property
    def game_title(self) -> str:
        return self._game_title

    @property
    def has_story_content(self) -> bool:
        return bool(
            self._total_lines or self._scene_summary or self._scene_log
            or self._facts or self._characters or self._viewer_notes or self._recent_comments
        )

    def bind_game_title(self, title: str) -> None:
        value = self._normalize(title)[:200]
        if value and value != self._game_title:
            previous_title = self._game_title
            previous_dirty = self._dirty_count
            self._game_title = value
            self._dirty_count += 1
            try:
                self.flush()
            except Exception:
                self._game_title = previous_title
                self._dirty_count = previous_dirty
                raise

    @property
    def scene_summary(self) -> str:
        return str(self._scene_summary)

    @property
    def scene_log(self) -> list[str]:
        return list(self._scene_log)

    @property
    def facts(self) -> list[str]:
        return list(self._facts)

    @property
    def characters(self) -> dict[str, dict]:
        return json.loads(json.dumps(self._characters, ensure_ascii=False))

    @property
    def summarized_line_count(self) -> int:
        return int(self._summarized_line_count)

    @property
    def last_requested_line_count(self) -> int:
        return int(self._last_requested_line_count)

    @property
    def total_lines(self) -> int:
        return int(self._total_lines)

    def scene_session_digest(self, *, max_chars: int = 700) -> str:
        """Compact plot context for the memory store (summary + newest scenes)."""
        parts: list[str] = []
        if self._scene_summary:
            parts.append(f"主线：{self._scene_summary}")
        for index, item in enumerate(self._scene_log[-3:], start=1):
            parts.append(f"近况{index}：{item}")
        return self._normalize("\n".join(parts))[: max_chars]

    def apply_evaluation(self, result: dict) -> None:
        if not isinstance(result, dict):
            return
        raw_viewer_notes = result.get("viewer_notes")
        if isinstance(raw_viewer_notes, list):
            self._viewer_notes = list(dict.fromkeys(
                self._normalize(item)[:140]
                for item in raw_viewer_notes
                if isinstance(item, str) and self._normalize(item)
            ))[:6]
        delta = self._normalize(result.get("scene_delta", ""))
        if delta:
            delta = delta[:600]
            if not self._scene_log or self._scene_log[-1] != delta:
                self._scene_log.append(delta)
                if len(self._scene_log) > self._scene_log_limit:
                    self._scene_log = self._scene_log[-self._scene_log_limit :]
        summary = self._normalize(result.get("scene_summary", ""))
        if summary:
            # The prompt requests a cumulative summary. Accept a newer compact
            # rewrite even when it is shorter; scene_log and facts preserve the
            # detailed history independently.
            self._scene_summary = summary[:1600]
        updated: list[str] = []
        raw_updates = result.get("fact_updates", [])
        if isinstance(raw_updates, list):
            for item in raw_updates:
                if not isinstance(item, dict):
                    continue
                existing = self._normalize(item.get("existing", ""))
                replacement = self._normalize(item.get("replacement", ""))[:240]
                if not existing or not replacement or existing == replacement:
                    continue
                try:
                    index = self._facts.index(existing)
                except ValueError:
                    continue
                self._facts[index] = replacement
                updated.append(replacement)
                if self._calibrator is not None:
                    self._calibrator.record(
                        {
                            "story": self.path.name,
                            "candidate": replacement[:160],
                            "matched": existing[:160],
                            "rule": "explicit_revision",
                            "decision": "replace",
                        }
                    )
        result["updated_characters"] = self._apply_character_updates(
            result.get("character_updates", [])
        )
        added: list[str] = []
        raw_facts = result.get("facts", [])
        if isinstance(raw_facts, list):
            added, _duplicates = self._merge_facts([str(value) for value in raw_facts])
        # Exposed so the caller can mirror only genuinely new knowledge.
        result["added_facts"] = added
        result["updated_facts"] = updated
        self._summarized_line_count = max(
            self._summarized_line_count,
            min(self._last_requested_line_count, self._total_lines),
        )
        self._dirty_count += 1
        self.flush()
