from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    @property
    def status(self) -> str: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class AttentionChunk:
    source: str
    index: int
    text: str
    semantic_score: float
    final_score: float
    attention_weight: float


@dataclass(frozen=True)
class AttentionResult:
    contexts: dict[str, str]
    source_weights: dict[str, float]
    selected_chunks: tuple[AttentionChunk, ...]
    applied: bool
    backend: str
    elapsed_ms: float

    @property
    def debug(self) -> str:
        selected = ",".join(
            f"{chunk.source}:{chunk.final_score:.2f}" for chunk in self.selected_chunks
        )
        return (
            f"semantic_attention={'on' if self.applied else 'fallback'};"
            f"backend={self.backend};selected={selected or 'none'};"
            f"elapsed_ms={self.elapsed_ms:.1f}"
        )


@dataclass(frozen=True)
class _RawChunk:
    source: str
    index: int
    text: str
    reliability: float
    recency: float


class OnnxTextEncoder:
    """Lazy local encoder for a Hugging Face tokenizer plus an ONNX transformer."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_length: int = 256,
        cpu_threads: int = 2,
    ) -> None:
        self._model_path = Path(model_path)
        self._max_length = max(32, min(512, int(max_length)))
        self._cpu_threads = max(1, min(16, int(cpu_threads)))
        self._tokenizer = None
        self._session = None
        self._status = "not_loaded"
        self._load_lock = threading.Lock()

    @property
    def status(self) -> str:
        return self._status

    def _find_model_file(self) -> Path | None:
        candidates = (
            self._model_path / "onnx" / "model_quantized.onnx",
            self._model_path / "onnx" / "model_int8.onnx",
            self._model_path / "model_quantized.onnx",
            self._model_path / "model_int8.onnx",
            self._model_path / "onnx" / "model.onnx",
            self._model_path / "model.onnx",
        )
        return next((path for path in candidates if path.is_file()), None)

    def _ensure_loaded(self) -> bool:
        if self._session is not None and self._tokenizer is not None:
            return True
        if self._status.startswith("unavailable:"):
            return False
        with self._load_lock:
            if self._session is not None and self._tokenizer is not None:
                return True
            model_file = self._find_model_file()
            if model_file is None:
                self._status = f"unavailable:model_missing:{self._model_path}"
                return False
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                options = ort.SessionOptions()
                options.intra_op_num_threads = self._cpu_threads
                options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self._model_path),
                    local_files_only=True,
                    use_fast=True,
                )
                session = ort.InferenceSession(
                    str(model_file),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                detail = re.sub(r"\s+", " ", str(exc)).strip()[:160]
                self._status = f"unavailable:{type(exc).__name__}:{detail}"
                return False
            self._tokenizer = tokenizer
            self._session = session
            self._status = f"onnx:{model_file.name}"
            return True

    def warmup(self) -> bool:
        if not self._ensure_loaded():
            return False
        try:
            self.encode(["本地语义注意力预热"])
            return True
        except Exception:
            return False

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if not self._ensure_loaded():
            raise RuntimeError(self._status)
        tokenizer = self._tokenizer
        session = self._session
        assert tokenizer is not None and session is not None

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        inputs: dict[str, np.ndarray] = {}
        for model_input in session.get_inputs():
            value = encoded.get(model_input.name)
            if value is None and model_input.name == "token_type_ids":
                value = np.zeros_like(encoded["input_ids"])
            if value is not None:
                inputs[model_input.name] = np.asarray(value, dtype=np.int64)
        outputs = session.run(None, inputs)
        if not outputs:
            raise RuntimeError("ONNX encoder returned no outputs")

        embeddings = None
        for output in outputs:
            array = np.asarray(output)
            if array.ndim == 2 and array.shape[0] == len(texts):
                embeddings = array
                break
        if embeddings is None:
            for output in outputs:
                array = np.asarray(output)
                if array.ndim == 3 and array.shape[0] == len(texts):
                    # BGE checkpoints use the first token representation for retrieval.
                    embeddings = array[:, 0, :]
                    break
        if embeddings is None:
            raise RuntimeError("ONNX encoder returned an unsupported output shape")

        normalized = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized /= np.maximum(norms, 1e-12)
        return normalized


class SemanticAttentionRouter:
    """Selects context chunks before the main LLM call using local embeddings."""

    _SOURCE_RELIABILITY = {
        "web": 0.90,
        "extra": 0.86,
        "recent": 0.72,
        "memory": 0.52,
    }

    def __init__(
        self,
        *,
        enabled: bool,
        model_path: Path,
        top_k: int = 8,
        min_score: float = 0.34,
        max_length: int = 256,
        cache_size: int = 512,
        cpu_threads: int = 2,
        encoder: TextEncoder | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.top_k = max(2, min(32, int(top_k)))
        self.min_score = max(0.0, min(1.0, float(min_score)))
        self.cache_size = max(0, min(10000, int(cache_size)))
        self._encoder: TextEncoder = encoder or OnnxTextEncoder(
            model_path,
            max_length=max_length,
            cpu_threads=cpu_threads,
        )
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._disabled_reason = ""

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        return self._disabled_reason or self._encoder.status

    def warmup(self) -> bool:
        if not self.enabled:
            return False
        warmup = getattr(self._encoder, "warmup", None)
        if not callable(warmup):
            try:
                self._encoder.encode(["本地语义注意力预热"])
                return True
            except Exception as exc:
                self._disabled_reason = f"unavailable:{type(exc).__name__}"
                return False
        ok = bool(warmup())
        if not ok:
            self._disabled_reason = self._encoder.status
        return ok

    def similarity(self, left: str, right: str) -> float | None:
        """Return local embedding cosine similarity, or None when unavailable."""
        scores = self.similarities(left, [right])
        return scores[0] if scores else None

    def similarities(self, query: str, texts: list[str]) -> list[float] | None:
        """Return cosine similarities for one query in a single local batch."""
        normalized_query = self._normalize_text(query)
        normalized_texts = [self._normalize_text(text) for text in texts]
        if not self.enabled or not normalized_query or not normalized_texts or any(not text for text in normalized_texts):
            return None
        # The desktop UI warms the encoder in a background thread. Do not make
        # the first visual-novel scan block on the model's cold start.
        if self._encoder.status == "not_loaded":
            return None
        try:
            vectors = self._encode_cached([normalized_query, *normalized_texts])
            return [
                max(0.0, min(1.0, float(vector @ vectors[0])))
                for vector in vectors[1:]
            ]
        except Exception as exc:
            detail = re.sub(r"\s+", " ", str(exc)).strip()[:140]
            self._disabled_reason = f"unavailable:{type(exc).__name__}:{detail}"
            return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        raw = str(text or "").lower()
        tokens = set(re.findall(r"[a-z0-9][a-z0-9._+-]{1,}", raw))
        for segment in re.findall(r"[\u4e00-\u9fff]+", raw):
            for size in (2, 3):
                tokens.update(
                    segment[index : index + size]
                    for index in range(max(0, len(segment) - size + 1))
                )
        return tokens

    @classmethod
    def _lexical_similarity(cls, left: str, right: str) -> float:
        left_tokens = cls._tokens(left)
        right_tokens = cls._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        return min(1.0, overlap / math.sqrt(len(left_tokens) * len(right_tokens)))

    @classmethod
    def _split_text(cls, text: str, *, max_chars: int = 260) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        pieces: list[str] = []
        for block in re.split(r"[\r\n]+", raw):
            block = cls._normalize_text(block)
            if not block:
                continue
            sentences = re.split(r"(?<=[。！？!?；;])\s*", block)
            current = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > max_chars:
                    if current:
                        pieces.append(current)
                        current = ""
                    pieces.extend(
                        sentence[index : index + max_chars]
                        for index in range(0, len(sentence), max_chars)
                    )
                    continue
                merged = f"{current} {sentence}".strip()
                if current and len(merged) > max_chars:
                    pieces.append(current)
                    current = sentence
                else:
                    current = merged
            if current:
                pieces.append(current)
        return pieces

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        value = str(text or "").strip()
        if len(value) <= max(0, int(limit)):
            return value
        return value[: max(0, int(limit))].rstrip()

    def _fallback(
        self,
        contexts: dict[str, str],
        budgets: dict[str, int],
        *,
        started: float,
        backend: str,
    ) -> AttentionResult:
        return AttentionResult(
            contexts={
                source: self._truncate(text, budgets.get(source, len(text)))
                for source, text in contexts.items()
            },
            source_weights={},
            selected_chunks=(),
            applied=False,
            backend=backend,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _encode_cached(self, texts: list[str]) -> np.ndarray:
        normalized = [self._normalize_text(text) for text in texts]
        found: dict[str, np.ndarray] = {}
        missing: list[str] = []
        with self._cache_lock:
            for text in normalized:
                cached = self._embedding_cache.get(text)
                if cached is None:
                    if text not in missing:
                        missing.append(text)
                else:
                    found[text] = cached
                    self._embedding_cache.move_to_end(text)
        if missing:
            vectors = self._encoder.encode(missing)
            if len(vectors) != len(missing):
                raise RuntimeError("embedding count mismatch")
            with self._cache_lock:
                for text, vector in zip(missing, vectors):
                    stored = np.asarray(vector, dtype=np.float32)
                    found[text] = stored
                    if self.cache_size > 0:
                        self._embedding_cache[text] = stored
                        self._embedding_cache.move_to_end(text)
                while len(self._embedding_cache) > self.cache_size:
                    self._embedding_cache.popitem(last=False)
        return np.stack([found[text] for text in normalized])

    def route(
        self,
        *,
        query: str,
        contexts: dict[str, str],
        budgets: dict[str, int],
        required_sources: set[str] | None = None,
        source_reliability: dict[str, float] | None = None,
    ) -> AttentionResult:
        started = time.perf_counter()
        cleaned_contexts = {
            source: str(text or "").strip() for source, text in contexts.items() if str(text or "").strip()
        }
        if not self.enabled or not cleaned_contexts or not str(query or "").strip():
            return self._fallback(
                cleaned_contexts,
                budgets,
                started=started,
                backend="disabled" if not self.enabled else "empty",
            )

        reliability_overrides = source_reliability or {}
        raw_chunks: list[_RawChunk] = []
        for source, text in cleaned_contexts.items():
            pieces = self._split_text(text)
            if source in {"recent", "memory"}:
                pieces = pieces[-12:]
            else:
                pieces = pieces[:12]
            total = max(1, len(pieces))
            for index, piece in enumerate(pieces):
                if source in {"recent", "memory"}:
                    recency = 0.35 + 0.65 * ((index + 1) / total)
                else:
                    recency = 1.0
                reliability = reliability_overrides.get(
                    source,
                    self._SOURCE_RELIABILITY.get(source, 0.60),
                )
                raw_chunks.append(
                    _RawChunk(
                        source=source,
                        index=index,
                        text=piece,
                        reliability=max(0.0, min(1.0, float(reliability))),
                        recency=recency,
                    )
                )
        if not raw_chunks:
            return self._fallback(cleaned_contexts, budgets, started=started, backend="no_chunks")

        try:
            vectors = self._encode_cached([str(query), *[chunk.text for chunk in raw_chunks]])
            query_vector = vectors[0]
            similarities = np.clip(vectors[1:] @ query_vector, 0.0, 1.0)
        except Exception as exc:
            detail = re.sub(r"\s+", " ", str(exc)).strip()[:140]
            self._disabled_reason = f"unavailable:{type(exc).__name__}:{detail}"
            return self._fallback(
                cleaned_contexts,
                budgets,
                started=started,
                backend=self._disabled_reason,
            )

        scored: list[tuple[float, float, _RawChunk]] = []
        for semantic_score, chunk in zip(similarities.tolist(), raw_chunks):
            lexical_score = self._lexical_similarity(query, chunk.text)
            final_score = (
                0.70 * float(semantic_score)
                + 0.12 * lexical_score
                + 0.11 * chunk.reliability
                + 0.07 * chunk.recency
            )
            # Reliability and recency may refine a relevant match, but must not
            # rescue a chunk that is weak both semantically and lexically.
            if float(semantic_score) < 0.34 and lexical_score < 0.06:
                final_score *= 0.55
            scored.append((final_score, float(semantic_score), chunk))
        scored.sort(key=lambda item: (-item[0], item[2].source, item[2].index))

        selected_rows: list[tuple[float, float, _RawChunk]] = []
        for row in scored:
            if row[0] < self.min_score:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= self.top_k:
                break

        for source in required_sources or set():
            if source not in cleaned_contexts or any(row[2].source == source for row in selected_rows):
                continue
            best = next((row for row in scored if row[2].source == source), None)
            if best is not None:
                selected_rows.append(best)

        if not selected_rows:
            selected_rows.append(scored[0])

        logits = np.asarray([row[0] for row in selected_rows], dtype=np.float32) / 0.18
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        weights /= max(float(weights.sum()), 1e-12)

        weighted_rows = list(zip(selected_rows, weights.tolist()))
        selected_chunks = tuple(
            AttentionChunk(
                source=row[2].source,
                index=row[2].index,
                text=row[2].text,
                semantic_score=row[1],
                final_score=row[0],
                attention_weight=float(weight),
            )
            for row, weight in weighted_rows
        )

        source_weights: dict[str, float] = {}
        for chunk in selected_chunks:
            source_weights[chunk.source] = source_weights.get(chunk.source, 0.0) + chunk.attention_weight
        total_source_weight = sum(source_weights.values()) or 1.0
        source_weights = {
            source: value / total_source_weight for source, value in source_weights.items()
        }

        selected_by_source: dict[str, list[tuple[int, str]]] = {}
        for chunk in selected_chunks:
            selected_by_source.setdefault(chunk.source, []).append((chunk.index, chunk.text))
        output_contexts: dict[str, str] = {}
        for source, rows in selected_by_source.items():
            # selected_chunks is already relevance-ordered. Keep that order for
            # factual sources so a low-scoring old memory cannot consume the
            # budget before the best match. Dialogue turns are the exception:
            # their chronology carries meaning and must remain intact.
            if source == "recent":
                rows.sort(key=lambda item: item[0])
            joined = "\n".join(text for _, text in rows)
            cap = max(0, int(budgets.get(source, len(joined))))
            strength = min(1.0, max(chunk.final_score for chunk in selected_chunks if chunk.source == source))
            minimum = min(cap, 240 if source in (required_sources or set()) else 80)
            effective_budget = max(minimum, int(cap * max(0.35, strength)))
            output_contexts[source] = self._truncate(joined, min(cap, effective_budget))

        return AttentionResult(
            contexts=output_contexts,
            source_weights=source_weights,
            selected_chunks=selected_chunks,
            applied=True,
            backend=self._encoder.status,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
