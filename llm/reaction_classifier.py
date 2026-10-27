from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np


FILTER_MODELS = {
    "semantic": "24条整理＋语义筛选",
    "window-linear-v64": "window-linear-v64",
    "target-gru-fusion-r8": "本轮最佳：BGE＋GRU融合",
}


def selected_filter_model(settings) -> str:
    selected = getattr(settings, "vn_reaction_filter_model", "")
    if selected in FILTER_MODELS:
        return selected
    return "window-linear-v64" if getattr(settings, "enable_vn_reaction_classifier", False) else "semantic"


class LocalReactionClassifier:
    """Frozen local window classifier; call only from a background worker."""

    def __init__(self, model_path: Path, cpu_threads: int = 4) -> None:
        self.model_path = Path(model_path)
        self.cpu_threads = max(1, min(8, int(cpu_threads)))
        self.status = "not_loaded"
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None

    def warmup(self) -> bool:
        with self._lock:
            if self._session is not None:
                return True
            if self.status.startswith("unavailable:"):
                return False
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                metadata = json.loads((self.model_path / "classifier.json").read_text(encoding="utf-8"))
                self.model_id = metadata["model_id"]
                if self.model_id not in {"window-linear-v64", "target-gru-fusion-r8"}:
                    raise ValueError("Unsupported reaction model")
                encoder_path = self.model_path / metadata.get("encoder_directory", ".")
                tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
                tokenizer.truncation_side = "left"
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.cpu_threads
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    str(encoder_path / "encoder.onnx"),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                if self.model_id == "target-gru-fusion-r8":
                    self._threshold = float(metadata["threshold"])
                    self._head_session = ort.InferenceSession(
                        str(self.model_path / "head.onnx"), sess_options=options,
                        providers=["CPUExecutionProvider"])
                else:
                    with np.load(self.model_path / "head.npz", allow_pickle=False) as head:
                        self._weight = head["weight"].copy()
                        self._bias = head["bias"].copy()
                    if self._weight.shape != (3, 1024) or self._bias.shape != (3,):
                        raise ValueError("Invalid reaction head shape")
                self._tokenizer, self._session = tokenizer, session
                self.status = f"ready:{self.model_id}"
                self._predict([{"id": 1, "text": "本地模型预热"}])
                return True
            except Exception as exc:
                self._session = None
                self.status = f"unavailable:{type(exc).__name__}:{str(exc)[:160]}"
                return False

    @staticmethod
    def input_pairs(rows: list[dict]) -> list[tuple[str, str]]:
        if not rows:
            return []
        latest = rows[-1]
        rows = [r for r in rows if r.get("branch") == latest.get("branch")]
        active = [r for r in rows if 0 <= int(latest["id"]) - int(r["id"]) < 5]
        return [
            ("\n".join(str(r["text"]) for r in part[:-1]) or "无前文", str(part[-1]["text"]))
            for part in (active, rows)
        ]

    def _predict(self, rows: list[dict]) -> dict:
        if self.model_id == "target-gru-fusion-r8":
            return self._predict_fusion(rows)
        tokenizer, session = self._tokenizer, self._session
        features = []
        # Serial views have a lower measured latency tail than a padded batch.
        for background, target in self.input_pairs(rows):
            encoded = tokenizer(
                background, target, truncation="only_first", max_length=512, return_tensors="np"
            )
            inputs = {item.name: np.asarray(encoded[item.name], dtype=np.int64)
                      for item in session.get_inputs()}
            hidden = session.run(["last_hidden_state"], inputs)[0]
            mask = ((encoded["token_type_ids"] == 1) & (encoded["attention_mask"] == 1)
                    & (encoded["input_ids"] != tokenizer.sep_token_id))
            feature = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1)[:, None], 1)
            feature = feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
            features.append(feature)
        joined = np.concatenate(features, axis=1)
        joined /= np.maximum(np.linalg.norm(joined, axis=1, keepdims=True), 1e-12)
        logits = (joined @ self._weight.T + self._bias)[0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return {"class": int(logits.argmax()), "probabilities": probabilities.tolist(),
                "model": "window-linear-v64"}

    @staticmethod
    def fusion_messages(rows: list[dict]) -> list[dict]:
        latest = rows[-1]
        messages, speakers = [], {}
        for row in [r for r in rows if r.get("branch") == latest.get("branch")][-8:]:
            text = str(row["text"])
            agent = bool(row.get("is_agent", False))
            speaker = str(row.get("speaker") or "")
            if not speaker and not agent:
                match = re.match(r'^([^「『：:\n]{1,24})\s*[「『：:](.*)$', text, re.S)
                if match:
                    speaker, text = match.group(1).strip(), match.group(2).rstrip('」』')
            if agent:
                speaker = "Assistant"
            else:
                speaker = speakers.setdefault(speaker, f"Speaker_{len(speakers)}")
            messages.append({"speaker": speaker, "text": text, "is_agent": agent,
                             "agent_mentioned": bool(row.get("agent_mentioned", False))})
        return messages

    def _encode(self, encoded):
        inputs = {item.name: np.asarray(encoded[item.name], dtype=np.int64)
                  for item in self._session.get_inputs()}
        return self._session.run(["last_hidden_state"], inputs)[0]

    def _predict_fusion(self, rows: list[dict]) -> dict:
        messages = self.fusion_messages(rows)
        parts = [m["speaker"] + ": " + m["text"] for m in messages]
        target = []
        for part in (parts[-5:], parts):
            encoded = self._tokenizer("\n".join(part[:-1]) or "无前文", part[-1],
                                      truncation="only_first", max_length=512, return_tensors="np")
            hidden = self._encode(encoded)
            mask = ((encoded["token_type_ids"] == 1) & (encoded["attention_mask"] == 1)
                    & (encoded["input_ids"] != self._tokenizer.sep_token_id))
            pooled = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1)[:, None], 1)
            target.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))
        target = np.concatenate(target, axis=1)
        target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
        encoded = self._tokenizer([m["text"] for m in messages], padding=True,
                                  truncation=True, max_length=512, return_tensors="np")
        sentences = self._encode(encoded)[:, 0]
        sentences /= np.maximum(np.linalg.norm(sentences, axis=1, keepdims=True), 1e-12)
        agents = [i for i, m in enumerate(messages) if m["is_agent"]]
        scalars = [len(messages)/8, len({m["speaker"] for m in messages})/8, len(agents)/8,
                   (len(messages)-1-agents[-1])/8 if agents else 1., bool(agents),
                   messages[-1]["is_agent"], messages[-1]["agent_mentioned"],
                   any(m["agent_mentioned"] for m in messages[:-1])]
        probability = float(self._head_session.run(None, {
            "target": target.astype(np.float32), "sentences": sentences[None].astype(np.float32),
            "kinds": np.array([[int(m["is_agent"]) for m in messages]], dtype=np.int64),
            "scalars": np.array([scalars], dtype=np.float32)})[0][0, 0])
        # Binary speak is passed to the existing full-comment gate, not a learned length label.
        return {"class": 2 if probability >= self._threshold else 0,
                "probabilities": [1-probability, probability], "model": self.model_id,
                "speak_probability": probability, "threshold": self._threshold, "binary": True}

    def classify(self, rows: list[dict]) -> dict | None:
        if not rows or not self.warmup():
            return None
        with self._lock:
            try:
                return self._predict(rows)
            except Exception as exc:
                self._session = None
                self.status = f"unavailable:{type(exc).__name__}:{str(exc)[:160]}"
                return None
