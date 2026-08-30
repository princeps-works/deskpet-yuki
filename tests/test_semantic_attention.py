from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from desktop_pet.llm.semantic_attention import SemanticAttentionRouter


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def status(self) -> str:
        return "fake-embedding"

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            if "报错" in text or "Python" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "魔裁" in text or "樱羽艾玛" in text or "二阶堂希罗" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "饿殍" in text:
                vectors.append([0.42, 0.9075, 0.0])
            elif "屏幕" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class _ColdEncoder:
    @property
    def status(self) -> str:
        return "not_loaded"

    def encode(self, texts: list[str]) -> np.ndarray:
        raise AssertionError("cold similarity must not load the model on the caller thread")


def test_semantic_attention_filters_irrelevant_memory_and_keeps_required_visual():
    with TemporaryDirectory() as temp_dir:
        encoder = _FakeEncoder()
        router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            top_k=3,
            min_score=0.34,
            encoder=encoder,
        )
        result = router.route(
            query="这个 Python 报错应该怎么修复？",
            contexts={
                "recent": "你: 我在运行 Python\n桌宠: 请把报错贴出来",
                "memory": "- 用户去年喜欢过一首与当前问题无关的歌曲",
                "extra": "当前屏幕中显示了异常堆栈",
            },
            budgets={"recent": 500, "memory": 300, "extra": 500},
            required_sources={"extra"},
        )

        assert result.applied is True
        assert "Python" in result.contexts["recent"] or "报错" in result.contexts["recent"]
        assert "memory" not in result.contexts
        assert "异常堆栈" in result.contexts["extra"]
        assert result.source_weights["recent"] > result.source_weights["extra"]


def test_semantic_attention_reuses_cached_embeddings():
    with TemporaryDirectory() as temp_dir:
        encoder = _FakeEncoder()
        router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            encoder=encoder,
        )
        kwargs = {
            "query": "Python 报错",
            "contexts": {"recent": "Python 出现报错"},
            "budgets": {"recent": 300},
        }
        first = router.route(**kwargs)
        second = router.route(**kwargs)

        assert first.applied is True
        assert second.applied is True
        assert len(encoder.calls) == 1


def test_semantic_attention_exposes_local_similarity_for_event_gating():
    with TemporaryDirectory() as temp_dir:
        router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            encoder=_FakeEncoder(),
        )

        related = router.similarity("魔裁出现剧情反转", "樱羽艾玛揭露了真相")
        unrelated = router.similarity("魔裁出现剧情反转", "当前屏幕显示普通按钮")

        assert related is not None and related > 0.9
        assert unrelated is not None and unrelated < 0.1


def test_semantic_attention_batches_multiple_similarity_candidates():
    with TemporaryDirectory() as temp_dir:
        encoder = _FakeEncoder()
        router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            encoder=encoder,
        )

        scores = router.similarities(
            "魔裁出现剧情反转",
            ["樱羽艾玛揭露了真相", "当前屏幕显示普通按钮"],
        )

        assert scores is not None
        assert scores[0] > 0.9
        assert scores[1] < 0.1
        assert len(encoder.calls) == 1


def test_semantic_similarity_skips_cold_model_instead_of_blocking_ui():
    with TemporaryDirectory() as temp_dir:
        router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            encoder=_ColdEncoder(),
        )

        assert router.similarity("当前台词", "剧情反转") is None


def test_semantic_attention_missing_model_falls_back_without_dropping_context():
    with TemporaryDirectory() as temp_dir:
        router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir) / "missing-model",
        )
        result = router.route(
            query="测试",
            contexts={"recent": "保留这段上下文", "memory": "也保留这段记忆"},
            budgets={"recent": 100, "memory": 100},
        )

        assert result.applied is False
        assert result.contexts == {"recent": "保留这段上下文", "memory": "也保留这段记忆"}
        assert "model_missing" in result.backend


def test_semantic_attention_packs_highest_scoring_memory_before_older_memory():
    with TemporaryDirectory() as temp_dir:
        router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            top_k=4,
            min_score=0.34,
            encoder=_FakeEncoder(),
        )
        result = router.route(
            query="魔裁的主要角色有哪些？",
            contexts={
                "memory": (
                    "以前查询《饿殍》时讨论过良和穗，以及其他历史题材游戏。\n"
                    "魔裁的主要角色包括樱羽艾玛、二阶堂希罗，还有典狱长和看守。"
                )
            },
            budgets={"memory": 80},
        )

        assert result.applied is True
        assert result.contexts["memory"].startswith("魔裁的主要角色")
        assert "樱羽艾玛" in result.contexts["memory"]
