"""Persona-driven comment timing: the character sheet and the second track.

Two changes are covered here.

**The character sheet was a lossy import.** ``data/persona_initial.json`` had the
card's *scenario* text pasted into its ``personality`` field (measured 0.966
similar against ``sister-high.scenario``, 0.116 against the real personality),
and the fields that actually define what she reacts to -- appearance, 喜好,
称呼, and the "信任会在涉及其他女生或受到欺骗时下降" rule -- were never
imported at all. The file now carries the card's fields, and the persona block is
rendered from them.

**Timing was plot-only.** The tracker scored each dialogue window against five
plot anchors (twist / humour / touching / choice / climax) and nothing else, so a
moment that mattered only to the character could never be nominated. There is now
a second anchor set taken from the character sheet, combined with OR: either
track alone can nominate.

The two behavioural tests use a stub similarity function, so they need no
embedding model and are fully deterministic.
"""

from __future__ import annotations

import ast
import json
from tempfile import TemporaryDirectory
from difflib import SequenceMatcher
from pathlib import Path

import desktop_pet.llm.visual_novel as vn_module
from desktop_pet.config.prompts import (
    INITIAL_PERSONA,
    get_persona_moment_anchors,
    get_persona_moment_criteria,
    get_system_chat_prompt,
    get_system_screen_comment_prompt,
    get_system_visual_novel_prompt,
)
from desktop_pet.llm.visual_novel import VisualNovelTracker

_ROOT = Path(__file__).resolve().parent.parent
_SOURCE = (Path(vn_module.__file__)).read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def test_repeated_reaction_stays_silent_but_keeps_story_update():
    from desktop_pet.llm.comment_engine import CommentEngine

    class Client:
        def chat(self, user_text, system_prompt):
            return json.dumps({"should_comment": True, "moment_type": "tender",
                               "reaction_repeats": True, "comment": "还是好温柔。",
                               "scene_summary": "他把伞留给了她。", "facts": ["他留下伞"]})

    result = CommentEngine(Client()).evaluate_visual_novel({"recent_dialogue": "他留下伞。"})
    assert result["should_comment"] is False
    assert result["scene_summary"] == "他把伞留给了她。"
    assert result["facts"] == ["他留下伞"]


def test_new_reaction_can_keep_the_same_emotion_and_ignores_rough_emotion_hint():
    from desktop_pet.llm.comment_engine import CommentEngine

    class Client:
        def chat(self, user_text, system_prompt):
            assert "本轮新增台词" in user_text and "把自己的伞也留下了" in user_text
            assert user_text.count("他把自己的伞也留下了。") == 1
            assert "当前情绪上下文参考:" not in user_text
            return json.dumps({"should_comment": True, "moment_type": "tender",
                               "reaction_repeats": False, "comment": "他自己怎么办呀，会淋湿的。"})

    result = CommentEngine(Client()).evaluate_visual_novel({
        "recent_dialogue": "他陪她走回家。\n他把自己的伞也留下了。",
        "new_dialogue": "他把自己的伞也留下了。",
        "recent_comments": ["有人陪着回家，真好。"],
    })
    assert result["should_comment"] is True


def test_new_dialogue_excludes_already_summarized_lines():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json", summary_batch_size=8)
        for text in ("他先陪她走回家。", "街灯亮了起来。", "路边的小店关门了。", "她忘带了雨伞。",
                     "天空传来雷声。", "两人停在屋檐下。", "他想起她明天要早起。", "她向他道谢。"):
            tracker.observe(text)
        first = tracker.build_evaluation_payload()
        assert first is not None
        tracker.apply_evaluation({"line_count": first["line_count"], "scene_summary": "他送她回家。"})
        for text in ("他把自己的伞也留下了。", "她目送他跑进雨中。", "那把伞的伞柄已经旧了。", "雨越下越大。",
                     "他的背影消失在拐角。", "她握紧手里的伞。", "她推开了家门。", "屋里传来母亲的呼唤。"):
            tracker.observe(text)
        second = tracker.build_evaluation_payload()
        assert second is not None
        assert "他先陪她走回家" in second["recent_dialogue"]
        assert "他先陪她走回家" not in second["new_dialogue"]
        assert "把自己的伞也留下了" in second["new_dialogue"]
        assert "把自己的伞也留下了" not in second["previous_dialogue"]


def _segment(name: str) -> str:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_SOURCE, node) or ""
    raise AssertionError(f"function {name} not found")


# --------------------------------------------------------------------------
# The character sheet
# --------------------------------------------------------------------------


def test_the_personality_field_is_a_personality_not_the_scenario():
    """The original defect: scenario text pasted into ``personality``.

    Asserts the *shape* rather than any particular wording -- the character text
    belongs to whoever owns the persona file and is expected to change.
    """
    personality = INITIAL_PERSONA["personality"]
    scenario = INITIAL_PERSONA["scenario"]
    assert personality and scenario
    assert personality != scenario
    # A pasted duplicate would be near-identical, whatever the wording.
    similarity = SequenceMatcher(None, personality, scenario).ratio()
    assert similarity < 0.8, f"personality looks like the scenario (ratio {similarity:.2f})"


def test_fields_the_import_used_to_drop_are_present():
    """The fields themselves must survive a persona rewrite."""
    for key in ("age", "identity", "appearance", "traits", "likes", "taboos"):
        assert INITIAL_PERSONA.get(key), f"{key} missing from the character sheet"
    for key in ("traits", "likes", "taboos"):
        assert isinstance(INITIAL_PERSONA[key], list)
        assert all(str(item).strip() for item in INITIAL_PERSONA[key])


def test_the_self_aware_virtual_character_line_is_not_used():
    """Deliberately excluded: it belongs to the other cards, not this one."""
    everything = json.dumps(INITIAL_PERSONA, ensure_ascii=False)
    assert "游戏中的角色" not in everything


def test_persona_block_reaches_all_three_prompts():
    """Each prompt must render whatever the sheet currently holds."""
    appearance = INITIAL_PERSONA["appearance"].strip()
    address = INITIAL_PERSONA["address_style"].strip()
    assert appearance and address, "the sheet needs these to be renderable"
    # The block may wrap the text, so compare on the first clause.
    appearance_probe = appearance.split("，")[0].strip()
    address_probe = address.split("；")[0].strip()
    for prompt in (
        get_system_chat_prompt(),
        get_system_screen_comment_prompt(),
        get_system_visual_novel_prompt(),
    ):
        assert "【人设】" in prompt
        assert appearance_probe in prompt, f"appearance {appearance_probe!r} not rendered"
        assert address_probe in prompt, f"address style {address_probe!r} not rendered"


def test_prompts_stay_distinct_and_within_budget():
    chat = get_system_chat_prompt()
    short = get_system_screen_comment_prompt()
    visual = get_system_visual_novel_prompt()
    assert chat != short and short != visual and chat != visual
    assert "沉默" in visual, "the existing silence-encouraging wording must survive"
    for label, prompt in (("chat", chat), ("short", short), ("visual", visual)):
        assert len(prompt) < 1600, f"{label} prompt grew unexpectedly: {len(prompt)}"


def test_tutor_mode_still_appends_on_top():
    plain = get_system_chat_prompt(tutor_enabled=False)
    tutor = get_system_chat_prompt(tutor_enabled=True)
    assert "【人设】" in tutor
    assert len(tutor) > len(plain)


# --------------------------------------------------------------------------
# Persona-relative judgement
# --------------------------------------------------------------------------


def test_persona_anchors_come_from_the_sheet():
    anchors = get_persona_moment_anchors()
    assert len(anchors) >= 3
    joined = "".join(anchors)
    assert "别的女生" in joined or "其他女生" in joined
    assert "说谎" in joined or "隐瞒" in joined


def test_persona_criteria_quote_her_sensitivities():
    criteria = get_persona_moment_criteria()
    assert INITIAL_PERSONA["name"] in criteria
    assert "女生" in criteria
    assert "沉默" in criteria, "she must still be allowed to stay quiet"


def test_the_visual_novel_judgement_asks_which_track_fired():
    from desktop_pet.llm.comment_engine import CommentEngine
    import inspect

    source = inspect.getsource(CommentEngine.evaluate_visual_novel)
    assert "reaction_source" in source
    assert "plot|persona|both|none" in source


# --------------------------------------------------------------------------
# OR-merged nomination
# --------------------------------------------------------------------------


def _stub_similarity(left: str, right: str) -> float:
    """1.0 when the anchor text appears verbatim in the window."""
    return 1.0 if right and right in left else 0.0


def _tracker(tmp_path: Path) -> VisualNovelTracker:
    return VisualNovelTracker(
        tmp_path / "story.json",
        similarity_fn=_stub_similarity,
        similarities_fn=None,
    )


def test_plot_track_still_nominates(tmp_path):
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        tracker = _tracker(Path(tmp))
        window = VisualNovelTracker._PLOT_ANCHORS[0]
        score = tracker._semantic_window_score(window)
        assert score is not None and score >= 0.6


def test_persona_track_nominates_when_the_plot_track_cannot():
    """The whole point: a persona-only moment must be scoreable.

    The same window scores at the floor once the persona anchors are removed,
    which proves the persona track is what lifts it.
    """
    from tempfile import TemporaryDirectory

    anchors = get_persona_moment_anchors()
    assert anchors, "persona anchors must be configured"
    window = anchors[0]

    saved = vn_module._PERSONA_ANCHORS_CACHE
    try:
        with TemporaryDirectory() as tmp:
            tracker = _tracker(Path(tmp))
            with_persona = tracker._semantic_window_score(window)
            assert with_persona is not None and with_persona >= 0.6

            # Plot anchors alone cannot see this window at all.
            plot_only = max(
                _stub_similarity(window, anchor)
                for anchor in VisualNovelTracker._PLOT_ANCHORS
            )
            assert plot_only == 0.0

            vn_module._PERSONA_ANCHORS_CACHE = ()
            without_persona = tracker._semantic_window_score(window)
            assert without_persona is not None
            assert with_persona > without_persona, (
                "removing the persona track must lower the score for a "
                "persona-only moment"
            )
    finally:
        vn_module._PERSONA_ANCHORS_CACHE = saved


def test_empty_persona_anchors_degrade_to_the_old_behaviour():
    """A missing persona file must not break tracking."""
    saved = vn_module._PERSONA_ANCHORS_CACHE
    try:
        vn_module._PERSONA_ANCHORS_CACHE = ()
        assert vn_module._persona_moment_anchors() == ()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tracker = _tracker(Path(tmp))
            window = VisualNovelTracker._PLOT_ANCHORS[1]
            assert tracker._semantic_window_score(window) is not None
    finally:
        vn_module._PERSONA_ANCHORS_CACHE = saved


def test_anchors_are_split_and_or_merged():
    assert not hasattr(VisualNovelTracker, "_MOMENT_ANCHORS"), (
        "the single plot-only anchor set must be gone"
    )
    assert len(VisualNovelTracker._PLOT_ANCHORS) == 5
    source = _segment("_semantic_window_score")
    assert "_PLOT_ANCHORS" in source
    assert "_persona_moment_anchors()" in source
    assert "max(available)" in source, "the two tracks must be OR-merged"


def test_reaction_source_survives_the_result_whitelist():
    """``evaluate_visual_novel`` builds its result from an explicit key list.

    Adding the field to the *requested* JSON is not enough: anything not named in
    the returned dict is silently dropped, which is exactly how this field first
    came back as ``None`` despite the model returning it.
    """
    from desktop_pet.llm.comment_engine import CommentEngine

    class _Client:
        def __init__(self, value: str):
            self.value = value

        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            return json.dumps(
                {
                    "should_comment": True,
                    "moment_type": "tension",
                    "reaction_source": self.value,
                    "comment": "……哥哥，那个女生是谁呀？",
                    "scene_delta": "出现了一名与主角亲近的女性角色。",
                    "scene_summary": "主角遇到一名女性角色。",
                    "facts": [],
                },
                ensure_ascii=False,
            )

    context = {"reason": "semantic_candidate", "recent_dialogue": "她挽住了他的手臂。"}

    assert CommentEngine(_Client("persona")).evaluate_visual_novel(context)[
        "reaction_source"
    ] == "persona"
    assert CommentEngine(_Client("both")).evaluate_visual_novel(context)[
        "reaction_source"
    ] == "both"
    # A missing or malformed value must normalise, never raise.
    for bad in ("", "PERSONA!", "unknown"):
        assert CommentEngine(_Client(bad)).evaluate_visual_novel(context)[
            "reaction_source"
        ] == "none"


def test_reaction_source_is_not_required_for_the_comment_to_land():
    """It is observability only; dropping it must not suppress a comment."""
    from desktop_pet.llm.comment_engine import CommentEngine

    class _Client:
        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            return json.dumps(
                {
                    "should_comment": True,
                    "moment_type": "tender",
                    "comment": "哥哥，这一段好温柔。",
                    "scene_delta": "",
                    "scene_summary": "",
                    "facts": [],
                },
                ensure_ascii=False,
            )

    result = CommentEngine(_Client()).evaluate_visual_novel(
        {"reason": "semantic_candidate", "recent_dialogue": "两人相视而笑。"}
    )
    assert result["should_comment"] is True
    assert result["comment"] == "哥哥，这一段好温柔。"
    assert result["reaction_source"] == "none"


def test_visual_novel_comment_uses_a_grounded_personal_reaction():
    from desktop_pet.llm.comment_engine import CommentEngine

    class _Client:
        prompt = ""

        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            self.prompt = user_text
            return json.dumps(
                {
                    "should_comment": True,
                    "moment_type": "twist",
                    "reaction_basis": "被怀疑的人承认留下了信",
                    "felt_reaction": "意外，也担心之前误会了他",
                    "comment": "诶，信居然是他留的……我之前是不是误会他了？",
                    "scene_delta": "信的作者得到确认。",
                    "scene_summary": "被怀疑的人承认留下了信。",
                    "facts": [],
                },
                ensure_ascii=False,
            )

    client = _Client()
    result = CommentEngine(client).evaluate_visual_novel(
        {"reason": "semantic_candidate", "recent_dialogue": "那封信，是我留下的。"}
    )
    assert "先分清本段确实发生了什么" in client.prompt
    assert result["reaction_basis"] == "被怀疑的人承认留下了信"
    assert result["felt_reaction"] == "意外，也担心之前误会了他"
    assert result["should_comment"] is True


def test_persona_anchor_lookup_is_cached_and_safe():
    source = _segment("_persona_moment_anchors")
    assert "_PERSONA_ANCHORS_CACHE" in source
    assert "except Exception" in source, "a broken persona file must not raise"
