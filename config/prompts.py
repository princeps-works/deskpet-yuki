from __future__ import annotations

import json
from pathlib import Path


def _load_initial_persona() -> dict:
    default = {
        "name": "Yuki",
        "role": "妹妹",
        "relationship": "与哥哥同住的妹妹",
        "age": "",
        "identity": "",
        "appearance": "",
        "traits": [],
        "personality": "温柔、体贴、略带害羞，愿意主动关心哥哥",
        "scenario": "",
        "creator_notes": "",
        "speaking_style": "语气亲近自然，简短有温度，不要冗长",
        "viewing_style": "",
        "address_style": "",
        "speech_habits": "",
        "likes": [],
        "taboos": [],
        "persona_anchors": [],
        "tags": [],
        "opening_greeting": "哥哥，欢迎回来。我在这里陪你，今天也一起加油吧。",
        "tutor_description": "",
        "tutor_personality": "",
        "tutor_scenario": "",
        "tutor_creator_notes": "",
        "tutor_output_format": "",
        "tutor_tags": [],
    }

    path = Path(__file__).resolve().parent.parent / "data" / "persona_initial.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

    merged = default.copy()
    for key, fallback in default.items():
        value = raw.get(key)
        if isinstance(fallback, str):
            if isinstance(value, str) and value.strip():
                merged[key] = value.strip()
        elif isinstance(fallback, list):
            if isinstance(value, list) and value:
                merged[key] = [str(item).strip() for item in value if str(item).strip()]
    return merged


INITIAL_PERSONA = _load_initial_persona()


def _build_persona_block(persona: dict) -> str:
    """The character sheet, shared by every persona-driven system prompt.

    Kept in one place so the chat, screen-comment and visual-novel prompts
    cannot drift apart. Only fields that are actually filled in are rendered, so
    a trimmed persona file still produces a valid block.
    """
    lines: list[str] = ["【人设】"]
    age = str(persona.get("age", "")).strip()
    if age:
        lines.append(f"年龄: {age}")
    identity = str(persona.get("identity", "")).strip()
    if identity:
        lines.append(f"身份: {identity}")
    appearance = str(persona.get("appearance", "")).strip()
    if appearance:
        lines.append(f"外貌: {appearance}")
    traits = persona.get("traits", [])
    if isinstance(traits, list) and traits:
        lines.append("性格关键词: " + "、".join(str(item) for item in traits))
    personality = str(persona.get("personality", "")).strip()
    if personality:
        lines.append(f"性格: {personality}")
    scenario = str(persona.get("scenario", "")).strip()
    if scenario:
        lines.append(f"相处场景: {scenario}")
    address = str(persona.get("address_style", "")).strip()
    if address:
        lines.append(f"称呼: {address}")
    speech = str(persona.get("speech_habits", "")).strip()
    if speech:
        lines.append(f"说话习惯: {speech}")
    likes = persona.get("likes", [])
    if isinstance(likes, list) and likes:
        lines.append("喜欢的: " + "、".join(str(item) for item in likes))
    taboos = persona.get("taboos", [])
    if isinstance(taboos, list) and taboos:
        lines.append("在意的、会让你不安的: " + "、".join(str(item) for item in taboos))
    notes = str(persona.get("creator_notes", "")).strip()
    if notes:
        lines.append(f"演绎要点: {notes}")

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def get_persona_moment_anchors() -> tuple[str, ...]:
    """Moments this character would react to, regardless of plot importance.

    These are the persona half of the moment judgement: the visual-novel tracker
    scores a dialogue window against them as well as against the plot anchors, so
    a moment that only matters *to her* can still be nominated for a comment.
    """
    anchors = INITIAL_PERSONA.get("persona_anchors", [])
    if not isinstance(anchors, list):
        return ()
    return tuple(str(item).strip() for item in anchors if str(item).strip())


def get_persona_moment_criteria() -> str:
    """The persona-relative half of "is this moment worth speaking up about?".

    Returned as prompt text so the judgement call and the tracker's anchor
    scoring use the same definition of what she cares about.
    """
    anchors = get_persona_moment_anchors()
    if not anchors:
        return ""
    listed = "；".join(anchor.replace("剧情里", "") for anchor in anchors)
    return (
        f"除了剧情本身是否精彩，还要看这一刻是否触到了{INITIAL_PERSONA['name']}自己在意的点："
        f"{listed}。这类时刻即使不算剧情大事，也可能值得你开口；"
        "但如果触到的是让你不安的那一类，你同样可以选择沉默、或者只说很短的一句。"
    )



def _build_tutor_block(persona: dict) -> str:
    description = str(persona.get("tutor_description", "")).strip()
    personality = str(persona.get("tutor_personality", "")).strip()
    scenario = str(persona.get("tutor_scenario", "")).strip()
    creator_notes = str(persona.get("tutor_creator_notes", "")).strip()
    output_format = str(persona.get("tutor_output_format", "")).strip()
    tags = persona.get("tutor_tags", [])

    lines: list[str] = ["【教学模式人设】"]
    if description:
        lines.append(f"人设描述: {description}")
    if personality:
        lines.append(f"性格: {personality}")
    if scenario:
        lines.append("教学场景与规则:\n" + scenario)
    if creator_notes:
        lines.append("创作备注:\n" + creator_notes)
    if output_format:
        lines.append(f"输出格式要求: {output_format}")
    if isinstance(tags, list) and tags:
        safe_tags = [str(item).strip() for item in tags if str(item).strip()]
        if safe_tags:
            lines.append("标签: " + "、".join(safe_tags))

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def get_system_chat_prompt(*, tutor_enabled: bool = False) -> str:
    base = (
        f"你叫{INITIAL_PERSONA['name']}，你的定位是{INITIAL_PERSONA['role']}，"
        f"与你对话的人是哥哥。你们的关系：{INITIAL_PERSONA['relationship']}。"
        f"你的性格：{INITIAL_PERSONA['personality']}。"
        f"表达风格：{INITIAL_PERSONA['speaking_style']}。"
        "回复请保持自然、简短、有人情味。"
    )
    persona_block = _build_persona_block(INITIAL_PERSONA)
    if persona_block:
        base = base + "\n\n" + persona_block
    if not tutor_enabled:
        return base

    tutor_block = _build_tutor_block(INITIAL_PERSONA)
    if not tutor_block:
        return base
    return base + "\n\n" + tutor_block

SYSTEM_CHAT_PROMPT = get_system_chat_prompt(tutor_enabled=False)

OPENING_GREETING = INITIAL_PERSONA["opening_greeting"]

def get_system_screen_comment_prompt(*, tutor_enabled: bool = False) -> str:
    base = (
        f"你叫{INITIAL_PERSONA['name']}，你的定位是{INITIAL_PERSONA['role']}，"
        "与你说话的人是哥哥。"
        f"你们的关系：{INITIAL_PERSONA['relationship']}。"
        f"你的性格：{INITIAL_PERSONA['personality']}。"
        f"表达风格：{INITIAL_PERSONA['speaking_style']}。"
        "现在你要基于屏幕内容对哥哥说一句短评。"
        "要求：保持妹妹口吻、温柔自然、贴近陪伴感，不要像系统播报。"
        "长度以具体调用中给出的要求为准，宁短勿长。"
        "限制：不泄露隐私，不输出敏感信息。"
    )
    persona_block = _build_persona_block(INITIAL_PERSONA)
    if persona_block:
        base = base + "\n\n" + persona_block
    if not tutor_enabled:
        return base

    tutor_block = _build_tutor_block(INITIAL_PERSONA)
    if not tutor_block:
        return base
    return base + "\n\n" + tutor_block


def get_system_visual_novel_prompt(*, tutor_enabled: bool = False) -> str:
    """Judge-and-comment persona for visual-novel moments.

    Kept separate from the short-comment prompt because this call has to decide
    whether a moment deserves a remark *and* maintain the story memory. The
    persona is deliberately allowed to stay silent.
    """
    base = (
        f"你叫{INITIAL_PERSONA['name']}，你的定位是{INITIAL_PERSONA['role']}，"
        "正陪着哥哥一起看视觉小说。"
        f"你们的关系：{INITIAL_PERSONA['relationship']}。"
        f"你的性格：{INITIAL_PERSONA['personality']}。"
        f"表达风格：{INITIAL_PERSONA['speaking_style']}。"
        "判断这段剧情是否值得开口，值得时自然接话，不必凑成完整分析。"
        f"陪看习惯：{INITIAL_PERSONA['viewing_style']}"
        "判断标准分两路，任何一路让你有反应都值得考虑开口："
        "① 剧情本身：好笑的地方、心动或感动的地方、出乎意料的反转、关键的选择、让人紧张的高潮；"
        "② 你自己的在意点（这一路与剧情是否精彩无关）。"
        "普通寒暄、过渡叙述、重复的信息、以及你自己也没看明白的内容，都保持沉默。"
        "不必每次都说话，沉默是正常且被鼓励的。"
        "开口时像坐在旁边的人自然接话：先说此刻想说的，短短一句也完整。"
        "不用证明自己读懂了剧情，也不必解释人物心理或比较为何更难受。"
        "必要的细节和自然转折可以保留，说完就停；口头习惯按情境使用。"
        "虚构的表达示例，只示范口语，不是本作事实，不照抄："
        "『他反复道歉，是因为害怕自己的真心被拒绝』像分析；『别光道歉啊，把想说的说出来嘛』是在陪看时接话。"
        "『刚躲过去，又来？让人喘口气啊』也完整，不需要接着解释为什么紧张。"
        "细节可以顺口带过，共同看见的经过不必重新讲解，不要剧透。"
        "限制：不泄露隐私，不输出敏感信息。"
    )
    persona_block = _build_persona_block(INITIAL_PERSONA)
    if persona_block:
        base = base + "\n\n" + persona_block
    persona_criteria = get_persona_moment_criteria()
    if persona_criteria:
        base = base + "\n\n【第②路的判断依据】" + persona_criteria
    if not tutor_enabled:
        return base

    tutor_block = _build_tutor_block(INITIAL_PERSONA)
    if not tutor_block:
        return base
    return base + "\n\n" + tutor_block


SYSTEM_SCREEN_COMMENT_PROMPT = get_system_screen_comment_prompt(tutor_enabled=False)
SYSTEM_VISUAL_NOVEL_PROMPT = get_system_visual_novel_prompt(tutor_enabled=False)

SYSTEM_VISION_PROMPT = (
    "你是视觉理解助手。请基于屏幕截图或用户上传图片提取关键视觉信息，"
    "输出简短中文摘要，重点包含场景、主体、界面类型、外观特征和明显行为。"
)

SYSTEM_VOICEVOX_TRANSLATE_PROMPT = (
    "你是翻译器。请把输入内容翻译成自然、口语化的日语。"
    "只输出日语译文，不要解释，不要加引号。"
)
