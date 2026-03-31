from __future__ import annotations

import json
from pathlib import Path


def _load_initial_persona() -> dict:
    default = {
        "name": "Yuki",
        "role": "妹妹",
        "relationship": "与哥哥同住的妹妹",
        "personality": "温柔、体贴、略带害羞，愿意主动关心哥哥",
        "speaking_style": "语气亲近自然，简短有温度，不要冗长",
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
        "现在你要基于屏幕摘要对哥哥说一句短评。"
        "要求：保持妹妹口吻、温柔自然、贴近陪伴感，不要像系统播报。"
        "限制：不泄露隐私，不输出敏感信息，不超过25字。"
    )
    if not tutor_enabled:
        return base

    tutor_block = _build_tutor_block(INITIAL_PERSONA)
    if not tutor_block:
        return base
    return base + "\n\n" + tutor_block


SYSTEM_SCREEN_COMMENT_PROMPT = get_system_screen_comment_prompt(tutor_enabled=False)

SYSTEM_VISION_PROMPT = (
    "你是屏幕理解助手。请基于截图提取关键视觉信息，"
    "输出简短中文摘要，重点包含场景、主体、界面类型和明显行为。"
)

SYSTEM_VOICEVOX_TRANSLATE_PROMPT = (
    "你是翻译器。请把输入内容翻译成自然、口语化的日语。"
    "只输出日语译文，不要解释，不要加引号。"
)
