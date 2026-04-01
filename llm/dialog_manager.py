from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from desktop_pet.config.prompts import INITIAL_PERSONA, get_system_chat_prompt
from desktop_pet.llm.client import LLMClient


class DialogManager:
    def __init__(self, llm_client: LLMClient, memory_path: Path, *, tutor_enabled: bool = False) -> None:
        self._client = llm_client
        self._memory_path = memory_path
        self._session_messages: list[dict[str, str]] = []
        self._tutor_enabled = bool(tutor_enabled)

    def _load_memory_entries(self) -> list[dict[str, str]]:
        try:
            raw = json.loads(self._memory_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        entries: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                continue
            ts = item.get("timestamp")
            entries.append(
                {
                    "timestamp": str(ts) if ts else "",
                    "summary": summary.strip(),
                }
            )
        return entries

    def _save_memory_entries(self, entries: list[dict[str, str]]) -> None:
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_path.write_text(
            json.dumps(entries[-80:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_long_memory_block(self) -> str:
        entries = self._load_memory_entries()
        if not entries:
            return ""
        picked = entries[-8:]
        lines = [f"- {item['summary']}" for item in picked]
        return "长期互动记忆要点:\n" + "\n".join(lines)

    def build_light_long_memory_hint(self, limit: int = 3) -> str:
        entries = self._load_memory_entries()
        if not entries:
            return ""

        max_items = max(1, limit)
        picked = entries[-max_items:]
        lines = [f"- {item['summary']}" for item in picked]
        return "长期记忆（低权重参考，可忽略）:\n" + "\n".join(lines)

    def _build_recent_session_block(self) -> str:
        if not self._session_messages:
            return ""
        picked = self._session_messages[-120:]
        lines: list[str] = []
        for item in picked:
            role = item.get("role", "")
            text = item.get("text", "")
            if role and text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def build_recent_session_hint(self, limit: int = 8) -> str:
        if not self._session_messages:
            return ""

        max_items = max(1, limit)
        picked = self._session_messages[-max_items:]
        lines: list[str] = []
        for item in picked:
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            if role and text:
                lines.append(f"{role}: {text}")
        if not lines:
            return ""
        return "近期对话片段（用于语气和连续性参考）:\n" + "\n".join(lines)

    def start_new_chat(self) -> int:
        self._session_messages = []
        return len(self._load_memory_entries())

    def record_session_message(self, role: str, text: str) -> None:
        role_text = role.strip()
        content = text.strip()
        if not role_text or not content:
            return
        self._session_messages.append({"role": role_text, "text": content})

    def set_tutor_enabled(self, enabled: bool) -> None:
        self._tutor_enabled = bool(enabled)

    def list_long_memory(self, limit: int = 12) -> list[dict[str, str]]:
        entries = self._load_memory_entries()
        if limit <= 0:
            return entries
        return entries[-limit:]

    def append_long_memory(self, summary: str) -> bool:
        text = summary.strip()
        if not text:
            return False

        if len(text) > 500:
            text = text[:500]

        entries = self._load_memory_entries()
        if entries and entries[-1].get("summary", "") == text:
            return False

        entries.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "summary": text,
            }
        )
        self._save_memory_entries(entries)
        return True

    def end_current_chat(self) -> str:
        if not self._session_messages:
            return ""

        transcript = self._build_recent_session_block()
        if not transcript.strip():
            self._session_messages = []
            return ""

        summary_prompt = (
            f"你是{INITIAL_PERSONA['name']}，定位是{INITIAL_PERSONA['role']}。"
            "请把以下本轮互动内容整理为一篇日记体长期记忆，长度50到500字。"
            "要求：保持妹妹口吻、自然有温度；保留关系进展、稳定偏好、重要约定与持续目标；"
            "不记录一次性噪声。只输出日记正文，不要额外解释。"
        )
        try:
            summary = self._client.chat(user_text=transcript, system_prompt=summary_prompt).strip()
        except Exception:
            summary = ""

        if not summary or summary.startswith("[离线回声]"):
            summary = f"今天和哥哥聊了很多，主要是：{transcript[:220]}"

        if len(summary) > 500:
            summary = summary[:500]
        if len(summary) < 50:
            padding = transcript[: (50 - len(summary))]
            summary = (summary + " " + padding).strip()
            if len(summary) > 500:
                summary = summary[:500]

        entries = self._load_memory_entries()
        entries.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
            }
        )
        self._save_memory_entries(entries)
        self._session_messages = []
        return summary

    def reply(self, user_text: str, extra_context: str = "") -> str:
        long_memory = self._build_long_memory_block()
        recent_session = self._build_recent_session_block()

        prompt_parts: list[str] = []
        if long_memory:
            prompt_parts.append(long_memory)
        if recent_session:
            prompt_parts.append("本次会话最近内容:\n" + recent_session)
        if extra_context.strip():
            prompt_parts.append("额外上下文:\n" + extra_context.strip())
        prompt_parts.append("当前用户输入:\n" + user_text)
        merged_user_text = "\n\n".join(prompt_parts)

        system_prompt = get_system_chat_prompt(tutor_enabled=self._tutor_enabled)
        reply = self._client.chat(user_text=merged_user_text, system_prompt=system_prompt)
        self.record_session_message("你", user_text)
        self.record_session_message("桌宠", reply)
        return reply
