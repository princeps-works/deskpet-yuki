from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AppState:
    auto_scan_enabled: bool = False
    muted: bool = False
    last_auto_comment_at: datetime | None = None
    recent_screen_summary: str = ""
    chat_history: list[tuple[str, str]] = field(default_factory=list)
