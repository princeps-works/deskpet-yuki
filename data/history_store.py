from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HistoryStore:
    messages: list[tuple[str, str]] = field(default_factory=list)

    def add(self, role: str, text: str) -> None:
        self.messages.append((role, text))
