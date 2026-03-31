from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryStore:
    topics: list[str] = field(default_factory=list)

    def remember(self, topic: str) -> None:
        if topic and topic not in self.topics:
            self.topics.append(topic)
