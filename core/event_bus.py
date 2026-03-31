from __future__ import annotations

from collections import defaultdict
from typing import Callable, DefaultDict, List


class EventBus:
    def __init__(self) -> None:
        self._handlers: DefaultDict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Callable) -> None:
        self._handlers[event_name].append(handler)

    def emit(self, event_name: str, *args, **kwargs) -> None:
        for handler in self._handlers[event_name]:
            handler(*args, **kwargs)
