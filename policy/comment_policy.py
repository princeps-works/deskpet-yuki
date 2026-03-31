from __future__ import annotations

from datetime import datetime, timedelta


def can_emit_comment(last_at: datetime | None, cooldown_sec: int) -> bool:
    if last_at is None:
        return True
    return datetime.now() - last_at >= timedelta(seconds=max(1, cooldown_sec))
