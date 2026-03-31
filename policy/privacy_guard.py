from __future__ import annotations


def is_sensitive_context(window_title: str) -> bool:
    title = window_title.lower()
    blocked = ["bank", "密码", "支付", "wallet"]
    return any(token in title for token in blocked)
