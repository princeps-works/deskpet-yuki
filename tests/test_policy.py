from datetime import datetime, timedelta

from desktop_pet.policy.comment_policy import can_emit_comment


def test_can_emit_comment_when_none():
    assert can_emit_comment(None, 60)


def test_can_emit_comment_after_cooldown():
    last = datetime.now() - timedelta(seconds=120)
    assert can_emit_comment(last, 60)
