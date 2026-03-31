from __future__ import annotations


class Live2DDriver:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:
        # Placeholder for Cubism runtime bridge.
        self.initialized = True

    def set_expression(self, expression_name: str) -> None:
        _ = expression_name

    def play_motion(self, motion_name: str) -> None:
        _ = motion_name
