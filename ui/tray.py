from __future__ import annotations

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QWidget


class AppTray(QSystemTrayIcon):
    def __init__(self, parent: QWidget, icon: QIcon | None = None):
        super().__init__(icon or QIcon(), parent)
        menu = QMenu(parent)

        self.show_action = QAction("显示桌宠", self)
        self.quit_action = QAction("退出", self)

        menu.addAction(self.show_action)
        menu.addSeparator()
        menu.addAction(self.quit_action)
        self.setContextMenu(menu)
