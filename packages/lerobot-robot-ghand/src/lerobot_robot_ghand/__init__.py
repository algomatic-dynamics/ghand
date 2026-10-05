"""LeRobot plugin entry point for the LEAP V1 KAIZEN hand.

Importing this module registers the ``ghand`` robot type with LeRobot. The
implementation lives in :mod:`ghand.lerobot_plugin`.
"""

from ghand.lerobot_plugin import GHandRobot, GHandRobotConfig

__all__ = ["GHandRobot", "GHandRobotConfig"]
