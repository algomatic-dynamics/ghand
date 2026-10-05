"""LeRobot plugin entrypoint for the LEAP V1 KAIZEN hand."""

from ghand.lerobot_plugin.config_ghand import GHandRobotConfig
from ghand.lerobot_plugin.robot import GHandRobot

__all__ = ["GHandRobot", "GHandRobotConfig"]
