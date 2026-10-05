#!/usr/bin/env python3
"""Regenerate the packaged LEAP V1 KAIZEN MuJoCo convex collision assets in place."""

from pathlib import Path

from ghand.sim.mujoco.collision_asset_generation import generate_mujoco_collision_assets
from ghand.sim.mujoco.collision_assets import DEFAULT_MUJOCO_COLLISION_ASSET_VERSION


if __name__ == "__main__":
    generate_mujoco_collision_assets(
        Path(__file__).resolve().parents[1]
        / "src/ghand/assets/collision/mujoco"
        / DEFAULT_MUJOCO_COLLISION_ASSET_VERSION
    )
