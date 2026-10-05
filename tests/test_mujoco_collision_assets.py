from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
import trimesh

from ghand.sim.mujoco import collision_assets as assets
from ghand.sim.mujoco.collision_assets import (
    DEFAULT_MUJOCO_COLLISION_ASSET_VERSION,
    MujocoCollisionAssetError,
    default_mujoco_collision_manifest_path,
    load_mujoco_collision_asset_manifest,
    sha256_file,
)
from ghand.hand import default_right_hand_urdf_path


def _is_convex(mesh: trimesh.Trimesh, tolerance_m: float = 1e-4) -> bool:
    """Return whether every vertex lies on or behind every face plane."""

    offsets = (mesh.vertices[None, :, :] - mesh.triangles[:, 0][:, None, :]) @ mesh.face_normals[:, :, None]
    return bool((offsets <= tolerance_m).all())


def _source_paths(urdf_path: Path) -> set[Path]:
    root = ET.parse(urdf_path).getroot()
    return {
        (urdf_path.parent / filename).resolve()
        for mesh in root.iter("mesh")
        if (filename := mesh.get("filename")) is not None
    }


def test_packaged_manifest_is_the_default_and_covers_the_source_urdf() -> None:
    urdf_path = default_right_hand_urdf_path().resolve()
    manifest_path = default_mujoco_collision_manifest_path()
    manifest = load_mujoco_collision_asset_manifest(manifest_path)
    source_paths = _source_paths(urdf_path)

    assert manifest_path.is_relative_to(urdf_path.parents[2] / "collision")
    assert manifest_path.parent.name == DEFAULT_MUJOCO_COLLISION_ASSET_VERSION
    assert manifest.asset_version == DEFAULT_MUJOCO_COLLISION_ASSET_VERSION
    assert manifest.source_urdf_sha256 == sha256_file(urdf_path)
    assert set(manifest.sources_by_sha256) == {
        sha256_file(path) for path in source_paths
    }
    assert len(source_paths) == 25
    assert (
        sum(len(source.parts) for source in manifest.sources_by_sha256.values()) == 43
    )
    assert not any("rev" in path.name for path in manifest.files)


def test_versioned_mujoco_collision_parts_are_positive_watertight_convex_meshes() -> (
    None
):
    manifest = load_mujoco_collision_asset_manifest()

    for source in manifest.sources_by_sha256.values():
        for part in source.parts:
            mesh = trimesh.load_mesh(Path(part.file), process=False)

            assert isinstance(mesh, trimesh.Trimesh)
            assert mesh.is_watertight
            assert _is_convex(mesh)
            assert mesh.volume > 0.0
            assert mesh.volume == pytest.approx(part.volume_m3, rel=1e-6)
            assert len(mesh.vertices) == part.vertex_count
            assert len(mesh.faces) == part.face_count


@pytest.fixture
def local_source(tmp_path, monkeypatch):
    urdf = tmp_path / "source/hand.urdf"
    urdf.parent.mkdir()
    ply = urdf.parent / "part.ply"
    mesh = trimesh.creation.box(extents=(0.01, 0.02, 0.03))
    mesh.export(ply)
    urdf.write_text(
        '<robot name="test"><link name="palm"><visual><geometry>'
        '<mesh filename="part.ply"/></geometry></visual></link></robot>'
    )
    monkeypatch.setattr("ghand.hand.default_right_hand_urdf_path", lambda: urdf)

    def write_asset(root):
        root.mkdir(parents=True, exist_ok=True)
        part = root / "part.obj"
        mesh.export(part)
        path = root / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": assets.MUJOCO_COLLISION_ASSET_SCHEMA_VERSION,
                    "asset_version": DEFAULT_MUJOCO_COLLISION_ASSET_VERSION,
                    "source_urdf": {"sha256": sha256_file(urdf)},
                    "sources": [
                        {
                            "source_file": ply.name,
                            "source_sha256": sha256_file(ply),
                            "method": "convex_hull",
                            "parts": [
                                {
                                    "file": part.name,
                                    "sha256": sha256_file(part),
                                    "vertex_count": len(mesh.vertices),
                                    "face_count": len(mesh.faces),
                                    "volume_m3": float(mesh.volume),
                                }
                            ],
                        }
                    ],
                }
            )
        )
        return path

    return SimpleNamespace(urdf=urdf, ply=ply, write_asset=write_asset)


def test_asset_matching_the_current_urdf_is_accepted(local_source, tmp_path) -> None:
    path = local_source.write_asset(tmp_path / "asset")

    assert assets._validate_default_manifest(path) == path


@pytest.mark.parametrize(
    "damage", ["missing_part", "corrupt_part", "wrong_urdf", "wrong_ply"]
)
def test_asset_not_matching_the_current_urdf_is_rejected(
    local_source, tmp_path, damage
) -> None:
    root = tmp_path / "asset"
    path = local_source.write_asset(root)
    if damage == "missing_part":
        (root / "part.obj").unlink()
    elif damage == "corrupt_part":
        (root / "part.obj").write_text("corrupt")
    else:
        data = json.loads(path.read_text())
        if damage == "wrong_urdf":
            data["source_urdf"]["sha256"] = "0" * 64
        else:
            data["sources"][0]["source_sha256"] = "0" * 64
        path.write_text(json.dumps(data))

    with pytest.raises(MujocoCollisionAssetError):
        assets._validate_default_manifest(path)
