from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from ghand.hand import DISPLAY_HAND_ROTATION_RPY
from ghand.viewer.urdf_meshes import prepare_rerun_urdf


def test_prepare_rerun_urdf_converts_local_ply_meshes_to_cached_glb(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    urdf_path = model_root / "urdf" / "hand.urdf"
    mesh_path = model_root / "meshes" / "triangle.ply"
    urdf_path.parent.mkdir(parents=True)
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
                "",
            )
        ),
        encoding="utf-8",
    )
    urdf_path.write_text(
        """
<robot name="mini">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="../meshes/triangle.ply" scale="1 1 1" />
      </geometry>
    </visual>
  </link>
</robot>
""".strip(),
        encoding="utf-8",
    )

    converted = prepare_rerun_urdf(urdf_path, cache_root=tmp_path / "cache")

    assert converted != urdf_path
    converted_tree = ElementTree.parse(converted)
    mesh_filename = converted_tree.getroot().find(".//mesh").attrib["filename"]
    converted_mesh = (converted.parent / mesh_filename).resolve()
    assert mesh_filename.endswith(".glb")
    assert converted_mesh.is_file()
    assert converted_mesh.read_bytes().startswith(b"glTF")
    assert ".ply" not in converted.read_text(encoding="utf-8")
    assert prepare_rerun_urdf(urdf_path, cache_root=tmp_path / "cache") == converted


def test_prepare_rerun_urdf_returns_original_when_no_ply_meshes(tmp_path: Path) -> None:
    urdf_path = tmp_path / "hand.urdf"
    urdf_path.write_text(
        """
<robot name="mini">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="../meshes/triangle.glb" />
      </geometry>
    </visual>
  </link>
</robot>
""".strip(),
        encoding="utf-8",
    )

    assert prepare_rerun_urdf(urdf_path, cache_root=tmp_path / "cache") == urdf_path


def test_prepare_rerun_urdf_can_add_display_root_rotation(tmp_path: Path) -> None:
    urdf_path = tmp_path / "hand.urdf"
    urdf_path.write_text(
        """
<robot name="mini">
  <link name="base" />
</robot>
""".strip(),
        encoding="utf-8",
    )

    converted = prepare_rerun_urdf(
        urdf_path,
        cache_root=tmp_path / "cache",
        root_rotation_rpy=DISPLAY_HAND_ROTATION_RPY,
    )

    assert converted != urdf_path
    root = ElementTree.parse(converted).getroot()
    display_root = root.find("./link[@name='ghand_display_root']")
    display_joint = root.find("./joint[@name='ghand_display_root_to_base']")
    assert display_root is not None
    assert display_joint is not None
    assert display_joint.attrib["type"] == "fixed"
    assert display_joint.find("parent").attrib["link"] == "ghand_display_root"
    assert display_joint.find("child").attrib["link"] == "base"
    rpy = tuple(float(part) for part in display_joint.find("origin").attrib["rpy"].split())
    assert rpy == pytest.approx(DISPLAY_HAND_ROTATION_RPY)
