"""Generate reusable MuJoCo convex collision assets from the source PLY meshes."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree as ET

import manifold3d  # type: ignore[import-not-found]
import numpy as np
import trimesh

from ghand.sim.mujoco.collision_assets import (
    DEFAULT_MUJOCO_COLLISION_ASSET_VERSION,
    MUJOCO_COLLISION_ASSET_SCHEMA_VERSION,
    sha256_file,
)
from ghand.hand import default_right_hand_urdf_path


COACD_THRESHOLD_M: Final[float] = 0.003
COACD_RESOLUTION: Final[int] = 1000
COACD_MCTS_NODES: Final[int] = 10
COACD_MCTS_ITERATIONS: Final[int] = 60
COACD_MCTS_MAX_DEPTH: Final[int] = 3
COACD_SEED: Final[int] = 0

# The high-ratio meshes account for the large false volume in the old aggregate
# hulls. Keeping the budget fixed prevents an unbounded geom/contact-pair count.
COACD_PART_BUDGET_BY_SOURCE: Final[dict[str, int]] = {
    "cover_mcp_side_right.ply": 2,
    "fpx330_h101.ply": 3,
    "fpx330_s101.ply": 3,
    "fpx330_s102.ply": 3,
    "palm_lower_right.ply": 12,
}


def generate_mujoco_collision_assets(output_root: Path) -> Path:
    urdf_path = default_right_hand_urdf_path().resolve()
    mesh_output = output_root / "meshes"
    staging_root = output_root.with_name(f".{output_root.name}.staging")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_mesh_output = staging_root / "meshes"
    staging_mesh_output.mkdir(parents=True)

    try:
        sources = _source_mesh_paths(urdf_path)
        source_entries = []
        for index, source in enumerate(sources, start=1):
            print(
                f"Collision assets: generating {index}/{len(sources)} {source.name}",
                file=sys.stderr,
                flush=True,
            )
            source_entries.append(_generate_source(source, staging_mesh_output))
        manifest = {
            "schema_version": MUJOCO_COLLISION_ASSET_SCHEMA_VERSION,
            "asset_version": DEFAULT_MUJOCO_COLLISION_ASSET_VERSION,
            "source_urdf": {
                "file": str(urdf_path.relative_to(urdf_path.parents[3])),
                "sha256": sha256_file(urdf_path),
            },
            "generator": {
                "coacd_version": importlib.metadata.version("coacd"),
                "manifold3d_version": importlib.metadata.version("manifold3d"),
                "trimesh_version": importlib.metadata.version("trimesh"),
                "settings": {
                    "threshold_m": COACD_THRESHOLD_M,
                    "real_metric": True,
                    "preprocess_mode": "off",
                    "resolution": COACD_RESOLUTION,
                    "mcts_nodes": COACD_MCTS_NODES,
                    "mcts_iterations": COACD_MCTS_ITERATIONS,
                    "mcts_max_depth": COACD_MCTS_MAX_DEPTH,
                    "seed": COACD_SEED,
                    "part_budget_by_source": dict(
                        sorted(COACD_PART_BUDGET_BY_SOURCE.items())
                    ),
                },
            },
            "sources": source_entries,
        }
        (staging_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_root.mkdir(parents=True, exist_ok=True)
        if mesh_output.exists():
            shutil.rmtree(mesh_output)
        shutil.move(str(staging_mesh_output), str(mesh_output))
        shutil.move(
            str(staging_root / "manifest.json"), str(output_root / "manifest.json")
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    total_parts = sum(len(source["parts"]) for source in source_entries)
    print(
        f"Generated {total_parts} convex meshes for {len(source_entries)} sources "
        f"in {output_root}.",
        file=sys.stderr,
        flush=True,
    )
    return output_root / "manifest.json"


def _source_mesh_paths(urdf_path: Path) -> tuple[Path, ...]:
    root = ET.parse(urdf_path).getroot()
    paths = {
        (urdf_path.parent / filename).resolve()
        for mesh in root.iter("mesh")
        if (filename := mesh.get("filename")) is not None
    }
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"URDF references missing meshes: {missing}")
    return tuple(sorted(paths, key=lambda path: path.name))


def _generate_source(source_path: Path, mesh_output: Path) -> dict[str, Any]:
    mesh = trimesh.load_mesh(source_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected one triangle mesh: {source_path}")
    if not mesh.is_watertight or mesh.volume <= 0.0:
        raise ValueError(
            f"Source mesh must be watertight with positive volume: {source_path}"
        )

    budget = COACD_PART_BUDGET_BY_SOURCE.get(source_path.name, 1)
    if budget == 1:
        method = "convex_hull"
        parts = [mesh.convex_hull]
    else:
        method = "coacd"
        parts = _coacd_parts(mesh, budget)
    parts = sorted(
        (_canonical_convex_mesh(part) for part in parts),
        key=_part_sort_key,
    )

    stem = _safe_name(source_path.stem)
    part_entries: list[dict[str, object]] = []
    for index, part in enumerate(parts):
        relative_file = Path("meshes") / f"{stem}_part_{index:03d}.obj"
        output_file = mesh_output / relative_file.name
        _write_obj(output_file, part)
        part_entries.append(
            {
                "file": relative_file.as_posix(),
                "sha256": sha256_file(output_file),
                "vertex_count": int(len(part.vertices)),
                "face_count": int(len(part.faces)),
                "volume_m3": float(part.volume),
            }
        )

    union = trimesh.boolean.union(parts, engine="manifold", check_volume=True)
    if not isinstance(union, trimesh.Trimesh) or union.volume <= 0.0:
        raise ValueError(f"Failed to validate convex-part union: {source_path}")
    return {
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "method": method,
        "part_budget": budget,
        "source_vertex_count": int(len(mesh.vertices)),
        "source_face_count": int(len(mesh.faces)),
        "source_volume_m3": float(mesh.volume),
        "source_convex_hull_volume_ratio": float(mesh.convex_hull.volume / mesh.volume),
        "decomposed_union_volume_ratio": float(union.volume / mesh.volume),
        "parts": part_entries,
    }


def _coacd_parts(mesh: trimesh.Trimesh, budget: int) -> list[trimesh.Trimesh]:
    try:
        import coacd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Collision asset generation requires CoACD; run "
            "`uv sync --group collision-assets` first."
        ) from exc
    coacd.set_log_level("off")
    result = coacd.run_coacd(
        coacd.Mesh(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        ),
        threshold=COACD_THRESHOLD_M,
        max_convex_hull=budget,
        preprocess_mode="off",
        resolution=COACD_RESOLUTION,
        mcts_nodes=COACD_MCTS_NODES,
        mcts_iterations=COACD_MCTS_ITERATIONS,
        mcts_max_depth=COACD_MCTS_MAX_DEPTH,
        seed=COACD_SEED,
        real_metric=True,
    )
    parts = [
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        for vertices, faces in result
    ]
    if not parts or len(parts) > budget:
        raise ValueError(f"CoACD returned {len(parts)} parts for budget {budget}.")
    return parts


def _canonical_convex_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    manifold_mesh = manifold3d.Manifold.hull_points(
        np.asarray(mesh.vertices, dtype=np.float64)
    ).to_mesh()
    hull = trimesh.Trimesh(
        vertices=np.asarray(manifold_mesh.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(manifold_mesh.tri_verts, dtype=np.int64),
        process=False,
    )
    if not hull.is_watertight or not hull.is_convex or hull.volume <= 0.0:
        raise ValueError(
            "Generated collision part is not a positive watertight convex mesh."
        )
    vertices = np.round(np.asarray(hull.vertices, dtype=np.float64), decimals=10)
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    faces = inverse[np.asarray(hull.faces, dtype=np.int64)]
    faces = np.asarray([_rotate_face(face) for face in faces], dtype=np.int64)
    face_order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    canonical = trimesh.Trimesh(
        vertices=vertices[order],
        faces=faces[face_order],
        process=False,
    )
    if canonical.volume < 0.0:
        canonical.invert()
    return canonical


def _rotate_face(face: np.ndarray) -> tuple[int, int, int]:
    start = int(np.argmin(face))
    values = np.roll(face, -start)
    return int(values[0]), int(values[1]), int(values[2])


def _part_sort_key(mesh: trimesh.Trimesh) -> tuple[float, ...]:
    center = np.asarray(mesh.centroid, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(-1)
    return (*center.tolist(), *bounds.tolist(), float(mesh.volume))


def _write_obj(path: Path, mesh: trimesh.Trimesh) -> None:
    lines = [
        *(
            "v " + " ".join(f"{float(value):.10g}" for value in vertex)
            for vertex in mesh.vertices
        ),
        *(
            "f " + " ".join(str(int(index) + 1) for index in face)
            for face in mesh.faces
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _safe_name(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    )
