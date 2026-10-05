"""Versioned convex collision assets used by the LEAP V1 KAIZEN MuJoCo exporter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree as ET


MUJOCO_COLLISION_ASSET_SCHEMA_VERSION: Final[str] = (
    "ghand-mujoco-convex-collision-assets-v1"
)
DEFAULT_MUJOCO_COLLISION_ASSET_VERSION: Final[str] = "v1"


class MujocoCollisionAssetError(RuntimeError):
    """Raised when versioned MuJoCo collision assets are missing or inconsistent."""


@dataclass(frozen=True)
class MujocoConvexPartAsset:
    """One validated convex mesh in its source-mesh coordinate frame."""

    file: Path
    sha256: str
    vertex_count: int
    face_count: int
    volume_m3: float


@dataclass(frozen=True)
class MujocoCollisionSourceAsset:
    """Convex decomposition associated with one source visual mesh."""

    source_file: str
    source_sha256: str
    method: str
    parts: tuple[MujocoConvexPartAsset, ...]


@dataclass(frozen=True)
class MujocoCollisionAssetManifest:
    """Validated, immutable collision-asset manifest."""

    path: Path
    schema_version: str
    asset_version: str
    source_urdf_sha256: str
    sources_by_sha256: dict[str, MujocoCollisionSourceAsset]

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the manifest and every referenced convex mesh."""

        parts = tuple(
            part.file
            for source in self.sources_by_sha256.values()
            for part in source.parts
        )
        return (self.path, *parts)


def default_mujoco_collision_manifest_path() -> Path:
    """Return the packaged collision-asset manifest after validating it."""

    path = Path(
        str(
            files("ghand.assets").joinpath(
                "collision",
                "mujoco",
                DEFAULT_MUJOCO_COLLISION_ASSET_VERSION,
                "manifest.json",
            )
        )
    )
    return _validate_default_manifest(path)


def _validate_default_manifest(path: Path) -> Path:
    from ghand.hand import default_right_hand_urdf_path

    manifest = load_mujoco_collision_asset_manifest(path)
    urdf = default_right_hand_urdf_path().resolve()
    source_paths = {
        (urdf.parent / node.attrib["filename"]).resolve()
        for node in ET.parse(urdf).getroot().iter("mesh")
    }
    if (
        manifest.asset_version != DEFAULT_MUJOCO_COLLISION_ASSET_VERSION
        or manifest.source_urdf_sha256 != sha256_file(urdf)
        or set(manifest.sources_by_sha256) != {sha256_file(p) for p in source_paths}
    ):
        raise MujocoCollisionAssetError(
            f"Collision assets do not match the current URDF/source PLY meshes: {path}"
        )
    return path


def load_mujoco_collision_asset_manifest(
    manifest_path: Path | str | None = None,
) -> MujocoCollisionAssetManifest:
    """Load and fully validate a convex collision-asset manifest."""

    path = (
        default_mujoco_collision_manifest_path()
        if manifest_path is None
        else Path(manifest_path).absolute()
    )
    if not path.is_file():
        raise MujocoCollisionAssetError(
            f"MuJoCo collision asset manifest not found: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MujocoCollisionAssetError(
            f"Failed to parse MuJoCo collision asset manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise MujocoCollisionAssetError(
            f"MuJoCo collision asset manifest must be an object: {path}"
        )

    schema_version = _required_string(payload, "schema_version", path)
    if schema_version != MUJOCO_COLLISION_ASSET_SCHEMA_VERSION:
        raise MujocoCollisionAssetError(
            "Unsupported MuJoCo collision asset schema_version "
            f"{schema_version!r}; expected {MUJOCO_COLLISION_ASSET_SCHEMA_VERSION!r}."
        )
    asset_version = _required_string(payload, "asset_version", path)
    source_urdf = _required_mapping(payload, "source_urdf", path)
    source_urdf_sha256 = _required_sha256(source_urdf, "sha256", path)
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise MujocoCollisionAssetError(
            f"MuJoCo collision asset manifest has no sources: {path}"
        )

    sources_by_sha256: dict[str, MujocoCollisionSourceAsset] = {}
    for index, raw_source in enumerate(raw_sources):
        context = f"{path}: sources[{index}]"
        if not isinstance(raw_source, dict):
            raise MujocoCollisionAssetError(f"{context} must be an object.")
        source_file = _required_string(raw_source, "source_file", context)
        source_sha256 = _required_sha256(raw_source, "source_sha256", context)
        method = _required_string(raw_source, "method", context)
        raw_parts = raw_source.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise MujocoCollisionAssetError(f"{context} has no convex parts.")
        parts = tuple(
            _load_part(path.parent, raw_part, f"{context}.parts[{part_index}]")
            for part_index, raw_part in enumerate(raw_parts)
        )
        if source_sha256 in sources_by_sha256:
            raise MujocoCollisionAssetError(
                f"{context} duplicates source SHA256 {source_sha256}."
            )
        sources_by_sha256[source_sha256] = MujocoCollisionSourceAsset(
            source_file=source_file,
            source_sha256=source_sha256,
            method=method,
            parts=parts,
        )

    return MujocoCollisionAssetManifest(
        path=path,
        schema_version=schema_version,
        asset_version=asset_version,
        source_urdf_sha256=source_urdf_sha256,
        sources_by_sha256=sources_by_sha256,
    )


def sha256_file(path: Path | str) -> str:
    """Return the SHA256 digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_part(
    manifest_root: Path,
    raw_part: object,
    context: str,
) -> MujocoConvexPartAsset:
    if not isinstance(raw_part, dict):
        raise MujocoCollisionAssetError(f"{context} must be an object.")
    relative_file = _required_string(raw_part, "file", context)
    relative_path = Path(relative_file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise MujocoCollisionAssetError(
            f"{context}.file escapes the collision asset directory: {relative_file}"
        )
    file = manifest_root / relative_path
    if not file.is_file():
        raise MujocoCollisionAssetError(f"{context}.file not found: {file}")
    expected_sha256 = _required_sha256(raw_part, "sha256", context)
    actual_sha256 = sha256_file(file)
    if actual_sha256 != expected_sha256:
        raise MujocoCollisionAssetError(
            f"{context}.file SHA256 mismatch: {actual_sha256} != {expected_sha256}."
        )
    vertex_count = _required_positive_int(raw_part, "vertex_count", context)
    face_count = _required_positive_int(raw_part, "face_count", context)
    volume_m3 = raw_part.get("volume_m3")
    if isinstance(volume_m3, bool) or not isinstance(volume_m3, (int, float)):
        raise MujocoCollisionAssetError(f"{context}.volume_m3 must be numeric.")
    if float(volume_m3) <= 0.0:
        raise MujocoCollisionAssetError(f"{context}.volume_m3 must be positive.")
    return MujocoConvexPartAsset(
        file=file,
        sha256=expected_sha256,
        vertex_count=vertex_count,
        face_count=face_count,
        volume_m3=float(volume_m3),
    )


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    context: Path | str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise MujocoCollisionAssetError(f"{context}: {key} must be an object.")
    return value


def _required_string(
    payload: dict[str, Any],
    key: str,
    context: Path | str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MujocoCollisionAssetError(f"{context}: {key} must be a non-empty string.")
    return value


def _required_sha256(
    payload: dict[str, Any],
    key: str,
    context: Path | str,
) -> str:
    value = _required_string(payload, key, context)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise MujocoCollisionAssetError(
            f"{context}: {key} must be a lowercase SHA256 digest."
        )
    return value


def _required_positive_int(
    payload: dict[str, Any],
    key: str,
    context: str,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MujocoCollisionAssetError(f"{context}.{key} must be a positive integer.")
    return value
