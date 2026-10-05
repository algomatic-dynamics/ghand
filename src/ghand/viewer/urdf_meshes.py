"""URDF mesh preparation helpers for Rerun visualization."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

RERUN_MESH_FORMAT = "glb"
RERUN_MESH_CACHE_VERSION = "v1"


class UrdfMeshConversionError(RuntimeError):
    """Raised when a URDF mesh cannot be converted for Rerun."""


def prepare_rerun_urdf(
    urdf_path: Path | str,
    *,
    cache_root: Path | str | None = None,
    root_rotation_rpy: tuple[float, float, float] | None = None,
) -> Path:
    """Return a Rerun-compatible URDF path, converting local PLY meshes to GLB.

    Rerun 0.32 can log URDF files, but its viewer does not render PLY assets.
    The packaged LEAP V1 KAIZEN URDF references PLY meshes, so this prepares a
    cached URDF that references converted GLB meshes while leaving the source
    assets intact.
    """

    source_urdf = Path(urdf_path)
    tree = ElementTree.parse(source_urdf)
    root = tree.getroot()
    mesh_entries = _collect_ply_meshes(source_urdf, root)
    if not mesh_entries and root_rotation_rpy is None:
        return source_urdf

    cache_dir = _cache_dir(
        source_urdf,
        [mesh_path for _, mesh_path in mesh_entries],
        cache_root,
        root_rotation_rpy,
    )
    mesh_cache_dir = cache_dir / "meshes"
    urdf_cache_dir = cache_dir / "urdf"
    try:
        mesh_cache_dir.mkdir(parents=True, exist_ok=True)
        urdf_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        if cache_root is not None:
            raise
        cache_dir = _cache_dir(
            source_urdf,
            [mesh_path for _, mesh_path in mesh_entries],
            _fallback_rerun_urdf_cache_root(),
            root_rotation_rpy,
        )
        mesh_cache_dir = cache_dir / "meshes"
        urdf_cache_dir = cache_dir / "urdf"
        mesh_cache_dir.mkdir(parents=True, exist_ok=True)
        urdf_cache_dir.mkdir(parents=True, exist_ok=True)

    converted_by_source: dict[Path, Path] = {}
    for _, mesh_path in mesh_entries:
        if mesh_path not in converted_by_source:
            converted_by_source[mesh_path] = _convert_mesh_to_glb(mesh_path, mesh_cache_dir)

    converted_urdf = urdf_cache_dir / source_urdf.name
    for mesh_element, mesh_path in mesh_entries:
        converted_mesh = converted_by_source[mesh_path]
        mesh_element.set("filename", os.path.relpath(converted_mesh, converted_urdf.parent))
    if root_rotation_rpy is not None:
        _add_root_rotation(root, root_rotation_rpy)
    tree.write(converted_urdf, encoding="utf-8", xml_declaration=True)
    return converted_urdf


def default_rerun_urdf_cache_root() -> Path:
    """Return the cache directory used for Rerun-compatible URDF assets."""

    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ghand" / "rerun_urdf"


def _fallback_rerun_urdf_cache_root() -> Path:
    return Path(tempfile.gettempdir()) / "ghand" / "rerun_urdf"


def _collect_ply_meshes(
    urdf_path: Path,
    root: ElementTree.Element,
) -> list[tuple[ElementTree.Element, Path]]:
    entries: list[tuple[ElementTree.Element, Path]] = []
    for mesh_element in root.iter("mesh"):
        filename = mesh_element.get("filename")
        if filename is None or not filename.lower().split("?", 1)[0].endswith(".ply"):
            continue
        mesh_path = _local_mesh_path(urdf_path, filename)
        if mesh_path is not None:
            entries.append((mesh_element, mesh_path))
    return entries


def _local_mesh_path(urdf_path: Path, filename: str) -> Path | None:
    parsed = urlparse(filename)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    return (urdf_path.parent / filename).resolve()


def _cache_dir(
    urdf_path: Path,
    mesh_paths: list[Path],
    cache_root: Path | str | None,
    root_rotation_rpy: tuple[float, float, float] | None,
) -> Path:
    digest = hashlib.sha256()
    digest.update(RERUN_MESH_CACHE_VERSION.encode("utf-8"))
    digest.update(RERUN_MESH_FORMAT.encode("utf-8"))
    digest.update(urdf_path.read_bytes())
    if root_rotation_rpy is not None:
        digest.update("root_rotation_rpy".encode("utf-8"))
        digest.update(" ".join(_format_float(value) for value in root_rotation_rpy).encode("utf-8"))
    for mesh_path in sorted(set(mesh_paths), key=lambda path: str(path)):
        stat = mesh_path.stat()
        digest.update(str(mesh_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    root = Path(cache_root) if cache_root is not None else default_rerun_urdf_cache_root()
    return root / digest.hexdigest()[:16]


def _add_root_rotation(
    root: ElementTree.Element,
    root_rotation_rpy: tuple[float, float, float],
) -> None:
    root_links = _root_link_names(root)
    if not root_links:
        return
    existing_names = _named_elements(root)
    wrapper_link = _unique_name("ghand_display_root", existing_names)
    existing_names.add(wrapper_link)
    root.append(ElementTree.Element("link", {"name": wrapper_link}))
    rpy = " ".join(_format_float(value) for value in root_rotation_rpy)
    for root_link in root_links:
        joint_name = _unique_name(f"{wrapper_link}_to_{root_link}", existing_names)
        existing_names.add(joint_name)
        joint = ElementTree.Element("joint", {"name": joint_name, "type": "fixed"})
        ElementTree.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": rpy})
        ElementTree.SubElement(joint, "parent", {"link": wrapper_link})
        ElementTree.SubElement(joint, "child", {"link": root_link})
        root.append(joint)


def _root_link_names(root: ElementTree.Element) -> list[str]:
    link_names = [link.get("name") for link in root.findall("link") if link.get("name")]
    child_links = {
        child.get("link")
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None and child.get("link")
    }
    return [name for name in link_names if name not in child_links]


def _named_elements(root: ElementTree.Element) -> set[str]:
    return {name for element in root.iter() if (name := element.get("name"))}


def _unique_name(base: str, existing_names: set[str]) -> str:
    if base not in existing_names:
        return base
    index = 1
    while f"{base}_{index}" in existing_names:
        index += 1
    return f"{base}_{index}"


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"


def _convert_mesh_to_glb(mesh_path: Path, mesh_cache_dir: Path) -> Path:
    name_hash = hashlib.sha256(str(mesh_path).encode("utf-8")).hexdigest()[:10]
    target_path = mesh_cache_dir / f"{mesh_path.stem}-{name_hash}.{RERUN_MESH_FORMAT}"
    if target_path.exists():
        return target_path

    try:
        import trimesh

        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.export(target_path, file_type=RERUN_MESH_FORMAT)
    except Exception as exc:
        msg = f"Failed to convert URDF mesh for Rerun: {mesh_path}: {exc}"
        raise UrdfMeshConversionError(msg) from exc
    return target_path
