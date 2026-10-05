"""Helpers for serving packaged model assets in the operator UI."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from types import TracebackType

from ghand.hand import DEFAULT_RIGHT_HAND_URDF


DEFAULT_MODEL_URL_PREFIX = "/models"
DEFAULT_UI_STATIC_URL_PREFIX = "/ghand-static"


@dataclass(frozen=True)
class ServedModelAssets:
    """URL information for model assets served by NiceGUI."""

    model_root: Path
    static_root: Path
    model_url_prefix: str = DEFAULT_MODEL_URL_PREFIX
    static_url_prefix: str = DEFAULT_UI_STATIC_URL_PREFIX
    urdf_relative_path: str = DEFAULT_RIGHT_HAND_URDF

    @property
    def urdf_url(self) -> str:
        """Return the browser URL for the selected URDF."""

        return f"{self.model_url_prefix.rstrip('/')}/{self.urdf_relative_path}"

    @property
    def viewer_js_url(self) -> str:
        """Return the browser URL for the operator URDF viewer module."""

        return f"{self.static_url_prefix.rstrip('/')}/urdf_viewer.js"


class ModelAssetServer:
    """Context manager that keeps packaged asset paths available for serving."""

    def __init__(
        self,
        *,
        model_root: Path | None = None,
        urdf_path: Path | None = None,
        model_url_prefix: str = DEFAULT_MODEL_URL_PREFIX,
        static_url_prefix: str = DEFAULT_UI_STATIC_URL_PREFIX,
    ) -> None:
        self._explicit_model_root = model_root
        self._explicit_urdf_path = urdf_path
        self._model_url_prefix = model_url_prefix
        self._static_url_prefix = static_url_prefix
        self._stack = ExitStack()

    def __enter__(self) -> ServedModelAssets:
        static_root = self._stack.enter_context(as_file(files("ghand.ui") / "static"))

        if self._explicit_model_root is not None:
            model_root = self._explicit_model_root
        elif self._explicit_urdf_path is not None:
            model_root = self._explicit_urdf_path.parent.parent
        else:
            model_root = self._stack.enter_context(as_file(files("ghand.assets") / "models"))

        if self._explicit_urdf_path is not None:
            try:
                urdf_relative_path = self._explicit_urdf_path.relative_to(model_root).as_posix()
            except ValueError as exc:
                raise ValueError("urdf_path must be inside model_root.") from exc
        else:
            urdf_relative_path = DEFAULT_RIGHT_HAND_URDF

        return ServedModelAssets(
            model_root=Path(model_root),
            static_root=Path(static_root),
            model_url_prefix=self._model_url_prefix,
            static_url_prefix=self._static_url_prefix,
            urdf_relative_path=urdf_relative_path,
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._stack.__exit__(exc_type, exc, traceback)
