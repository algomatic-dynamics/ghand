from pathlib import Path

from ghand.ui.model_assets import (
    DEFAULT_RIGHT_HAND_URDF,
    ModelAssetServer,
)


def test_packaged_model_assets_resolve_default_urdf_and_ply_meshes() -> None:
    with ModelAssetServer() as assets:
        assert assets.urdf_relative_path == DEFAULT_RIGHT_HAND_URDF
        assert assets.urdf_url == f"/models/{DEFAULT_RIGHT_HAND_URDF}"
        assert assets.viewer_js_url == "/ghand-static/urdf_viewer.js"
        assert (assets.model_root / assets.urdf_relative_path).is_file()
        assert (assets.model_root / "meshes" / "palm_lower_right.ply").is_file()
        assert (assets.static_root / "urdf_viewer.js").is_file()


def test_model_assets_accept_explicit_model_root_and_urdf(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    urdf = model_root / "urdf" / "hand.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot name='test' />", encoding="utf-8")

    with ModelAssetServer(model_root=model_root, urdf_path=urdf) as assets:
        assert assets.model_root == model_root
        assert assets.urdf_relative_path == "urdf/hand.urdf"
        assert assets.urdf_url == "/models/urdf/hand.urdf"
