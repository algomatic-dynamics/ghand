"""NiceGUI-based operator UI for motor bring-up and hand inspection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass
from typing import Any

from ghand.control import HandOperatorRuntime, JointRuntimeState
from ghand.hand import (
    DISPLAY_HAND_ROTATION_EULER_X_RAD,
    UrdfPoseModel,
)
from ghand.ui.model_assets import ModelAssetServer, ServedModelAssets


JOINT_GRID_COLUMNS = (
    "minmax(170px, 1.1fr) "
    "minmax(340px, 1.8fr) "
    "minmax(110px, 0.7fr) "
    "minmax(140px, 0.8fr) "
    "minmax(180px, 1.1fr)"
)


class OperatorUIDependencyError(RuntimeError):
    """Raised when optional UI dependencies are not installed."""


@dataclass
class _JointWidgets:
    dxl_id: int
    urdf_dxl_id: int
    slider: Any
    target: Any
    measured: Any
    target_tick: Any
    telemetry: Any


def run_operator_ui(
    runtime: HandOperatorRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "ghand Operator",
    enable_3d: bool = True,
    model_root: Path | None = None,
    urdf_path: Path | None = None,
    viewer_angle_scale: float = 1.0,
    reload: bool = False,
    show: bool = False,
) -> None:
    """Run the NiceGUI operator UI."""

    try:
        from nicegui import app, ui
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise OperatorUIDependencyError(
            "NiceGUI is required for the operator UI. "
            "Install with `uv sync --extra ui` or run `uv run --extra ui ghand ui`."
        ) from exc

    if enable_3d:
        with ModelAssetServer(model_root=model_root, urdf_path=urdf_path) as assets:
            pose_model = UrdfPoseModel.from_urdf_file(
                assets.model_root / assets.urdf_relative_path
            )
            app.add_static_files(
                assets.model_url_prefix, assets.model_root, max_cache_age=0
            )
            app.add_static_files(
                assets.static_url_prefix, assets.static_root, max_cache_age=0
            )

            @ui.page("/")
            def index() -> None:
                _build_operator_page(
                    ui,
                    runtime,
                    title,
                    assets=assets,
                    pose_model=pose_model,
                    viewer_angle_scale=viewer_angle_scale,
                )

            ui.run(title=title, host=host, port=port, reload=reload, show=show)
    else:

        @ui.page("/")
        def index() -> None:
            _build_operator_page(
                ui,
                runtime,
                title,
                assets=None,
                pose_model=UrdfPoseModel.empty(),
                viewer_angle_scale=viewer_angle_scale,
            )

        ui.run(title=title, host=host, port=port, reload=reload, show=show)


def _build_operator_page(
    ui: Any,
    runtime: HandOperatorRuntime,
    title: str,
    *,
    assets: ServedModelAssets | None,
    pose_model: UrdfPoseModel,
    viewer_angle_scale: float,
) -> None:
    ui.colors(primary="#2563eb", secondary="#475569", accent="#059669")
    status = ui.label()
    error = ui.label().classes("text-red-700")
    joint_widgets: dict[str, _JointWidgets] = {}
    viewer_global = f"ghandUrdfViewer_{uuid4().hex}"

    def refresh_status() -> None:
        snapshot = runtime.snapshot()
        connection = "connected" if snapshot.connected else "disconnected"
        dry_run = "dry-run" if snapshot.dry_run else "writes enabled"
        running = any(joint.torque_enabled for joint in snapshot.joints)
        state = (
            "E-stop" if snapshot.emergency_stop else "running" if running else "stopped"
        )
        status.set_text(f"{snapshot.backend} | {connection} | {dry_run} | {state}")
        error.set_text(snapshot.last_error or "")

    def start() -> None:
        try:
            snapshot = runtime.start()
            for joint in snapshot.joints:
                joint_widgets[joint.joint_name].slider.value = math.degrees(
                    joint.target
                )
        except Exception as exc:  # pragma: no cover - depends on concrete backend.
            runtime.last_error = str(exc)
        refresh()

    def disconnect() -> None:
        try:
            runtime.disconnect()
        except Exception as exc:  # pragma: no cover - depends on concrete backend.
            runtime.last_error = str(exc)
        refresh()

    def stop() -> None:
        try:
            runtime.set_torque_enabled(False)
        except Exception as exc:  # pragma: no cover - depends on concrete backend.
            runtime.last_error = str(exc)
        refresh()

    def home() -> None:
        runtime.home()
        for joint_name, widgets in joint_widgets.items():
            runtime.set_target(joint_name, 0.0)
            widgets.slider.value = 0.0
            widgets.slider.update()
        refresh()

    def emergency_stop() -> None:
        try:
            runtime.emergency_stop_now()
        except Exception as exc:  # pragma: no cover - depends on concrete backend.
            runtime.last_error = str(exc)
        refresh()

    def set_dry_run(event: object) -> None:
        try:
            value = bool(getattr(event, "value"))
            if runtime.connected:
                runtime.disconnect()
            runtime.set_dry_run(value)
        except Exception as exc:  # pragma: no cover - depends on concrete backend.
            runtime.last_error = str(exc)
        refresh()

    def set_joint_target(joint_name: str, urdf_dxl_id: int, event: object) -> None:
        _ = urdf_dxl_id
        runtime.set_target(joint_name, math.radians(float(getattr(event, "value"))))

    def refresh() -> None:
        snapshot = runtime.step()
        dry_run_switch.value = snapshot.dry_run
        dry_run_switch.update()
        refresh_status()
        for state in snapshot.joints:
            widgets = joint_widgets[state.joint_name]
            widgets.target.set_text(_format_joint_degrees(math.degrees(state.target)))
            widgets.measured.set_text(
                _format_joint_degrees(math.degrees(state.measured))
            )
            widgets.target_tick.set_text(str(state.target_tick))
            widgets.telemetry.set_text(_format_telemetry(state))
        update_3d(snapshot)

    def update_3d(snapshot: Any) -> None:
        if assets is None:
            return
        values = {
            f"id{state.pose_dxl_id}": _viewer_joint_angle(
                snapshot,
                state,
                viewer_angle_scale,
                pose_model,
            )
            for state in snapshot.joints
        }
        ui.run_javascript(
            f"window[{json.dumps(viewer_global)}]?.setJointValues({json.dumps(values)});",
            timeout=0.1,
        )

    with ui.header().classes("items-center justify-between bg-slate-900 text-white"):
        ui.label(title).classes("text-lg font-semibold")
        status.classes("text-sm")

    with ui.row().classes("w-full items-center gap-2 p-3"):
        ui.button("Start / Resume", icon="play_arrow", on_click=start).props(
            "color=positive"
        )
        ui.button("Stop", icon="pause", on_click=stop).props("outline")
        ui.button("Home", icon="home", on_click=home)
        ui.button("E-stop", icon="warning", on_click=emergency_stop).props(
            "color=negative"
        )
    with ui.expansion("Connection settings").classes("w-full px-3"):
        ui.button("Disconnect", icon="link_off", on_click=disconnect).props("outline")
        dry_run_switch = ui.switch(
            "Dry run", value=runtime.dry_run, on_change=set_dry_run
        )

    error.classes("px-3 min-h-6")

    if assets is not None:
        _add_urdf_viewer(ui, assets=assets, viewer_global=viewer_global)

    with ui.column().classes("w-full gap-1 px-3 pb-4"):
        with (
            ui.grid()
            .style(f"grid-template-columns: {JOINT_GRID_COLUMNS}")
            .classes(
                "w-full items-center gap-x-3 gap-y-1 text-sm font-semibold text-slate-600"
            )
        ):
            ui.label("Joint")
            ui.label("Target (deg)")
            ui.label("Measured (deg)")
            ui.label("Target tick")
            ui.label("Telemetry")

        for state in runtime.snapshot().joints:
            slider_min, slider_max = pose_model.degree_range(dxl_id=state.pose_dxl_id)
            with (
                ui.grid()
                .style(f"grid-template-columns: {JOINT_GRID_COLUMNS}")
                .classes(
                    "w-full items-center gap-x-3 gap-y-1 border-t border-slate-200 py-1"
                )
            ):
                ui.label(_short_joint_name(state.joint_name)).classes(
                    "text-sm font-mono"
                )
                with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
                    slider = (
                        ui.slider(
                            min=slider_min,
                            max=slider_max,
                            step=1.0,
                            value=math.degrees(state.target),
                            on_change=lambda e, name=state.joint_name, urdf_dxl_id=state.pose_dxl_id: (
                                set_joint_target(
                                    name,
                                    urdf_dxl_id,
                                    e,
                                )
                            ),
                        )
                        .props("dense")
                        .classes("flex-1 min-w-40")
                    )
                    target = ui.label(
                        _format_joint_degrees(math.degrees(state.target))
                    ).classes(
                        "w-20 shrink-0 text-right font-mono text-sm text-slate-700"
                    )
                measured = ui.label(
                    _format_joint_degrees(math.degrees(state.measured))
                ).classes("font-mono")
                target_tick = ui.label(str(state.target_tick)).classes("font-mono")
                telemetry = ui.label(_format_telemetry(state)).classes(
                    "font-mono text-xs"
                )
                joint_widgets[state.joint_name] = _JointWidgets(
                    dxl_id=state.dxl_id,
                    urdf_dxl_id=state.pose_dxl_id,
                    slider=slider,
                    target=target,
                    measured=measured,
                    target_tick=target_tick,
                    telemetry=telemetry,
                )

    ui.timer(1.0 / runtime.target_hz, refresh)
    refresh_status()


def _add_urdf_viewer(ui: Any, *, assets: ServedModelAssets, viewer_global: str) -> None:
    container_id = f"ghand-urdf-viewer-{uuid4().hex}"
    status_id = f"{container_id}-status"
    ui.add_head_html(
        f"""
        <script type="module" src="{assets.viewer_js_url}"></script>
        <style>
          .ghand-urdf-viewer {{
            width: 100%;
            height: min(48vh, 520px);
            min-height: 320px;
            position: relative;
            overflow: hidden;
            border-top: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
            background: #f8fafc;
          }}
          .ghand-urdf-viewer canvas {{
            display: block;
            width: 100%;
            height: 100%;
          }}
          .ghand-urdf-viewer-status {{
            position: absolute;
            left: 12px;
            bottom: 10px;
            padding: 3px 8px;
            border-radius: 4px;
            background: rgba(15, 23, 42, 0.72);
            color: white;
            font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            pointer-events: none;
          }}
        </style>
        """
    )
    ui.html(
        f"""
        <div class="px-3">
          <div id="{container_id}" class="ghand-urdf-viewer">
            <div id="{status_id}" class="ghand-urdf-viewer-status">Loading viewer...</div>
          </div>
        </div>
        """
    ).classes("w-full")
    ui.run_javascript(
        f"""
        (async () => {{
          const status = document.getElementById({json.dumps(status_id)});
          try {{
            for (let i = 0; i < 200 && !window.GHandUrdfViewer; i += 1) {{
              await new Promise(resolve => setTimeout(resolve, 50));
            }}
            if (!window.GHandUrdfViewer) throw new Error("URDF viewer module did not load.");
            window[{json.dumps(viewer_global)}] = await window.GHandUrdfViewer.create({{
              containerId: {json.dumps(container_id)},
              urdfUrl: {json.dumps(assets.urdf_url)},
              rootRotationXRad: {json.dumps(DISPLAY_HAND_ROTATION_EULER_X_RAD)},
            }});
          }} catch (error) {{
            console.error(error);
            if (status) status.textContent = `3D viewer failed: ${{error.message || error}}`;
          }}
        }})();
        """,
        timeout=1.0,
    )


def _viewer_pose_value(snapshot: Any, state: JointRuntimeState) -> float:
    if snapshot.connected and not snapshot.dry_run:
        return state.measured
    return state.target


def _viewer_joint_angle(
    snapshot: Any,
    state: JointRuntimeState,
    viewer_angle_scale: float,
    pose_model: UrdfPoseModel,
) -> float:
    _ = pose_model
    return _viewer_pose_value(snapshot, state) * viewer_angle_scale


def _short_joint_name(joint_name: str) -> str:
    return joint_name.removeprefix("right_hand.").removeprefix("left_hand.")


def _format_joint_degrees(value_degrees: float) -> str:
    if math.isclose(value_degrees, 0.0, abs_tol=0.05):
        value_degrees = 0.0
    return f"{value_degrees:+.1f} deg"


def _format_telemetry(state: JointRuntimeState) -> str:
    limit = (
        "--" if state.current_limit_ma is None else f"{state.current_limit_ma:.0f}mA"
    )
    current = (
        "--"
        if state.present_current_ma is None
        else f"{state.present_current_ma:.0f}mA"
    )
    temperature = (
        "--" if state.temperature_celsius is None else f"{state.temperature_celsius}C"
    )
    voltage = "--" if state.input_voltage is None else f"{state.input_voltage:.1f}V"
    error = (
        "--"
        if state.hardware_error_status is None
        else str(state.hardware_error_status)
    )
    return f"{voltage} {temperature} {current} lim={limit} err={error}"
