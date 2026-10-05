from importlib.resources import files
from xml.etree import ElementTree

from ghand.data.lerobot_schema import (
    ACTION_KEY,
    AUX_TELEOP_HANDEDNESS_KEY,
    AUX_TELEOP_LANDMARKS_KEY,
    AUX_TELEOP_SCORE_KEY,
    AUX_TELEOP_VALID_KEY,
    AUX_TELEOP_WORLD_LANDMARKS_KEY,
    CameraSpec,
    DEFAULT_FPS,
    DEFAULT_FRONT_IMAGE_SHAPE,
    FRONT_IMAGE_KEY,
    GHAND_LEROBOT_FEATURES,
    HAND_LANDMARK_COUNT,
    HANDEDNESS_LABEL_TO_ID,
    HANDEDNESS_LEFT,
    HANDEDNESS_RIGHT,
    HANDEDNESS_UNKNOWN,
    LANDMARK_DIMS,
    OBSERVATION_STATE_KEY,
    TELEOP_HAND_SLOT_NAMES,
    TELEOP_HAND_SLOTS,
    camera_feature_key,
    copy_lerobot_features,
    make_camera_features,
    make_hand_joint_contact_names,
    make_hand_joint_current_limit_action_names,
    make_hand_joint_current_residual_names,
    make_hand_joint_present_current_names,
    make_hand_joint_position_names,
    make_hand_lerobot_features,
    make_hand_position_current_action_names,
    make_hand_position_current_lerobot_features,
    make_hand_position_current_state_names,
    make_joint_lerobot_features,
    make_lerobot_features,
)
from ghand.hand import (
    BIMANUAL_HAND_SIDES,
    DEFAULT_HAND_SIDES,
    GHAND_HAND_JOINT_POSITION_NAMES,
    GHAND_HAND_JOINTS,
    JOINT_CONTACT_KEY,
    JOINT_CURRENT_LIMIT_KEY,
    JOINT_CURRENT_RESIDUAL_KEY,
    JOINT_POSITION_KEY,
    JOINT_PRESENT_CURRENT_KEY,
    LEFT_HAND_SIDE,
    JointComponent,
    RIGHT_HAND_SIDE,
    compose_joint_position_names,
    make_hand_joint_component,
    make_hand_joint_names,
)


def test_ghand_hand_joint_order_is_fixed() -> None:
    assert DEFAULT_HAND_SIDES == (RIGHT_HAND_SIDE,)
    assert BIMANUAL_HAND_SIDES == (LEFT_HAND_SIDE, RIGHT_HAND_SIDE)
    assert len(GHAND_HAND_JOINTS) == 16
    assert len(set(GHAND_HAND_JOINTS)) == len(GHAND_HAND_JOINTS)
    assert GHAND_HAND_JOINTS == (
        "ring_mcp_side",
        "ring_mcp_flex",
        "ring_pip_flex",
        "ring_dip_flex",
        "middle_mcp_side",
        "middle_mcp_flex",
        "middle_pip_flex",
        "middle_dip_flex",
        "index_mcp_side",
        "index_mcp_flex",
        "index_pip_flex",
        "index_dip_flex",
        "thumb_cmc_side",
        "thumb_cmc_flex",
        "thumb_mcp_flex",
        "thumb_ip_flex",
    )
    assert GHAND_HAND_JOINT_POSITION_NAMES == tuple(f"{joint}.pos" for joint in GHAND_HAND_JOINTS)
    assert JOINT_POSITION_KEY == "pos"
    assert JOINT_PRESENT_CURRENT_KEY == "present_current_ma"
    assert JOINT_CURRENT_LIMIT_KEY == "current_limit_ma"
    assert JOINT_CURRENT_RESIDUAL_KEY == "current_residual_ma"
    assert JOINT_CONTACT_KEY == "contact"


def test_make_hand_joint_names_adds_side_namespace() -> None:
    assert make_hand_joint_names("right")[0] == "right_hand.ring_mcp_side.pos"
    assert make_hand_joint_names("right")[-1] == "right_hand.thumb_ip_flex.pos"
    assert make_hand_joint_names("left")[0] == "left_hand.ring_mcp_side.pos"
    assert make_hand_joint_names("left")[-1] == "left_hand.thumb_ip_flex.pos"


def test_make_hand_joint_component_wraps_ghand_hand_joints() -> None:
    component = make_hand_joint_component("right")

    assert component.prefix == "right_hand"
    assert component.joints == GHAND_HAND_JOINTS
    assert component.position_key == "pos"
    assert component.position_names() == make_hand_joint_names("right")


def test_make_hand_current_channel_names_keep_side_and_joint_order() -> None:
    present_current_names = make_hand_joint_present_current_names(("right",))
    current_limit_names = make_hand_joint_current_limit_action_names(("right",))
    residual_names = make_hand_joint_current_residual_names(("right",))
    contact_names = make_hand_joint_contact_names(("right",))

    assert len(present_current_names) == 16
    assert len(current_limit_names) == 16
    assert len(residual_names) == 16
    assert len(contact_names) == 16
    assert present_current_names[9] == "right_hand.index_mcp_flex.present_current_ma"
    assert current_limit_names[9] == "right_hand.index_mcp_flex.current_limit_ma"
    assert residual_names[9] == "right_hand.index_mcp_flex.current_residual_ma"
    assert contact_names[9] == "right_hand.index_mcp_flex.contact"


def test_hand_joint_order_matches_packaged_urdf_finger_hierarchy() -> None:
    urdf_path = files("ghand.assets") / "models" / "urdf" / "ghand_right.urdf"
    id_groups = _active_joint_ids_by_fixed_finger(urdf_path)

    assert id_groups == {
        "fixed_ring": (0, 1, 2, 3),
        "fixed_middle": (4, 5, 6, 7),
        "fixed_index": (8, 9, 10, 11),
        "fixed_thumb": (12, 13, 14, 15),
    }


def _active_joint_ids_by_fixed_finger(urdf_path: object) -> dict[str, tuple[int, ...]]:
    root = ElementTree.parse(urdf_path).getroot()
    children_by_parent: dict[str, list[tuple[str, str]]] = {}
    fixed_finger_roots: dict[str, str] = {}
    for joint in root.findall("joint"):
        joint_name = joint.attrib["name"]
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.attrib["link"]
        child_link = child.attrib["link"]
        children_by_parent.setdefault(parent_link, []).append((joint_name, child_link))
        if joint_name in {"fixed_ring", "fixed_middle", "fixed_index", "fixed_thumb"}:
            fixed_finger_roots[joint_name] = child_link

    groups: dict[str, tuple[int, ...]] = {}
    for fixed_joint_name, root_link in fixed_finger_roots.items():
        groups[fixed_joint_name] = tuple(
            sorted(_active_joint_ids_under(root_link, children_by_parent))
        )
    return groups


def _active_joint_ids_under(
    root_link: str,
    children_by_parent: dict[str, list[tuple[str, str]]],
) -> list[int]:
    active_ids: list[int] = []
    stack = [root_link]
    visited: set[str] = set()
    while stack:
        link = stack.pop()
        if link in visited:
            continue
        visited.add(link)
        for joint_name, child_link in children_by_parent.get(link, []):
            if joint_name.startswith("id") and joint_name[2:].isdigit():
                active_ids.append(int(joint_name[2:]))
            stack.append(child_link)
    return active_ids


def test_make_hand_joint_position_names_uses_requested_order() -> None:
    names = make_hand_joint_position_names(("left", "right"))

    assert len(names) == 32
    assert names[:16] == make_hand_joint_names("left")
    assert names[16:] == make_hand_joint_names("right")


def test_joint_component_composes_external_robot_parts() -> None:
    right_arm = JointComponent(prefix="right_arm", joints=("shoulder_pan", "elbow_flex", "wrist_roll"))
    right_hand = make_hand_joint_component("right")

    names = compose_joint_position_names((right_arm, right_hand))

    assert len(names) == 19
    assert names[:3] == (
        "right_arm.shoulder_pan.pos",
        "right_arm.elbow_flex.pos",
        "right_arm.wrist_roll.pos",
    )
    assert names[3:] == make_hand_joint_names("right")


def test_joint_lerobot_features_accept_composed_components() -> None:
    right_arm = JointComponent(prefix="right_arm", joints=("shoulder_pan", "elbow_flex"))
    right_hand = make_hand_joint_component("right")
    features = make_joint_lerobot_features(components=(right_arm, right_hand), include_aux=False)
    expected_names = list(compose_joint_position_names((right_arm, right_hand)))

    assert features[OBSERVATION_STATE_KEY]["shape"] == (18,)
    assert features[ACTION_KEY]["shape"] == (18,)
    assert features[OBSERVATION_STATE_KEY]["names"] == expected_names
    assert features[ACTION_KEY]["names"] == expected_names
    assert AUX_TELEOP_LANDMARKS_KEY not in features


def test_lerobot_state_and_action_share_joint_names() -> None:
    features = GHAND_LEROBOT_FEATURES
    state = features[OBSERVATION_STATE_KEY]
    action = features[ACTION_KEY]

    assert DEFAULT_FPS == 30
    assert state["dtype"] == "float32"
    assert action["dtype"] == "float32"
    assert state["shape"] == (16,)
    assert action["shape"] == (16,)
    assert state["names"] == list(make_hand_joint_names("right"))
    assert action["names"] == list(make_hand_joint_names("right"))


def test_bimanual_lerobot_features_concatenate_left_then_right() -> None:
    features = make_hand_lerobot_features(sides=BIMANUAL_HAND_SIDES, include_aux=False)
    state = features[OBSERVATION_STATE_KEY]
    action = features[ACTION_KEY]

    expected_names = list(make_hand_joint_position_names(("left", "right")))
    assert state["shape"] == (32,)
    assert action["shape"] == (32,)
    assert state["names"] == expected_names
    assert action["names"] == expected_names


def test_position_current_lerobot_features_extend_state_and_action() -> None:
    features = make_hand_position_current_lerobot_features(
        sides=("right",),
        front_image_shape=(240, 320, 3),
        image_dtype="image",
        include_aux=False,
    )
    state = features[OBSERVATION_STATE_KEY]
    action = features[ACTION_KEY]

    expected_state_names = list(make_hand_position_current_state_names(("right",)))
    expected_action_names = list(make_hand_position_current_action_names(("right",)))
    assert state["dtype"] == "float32"
    assert action["dtype"] == "float32"
    assert state["shape"] == (32,)
    assert action["shape"] == (32,)
    assert state["names"] == expected_state_names
    assert action["names"] == expected_action_names
    assert state["names"][:16] == list(make_hand_joint_position_names(("right",)))
    assert state["names"][16:] == list(make_hand_joint_present_current_names(("right",)))
    assert action["names"][:16] == list(make_hand_joint_position_names(("right",)))
    assert action["names"][16:] == list(make_hand_joint_current_limit_action_names(("right",)))
    assert features[FRONT_IMAGE_KEY]["dtype"] == "image"
    assert AUX_TELEOP_VALID_KEY not in features


def test_position_current_lerobot_features_support_aux_and_bimanual_order() -> None:
    features = make_hand_position_current_lerobot_features(
        sides=BIMANUAL_HAND_SIDES,
        include_aux=True,
    )

    assert features[OBSERVATION_STATE_KEY]["shape"] == (64,)
    assert features[ACTION_KEY]["shape"] == (64,)
    assert features[OBSERVATION_STATE_KEY]["names"] == list(
        make_hand_position_current_state_names(BIMANUAL_HAND_SIDES)
    )
    assert features[ACTION_KEY]["names"] == list(
        make_hand_position_current_action_names(BIMANUAL_HAND_SIDES)
    )
    assert AUX_TELEOP_VALID_KEY in features


def test_position_current_lerobot_features_can_include_current_diagnostics() -> None:
    features = make_hand_position_current_lerobot_features(
        sides=("right",),
        include_aux=False,
        include_current_diagnostics=True,
    )

    state_names = features[OBSERVATION_STATE_KEY]["names"]
    assert features[OBSERVATION_STATE_KEY]["shape"] == (64,)
    assert features[ACTION_KEY]["shape"] == (32,)
    assert state_names[:16] == list(make_hand_joint_position_names(("right",)))
    assert state_names[16:32] == list(make_hand_joint_present_current_names(("right",)))
    assert state_names[32:48] == list(make_hand_joint_current_residual_names(("right",)))
    assert state_names[48:] == list(make_hand_joint_contact_names(("right",)))


def test_front_image_feature_uses_lerobot_video_key() -> None:
    image = GHAND_LEROBOT_FEATURES[FRONT_IMAGE_KEY]

    assert image["dtype"] == "video"
    assert image["shape"] == DEFAULT_FRONT_IMAGE_SHAPE
    assert image["names"] == ["height", "width", "channels"]


def test_camera_features_support_multiple_external_cameras() -> None:
    cameras = (
        CameraSpec("front", (240, 320, 3), "image"),
        CameraSpec("external", (720, 1280, 3), "video"),
    )
    features = make_hand_lerobot_features(cameras=cameras, include_aux=False)

    assert camera_feature_key("front") == FRONT_IMAGE_KEY
    assert features[FRONT_IMAGE_KEY] == {
        "dtype": "image",
        "shape": (240, 320, 3),
        "names": ["height", "width", "channels"],
    }
    assert features["observation.images.external"] == {
        "dtype": "video",
        "shape": (720, 1280, 3),
        "names": ["height", "width", "channels"],
    }


def test_make_camera_features_preserves_front_image_compatibility() -> None:
    features = make_camera_features(front_image_shape=(120, 160, 3), image_dtype="image")

    assert features == {
        FRONT_IMAGE_KEY: {
            "dtype": "image",
            "shape": (120, 160, 3),
            "names": ["height", "width", "channels"],
        }
    }


def test_camera_features_can_be_omitted_explicitly() -> None:
    features = make_hand_lerobot_features(cameras=(), include_aux=False)

    assert OBSERVATION_STATE_KEY in features
    assert ACTION_KEY in features
    assert not any(key.startswith("observation.images.") for key in features)


def test_aux_teleop_features_are_not_policy_observations() -> None:
    features = GHAND_LEROBOT_FEATURES

    for key in (
        AUX_TELEOP_LANDMARKS_KEY,
        AUX_TELEOP_WORLD_LANDMARKS_KEY,
        AUX_TELEOP_HANDEDNESS_KEY,
        AUX_TELEOP_SCORE_KEY,
        AUX_TELEOP_VALID_KEY,
    ):
        assert key in features
        assert not key.startswith("observation.")
        assert not key.startswith("action")

    assert features[AUX_TELEOP_LANDMARKS_KEY]["shape"] == (
        TELEOP_HAND_SLOTS,
        HAND_LANDMARK_COUNT,
        LANDMARK_DIMS,
    )
    assert features[AUX_TELEOP_WORLD_LANDMARKS_KEY]["shape"] == (
        TELEOP_HAND_SLOTS,
        HAND_LANDMARK_COUNT,
        LANDMARK_DIMS,
    )
    assert features[AUX_TELEOP_HANDEDNESS_KEY]["dtype"] == "int64"
    assert features[AUX_TELEOP_SCORE_KEY]["dtype"] == "float32"
    assert features[AUX_TELEOP_VALID_KEY]["dtype"] == "int64"
    assert features[AUX_TELEOP_HANDEDNESS_KEY]["names"] == list(TELEOP_HAND_SLOT_NAMES)
    assert features[AUX_TELEOP_SCORE_KEY]["names"] == list(TELEOP_HAND_SLOT_NAMES)
    assert features[AUX_TELEOP_VALID_KEY]["names"] == list(TELEOP_HAND_SLOT_NAMES)


def test_handedness_ids_are_stable() -> None:
    assert HANDEDNESS_UNKNOWN == 0
    assert HANDEDNESS_LEFT == 1
    assert HANDEDNESS_RIGHT == 2
    assert HANDEDNESS_LABEL_TO_ID == {
        "Unknown": HANDEDNESS_UNKNOWN,
        "Left": HANDEDNESS_LEFT,
        "Right": HANDEDNESS_RIGHT,
    }


def test_make_lerobot_features_can_disable_aux_and_change_image_dtype() -> None:
    features = make_lerobot_features(
        sides=("left",),
        front_image_shape=(240, 320, 3),
        image_dtype="image",
        include_aux=False,
    )

    assert features[FRONT_IMAGE_KEY]["dtype"] == "image"
    assert features[FRONT_IMAGE_KEY]["shape"] == (240, 320, 3)
    assert features[OBSERVATION_STATE_KEY]["names"] == list(make_hand_joint_names("left"))
    assert AUX_TELEOP_LANDMARKS_KEY not in features


def test_copy_lerobot_features_returns_deep_copy() -> None:
    features = copy_lerobot_features()
    names = features[OBSERVATION_STATE_KEY]["names"]
    assert isinstance(names, list)

    names.append("unexpected.pos")

    assert GHAND_LEROBOT_FEATURES[OBSERVATION_STATE_KEY]["names"] == list(make_hand_joint_names("right"))


def test_invalid_or_duplicate_hand_sides_raise() -> None:
    try:
        make_hand_joint_names("center")
    except ValueError as exc:
        assert "Unsupported hand side" in str(exc)
    else:
        raise AssertionError("Expected invalid hand side to raise.")

    try:
        make_hand_joint_position_names(("right", "right"))
    except ValueError as exc:
        assert "Duplicate hand side" in str(exc)
    else:
        raise AssertionError("Expected duplicate hand sides to raise.")


def test_invalid_or_duplicate_joint_components_raise() -> None:
    for kwargs in (
        {"prefix": "", "joints": ("joint",)},
        {"prefix": "right/arm", "joints": ("joint",)},
        {"prefix": "right_arm", "joints": ()},
        {"prefix": "right_arm", "joints": ("joint", "joint")},
        {"prefix": "right_arm", "joints": ("bad/joint",)},
    ):
        try:
            JointComponent(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid component to raise: {kwargs}")

    try:
        compose_joint_position_names(
            (
                JointComponent(prefix="right_arm", joints=("joint",)),
                JointComponent(prefix="right_arm", joints=("joint",)),
            )
        )
    except ValueError as exc:
        assert "Duplicate joint position channel" in str(exc)
    else:
        raise AssertionError("Expected duplicate composed names to raise.")


def test_invalid_or_duplicate_cameras_raise() -> None:
    invalid_specs = (
        {"name": "", "shape": (64, 64, 3), "dtype": "image"},
        {"name": "bad/cam", "shape": (64, 64, 3), "dtype": "image"},
        {"name": "front", "shape": (64, 64), "dtype": "image"},
        {"name": "front", "shape": (64, 64, 1), "dtype": "image"},
        {"name": "front", "shape": (64, 0, 3), "dtype": "image"},
        {"name": "front", "shape": (64, 64, 3), "dtype": "tensor"},
    )
    for kwargs in invalid_specs:
        try:
            CameraSpec(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid camera spec to raise: {kwargs}")

    try:
        make_camera_features((CameraSpec("front"), CameraSpec("front")))
    except ValueError as exc:
        assert "Duplicate camera feature key" in str(exc)
    else:
        raise AssertionError("Expected duplicate camera features to raise.")
