from ghand.teleop.hand_pose import HandDetection, HandFrame, Landmark


def test_hand_frame_to_dict_is_json_serializable_shape() -> None:
    frame = HandFrame(
        timestamp_ns=123,
        image_width=640,
        image_height=480,
        hands=(
            HandDetection(
                handedness="Right",
                score=0.99,
                landmarks=(Landmark(0.1, 0.2, 0.3),),
                world_landmarks=None,
            ),
        ),
    )

    data = frame.to_dict()

    assert data["timestamp_ns"] == 123
    assert data["image_width"] == 640
    assert data["image_height"] == 480
    assert data["hands"][0]["handedness"] == "Right"
    assert data["hands"][0]["landmarks"][0] == {"x": 0.1, "y": 0.2, "z": 0.3}


def test_hand_frame_from_dict_round_trips_json_shape() -> None:
    data = {
        "timestamp_ns": 123,
        "image_width": 640,
        "image_height": 480,
        "hands": [
            {
                "handedness": "Right",
                "score": 0.99,
                "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}],
                "world_landmarks": [{"x": 1.0, "y": 2.0, "z": 3.0}],
            }
        ],
    }

    frame = HandFrame.from_dict(data)

    assert frame.timestamp_ns == 123
    assert frame.image_width == 640
    assert frame.image_height == 480
    assert frame.hands[0].handedness == "Right"
    assert frame.hands[0].landmarks[0] == Landmark(0.1, 0.2, 0.3)
    assert frame.hands[0].world_landmarks == (Landmark(1.0, 2.0, 3.0),)
