"""The walkie_vision_msgs fields the people and seat tools read (no ROS graph).

A pin move in ros2_ws.repos that renames or resizes one of these fields fails
here, before a tool reads a wrong field on the robot.
"""

import pytest
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)
from walkie_vision_msgs.srv import EmbedText


@pytest.mark.parametrize(
    ("message", "fields"),
    [
        (
            PersonDetectionArray,
            {
                "header": "std_msgs/Header",
                "camera_info": "sensor_msgs/CameraInfo",
                "people": "sequence<walkie_vision_msgs/PersonDetection>",
            },
        ),
        (
            PersonDetection,
            {
                "score": "float",
                "track_id": "uint32",
                "bbox": "vision_msgs/BoundingBox2D",
                # COCO's 17 keypoints, as x and y pairs.
                "keypoints": "float[34]",
                "keypoint_scores": "float[17]",
                "face_bbox": "vision_msgs/BoundingBox2D",
                "face_det_score": "float",
                "face_model": "string",
                "face_embedding": "sequence<float>",
                "appearance_model": "string",
                "appearance_embedding": "sequence<float>",
            },
        ),
        (
            ObjectDetectionArray,
            {
                "header": "std_msgs/Header",
                "camera_info": "sensor_msgs/CameraInfo",
                "classes": "sequence<string>",
                "detections": "sequence<walkie_vision_msgs/ObjectDetection>",
            },
        ),
        (
            ObjectDetection,
            {
                "label": "string",
                "score": "float",
                "track_id": "uint32",
                "bbox": "vision_msgs/BoundingBox2D",
                "points": "sensor_msgs/PointCloud2",
            },
        ),
    ],
    ids=["PersonDetectionArray", "PersonDetection", "ObjectDetectionArray", "ObjectDetection"],
)
def test_fields_the_tools_read(message, fields):
    types = message.get_fields_and_field_types()
    assert {name: types.get(name) for name in fields} == fields


def test_embed_text_service_imports():
    assert EmbedText.Request and EmbedText.Response
