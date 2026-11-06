"""Rules that read walkie-vision's detections. No ROS.

The modules here take plain arrays, such as a person's keypoints, so the
capabilities, the atomics and the simulation's oracle share one reading of
them, and the tests run without a ROS installation.
"""

# walkie-vision's output topics (walkie-vision docs/architecture/vision.md). The
# simulation's oracle publishes on them too, so a consumer cannot tell the two apart.
PEOPLE_TOPIC = "/walkie_vision/people"
OBJECTS_TOPIC = "/walkie_vision/objects"
