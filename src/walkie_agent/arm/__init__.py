"""The arms: named poses, which pose each arm is in, and the commander's client.

poses loads config/arm_poses.toml and checks it against the robot-model
snapshot (ADR-0022); state matches joint readings to those poses and says
whether the robot may drive (ADR-0024, ADR-0025). Both are plain logic and
run without ROS. commander is the `arm` backend that the arm atomics and the
navigation interlock call (ADR-0021).
"""
