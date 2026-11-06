"""The arms: named poses, and which pose each arm is in.

poses loads config/arm_poses.toml and checks it against the robot-model
snapshot (ADR-0022); state matches joint readings to those poses and says
whether the robot may drive (ADR-0024, ADR-0025). Both are plain logic, and
nothing in this package imports ROS. The client of walkie-ros's commander,
which sends the poses and reads the joints, is walkie_agent.ros.commander
(ADR-0021).
"""
