"""The ROS-backed capability verbs and the operator's topics, one module each.

The modules load rclpy; this package init imports nothing, so callers import each class
from its own module. The foundation (capabilities, atomics, adapter, runtime) never
imports this package.
"""
