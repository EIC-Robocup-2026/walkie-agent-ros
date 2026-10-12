"""A stand-in for the Walkie robot, for testing agents without hardware or a simulator.

Provides the same ROS 2 interfaces the agent uses on the real robot:
  - /navigate_to_pose  nav2_msgs/action/NavigateToPose (drives in a straight line)
  - TF map -> base_link at 20 Hz
  - /walkie/speech     std_msgs/msg/String (logged instead of voiced)

Goals farther than `max_range` meters from the origin are aborted, so the
failure path can be tested too.
"""

import math
import threading
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class FakeWalkie(Node):
    def __init__(
        self,
        drive_time_sec: float = 0.5,
        max_range: float = 10.0,
        speech_topic: str = "/walkie/speech",
    ):
        super().__init__("fake_walkie")
        self.drive_time_sec = drive_time_sec
        self.max_range = max_range
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.heard: list[str] = []
        self._lock = threading.Lock()

        group = ReentrantCallbackGroup()
        self._tf = TransformBroadcaster(self)
        self.create_timer(0.05, self._publish_tf, callback_group=group)
        self.create_subscription(String, speech_topic, self._on_speech, 10)
        self._nav_server = ActionServer(
            self,
            NavigateToPose,
            "navigate_to_pose",
            execute_callback=self._execute_nav,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )

    def destroy_node(self) -> None:
        # Node.destroy_node() does not destroy action servers; without this a
        # destroyed FakeWalkie leaves an unserved navigate_to_pose on the graph.
        self._nav_server.destroy()
        super().destroy_node()

    def pose(self) -> tuple[float, float, float]:
        with self._lock:
            return self.x, self.y, self.yaw

    def _publish_tf(self) -> None:
        x, y, yaw = self.pose()
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._tf.sendTransform(t)

    def _on_speech(self, msg: String) -> None:
        self.heard.append(msg.data)
        self.get_logger().info(f"[Walkie says] {msg.data}")

    def _execute_nav(self, goal_handle):
        target = goal_handle.request.pose.pose
        tx, ty = target.position.x, target.position.y
        q = target.orientation
        _, _, tyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.get_logger().info(f"Navigating to x={tx:.2f} y={ty:.2f} yaw={tyaw:.2f}")

        result = NavigateToPose.Result()
        if math.hypot(tx, ty) > self.max_range:
            result.error_code = 1
            result.error_msg = f"Goal is outside the map ({self.max_range} m limit)."
            goal_handle.abort()
            return result

        sx, sy, _ = self.pose()
        steps = 10
        for i in range(1, steps + 1):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result
            frac = i / steps
            with self._lock:
                self.x = sx + (tx - sx) * frac
                self.y = sy + (ty - sy) * frac
            feedback = NavigateToPose.Feedback()
            feedback.distance_remaining = math.hypot(tx - self.x, ty - self.y)
            goal_handle.publish_feedback(feedback)
            time.sleep(self.drive_time_sec / steps)

        with self._lock:
            self.yaw = tyaw
        goal_handle.succeed()
        return result


def main() -> None:
    rclpy.init()
    node = FakeWalkie()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    node.get_logger().info("Fake Walkie is up (navigate_to_pose, TF, /walkie/speech).")
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
