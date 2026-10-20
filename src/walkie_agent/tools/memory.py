"""Memory tools: ask walkie-database's node what Walkie remembers.

The database node owns Walkie's memory and answers over ROS 2 services. Their
types come from walkie_database_msgs, which `pixi run build-ros2` builds from
the walkie-database commit pinned in ros2_ws.repos (ADR-0035).
"""

from typing import Type

from pydantic import BaseModel, Field, PrivateAttr
from rai.tools.ros2.base import BaseROS2Tool
from rclpy.client import Client
from walkie_database_msgs.msg import RememberedObject
from walkie_database_msgs.srv import FindObjects

from walkie_agent.tools.waiters import wait_for_future


class FindObjectsToolInput(BaseModel):
    label: str = Field(
        description=(
            "The kind of object to look for, as the object detector names it: "
            "in English and singular, for example 'cup'. It must match the "
            "stored name exactly (case is ignored), so a synonym finds nothing."
        )
    )


class FindObjectsTool(BaseROS2Tool):
    """Asks the database node where objects of one kind were last seen.

    Like the navigation tools, it bounds every wait: a stopped database node
    makes the tool answer within `server_timeout_sec` instead of freezing the
    agent. The request names no map, so the node's map_id parameter decides
    which map is searched.
    """

    name: str = "find_objects"
    description: str = (
        "Look up where Walkie remembers seeing objects of one kind, for example 'cup'. "
        "Returns each one's room and position on the map, most recently seen first. "
        "Only finds objects Walkie has already seen."
    )
    args_schema: Type[FindObjectsToolInput] = FindObjectsToolInput
    service_name: str = "/walkie_database/find_objects"
    max_results: int = 5
    server_timeout_sec: float = 2.0
    call_timeout_sec: float = 5.0

    _client: Client | None = PrivateAttr(default=None)

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = self.connector.node.create_client(
                FindObjects, self.service_name
            )
        return self._client

    def _run(self, label: str) -> str:
        if not self.is_writable(self.service_name):
            return f"Service {self.service_name} is not callable by this agent."
        label = label.strip()
        if not label:
            return "Say what kind of object to look for, for example 'cup'."

        client = self._get_client()
        if not client.wait_for_service(timeout_sec=self.server_timeout_sec):
            return (
                f"Memory is unavailable: no node offers '{self.service_name}'. "
                "The walkie-database node is not running, or not on this ROS "
                "domain."
            )

        future = client.call_async(
            FindObjects.Request(label=label, max_results=self.max_results)
        )
        response = wait_for_future(future, self.call_timeout_sec)
        if response is None:
            client.remove_pending_request(future)
            # rmw_zenoh_cpp finds a server by service name and type name, but
            # sends the request to a key that includes the type hash. A server
            # built from another version of walkie_database_msgs therefore
            # passes wait_for_service and never receives the request
            # (ADR-0035). A busy node looks the same, so the hint goes to the
            # log for a developer, not to the LLM.
            self.connector.node.get_logger().warning(
                f"{self.service_name} did not answer within "
                f"{self.call_timeout_sec:g} s. If the database node is not busy, "
                "it was probably built from another walkie_database_msgs than "
                "the walkie-database commit pinned in ros2_ws.repos."
            )
            return (
                f"Memory did not answer within {self.call_timeout_sec:g} s "
                f"when asked for '{label}'. It may be busy; try again once."
            )
        if not response.success:
            return f"Memory search for '{label}' failed. {response.message}"
        if not response.objects:
            # The node matches the label exactly, so "phone" does not find an
            # object the detector called "cell phone".
            return (
                f"No remembered object is labelled '{label}'. Labels must match "
                "the detector's name exactly; the object may be stored under "
                "another name."
            )
        return "\n".join(
            [f"Remembered '{label}', most recently seen first:"]
            + [f"- {_describe(o)}" for o in response.objects]
        )


def _describe(obj: RememberedObject) -> str:
    """One line for the LLM, for example "cup (a red mug) in the kitchen, at
    x=3.60 m, y=0.40 m in the map frame"."""
    text = obj.label
    if obj.caption:
        text += f" ({obj.caption})"
    if obj.room:
        text += f" in the {obj.room}"
    return (
        f"{text}, at x={obj.position.x:.2f} m, y={obj.position.y:.2f} m "
        f"in the {obj.frame_id} frame"
    )
