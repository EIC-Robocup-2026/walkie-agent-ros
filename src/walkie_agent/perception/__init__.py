"""Plain perception logic behind the people and seat atomics (ADR-0050).

The atomics in walkie_agent.tools.atomic read walkie-vision's people and
object arrays; the logic that turns those messages into answers lives here
so that it runs in tests without LangChain and, apart from geometry's TF
lookup and vision_feed's subscriptions, without ROS. Gestures, seating and
identity are ported from walkie-agent-v2 with its defaults, which were tuned
on the robot.
"""
