# 5. Limit What Tools Can Touch

**Goal:** make sure no tool lets the LLM use ROS topics it should never
touch, such as `/cmd_vel`. This follows RAI's
[tools tutorial](https://robotecai.github.io/rai/tutorials/tools/).

## Background

- The agent can only act through its tools. Our tools each use one fixed
  topic or action (for example, `speak` only publishes to `/to_human`).
- RAI also has generic tools, such as `PublishROS2MessageTool`, that publish
  **any** message to **any** topic. If one of those is added, the LLM could
  send velocity commands to `/cmd_vel` and drive the base without Nav2's
  obstacle avoidance.
- Every RAI ROS 2 tool (a subclass of `BaseROS2Tool`) accepts three lists:

| Field       | Meaning                                                              |
| ----------- | -------------------------------------------------------------------- |
| `forbidden` | Names the tool may never read or write. Always wins.                 |
| `writable`  | If set, the only topics, services and actions the tool may write to. |
| `readable`  | If set, the only topics the tool may read.                           |

Tools that list topics, such as `get_ros2_topics_names_and_types`, also hide
the names the tool is not allowed to use, so the LLM never sees them.

!!! warning "This is a guard rail, not security"
    Each tool checks these lists itself, in Python. A tool that is not given
    the list, or a plain `@tool` function, is not limited at all. Real access
    control in ROS 2 is done with [SROS2](https://docs.ros.org/en/jazzy/Tutorials/Advanced/Security/Introducing-ros2-security.html).

## Step 1: One deny list for every tool

We keep a single list in `src/walkie_agent/tools/__init__.py` and pass it to
every ROS 2 tool:

```python title="src/walkie_agent/tools/__init__.py"
FORBIDDEN_NAMES = ["/cmd_vel", "/walkie/reset"]


def build_walkie_tools(connector, locations, whoami_dir=None):
    tools = [
        SpeakTool(connector=connector, forbidden=FORBIDDEN_NAMES),
        NavigateToLocationTool(connector=connector, locations=locations, forbidden=FORBIDDEN_NAMES),
        GetROS2TopicsNamesAndTypesTool(connector=connector, forbidden=FORBIDDEN_NAMES),
        ...
    ]
```

- `/cmd_vel`: the robot should only move through Nav2.
- `/walkie/reset`: the agent must not wipe its own memory.

We use `forbidden` and not `writable` here. When only `writable` is set,
the topic list tool shows **only** the writable topics, so the agent could
no longer see the topics it is allowed to read.

## Step 2: Check it

```bash
pixi run test
```

`test_tools_hide_forbidden_topics` publishes on `/cmd_vel` and checks that
the agent's topic list does not show it.

## Adding a new tool

- If it subclasses `BaseROS2Tool`, pass `forbidden=FORBIDDEN_NAMES` in
  `build_walkie_tools`.
- If it writes to a topic, check `self.is_writable(topic)` before
  publishing, like `SpeakTool` does.
- If you add a forbidden name, say why in the comment above the list.
