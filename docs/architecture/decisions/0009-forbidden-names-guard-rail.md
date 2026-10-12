# ADR-0009: Forbidden ROS names are a guard rail, not a security boundary

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-12

**Context**

RAI has generic tools, such as `PublishROS2MessageTool`, that publish any
message to any topic. With one of them, the LLM could send velocity commands
to `/cmd_vel` and drive the base without Nav2's obstacle avoidance. Every
`BaseROS2Tool` accepts `forbidden`, `writable` and `readable` lists, and
checks them itself, in Python.

**Decision**

We keep one `FORBIDDEN_NAMES` list in `src/walkie_agent/tools/__init__.py`
and pass it to every ROS 2 tool. Each entry has a comment saying why it is
there. We use `forbidden` rather than `writable`, because setting `writable`
also hides every readable topic from the topic-listing tool.

**Consequences**

- Tools that list topics hide forbidden names, so the LLM never sees them.
- It only protects against mistakes. A tool that is not given the list, or a
  plain `@tool` function, is not limited at all. Real access control in ROS 2
  is SROS2.
- Matching is by exact string. Every derived name must be listed separately,
  including an action's `/_action/feedback` and `/_action/status` topics.
- Connecting to the real robot exposes many more motion inputs than
  `/cmd_vel`. They must all be added; the
  [manipulation design](/architecture/manipulation.md#forbidden-names) lists
  them.

**See also:** [Limit what tools can touch](/setup/05-tool-access.md)
<!-- --8<-- [end:body] -->
