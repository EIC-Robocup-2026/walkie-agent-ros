"""Typed atomic tools, one module per tool (ADR-0037).

Each module defines META (a walkie_agent.toolbox.ToolMeta whose name is the
module's name), Args (a pydantic model), Out (a ToolOutput subclass) and
run(ctx, args), which returns an Out or raises StepError. The Toolbox finds
every module here at startup and refuses to start if one is broken. Modules
whose names start with "_" are helpers and are not loaded as tools.

An atomic reaches ROS, its backends and the handle store only through ctx,
a walkie_agent.toolbox.ToolContext. The first atomics are HRI's people and
seat tools (docs/architecture/hri.md); _perception holds what they share.
"""
