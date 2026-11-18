"""Offline benchmarks over FakeRobot, each with its own pixi environment.

The robot's process never imports this package, which the gym-stays-out contract in
pyproject.toml enforces. mahoraga is the script-writing gym for GPSR commands.
"""
