"""The agent's command line (main.py), parsed without starting ROS."""

from datetime import datetime
from pathlib import Path

from walkie_agent.main import build_parser, tool_log_path

NOW = datetime(2026, 10, 24, 9, 5, 7)


def test_no_tool_log_without_the_flag():
    args = build_parser().parse_args([])

    assert tool_log_path(args.tool_log, NOW) is None


def test_the_bare_tool_log_flag_names_the_log_by_time():
    args = build_parser().parse_args(["--tool-log", "--arms"])

    assert args.arms
    assert tool_log_path(args.tool_log, NOW) == Path("logs/agent_tools_20261024-090507.jsonl")


def test_the_tool_log_flag_takes_a_path():
    args = build_parser().parse_args(["--tool-log", "/tmp/run1/tools.jsonl"])

    assert tool_log_path(args.tool_log, NOW) == Path("/tmp/run1/tools.jsonl")
