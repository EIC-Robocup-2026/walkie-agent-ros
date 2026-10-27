"""Entry point: runs the Walkie agent as a ROS 2 node.

The agent reads tasks from /from_human (rai_interfaces/msg/HRIMessage). What
it says out loud goes to /to_human through the speak tool, where RAI's
text-to-speech agent picks it up; its plain-text replies, and one line per
todo status change, go to the operator console topic, /walkie/console. Any
message on /walkie/reset (std_msgs/msg/Empty) makes it forget the
conversation so far.

The operator can also send a task with its todo list seeded on /walkie/task
(std_msgs/msg/String holding JSON). The current todo list is published as
JSON on /walkie/todos (std_msgs/msg/String, latched). Task templates are read
from config/task_templates.toml when it exists. With --tool-log, every tool
call also goes to a JSON lines record, which the HRI pass check reads.

The LLM is selected by config.toml (see RAI's get_llm_model), which is read
from the current working directory. The same file's [walkie.perception] and
[walkie.gaze] tables set the perception and gaze values to tune on the robot,
and [walkie.models.agent] the bounds on each LLM call (see tuning.py and
llm_deadline.py). API keys are read from the environment, or from a .env file
in the working directory (see .env.example).
"""

import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from rai import get_llm_model, get_llm_model_config_and_vendor
from rai.agents import AgentRunner
from rai.communication.ros2 import ROS2Connector, ROS2Context, ROS2HRIConnector

from walkie_agent.agent import create_walkie_agent
from walkie_agent.llm_deadline import AgentModelConfig, chat_openai_kwargs
from walkie_agent.locations import load_locations
from walkie_agent.tasks import load_task_templates
from walkie_agent.tuning import load_tuning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", default="config/locations.toml")
    parser.add_argument(
        "--task-templates",
        default="config/task_templates.toml",
        help="Task templates; a missing file means none.",
    )
    parser.add_argument("--input-topic", default="/from_human")
    parser.add_argument("--output-topic", default="/walkie/console")
    parser.add_argument("--reset-topic", default="/walkie/reset")
    parser.add_argument("--task-topic", default="/walkie/task")
    parser.add_argument("--todos-topic", default="/walkie/todos")
    parser.add_argument("--whoami-dir", type=Path, default=Path("whoami"))
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="Use the simulator's /clock, so goal timestamps match the simulated robot.",
    )
    parser.add_argument(
        "--mock-people-memory",
        action="store_true",
        help=(
            "Keep people in an in-memory mock, for sim and tests. Without it, the people "
            "tools use walkie-database's services."
        ),
    )
    parser.add_argument(
        "--resume-people-session",
        action="store_true",
        help=(
            "Keep walkie-database's open session for the first task after a restart, so "
            "the guests met before it are still known. By default the agent forgets them "
            "at start-up and at each new task (walkie-database ADR-0010)."
        ),
    )
    parser.add_argument(
        "--arms",
        action="store_true",
        help=(
            "Drive the arms through walkie-ros's bimanual_commander: loads the arm tools, "
            "and the navigation tools then refuse to drive with an arm out. With "
            "--use-sim-time, poses not yet tried on the robot are offered too."
        ),
    )
    parser.add_argument(
        "--tool-log",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Record every tool call as JSON lines, for the HRI pass check. Without a path, "
            "logs/agent_tools_<time>.jsonl."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with the flags that cannot go together refused."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume_people_session and args.mock_people_memory:
        parser.error(
            "--resume-people-session needs walkie-database; it does nothing with "
            "--mock-people-memory"
        )
    return args


def tool_log_path(value: str | None, now: datetime | None = None) -> Path | None:
    """The --tool-log path: None without the flag, the default name for the bare flag."""
    if value is None:
        return None
    if value:
        return Path(value)
    # The scenario's event log is named the same way, so a run's two logs sort together.
    return Path("logs") / f"agent_tools_{now or datetime.now():%Y%m%d-%H%M%S}.jsonl"


def build_llm(config: AgentModelConfig) -> BaseChatModel:
    """The agent's LLM, RAI's complex_model, streaming, with each call bounded by `config`.

    Only RAI's openai vendor, whose ChatOpenAI hands the bounds to the
    OpenAI SDK, gets them. ChatOllama has neither a timeout nor a
    max_retries field and drops both (ADR-0042), and the other vendors take
    no OpenAI SDK client, so their calls keep their own timeouts and
    retries.
    """
    _, vendor = get_llm_model_config_and_vendor("complex_model")
    bounds = chat_openai_kwargs(config) if vendor == "openai" else {}
    return get_llm_model("complex_model", vendor=vendor, streaming=True, **bounds)


def main() -> None:
    args = parse_args()

    # Variables already set in the shell take precedence over .env.
    load_dotenv()

    # Built before ROS starts so a config or credential error fails fast.
    # The tuning comes first: it holds the bounds on the LLM's calls.
    tuning = load_tuning()
    llm = build_llm(tuning.agent_model)
    templates = load_task_templates(args.task_templates)
    locations = load_locations(args.locations)

    with ROS2Context():
        connector = ROS2Connector(node_name="walkie_agent", use_sim_time=args.use_sim_time)
        hri_connector = ROS2HRIConnector(node_name="walkie_agent_hri")
        agent = create_walkie_agent(
            llm=llm,
            connector=connector,
            locations=locations,
            target_connectors={args.output_topic: hri_connector},
            whoami_dir=args.whoami_dir,
            templates=templates,
            people_memory="mock" if args.mock_people_memory else "database",
            resume_people_session=args.resume_people_session,
            arms=args.arms,
            tool_log_path=tool_log_path(args.tool_log),
            tuning=tuning,
        )
        agent.subscribe_source(args.input_topic, hri_connector)
        agent.subscribe_reset(connector, args.reset_topic)
        agent.subscribe_task(connector, args.task_topic)
        agent.publish_todos(connector, args.todos_topic)

        runner = AgentRunner([agent])
        try:
            runner.run_and_wait_for_shutdown()
        finally:
            connector.shutdown()
            hri_connector.shutdown()


if __name__ == "__main__":
    main()
