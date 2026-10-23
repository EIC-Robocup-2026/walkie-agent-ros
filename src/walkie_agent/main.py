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
from config/task_templates.toml when it exists.

The LLM is selected by config.toml (see RAI's get_llm_model), which is read
from the current working directory. API keys are read from the environment, or
from a .env file in the working directory (see .env.example).
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rai import get_llm_model
from rai.agents import AgentRunner
from rai.communication.ros2 import ROS2Connector, ROS2Context, ROS2HRIConnector

from walkie_agent.agent import create_walkie_agent
from walkie_agent.locations import load_locations
from walkie_agent.tasks import load_task_templates


def main() -> None:
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
            "Remember people in memory, cleared at every new task, so recognize_person "
            "and remember_person load. For sim and tests; without it they do not load."
        ),
    )
    args = parser.parse_args()

    # Variables already set in the shell take precedence over .env.
    load_dotenv()

    # Built before ROS starts so a config or credential error fails fast.
    llm = get_llm_model("complex_model", streaming=True)
    templates = load_task_templates(args.task_templates)

    with ROS2Context():
        connector = ROS2Connector(node_name="walkie_agent", use_sim_time=args.use_sim_time)
        hri_connector = ROS2HRIConnector(node_name="walkie_agent_hri")
        agent = create_walkie_agent(
            llm=llm,
            connector=connector,
            locations=load_locations(args.locations),
            target_connectors={args.output_topic: hri_connector},
            whoami_dir=args.whoami_dir,
            templates=templates,
            mock_people_memory=args.mock_people_memory,
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
