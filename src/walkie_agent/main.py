"""Entry point: runs the Walkie agent as a ROS 2 node.

The agent reads tasks from /from_human (rai_interfaces/msg/HRIMessage). What
it says out loud goes to /to_human through the speak tool, where RAI's
text-to-speech agent picks it up; its plain-text replies go to the operator
console topic, /walkie/console. Any message on /walkie/reset
(std_msgs/msg/Empty) makes it forget the conversation so far.

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", default="config/locations.toml")
    parser.add_argument("--input-topic", default="/from_human")
    parser.add_argument("--output-topic", default="/walkie/console")
    parser.add_argument("--reset-topic", default="/walkie/reset")
    parser.add_argument("--whoami-dir", type=Path, default=Path("whoami"))
    args = parser.parse_args()

    # Variables already set in the shell take precedence over .env.
    load_dotenv()

    # Built before ROS starts so a config or credential error fails fast.
    llm = get_llm_model("complex_model", streaming=True)

    with ROS2Context():
        connector = ROS2Connector(node_name="walkie_agent")
        hri_connector = ROS2HRIConnector(node_name="walkie_agent_hri")
        agent = create_walkie_agent(
            llm=llm,
            connector=connector,
            locations=load_locations(args.locations),
            target_connectors={args.output_topic: hri_connector},
            whoami_dir=args.whoami_dir,
        )
        agent.subscribe_source(args.input_topic, hri_connector)
        agent.subscribe_reset(connector, args.reset_topic)

        runner = AgentRunner([agent])
        try:
            runner.run_and_wait_for_shutdown()
        finally:
            connector.shutdown()
            hri_connector.shutdown()


if __name__ == "__main__":
    main()
