# Setup Log

This section records every step used to build this repository, in order.
Each page explains **what** we ran and **why**, so you can repeat the same
steps in your own project or when you add a new ROS package.

| Step                                                 | What it does                                                    |
| ---------------------------------------------------- | --------------------------------------------------------------- |
| [1. Create the Pixi workspace](01-pixi-workspace.md) | Starts the project and connects it to ROS 2 Jazzy packages      |
| [2. Add the documentation site](02-documentation.md) | Adds this website, built with MkDocs Material                   |
| [3. Install RAI](03-ros-and-rai.md)                  | Installs ROS 2 and RAI together and builds RAI's messages       |
| [4. Trace with local Langfuse](04-langfuse.md)       | Runs a Langfuse server in Docker to see what the agent does     |
