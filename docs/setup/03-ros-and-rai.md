# 3. Install RAI

**Goal:** install [RAI](https://github.com/RobotecAI/rai) and ROS 2 Jazzy in
the Pixi workspace, so the Walkie agent can use RAI's ROS 2 connector and speech
agents.

## Background

- **RAI** is a framework from Robotec.AI for building LLM agents that control
  ROS 2 robots. We keep two parts of it: `ROS2Connector`, which owns the agent's
  node, executor and TF buffer, and `rai_s2s`, which runs the voice nodes. Its
  messages, such as `HRIMessage`, come along.
- **The agent does not run on RAI's agent classes.** It runs on LangChain's
  `create_agent`, with its own runtime and its own model loader
  ([ADR-0061](../architecture/decisions/0061-create-agent-runtime-and-cancellation.md),
  [ADR-0062](../architecture/decisions/0062-bounded-model-calls.md)). Only the
  voice entry points use RAI's `AgentRunner`.
- ROS comes from **RoboStack** (conda packages). RAI is published on
  **PyPI** as `rai-core`. Pixi installs both into one environment: conda
  packages go under `[dependencies]`, PyPI packages under
  `[pypi-dependencies]`.

## Step 1: Install ROS 2 Jazzy

Add to `pixi.toml`:

```toml title="pixi.toml"
[dependencies]
# RoboStack Jazzy is built for Python 3.12.
python = "3.12.*"
ros-jazzy-ros-base = "*"
```

`ros-jazzy-ros-base` is the core of ROS 2: `rclpy`, the `ros2` command and
common messages. It has no GUI tools (see [Using rqt](#using-ros-gui-tools-rqt)).

Pixi activates ROS for you inside `pixi run` and `pixi shell`, so there is no
`source /opt/ros/jazzy/setup.bash` step.

## Step 2: Install `rai-core`

`rai-core` is on PyPI, so it goes under `[pypi-dependencies]`. We also limit
two conda packages to versions RAI accepts:

```toml title="pixi.toml"
[dependencies]
numpy = "<2"        # required by rai-core
packaging = "<26"   # required by langfuse 3.x, which rai-core uses

[pypi-dependencies]
rai-core = "==2.12.1"
```

```bash
pixi install
```

The agent's runtime libraries, such as `langchain` and `langgraph`, are direct
dependencies with exact pins in `pixi.toml`. The agent's behaviour depends on
theirs, so it does not take them through `rai-core`.

Pixi chooses the conda packages (ROS) first, then fits the PyPI packages
(RAI) around them. Without the limits, conda picks versions that `rai-core`
refuses, and `pixi install` fails.

??? info "How we found these limits"

    We added `rai-core` without any limits first. `pixi install` failed with
    this error:

    ```text
    ╰─▶ Because rai-core==2.12.1 depends on numpy<2.0 and numpy==2.5.4, we can
        conclude that rai-core==2.12.1 cannot be used.
    help: The following PyPI packages have been pinned by the conda solve, and
          this version may be causing a conflict:
          numpy==2.5.4
    ```

    How to read it:

    - `rai-core depends on numpy<2.0`: RAI needs numpy below 2.
    - `pinned by the conda solve ... numpy==2.5.4`: conda already chose numpy
      2.5.4 for ROS, and Pixi cannot change a conda choice to suit a PyPI
      package.

    The fix is to tell conda what RAI needs: add `numpy = "<2"` under
    `[dependencies]`. We ran `pixi install` again and got the same kind of
    error for `packaging`:

    ```text
    ╰─▶ Because langfuse>=3.8.0,<=3.15.0 depends on packaging>=23.2,<26.0 ...
    ```

    `langfuse` is not ours. `rai-core` requires it, so its limits apply to us
    too. Adding `packaging = "<26"` fixed it. Repeat until `pixi install`
    succeeds.

    To see every package `rai-core` requires, with its version limits:

    ```bash
    pixi run python -c "from importlib.metadata import requires; print(*requires('rai-core'), sep='\n')"
    ```

    The same list is on the [rai-core PyPI page](https://pypi.org/project/rai-core/)
    under **Requires**.

## Step 3: Add the ROS packages RAI imports

RAI's code imports some ROS packages that `ros-jazzy-ros-base` does not
include. Add them, plus one more version limit:

```toml title="pixi.toml"
[dependencies]
ros-jazzy-cv-bridge = "*"
ros-jazzy-nav2-msgs = "*"
ros-jazzy-tf2-geometry-msgs = "*"
ros-jazzy-tf-transformations = "*"
py-opencv = "<5"    # cv_bridge brings OpenCV from conda; rai-core needs OpenCV <5
```

```bash
pixi install
```

??? info "How we found these packages"

    `rai-core` comes from PyPI, and PyPI cannot install ROS packages. So RAI's
    package does not list the ROS packages it needs. We found them by
    searching RAI's source code for `import` lines. After Step 2, RAI's code
    is in `.pixi/envs/default/lib/python3.12/site-packages/rai/`:

    ```bash
    grep -rhoE "^\s*(from|import) [a-z0-9_]+" .pixi/envs/default/lib/python3.12/site-packages/rai \
      | awk '{print $2}' | sort -u
    ```

    This prints every module RAI imports. Most are standard Python (`os`,
    `json`) or PyPI packages that come with `rai-core` (`langchain`,
    `numpy`). The ROS ones are easy to spot: `rclpy`, names ending in
    `_msgs`, `cv_bridge`, `tf2_*`. `ros-jazzy-ros-base` already has some of
    them. To list the ones that are still missing:

    ```bash
    pixi run python -c "import importlib.util as u; print([m for m in 'cv_bridge nav2_msgs tf2_ros tf2_geometry_msgs tf_transformations'.split() if u.find_spec(m) is None])"
    ```

    It prints the missing modules, or `[]` when everything is installed. A
    ROS Python module has the same name as its ROS package, and RoboStack
    adds `ros-jazzy-` in front, with `-` instead of `_`. So `cv_bridge` is
    `ros-jazzy-cv-bridge`. Check that it exists before adding it:

    ```bash
    pixi search ros-jazzy-cv-bridge
    ```

    If you miss a package, Python stops with
    `ModuleNotFoundError: No module named '...'`. The module name in that
    error tells you which `ros-jazzy-*` package to add.

??? info "How we found the `py-opencv` limit"

    After adding the four packages, `pixi install` failed again:

    ```text
    ╰─▶ Because rai-core==2.12.1 depends on opencv-python-headless>=4.9.0.80,<5.0.0.0
        and opencv-python-headless==5.0.0, we can conclude that rai-core==2.12.1
        cannot be used.
    ```

    This is the same kind of error as in Step 2. One of the four new ROS
    packages brings OpenCV 5 from conda, but RAI needs OpenCV below 5.
    OpenCV's conda package is called `py-opencv`, so we added
    `py-opencv = "<5"` and `pixi install` succeeded.

    Once the install works, `pixi tree --invert` shows which package brings
    a dependency:

    ```bash
    pixi tree --invert py-opencv
    ```

    ```text
    py-opencv 4.13.0
    └── ros2-cv-bridge 4.1.0
        └── ros-jazzy-cv-bridge 4.1.0
    ```

    So it is `cv_bridge`.

!!! warning
    If `pixi install` fails after upgrading `rai-core` or ROS, check the
    three limits from Steps 2 and 3 first: `numpy`, `packaging` and
    `py-opencv`.

## Step 4: Create `config.toml`

`config.toml` holds the settings, in the folder you start the agent in. It keeps the
shape of RAI's own config, which started as the output of this command:

```bash
pixi run rai-config-init
```

It writes settings for every LLM vendor RAI supports. It refuses to
overwrite an existing file unless you pass `--force`.

We use [OpenRouter](https://openrouter.ai) for the LLM. OpenRouter has the
same API as OpenAI, so we use the `openai` vendor and change its model and URL:

```toml title="config.toml"
[vendor]
simple_model = "openai"
complex_model = "openai"
embeddings_model = "openai"

[openai]
simple_model = "qwen/qwen3.8-27b:nitro"
complex_model = "deepseek/deepseek-v4.1-flash:nitro"
embeddings_model = "openai/text-embedding-3-small"
base_url = "https://openrouter.ai/api/v1"
```

To try another model, change `complex_model` to any
[OpenRouter model ID](https://openrouter.ai/models).

`rai-config-init` does not write the `[team]` table, so add it. The agent's
`get_facts` tool reads the team from it, and the time from the computer's clock in
its timezone, an IANA name. The agent does not start without it:

```toml title="config.toml"
[team]
name = "EIC"
country = "Thailand"
affiliation = "Chulalongkorn University"
timezone = "Asia/Bangkok"
```

The same file has more than one reader:

| Section | Who reads it |
| ------- | ------------ |
| `[vendor]`, `[openai]`, `[ollama]` | `walkie_agent.runtime.model` reads `complex_model` and `base_url` of the vendor that `[vendor]` names |
| `[agent]`, optional | `run_limit`, `deadline_s`, `fallback_vendor` and `fallback_model` for the agent. The defaults are provisional, and no fallback is selected |
| `[team]` | `walkie_agent.capabilities.facts` reads `name`, `country`, `affiliation` and `timezone` for `get_facts`. Required |
| `[tracing.langfuse]` | The agent's Langfuse callback, when `use_langfuse` is true |
| `[asr]`, `[tts]` | The voice entry points, through `rai_s2s` |

Nothing in this repository reads `simple_model`, `embeddings_model`,
`[tracing]`'s project or `[tracing.langsmith]`. They stay so the file keeps the shape of
RAI's config. The embeddings fields were for the retired documentation search
([ADR-0068](../architecture/decisions/0068-static-robot-description.md)).

We deleted `[aws]` and `[google]`, because we do not use those vendors.

!!! note "Why the agent has its own loader"
    `get_llm_model` would hide provider options we need. `ChatOllama` silently ignores
    `timeout` and `max_retries`, and honours only `client_kwargs`. The agent's loader
    sets the timeout for each provider and turns SDK retries off. These settings and
    numbers have not met a real provider yet
    ([ADR-0062](../architecture/decisions/0062-bounded-model-calls.md)).

### API keys go in `.env`

The provider SDKs read API keys from environment variables, never from `config.toml`.
We keep them in a `.env` file, which Git ignores. Add the loader:

```toml title="pixi.toml"
[pypi-dependencies]
python-dotenv = ">=1.0"
```

`src/walkie_agent/main.py` calls `load_dotenv()` before building the agent.
Variables already set in your shell win over `.env`. To set up your own
`.env`, see [Getting Started](../getting-started.md#4-add-your-api-keys).

## Step 5: Build `rai_interfaces`

RAI's ROS message types (such as `HRIMessage`) are in a separate package,
`rai_interfaces`. RoboStack does not have it, so we build it from source with
colcon, in a small workspace called `ros2_ws/`. First, the build tools:

```toml title="pixi.toml"
[dependencies]
colcon-common-extensions = "*"
vcstool = "*"
c-compiler = "*"
cxx-compiler = "*"
cmake = "*"
pkg-config = "*"
make = "*"
```

The repository and its exact commit are listed in `ros2_ws.repos`. The file
now also lists walkie-database, walkie-vision and walkie-ros, which were added
later (see the note below):

```yaml title="ros2_ws.repos"
repositories:
  ros2_ws/src/rai_interfaces:
    type: git
    url: https://github.com/RobotecAI/rai_interfaces.git
    version: f03dc0f217679980447dfead177979f485eb44f6
  ros2_ws/src/walkie-database:
    type: git
    url: https://github.com/EIC-Robocup-2026/walkie-database.git
    version: ...  # the commit of the database node the robot runs
  ros2_ws/src/walkie-vision:
    type: git
    url: https://github.com/EIC-Robocup-2026/walkie-vision.git
    version: ...  # the commit of the vision node the robot runs
  ros2_ws/src/walkie-ros:
    type: git
    url: https://github.com/EIC-Robocup-2026/walkie-ros.git
    version: ...  # the commit of the arm commander the robot runs
```

Two tasks clone and build them:

```toml title="pixi.toml"
[tasks.clone-ros2-ws]
cmd = "vcs import --input ros2_ws.repos"
inputs = ["ros2_ws.repos"]
outputs = [
  "ros2_ws/src/rai_interfaces/package.xml",
  "ros2_ws/src/walkie-database/ros2_ws/src/walkie_database_msgs/package.xml",
  "ros2_ws/src/walkie-vision/ros2_ws/src/walkie_vision_msgs/package.xml",
  "ros2_ws/src/walkie-ros/manipulate/my_robot_interfaces/package.xml",
]

[tasks.build-ros2]
cmd = """colcon build --symlink-install \
  --packages-select rai_interfaces walkie_database_msgs walkie_vision_msgs my_robot_interfaces \
  --cmake-args -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python"""
cwd = "ros2_ws"
depends-on = ["clone-ros2-ws"]
```

- `vcs import` (from `vcstool`) clones every repository listed in the file.
- `inputs` and `outputs` let Pixi skip the clone when the folder already
  exists and `ros2_ws.repos` has not changed. A changed pin runs
  `vcs import` again, which moves the existing clone to the new commit. With
  `outputs` alone, Pixi would skip the task after a pin change, and
  `build-ros2` would keep building the old commit.
- `-DPython3_EXECUTABLE=...` forces CMake to use Pixi's Python. Without it,
  a `~/.colcon/defaults.yaml` made for a system ROS install can point the
  build at the wrong Python.

!!! note "Later: walkie-database, walkie-vision and walkie-ros"
    [walkie-database](https://github.com/EIC-Robocup-2026/walkie-database)
    was added for the messages of Walkie's memory, `walkie_database_msgs`
    ([ADR-0035](../architecture/decisions/0035-memory-through-walkie-database.md)),
    and [walkie-vision](https://github.com/EIC-Robocup-2026/walkie-vision)
    for the messages of its vision node, `walkie_vision_msgs`
    ([ADR-0034](../architecture/decisions/0034-vision-models-in-walkie-vision.md)).
    On `feat/mahoraga-demo`,
    [walkie-ros](https://github.com/EIC-Robocup-2026/walkie-ros) was added
    for the actions and services of its arm commander, `my_robot_interfaces`
    ([ADR-0018](../architecture/decisions/0018-pin-robot-interfaces.md)).
    Each clone holds a whole repository, so `build-ros2` builds only the
    packages named after `--packages-select`. That list is an allow-list: a
    message package from `ros2_ws.repos` that is missing from it is not
    built.

```bash
pixi run build-ros2
```

The cloned source and the build output are in `.gitignore`.

### Local message pins

On the `feat/mahoraga-demo` prototype branch, the walkie-vision and
walkie-database pins name commits that exist only on the local
`feat/scene-graph-prep` branches of those repositories (walkie-vision
`64811fe`, walkie-database `522ab92`). `vcs import` fetches from GitHub,
which does not have them. Put the commits into the clones first, then build
as usual:

```bash
# Skip a clone that already exists.
git clone --no-checkout https://github.com/EIC-Robocup-2026/walkie-vision.git ros2_ws/src/walkie-vision
git clone --no-checkout https://github.com/EIC-Robocup-2026/walkie-database.git ros2_ws/src/walkie-database

# Copy the local branches into the clones.
git -C ros2_ws/src/walkie-vision fetch /var/tmp/wv-prep feat/scene-graph-prep:feat/scene-graph-prep
git -C ros2_ws/src/walkie-database fetch /var/tmp/wd-prep feat/scene-graph-prep:feat/scene-graph-prep

pixi run --locked build-ros2
```

- `/var/tmp/wv-prep` and `/var/tmp/wd-prep` are the checkouts holding the
  two `feat/scene-graph-prep` branches. Use the paths of your own clones.
- The walkie-ros pin, `6b26daa`, is on GitHub, so `vcs import` clones it with
  no extra step. The clone is about 250 MB, without walkie-ros's submodules,
  which `my_robot_interfaces` does not need.
- `vcs import`, run by `clone-ros2-ws`, fetches each existing clone from
  GitHub and then checks out the pinned commit. The commit is already in the
  clone, so the checkout succeeds although GitHub does not have it.
- Fetching into a named branch, rather than only into `FETCH_HEAD`, keeps the
  commits referenced, so a later `git gc` does not remove them.
- `--locked` stops Pixi from rewriting `pixi.lock`.
- Do not add `--clean-env`: it drops the system `PATH`, and `vcs import`
  then cannot find `git`, which the environment does not provide.

Check that each clone is at its pin:

```bash
git -C ros2_ws/src/walkie-vision rev-parse HEAD    # 64811feb79ca...
git -C ros2_ws/src/walkie-database rev-parse HEAD  # 522ab92f4b1a...
```

Once the two branches are pushed, `pixi run build-ros2` works without the
fetch step, and this section can go.

A colcon workspace must be "sourced" before Python can import it. Pixi runs
`scripts/activate.sh` every time it activates the environment, and that
script does it:

```toml title="pixi.toml"
[activation]
scripts = ["scripts/activate.sh"]
```

```bash title="scripts/activate.sh"
if [[ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh" ]]; then
    source "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh"
fi
```

## Step 6: Add our own package

Our code lives in `src/walkie_agent/` and is described by `pyproject.toml`.
We install it into the environment in editable mode, so code changes take
effect without reinstalling:

```toml title="pixi.toml"
[pypi-dependencies]
walkie-agent = { path = ".", editable = true }

[dependencies]
pytest = "*"

[tasks.fake-walkie]
cmd = "python -m walkie_agent.sim.fake_walkie"

[tasks.agent]
cmd = "python -m walkie_agent.main"

[tasks.test]
cmd = "pytest"
```

| Command                | What it does                                                 |
| ---------------------- | ------------------------------------------------------------ |
| `pixi run fake-walkie` | A fake robot: Nav2 action, TF and speech topic, no hardware. |
| `pixi run agent`       | The agent, from `walkie_agent.main`. Reads tasks on `/from_human` and `/walkie/task`. |
| `pixi run test`        | All tests, with a scripted LLM, so no API key is needed.     |

`pyproject.toml` turns off the pytest plugins that ROS 2 installs
(`launch_testing`, `ament_*`), because our tests do not use them.

!!! warning "Editable installs and a second worktree"
    The editable install points at one checkout. A second Git worktree that borrows
    another checkout's environment imports that other checkout's code, and its tests
    pass or fail on the wrong files. Set `PYTHONPATH` to the worktree's `src/` there,
    and print `walkie_agent.__file__` once to check.

## Adding a new dependency

Use this order whenever you need a new package. Try each place until the
search finds it:

| What you need                               | Where to look      | Command                                                          |
| ------------------------------------------- | ------------------ | ---------------------------------------------------------------- |
| A ROS package (`*_msgs`, `nav2_*`, `rqt_*`) | RoboStack          | `pixi search ros-jazzy-<name>`, then `pixi add ros-jazzy-<name>` |
| A Python library                            | conda-forge        | `pixi search <name>`, then `pixi add <name>`                     |
| A Python library that is not on conda-forge | PyPI               | `pixi add --pypi <name>`                                         |
| A ROS package that is not packaged anywhere | Its Git repository | Add it to `ros2_ws.repos` and to `--packages-select` in `build-ros2`, then `pixi run build-ros2` |

Why conda-forge before PyPI: ROS packages also use common libraries like
`numpy` and OpenCV, and they come from conda. A PyPI package that needs a
different version causes the errors described in
[Step 2](#step-2-install-rai-core).

After adding a Python package from PyPI, check whether it needs ROS modules
that are not installed yet, as in
[Step 3](#step-3-add-the-ros-packages-rai-imports). Then run
`pixi run test`.

`errors`, `capabilities`, `atomics`, `adapter` and `runtime` must not import rclpy, RAI
or any ROS message, and `tests/test_ros_free.py` checks that. A package that needs ROS
belongs behind `walkie_agent/ros/`.

Changing a pin of `langchain`, `langgraph` or `rai-core` can change behaviour without any
error. Serial tool calls rest on `max_concurrency=1`, which is observed and not
documented. So run the whole suite after any such change. To check a `rai-core` upgrade,
see [Read the Code Next](../learn/read-the-code.md#checking-a-rai-core-upgrade).

## Using ROS GUI tools (rqt)

`ros-jazzy-ros-base` has no GUI tools. To use one, add its package, for
example `rqt_graph`:

```bash
pixi add ros-jazzy-rqt-graph
pixi run rqt_graph
```

!!! danger "Do not mix with a system ROS install"
    If your `~/.bashrc` or `~/.zshrc` sources `/opt/ros/jazzy/setup.*`, the
    system ROS tools stay on your `PATH` inside `pixi shell`. Running one of
    them there fails with errors like
    `PackageNotFoundError: No package metadata was found for rqt-graph`,
    because Pixi has replaced the Python paths. Install the tool with Pixi as
    shown above, or remove the `source /opt/ros/...` line from your shell
    config.
