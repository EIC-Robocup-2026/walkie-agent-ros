# 3. Install RAI

**Goal:** install [RAI](https://github.com/RobotecAI/rai) and ROS 2 Jazzy in
the Pixi workspace, so we can build the Walkie agent on top of RAI.

## Background

- **RAI** is a framework from Robotec.AI for building LLM agents that control
  ROS 2 robots. It gives us ready-made agents (`ReActAgent`), ROS 2
  connectors, and tools, so we write less code ourselves.
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
rai-core = ">=2.12.1, <2.13"
```

```bash
pixi install
```

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

RAI reads its settings from `config.toml` in the folder you start the agent
in. Create a default one:

```bash
pixi run rai-config-init
```

It writes settings for every LLM vendor RAI supports. It refuses to
overwrite an existing file unless you pass `--force`.

We use [OpenRouter](https://openrouter.ai) for the LLM. OpenRouter has the
same API as OpenAI, so we use RAI's `openai` vendor and change its model and
URL:

```toml title="config.toml"
[vendor]
simple_model = "openai"
complex_model = "openai"
embeddings_model = "openai"

[openai]
simple_model = "google/gemini-3-flash-preview"
complex_model = "google/gemini-3-flash-preview"
embeddings_model = "openai/text-embedding-3-small"
base_url = "https://openrouter.ai/api/v1"
```

To try another model, change `complex_model` to any
[OpenRouter model ID](https://openrouter.ai/models).

We kept these sections from the default file, unchanged, for later:

| Section    | What it is for                                                                                 |
| ---------- | ---------------------------------------------------------------------------------------------- |
| `[ollama]` | Local LLMs served by [Ollama](https://ollama.com). Used when a `[vendor]` entry is `"ollama"`. |
| `[asr]`    | Speech-to-text: microphone, Whisper model, voice detection.                                    |
| `[tts]`    | Text-to-speech: voice and speaker.                                                             |

We deleted `[aws]` and `[google]`, because we do not use those vendors. RAI
uses defaults for missing vendor sections.

### API keys go in `.env`

RAI reads API keys from environment variables, never from `config.toml`.
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

The repository and its exact commit are listed in `ros2_ws.repos`:

```yaml title="ros2_ws.repos"
repositories:
  ros2_ws/src/rai_interfaces:
    type: git
    url: https://github.com/RobotecAI/rai_interfaces.git
    version: f03dc0f217679980447dfead177979f485eb44f6
```

Two tasks clone and build it:

```toml title="pixi.toml"
[tasks.clone-ros2-ws]
cmd = "vcs import --input ros2_ws.repos"
outputs = ["ros2_ws/src/rai_interfaces/package.xml"]

[tasks.build-ros2]
cmd = "colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python"
cwd = "ros2_ws"
depends-on = ["clone-ros2-ws"]
```

- `vcs import` (from `vcstool`) clones every repository listed in the file.
- `outputs` lets Pixi skip the clone when the folder already exists.
- `-DPython3_EXECUTABLE=...` forces CMake to use Pixi's Python. Without it,
  a `~/.colcon/defaults.yaml` made for a system ROS install can point the
  build at the wrong Python.

```bash
pixi run build-ros2
```

The cloned source and the build output are in `.gitignore`.

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
| `pixi run agent`       | The agent. Reads tasks on `/from_human`.                     |
| `pixi run test`        | All tests, with a scripted LLM, so no API key is needed.     |

`pyproject.toml` turns off the pytest plugins that ROS 2 installs
(`launch_testing`, `ament_*`), because our tests do not use them.

## Adding a new dependency

Use this order whenever you need a new package. Try each place until the
search finds it:

| What you need                               | Where to look      | Command                                                          |
| ------------------------------------------- | ------------------ | ---------------------------------------------------------------- |
| A ROS package (`*_msgs`, `nav2_*`, `rqt_*`) | RoboStack          | `pixi search ros-jazzy-<name>`, then `pixi add ros-jazzy-<name>` |
| A Python library                            | conda-forge        | `pixi search <name>`, then `pixi add <name>`                     |
| A Python library that is not on conda-forge | PyPI               | `pixi add --pypi <name>`                                         |
| A ROS package that is not packaged anywhere | Its Git repository | Add it to `ros2_ws.repos`, then `pixi run build-ros2`            |

Why conda-forge before PyPI: ROS packages also use common libraries like
`numpy` and OpenCV, and they come from conda. A PyPI package that needs a
different version causes the errors described in
[Step 2](#step-2-install-rai-core).

After adding a Python package from PyPI, check whether it needs ROS modules
that are not installed yet, as in
[Step 3](#step-3-add-the-ros-packages-rai-imports). Then run
`pixi run test`.

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
