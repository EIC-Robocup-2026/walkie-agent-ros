# RAI's Connector, Our Runtime

RAI connects Walkie to ROS, and runs the voice stack. It no longer hosts the agent. The
agent is LangChain's `create_agent` with Walkie's middlewares, on a thread that Walkie's
Runner owns. This split is why most of the agent can be tested with no ROS installed.

## The two layers

| Layer | Whose | What it does | Files |
| ----- | ----- | ------------ | ----- |
| Connector | RAI | Owns the ROS node and its executor, topics and actions | Used by `src/walkie_agent/ros/` and `main.py` |
| Voice | RAI's `rai_s2s` | Microphone, speech recognition, text to speech | Used by `src/walkie_agent/voice.py` and `half_duplex.py` |
| Graph | LangChain | The model and tools loop, and the checkpoint | `create_agent`, called in `runtime/agent.py` |
| Runtime | Walkie | The middlewares, the todo list and the Runner | `src/walkie_agent/runtime/` |
| Tools | Walkie | Plain typed functions over `ctx`, and the fixed tool sets | `src/walkie_agent/tools/atomic/` |
| Verbs | Walkie | `ctx.base`, `ctx.speech` and the rest, with their guards, and `FakeRobot` | `src/walkie_agent/capabilities/` |
| ROS wrappers | Walkie | The verbs over ROS, and the operator's topics | `src/walkie_agent/ros/` |

The first thing to check is the claim itself. This reads each file's imports, without
running it:

```pycon
>>> import ast, pathlib
>>> import walkie_agent
>>> root = pathlib.Path(walkie_agent.__file__).parent
>>> ROS_NAMES = {"rai", "rai_interfaces", "rai_s2s", "rclpy"}
>>> def ros_imports(path):
...     names = set()
...     for node in ast.walk(ast.parse(path.read_text())):
...         if isinstance(node, ast.Import):
...             names |= {alias.name.split(".")[0] for alias in node.names}
...         elif isinstance(node, ast.ImportFrom) and node.module:
...             names.add(node.module.split(".")[0])
...     return sorted(names & ROS_NAMES)
...
>>> for layer in ["capabilities", "atomics", "runtime"]:
...     print(layer, [name for path in sorted((root / layer).glob("*.py")) for name in ros_imports(path)])
...
capabilities []
atomics []
runtime []
>>> ros_imports(root / "adapter.py"), ros_imports(root / "tasks.py")
([], [])

```

And the other side of the line:

```pycon
>>> ros_imports(root / "main.py")
['rai', 'rclpy']
>>> ros_imports(root / "ros" / "base.py")
['rai', 'rclpy']
>>> ros_imports(root / "voice.py")
['rai', 'rai_s2s']

```

`tests/test_ros_free.py` holds the same line with a stronger check. It imports every module
of those layers in a fresh interpreter, and fails if any ROS module got loaded.

## What RAI still does

- **The connector.** `main.py` creates one `ROS2Connector`. It owns the node and spins its
  executor. `RosBase`, `RosSpeech` and `RosBridge` make their clients, publishers and
  subscriptions on that node.
- **The messages.** `rai_interfaces/msg/HRIMessage` is what `/from_human` and `/to_human`
  carry.
- **Voice.** `voice-asr` and `voice-tts` are separate processes. They use RAI's speech
  agents, and `voice.py` runs them with RAI's `AgentRunner`. That is the one place where
  an RAI agent host remains. Those agents do no LLM work.

## Why the runtime left RAI's agent host

Walkie used to subclass RAI's `LangChainAgent`, and copied parts of its private loop to
queue messages and interrupt a run. Every RAI upgrade could break that without a warning.

`create_agent` offers named hooks for the same work. So the rules became small middlewares,
which need no private member ([ADR-0061](../architecture/decisions/0061-create-agent-runtime-and-cancellation.md)).
One assumption is left. A message's tool calls run one at a time because of
`max_concurrency=1`, which is observed behaviour and not documented.
`tests/test_runtime.py` pins it, so rerun that file on any dependency update.

## What the Runner owns

The Runner is one thread with one asyncio loop, named `walkie-runtime`. It takes one queued
task at a time. Only this thread touches the agent and its checkpoint.

| Thread | Who owns it | What runs there |
| ------ | ----------- | ---------------- |
| ROS executor | RAI's connector | Topic callbacks, which only queue work on the Runner and return |
| `walkie-runtime` | The Runner | The graph, the middlewares, `update_todos` and the listeners |
| Worker threads | asyncio | Atomics, which are sync tools under `ainvoke`, and the stop line |

Here is the Runner with a scripted model and a fake robot. The model plays its replies in
order, and the block defines it once, so the page needs no LLM key:

```pycon
>>> import threading
>>> from langchain_core.language_models import BaseChatModel
>>> from langchain_core.messages import AIMessage
>>> from langchain_core.outputs import ChatGeneration, ChatResult
>>> from walkie_agent.capabilities.fake import FakeRobot
>>> from walkie_agent.runtime import Runner, build_agent
>>> class Scripted(BaseChatModel):
...     script: list
...     seen: list = []
...     @property
...     def _llm_type(self):
...         return "scripted"
...     def bind_tools(self, tools, **kwargs):
...         return self
...     def _generate(self, *args, **kwargs):
...         raise NotImplementedError("The runtime only uses the async path.")
...     async def _agenerate(self, messages, *args, **kwargs):
...         self.seen.append(list(messages))
...         reply = self.script.pop(0).model_copy(update={"id": None})
...         return ChatResult(generations=[ChatGeneration(message=reply)])
...
>>> robot = FakeRobot()
>>> threads = {"caller": threading.current_thread().name}
>>> def on_reply(text):
...     threads["on_reply"] = threading.current_thread().name
...
>>> agent = build_agent(Scripted(script=[AIMessage("Hello.")]), robot.ctx)
>>> runner = Runner(agent, robot.ctx, on_reply=on_reply)
>>> runner.start()
>>> runner.submit("Hi.")
>>> runner.wait_idle(5)
True
>>> threads["on_reply"]
'walkie-runtime'
>>> runner.shutdown(5)
True

```

`submit` returned at once, on the caller's thread. That's the point of the design. A ROS
callback must not wait for an LLM, or it blocks everything else on that executor.

The price is that a listener such as `on_reply` runs on the Runner's thread, so it must not
block. [Watching Progress](watching-progress.md) comes back to this.

!!! example "Try it"
    Run `ros_imports(root / "ros" / "policy.py")`. This file holds the check that refuses
    forbidden topic names. Then read `assemble` in `src/walkie_agent/main.py`, and find
    where the verbs on `ctx` are built for the robot.

Previous: [Learn the Agent](index.md) · Next: [The Graph: create_agent and Middleware](the-graph.md)
