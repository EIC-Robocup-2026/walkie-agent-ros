# ADR-0035: Read Walkie's memory through walkie-database's ROS 2 services

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-20

**Context**

[walkie-database](https://github.com/EIC-Robocup-2026/walkie-database) holds
Walkie's long-term memory in one SQLite file. Its node, `walkie_database`,
is the only program that writes the file. Other programs store and search
memory through its ROS 2 topics and services (walkie-database ADR-0005). The
agent needs this memory to answer questions such as "where is my cup?".

Checked against the walkie-database commit that this ADR first pins in
`ros2_ws.repos`, the node offers
`/walkie_database/find_objects`. Its types are in `walkie_database_msgs`, a
package in walkie-database that holds only interface files
(walkie-database ADR-0015):

- `FindObjects.srv`: the request has `map_id`, `label` and `max_results`.
  The response has `success`, `message` and a list of `RememberedObject`,
  most recently seen first.
- `RememberedObject.msg`: an object's label, caption, map and TF frame,
  bounding box (position, orientation and size), room, number of merged
  detections, and when it was last seen.
- An empty `map_id` in the request means the node's `map_id` parameter, the
  map the robot is using. When both are empty, or the map is not in the
  database, the search fails with `success` false. Each map has its own
  frame, so a search covers one map (walkie-database ADR-0008).

RoboStack does not publish the package. The node runs one callback at a
time, so a slow search delays the next request (walkie-database ADR-0005).

Observed with rmw_zenoh_cpp 0.2.11 on 2026-10-19: a client built from a
different definition of the service passes `wait_for_service`, because it
matches the service name and the type name. The request then goes to a key
expression that includes the type's RIHS01 hash. The server's key has a
different hash, so the server never receives the request, and no answer
comes. `ros2 service call` hangs the same way. A change to comments only
keeps the hash.

[ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md) says that a
node built from a different message definition "never connects", and
[ADR-0018](/architecture/decisions/0018-pin-robot-interfaces.md) says that
such a client "never matches the server". For services, that is not what
happens: the client finds the server, and its requests are never answered.
Actions were not tested. Their goals and results are sent as service calls,
so they probably behave the same way.

**Decision**

- The agent reads Walkie's memory only through the database node's ROS 2
  services. It does not open the SQLite file.
- `ros2_ws.repos` lists walkie-database at a pinned commit. The pin must
  match the commit of the database node the robot runs.
- `vcs import` clones the whole repository. `pixi run build-ros2` builds
  only the message package, with
  `colcon build --packages-select rai_interfaces walkie_database_msgs`. This
  is the pattern of
  [ADR-0004](/architecture/decisions/0004-build-rai-interfaces-from-source.md)
  for `rai_interfaces`, and the one ADR-0018 plans for walkie-ros's message
  packages.
- The first tool is `find_objects`. It sends a label and `max_results` 5,
  and leaves `map_id` empty. The database node's `map_id` parameter decides
  which map is searched (walkie-database ADR-0008, and "The node's
  interface" in walkie-database's `docs/architecture/database.md`).
- When no node offers the service within 2 seconds, the tool reports that
  memory is unavailable.
- When a call gets no answer within 5 seconds, the tool drops the request.
  It tells the LLM only that memory did not answer and may be busy. It logs
  a warning for developers: if the node is not busy, it was probably built
  from a different `walkie_database_msgs` than the pinned commit. A busy
  node and a mismatched one look the same to the tool, and the LLM can do
  nothing about a mismatch.
- `tests/conftest.py` stops with a message to run `pixi run build-ros2` when
  `rai_interfaces` or `walkie_database_msgs` is missing.

This keeps the repository boundary of
[ADR-0015](/architecture/decisions/0015-repository-boundary.md): the
database and its code stay in walkie-database, and this repository holds
only the tool that calls it. Like the vision models of
[ADR-0034](/architecture/decisions/0034-vision-models-in-walkie-vision.md),
memory is a robot-side service that the agent calls over ROS 2.

We rejected copying the interface files into this repository. The copies
would drift from walkie-database, and a drifted copy fails without an
error, as described above. We also rejected opening the SQLite file from the
agent, even read-only. The agent would then depend on a schema that
walkie-database may change at any time. The ROS 2 messages are the
interface (walkie-database ADR-0005).

**Consequences**

- A change to `walkie_database_msgs` needs a coordinated update: push the
  walkie-database commit, bump the pin in `ros2_ws.repos`, run
  `pixi run build-ros2`, and restart the agent and the database node.
- CI clones walkie-database from GitHub, so the pinned commit must be pushed
  before the new pin is pushed here.
- `--packages-select` is an allow-list. It must name every message package
  in `ros2_ws.repos`. A package missing from it is not built, and the agent
  fails only when it imports the package.
- The whole walkie-database repository is cloned, although only one package
  is built.
- `find_objects` works only while the database node runs on the agent's
  domain, with its `map_id` parameter set to the map the robot uses. Without
  the parameter, or with a map the database does not have, every search
  fails, and the tool passes the node's message to the LLM.
- To the LLM, a mismatched pin looks like a busy node. A developer finds the
  hint in the agent's log.
- The node matches the label exactly, ignoring case. A synonym, such as
  "phone" for an object the detector called "cell phone", finds nothing. The
  tool's input description says so, and its answer when nothing is found
  says that the object may be stored under another name.
- The tests use a fake database node, built from the same pinned messages
  ([ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md)).
- ADR-0017 and ADR-0018 are corrected for services and point here.
  ADR-0018 also says that tool error messages should name a stale pin.
  `find_objects` puts that hint in the log instead, for the reason above.

**See also:** [The Agent: tools](/architecture/agent.md#tools),
[walkie-database Getting Started](https://eic-robocup-2026.github.io/walkie-database/getting-started/)
<!-- --8<-- [end:body] -->
