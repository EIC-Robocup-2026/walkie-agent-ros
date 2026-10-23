# ADR-0044: Mock run_skill first, and let generated tools move the arms only through the arm atomics

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

The [self-evolving agent](/architecture/self-evolving-agent.md) wires typed
atomic tools into bundles, and writes new tools that run in a sandbox
([ADR-0037](/architecture/decisions/0037-tool-tiers-bundles-effects.md)).
Its presets pick objects up and hand them over, so it needs arm tools that
return typed results from its first phases. The arm design already fixes
the contracts:

- the arm tools move the arms only through `bimanual_commander`
  ([ADR-0021](/architecture/decisions/0021-arm-control-through-commander.md));
- picks and other skills run behind a robot-side `RunSkill` action with seven
  outcomes ([ADR-0027](/architecture/decisions/0027-runskill-action.md));
- each walkie-ros node the agent calls gets a ROS-level fake for tests
  ([ADR-0019](/architecture/decisions/0019-test-fakes-fast-dds.md)).

walkie-ros has no `RunSkill` interface (W4) and no skill server (W5), so
`run_skill` has no real backend. The commander does run, with MoveIt, in
walkie-ros's MuJoCo simulation (checked on 2026-10-23), and the agent pins
its `my_robot_interfaces` messages.

Generated tools raise a second question. The MuJoCo simulation spawns the
arm, gripper, lift and head controllers, so generated code at the
Unrestricted level could command them directly through `ctx.ros`.

**Decision**

- The arm tools become typed atomics. `run_skill`, `move_arm_to_pose` and
  `set_gripper` have effect M, and `get_arm_state` and `list_arm_poses` have
  effect R. Each returns a typed result whose summary is the sentence on the
  [manipulation page](/architecture/manipulation.md#tools), and raises a
  `StepError` with a shared code on failure.
- `run_skill` maps the ADR-0027 outcomes. `SUCCEEDED` and
  `FINISHED_UNVERIFIED` return a `SkillResult`. `FAILED`,
  `PRECONDITION_FAILED`, `CANCELED`, `SAFETY_STOP` and `TIMED_OUT` raise
  `SKILL_FAILED`, `PRECONDITION_FAILED`, `CANCELLED`, `SAFETY_STOP` and
  `TIMEOUT`.
- The atomics reach the arms through backends. `move_arm_to_pose`,
  `set_gripper`, `get_arm_state` and `list_arm_poses` call the real
  commander through `CommanderClient`, the `arm` backend (ADR-0021). Their
  tests run against the ROS-level `FakeCommander` (ADR-0019). `run_skill`'s
  first backend is `MockSkillServer`, a mock at the ADR-0027 contract. The
  skill-server client (ADR-0027) and the MotionSupervisor
  ([ADR-0023](/architecture/decisions/0023-motion-supervisor.md)) are built
  on the real-backends track, from phase 2 of the self-evolving agent. It is
  done when the bundle tests rerun against the real clients.
- Each atomic declares its backends in `META.requires`. The registry hides a
  tool whose backend is a mock unless the agent runs in simulation, and a
  real run refuses to start if a required backend is a mock.
- The sim harvest serves `run_skill` with a ROS-level `FakeSkillServer`
  (`src/walkie_agent/sim/fake_skill_server.py`). It answers with the
  ADR-0027 outcomes, and mirrors picked and placed objects into MuJoCo
  through `SetFreeJointState`.
- Generated tools are to move the arms only through these atomics. At
  Standard and Unrestricted they call `run_skill`, `move_arm_to_pose` and
  `set_gripper` through `ctx.tools`. At Competition they have no M effect,
  and `get_arm_state` is the only arm atomic they can call.
- Every arm, gripper, head and lift controller name is on the never-list at
  every access level, for every verb, together with `/controller_manager/*`,
  the MoveIt and commander planning names, and the commander's `/go_to_home`
  and collision services. `/set_joint_position`, `/control_gripper` and the
  `RunSkill` action, once its name is pinned, are on it too, except in the
  human-written arm atomics' manifests
  ([ADR-0039](/architecture/decisions/0039-access-levels.md)). The reviewed
  manifests of git-tier tools obey the same list.

We rejected:

| Option | Why not |
| ------ | ------- |
| Raw sim arm controllers at Unrestricted | They would show which arm atomics are missing. But code that drives the controllers skips MoveIt's planning and collision checking (ADR-0021), so it is never promotable as-is. Cleaning up after it needs controller-aware stops, and the ROS bridge only knows stop messages for `Twist` and `TwistStamped`. Phase 4b, which ships only on harvest evidence, therefore adds no raw arm controllers. They can be reopened on evidence from the harvest's atomic wishlist. |
| Grasp planning in the agent, from poses and grippers | ADR-0021 and ADR-0027 put picks robot-side, and the system prompt says never to improvise a pick. The code would be thrown away when the skill server lands. walkie-ros already serves `/grasp/from_cloud` (GraspNet), which belongs behind the robot-side pick skill. |
| Building the real skill-server client first | Presets, bundles and the harvest would wait for W4 and W5. The mock lets that work start now, at the same contract. |
| A `MockArmBackend` for the pose and gripper atomics | The commander already runs with MoveIt in the MuJoCo simulation, so a mock would only add a second behavior to keep in step. The `FakeCommander` gives the tests the commander's interface without MoveIt. |

This ADR does not change ADR-0021 or ADR-0027, which stay the contracts for
the real backends. It changes the build order, with one narrowing: the
presets give `run_skill` an object handle as `target`, while `RunSkill`'s
`target` is free text. How the real client passes the handle is open (open
question 10 on the [manipulation page](/architecture/manipulation.md#open-questions)).
`run_skill`'s unit tests use a protocol mock, not the ROS-level fake
ADR-0019 asks for; ADR-0037 records that gap, and the `FakeSkillServer`
closes it.

**Consequences**

- Presets with arm steps (`pick_object`, `pickup_object_in_view`,
  `hand_over`) can be built, linted and tested now, with no robot and no
  walkie-ros change. Their pose and gripper steps run on the `FakeCommander`
  in tests.
- A mocked `run_skill` success says nothing about real grasps. Harvest
  results measure the agent's choices, not manipulation. The presets stay
  hidden on the real robot until their backends are real.
- `MockSkillServer` can drift from the real interface. It follows the
  ADR-0027 outcomes and the manipulation page's sentences, and the
  real-client reruns catch the rest.
- "Pick, then drive" exercises the
  [ADR-0025](/architecture/decisions/0025-travel-safe-interlock.md) interlock
  against the commander. The pick presets end in `carry`, so it must be
  travel-safe.
- Until the MotionSupervisor lands, a reset trips the session's cancel token.
  The bundle runner checks it between steps. A running arm atomic cancels its
  commander goal; any other running step stops only when its own wait checks
  the token.
- Through the atomics, generated tools can do with the arms only what the
  atomics express. A new arm capability needs a human to add an atomic, a
  skill or a named pose; the harvest's atomic wishlist reports the need.
- Which ROS interface the `FakeSkillServer` serves before walkie-ros adds
  `RunSkill` (W4) is open (open question 9 on the
  [manipulation page](/architecture/manipulation.md#open-questions)). W4 is
  not a precondition of the harvest. Whatever it serves needs the same
  never-list entry as `RunSkill`, or Unrestricted code could start it
  directly, around the atomics.
- The commander and skill entries land in phase 4a, with raw actions at
  Unrestricted ([The never-list](/architecture/access-levels.md#the-never-list)).

**See also:**
[Arm Manipulation](/architecture/manipulation.md#arms-in-the-self-evolving-agent),
[Access Levels and Generated Tools](/architecture/access-levels.md),
[Sim Harvest](/architecture/harvest.md)
<!-- --8<-- [end:body] -->
