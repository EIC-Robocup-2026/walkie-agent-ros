# Mahoraga Demo Branches

The pick and web search work is pushed as branches, not pull requests. This page holds what each pull request would have said. The reasons behind each choice are in [Mahoraga Demo Decisions](mahoraga-demo-decisions.md#pick-and-web-search).

| Repository | Branch | Base |
|---|---|---|
| walkie-ros | `feat/moveit-full-body` | `main` |
| walkie-ros | `feat/mujoco-chula-world` | `feat/moveit-full-body` |
| walkie-ros | `feat/sim-bottle-pick` | `feat/mujoco-chula-world` |
| walkie-agent-ros | `feat/mahoraga-demo` | its earlier push |

`feat/moveit-full-body` was pushed before this work and is described on its own.

## walkie-ros: feat/mujoco-chula-world

### Summary

Named MuJoCo worlds, starting with the Chula PAT14-6 lab room, ported from the older navigation branch onto the whole-body stack.

```text
whole_body_demo.launch.py world:=chula_pat14_6 spawn_x:= spawn_y:= spawn_yaw:=
  mujoco_omnibot.launch.py
    walkie_mujoco_worlds/chula_pat14_6   # model, map, Nav2 copy with the spawn pose
    AMCL only when truth_map_odom is off
```

Main's navigation launch features stay: `nav2_config` with RewrittenYaml, the map directory, keepout, `nav_debug`, `nav_commander` and `truth_map_odom`. One change applies without a world too: on Linux with both `DISPLAY` and `WAYLAND_DISPLAY` set, the simulator viewer now uses X11.

### Evidence

- **Before:** the whole-body stack launched only the empty floor.
  **After:** headless `chula_pat14_6` runs succeeded for a whole-body GoToPose, Nav2 goals with ground truth, and Nav2 goals with AMCL. `robot_bringup` and `walkie_mujoco_worlds` pass 145 tests, and `mkdocs build --strict` passes.

### Merge Danger

**Door:** two-way

**Blast Radius:** simulation

Launches without `world:=` behave as before, apart from the viewer backend above.

## walkie-ros: feat/sim-bottle-pick

### Summary

The Chula water bottles get contacts the gripper can hold, an offline check proves the hold, and `sim_bottle_pick.py` runs the whole pick in the simulation.

```text
sim_bottle_pick.py
  gate: /clock advances, use_sim_time on the commander and bt_navigator,
        /controller_manager mujoco_world == chula_pat14_6          else exit 2
  snapshot allow_gripper_vs_octomap and grasp_scene_action
  Nav2 to the stand-off       retry only ABORTED 103/105/107/207, 3 attempts
  tuck the left arm, whole_body_right to the pre-grasp, hand within 0.05 m
  approach, close, lift 0.10 m on right_arm_lift, hold 5 s
  verdict: grasp band + head depth shows the bottle left the table  else exit 3
  on every exit: cancel goals, restore both parameters, read them back
```

Exit codes: 0 passed, 1 failed, 2 gate or prerequisite, 3 inconclusive, 4 cleanup failed or a goal's end unknown, 130 interrupted.

### Evidence

- **Before:** in the exported world the bottle slid out between the fingers, and `check_grasp.py` fails on that export.
  **After:** the check passes on the lowpoly and baked worlds. Six fresh-launch runs after the review fixes: 3 passed, 1 inconclusive camera view, 1 Nav2 code 103 (now retried), 1 pre-grasp miss caught by the new hand check. With no simulation running, the script exits 2 in about 11 s and changes nothing. 90 unit tests and 236 colcon tests pass.

### Merge Danger

**Door:** two-way

**Blast Radius:** simulation

`robot_bringup` installs the script on the robot too; the gate keeps it off real hardware. Any other launch of the MuJoCo `ros2_control_node` must set `mujoco_world`, or the pick refuses to start.

## walkie-agent-ros: feat/mahoraga-demo

### Summary

"Bring me the water bottle" and answers from the web.

```diff
 hri tool set
+  pick_object(name)        # with --arms and --object-memory
+  search_web(query)        # with --web-search and TAVILY_API_KEY
 ctx
+  arms.reach(hand_pose)    # hand pose goals on /go_to_pose
+  web.search(query)        # Tavily, 3 results, a separate tool-less summarizer
```

```text
pick_object(name)
  recall the object, drive to a stand-off facing it
  wait up to 6 s for a sighting at most 5 s old, moved at most 0.15 m, in reach
  allowance on: whole_body_right pre-grasp, straight approach, close, lift
  allowance off in a finally; move clear beside the base; both arms home
```

A web search is bounded at 5 s for the request and 10 s for the summary, and the agent sees only the summary and the source URLs. The robot commander profile gains buttons for the whole-body stack, the object memory and this agent, each motion button in its own group.

### Evidence

- **Before:** the agent had no pick and no web tool.
  **After:** in the Chula simulation, a real-LLM run picked the water bottle in 260 s, carried it to the living room and handed it over; `allow_gripper_vs_octomap` read false afterwards. The full suite passes (3060 passed, 1 skipped), as do `lint-imports` and the strict docs build. No live Tavily call and no robot run yet.

### Merge Danger

**Door:** two-way

**Blast Radius:** hri

Both tools stay unbound without their flags. At startup the agent resets the commander's `allow_gripper_vs_octomap` to false, so it overrides anyone else who turned it on.
