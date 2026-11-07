"""The agent's tools, in four tiers, one folder each.

- atomic: plain typed functions over ctx, and the fixed tool sets (DEFAULT, GPSR,
  HRI, ARMS). Reviewed and in git. Package walkie_agent.tools.atomic.
- bundled_preset: functions that combine atomics. Reviewed and in git. Package
  walkie_agent.tools.bundled_preset holds only stand-ins for the Mahoraga gym's arm D.
- bundled: a data folder holding the last verified version of each saved function.
  Gitignored, never imported.
- temporary: a data folder holding the latest saved version of each function.
  Gitignored, never imported.

The two data folders keep only their README.md and .gitignore in git. This folder
is the default store root of walkie_agent.scripting.shelf, which writes the two
data folders and ledger.jsonl here unless WALKIE_TOOLS_HOME names another root.
"""
