"""The agent's tools, in four tiers, one folder each, and two modules beside them.

- atomic: plain typed functions over ctx, one module per tool family, with the
  catalogue of every atomic in its init. Reviewed and in git. Package
  walkie_agent.tools.atomic.
- bundled_preset: functions that combine atomics. Reviewed and in git. Package
  walkie_agent.tools.bundled_preset holds only stand-ins for the Mahoraga gym's arm D.
- bundled: a data folder holding the last verified version of each saved function.
  Gitignored, never imported.
- temporary: a data folder holding the latest saved version of each function.
  Gitignored, never imported.

Beside the tiers:

- sets.py: which atomics each run binds, the fixed tool sets (DEFAULT, GPSR, HRI,
  ARMS) and the atomics a missing flag drops. Module walkie_agent.tools.sets.
- adapter.py: turns an atomic into a LangChain tool or a plain callable that hides
  ctx. Module walkie_agent.tools.adapter.

The two data folders keep only their README.md and .gitignore in git. This folder
is the default store root of walkie_agent.scripting.shelf, which writes the two
data folders and ledger.jsonl here unless WALKIE_TOOLS_HOME names another root.
"""
