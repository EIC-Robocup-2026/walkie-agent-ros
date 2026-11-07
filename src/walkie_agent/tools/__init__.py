"""The agent's tools, in four tiers, one folder each.

- atomic: plain typed functions over ctx, and the fixed tool sets (DEFAULT, GPSR,
  HRI, ARMS). Reviewed and in git. Package walkie_agent.tools.atomic.
- bundled_preset: functions that combine atomics. Reviewed and in git. Package
  walkie_agent.tools.bundled_preset holds only stand-ins for the Mahoraga gym's arm D.
- bundled: a data folder for saved functions that a passing run has proven.
  Gitignored, never imported.
- temporary: a data folder for functions the model saved and no run has proven yet.
  Gitignored, never imported.

The two data folders keep only their README.md and .gitignore in git. Nothing
writes to them yet.
"""
