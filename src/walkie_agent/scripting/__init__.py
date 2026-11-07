"""run_script's runtime: the driver, the Monty sandbox, the evidence counters and the shelf.

A model-written script runs on a Monty worker and reaches only the functions on its
allow-list (driver). Each worker runs inside bubblewrap (sandbox). Counters inside
the worker tell which saved functions ran and returned (evidence). Functions a script
saves land on a shelf, in the git-ignored tier folders of walkie_agent.tools or of
WALKIE_TOOLS_HOME, with a ledger of how they fared (shelf).

This package imports no ROS and nothing from walkie_agent.mahoraga, so the agent can
use it in the default environment. The Mahoraga gym builds on it.
"""
