# temporary

The latest saved version of each function the model saved from its own scripts,
as `<name>.py`. The shelf (`walkie_agent.scripting.shelf`) writes one here when a
script run with `save=[names]` succeeds, and logs it in `../ledger.jsonl`. A
writable shelf, as the Mahoraga gym's `induct` arm has, offers everything here.

Everything here except this file and `.gitignore` is ignored by git and is never
committed. This folder is data: no code imports it as a package. Set
`WALKIE_TOOLS_HOME` to keep a shelf somewhere else.
