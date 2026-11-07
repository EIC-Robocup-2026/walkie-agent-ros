# bundled

The last verified version of each saved function, as `<name>.py`. The shelf
(`walkie_agent.scripting.shelf`) writes one here when a passing episode verifies a
version, with exactly the text that ran, so its sha is the one in the verified
event of `../ledger.jsonl`. A frozen shelf, as the Mahoraga gym's arms C and C2
have, loads its functions from here, and only while `temporary/` still holds the
same version.

Everything here except this file and `.gitignore` is ignored by git and is never
committed. This folder is data: no code imports it as a package. Set
`WALKIE_TOOLS_HOME` to keep a shelf somewhere else.
