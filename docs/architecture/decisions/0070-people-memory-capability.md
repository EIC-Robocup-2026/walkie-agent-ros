# ADR-0070: Remember people through ctx.memory, with live ids and no vectors

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-11-07

**Context**

The HRI task needs Walkie to learn a guest's name, and to know them again later in the
same task. walkie-database stores people with their facts and their face and
appearance vectors, and matches vectors against them (its `StartSession`,
`RememberPerson` and `MatchPeople` services). The experiment line built this over its
own toolbox, with per-track vector collection in its vision feed and handles that
carried a collection's generation. This tree has neither, and its find_people ids are
plain strings, `person_<track_id>`. Perception questions 2 (the services' session
rules and who tunes the distance limits) and 3 (no effect covers a memory write) are
still open.

**Decision**

- A `Memory` capability on `Ctx`, `ctx.memory`, with two verbs:
  `remember_person(person_id, name, facts)` and `recognize(person_id)`. Its default is
  `UnavailableMemory`, which refuses with UNAVAILABLE. The records it returns,
  `Remembered` and `Recognized`, hold no vector and no stored id.
- `person_id` is always a live id from find_people. The backend's own ids
  (`stored_<n>`) never reach the model. An id with no per-track samples in the last
  60 s, or an untracked person, is NOT_FOUND.
- On the robot, `RosVision` keeps a bounded buffer of each track's face and
  appearance samples from its existing subscription, and `TrackMemory` matches and
  stores them. The store is `RosPeopleMemory` over walkie-database, or
  `InMemoryPeopleMemory` in simulation, chosen with `--people-memory ros` or `mock`.
- Recognition is face first, then appearance, each model with its own distance limit,
  and one stored person per person in view. find_people shows the names of the tracked
  people in view the same way, so a name query such as `find_people("Sam")` works.
- The people remembered are forgotten at each reset on `/walkie/reset`, which is a new
  task. `--resume-people-session` keeps the database's open session for the first task
  after a restart.
- A database failure (a message that starts with "The database could not") is retried
  for up to 2 s and then reported as UNAVAILABLE. A store that does not answer is
  TIMEOUT, and the tool says that the person may have been stored already.
- `remember_person` and `recognize_person` are in the HRI tool set only. No effect
  letter is given to them, and they stay out of GPSR and Mahoraga until question 3 is
  answered.

**Consequences**

- A track id that walkie-vision reuses within 60 s without a clock jump, as after a
  restart of walkie-vision alone, can put two people in one collection. A clock jump,
  as after a simulation restart, clears the buffer.
- The distance limits are the experiment's: 0.4 for ArcFace faces and 0.5 for OSNet
  appearances, the second untuned. A model without a limit never names anyone.
- find_people on the robot makes up to two match calls when memory is on. A memory
  failure there leaves the names null and is logged; it does not fail find_people.
- In simulation the store is the mock, so the database path, its type hashes and its
  persistence are tested only against a fake database node.
- Stored faces raise a privacy and retention question on the robot, which this
  decision does not answer.

**See also:** [Perception: open questions](/architecture/perception.md#open-questions),
[See, remember and act: S3](/architecture/see-remember-act.md#s3-remember-people-by-name),
[ADR-0052](/architecture/decisions/0052-capability-verbs-and-backends.md)
<!-- --8<-- [end:body] -->
