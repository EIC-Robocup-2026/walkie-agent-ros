# ADR-0054: Detect the doorbell with YAMNet inside voice-asr and wait for it in a bounded atomic

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-27

**Context**

The HRI task scores detecting the doorbell, 30 points for each guest
([HRI Task](/architecture/hri.md#the-task)). The rulebook says:

- A doorbell sound is played when a guest is at the door. It is not
  provided beforehand and may be any traditional doorbell sound. If no
  doorbell sound is available, the guest may knock instead.
- Waiting for the doorbell is optional. A robot may go to the door
  without it, and the item then scores 0. No penalty is listed for a
  missed ring or for a false one.
- The ring counts only while the robot waits at the start position and
  reacts at once. The rulebook gives no number for "at once".
- Wireless microphones are not allowed, so the robot hears the ring
  through its own microphone, the one voice-asr already reads.

So the detector must know door chimes, buzzers and knocks it has never
heard, run offline, and not ring on speech, music or the robot's own
voice.

The audio reaches voice-asr through RAI. RAI opens the microphone as int16
at the device's default rate (rai_s2s `sound_device/api.py`). When that
rate is not 16 kHz, as with the PipeWire `default` device at 48 kHz, RAI
resamples each 80 ms block on its own with `scipy.signal.resample`, and
the callback receives float64 at int16 scale. That resample treats each
block as one period, so every block edge is a discontinuity.

YAMNet (tensorflow/models, `research/audioset/yamnet`) scores the 521
AudioSet classes on 16 kHz audio. Three facts of the AudioSet ontology
shape the rules:

- Bell sits under Musical instrument, and Chime is a child of Bell. So a
  real door chime also lifts Music.
- Doorbell has two parents, Door and Alarm, and its child is Ding-dong.
  Neither is a musical class.
- Buzzer sits under Alarm, and Ding under Brief tone.

This ADR records the detector. The tap in voice-asr, the armed flag and
the wait are not built yet, and later changes add them here.

**Decision**

- **The model.** YAMNet's core network runs on ONNX Runtime, on one CPU
  thread. Its log-mel features are computed in numpy, ported from YAMNet's
  `features.py`: a 25 ms window every 10 ms, 64 bands from 125 to
  7500 Hz. A window is 0.975 s of audio, and a window starts every
  0.48 s. `src/walkie_agent/doorbell.py` holds both.
- **The weights.** `tools/export_yamnet.py` exports the TF-Hub weights
  (google/yamnet/1, Apache 2.0) once, with tf2onnx, in a scratch
  environment. The repository holds no weights.
    - The exported `yamnet.onnx` and `yamnet_class_map.csv` are assets of
      the walkie-agent-ros GitHub release `doorbell-model-v1`, beside a
      NOTICE that names the weights' licence (Apache 2.0) and the AudioSet
      ontology's (CC BY-SA 4.0). Only the ontology's class names are used.
    - `pixi run fetch-doorbell-model` downloads both into `models/`, which
      git ignores. It checks each file against a SHA-256 pinned in
      `pixi.toml`. On a mismatch it fails, and moves neither file into
      place.
    - `onnxruntime` is a direct dependency, in the range rai_s2s already
      locks.
- **Classes by index.** `doorbell.py` pins each class it reads to its
  index in YAMNet's class map. Loading the model checks every name against
  the shipped map, and a mismatch fails the load. Names are never looked
  up at run time, since "Beep, bleep" holds a comma. A `SoundClassifier`
  protocol keeps the model swappable: another model puts its scores at
  the same indices.
- **The input.** The detector takes the int16 blocks of a 16 kHz device
  and the float64 blocks of RAI's resample. Both become float32 in
  [−1, 1], on Silero VAD's scale.
- **The rules.** Thresholds are in `[walkie.doorbell.detector]` of
  `config.toml`. The window counts (1, 2 of the newest 3, 4 of 6) are
  fixed in code, so the table holds only numbers and one bool. The first
  rule that holds wins:
    - **bell:** Doorbell or Ding-dong at `bell_single` (0.25) in the
      newest window, or at `bell_pair` (0.15) in 2 of the newest 3;
    - **chime:** Ding, Chime, Bell or Buzzer at `chime_pair` (0.25) in 2
      of the newest 3, unless Music or Television reached `context_veto`
      (0.3) in 4 of the 6 windows before the first hit;
    - **knock:** Knock at `knock_pair` (0.35) in 2 of the newest 3.
- In each pair rule, the newest window must be a hit. A score threshold
  of 1.1 turns its rule off. Windows below `min_level_dbfs` (−55) are not
  classified and count as all zeros. After a ring, no ring follows for
  `refractory_s` (3.0).
- **No Music veto on the bell rule,** since Doorbell and Ding-dong are not
  musical. The chime rule looks only at the windows before the chime, so
  a chime after silence passes and a chime inside a song is vetoed.
- **`chime_pair` is 0.25,** not the design's 0.45. Through RAI's resample
  a door chime scores about half: in 2 of 3 windows, the eval's
  `freesound_491404` reaches 0.69 at 16 kHz, but 0.32 and 0.28 from 48
  and 44.1 kHz devices. The eval's speech, music and room tone reach at
  most 0.18 on the chime classes, at any onset.
- **Muted.** Each 80 ms block carries a speaking flag: whether the robot
  was speaking, or within its echo tail, when the block was captured. A
  ring is muted when a block of any of its hit windows carries the flag.
- **The event.** Each ring is a pydantic `DoorbellEvent`, the JSON of a
  `std_msgs/String` on `/walkie/doorbell` (`DOORBELL_TOPIC`), reliable,
  depth 10. It holds the label, the score, the window's three highest
  classes, `muted`, `armed`, the level, the onset time and the model's
  name. `DOORBELL_ARMED_TOPIC` names `/walkie/doorbell_armed`, a
  `std_msgs/Bool`. Both constants live in `doorbell.py`, so the detector,
  the agent and the sim scenario share one definition.
- **The config.** The detector reads its table with
  `doorbell.load_detector_config(path)`, without the agent's loader. The
  agent's `parse_tuning` checks the same table with the same parser, so a
  typo in it stops the agent too. `[walkie.doorbell]` holds the wait's
  values (`timeout_s`, from 3 to 25 s, `pre_roll_s`, `start_radius_m` and
  `drop_tail_s`), which nothing reads yet.
- **The eval.** `python -m walkie_agent.doorbell eval tests/data/audio`
  (`pixi run doorbell-eval`) scores each clip as a 16 kHz device delivers
  it, and as RAI's resample delivers it from a 48 kHz and a 44.1 kHz
  device.
    - At each rate it plays the clip at 8 onsets, after 0 to 7 eighths of
      a hop of silence, since a ring starts anywhere between two windows.
    - Each clip in `bell/`, `chime/` and `knock/` must ring at every rate
      and onset, and no clip in `none/` may. Clips in other directories,
      such as `buzzer/`, are scored and not judged.
    - `tests/data/audio/known_failures.toml` lists the cases known to
      fail, each with its reason. A failing case it does not list fails
      the eval, and so does a listed case that passes, so the file follows
      the rules. When only listed cases fail, the eval says so instead of
      passing.
    - A ring directory whose rule the config turns off is not judged, and
      the eval says so.
    - It also reports the highest Knock and Tap scores in `none/`. If
      speech or silence scored as knocks, the knock rule would ship off.
    - The clips are CC0 or public domain (`tests/data/audio/LICENSES.md`).
- **Parity.** `tests/data/audio/yamnet_reference.npz` holds the TF-Hub
  model's scores on every clip at all three rates. The tests require the
  ONNX model and the numpy features within 1e-3 of them.
- **Imports.** `doorbell.py` imports only numpy, pydantic, ONNX Runtime and
  `walkie_agent.config_tables`, so voice-asr loads it without the agent
  and without ROS. It imports ONNX Runtime only when it loads a model.

**Consequences**

- Nothing publishes on `/walkie/doorbell` yet, and the agent does not
  wait for a ring. The HRI task still goes straight to the door.
- The model lives in a release, not in the repository. Where the release
  cannot be reached, `pixi run fetch-doorbell-model` fails with a message
  that names it, and the tests that run the model skip with a reason that
  names the task. The rules, the audio path, the config and the eval's
  bookkeeping are tested without it. CI does not fetch the model, so it
  runs none of the model's tests. `docs/setup/07-voice.md` says how a
  maintainer exports and publishes the release.
- The SHA-256 pins the published file, not an export. A second export
  gives the same weights and the same scores, but tf2onnx names its folded
  constants differently on each run, so the file's bytes differ.
- The margins are thin, and the rules' values are starting points to
  tune on the robot:
    - Shifting a clip's onset within one hop moves its scores. At some
      onsets the rules fail on 4 of the eval's 20 judged clips. The music
      box rings as a bell, with 0.43 on Ding-dong in one window at 16 kHz,
      above `bell_single`. The door chime `freesound_491404` falls below
      `chime_pair` from a 48 kHz or 44.1 kHz device. A distant doorbell
      and a knock miss at one or two onsets.
    - `known_failures.toml` lists those 15 cases. No change to the
      thresholds clears them: a `bell_single` above the music box's 0.43
      misses the distant doorbell, which rings on 0.40 from a 44.1 kHz
      device.
    - A buzzer often scores high in only one window, and Buzzer is read
      only by the chime rule's 2 of 3. Of the four CC0 buzzers in
      `tests/data/audio/buzzer/`, one rings, as a chime, at about half its
      onsets, and three never ring. They are scored and not judged, so the
      eval shows the gap without failing on it.
- Opening the microphone at 16 kHz, so that PipeWire resamples, would
  remove the block-edge discontinuities and restore the chime scores. It
  also changes Whisper's input, so it is a separate voice change,
  measured on speech recognition first.
<!-- --8<-- [end:body] -->
