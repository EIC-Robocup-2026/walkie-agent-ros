# The doorbell test clips and their licences

The clips in this directory are the doorbell detector's eval and parity set
(`python -m walkie_agent.doorbell eval tests/data/audio`, ADR-0054). Each one is
dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), or is in the public
domain and marked with the
[Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/). No
clip needs attribution; the authors are named here as a courtesy, and so that each
clip can be traced to its source.

The clips in `bell/`, `chime/` and `knock/` must ring, and the clips in `none/` must
not, except in the cases that `known_failures.toml` lists. The clips in `buzzer/` are
scored and not judged: the rules miss most buzzers (ADR-0054), and the eval shows how
they score. Every clip is 16 kHz mono 16-bit PCM. Each was decoded from its source with
ffmpeg, resampled to 16 kHz and mixed to mono, cut to the excerpt given below (in
seconds from the start of the source file), and scaled down only when its peak was
above -1 dBFS.

## Freesound

The Freesound clips are cut from each sound's high-quality MP3 preview, as
Freesound serves it without an account.

| File | Sound | Author | Licence | Excerpt |
|---|---|---|---|---|
| `bell/freesound_69384_doorbell1.wav` | [doorbell1.wav](https://freesound.org/people/guitarguy1985/sounds/69384/) | guitarguy1985 | CC0 1.0 | 0.0 to 2.5, -1.7 dB |
| `bell/freesound_69386_doorbell_distant.wav` | [doorbell_distant.wav](https://freesound.org/people/guitarguy1985/sounds/69386/) | guitarguy1985 | CC0 1.0 | 0.5 to 3.0 |
| `bell/freesound_157252_dingdong1.wav` | [dingdong1.wav](https://freesound.org/people/baujahr66/sounds/157252/) | baujahr66 | CC0 1.0 | 0.0 to 2.5, -1.2 dB |
| `bell/freesound_361564_ding_dong.wav` | [Ding Dong](https://freesound.org/people/MatthewWong/sounds/361564/) | MatthewWong | CC0 1.0 | 0.0 to 2.5, -3.6 dB |
| `bell/freesound_125967_dorrklocka.wav` | [Dörrklocka.wav](https://freesound.org/people/Ryding/sounds/125967/) | Ryding | CC0 1.0 | 0.0 to 2.5 |
| `buzzer/freesound_164090_buzzer.wav` | [buzzer.wav](https://freesound.org/people/hypocore/sounds/164090/) | hypocore | CC0 1.0 | the whole sound, 1.5 s |
| `buzzer/freesound_52004_door_buzzer.wav` | [door_buzzzzer.aif](https://freesound.org/people/scarfield/sounds/52004/) | scarfield | CC0 1.0 | 0.0 to 2.5 |
| `buzzer/freesound_332563_buzzer.wav` | [buzzer.wav](https://freesound.org/people/011-_11919_1-1011111/sounds/332563/) | 011-_11919_1-1011111 | CC0 1.0 | 1.5 to 4.0 |
| `buzzer/freesound_455746_door_apartment_buzzer.wav` | [door apartment buzzer unlock ext.flac](https://freesound.org/people/kyles/sounds/455746/) | kyles | CC0 1.0 | 0.0 to 2.5 |
| `chime/freesound_491404_door_chimes.wav` | [Door chimes](https://freesound.org/people/IENBA/sounds/491404/) | IENBA | CC0 1.0 | 0.0 to 2.5 |
| `chime/freesound_275629_doorbell_apartment.wav` | [Doorbell apartment front door](https://freesound.org/people/hanneswannerberger/sounds/275629/) | hanneswannerberger | CC0 1.0 | 0.5 to 3.0 |
| `knock/freesound_256513_knock_on_the_door.wav` | [Knock on the door](https://freesound.org/people/HunteR4708/sounds/256513/) | HunteR4708 | CC0 1.0 | 0.0 to 2.5, -3.2 dB |
| `knock/freesound_540770_door_knock.wav` | [door knock](https://freesound.org/people/SubwaySandwitch420/sounds/540770/) | SubwaySandwitch420 | CC0 1.0 | the whole sound, 1.8 s |
| `knock/freesound_556709_knock_on_door.wav` | [Knock on Door](https://freesound.org/people/NachtmahrTV/sounds/556709/) | NachtmahrTV | CC0 1.0 | 0.0 to 2.0, -0.1 dB |
| `knock/freesound_268500_door_knock.wav` | [door_knock.wav](https://freesound.org/people/wjtaylor/sounds/268500/) | wjtaylor | CC0 1.0 | the whole sound, 2.1 s, -4.1 dB |
| `none/freesound_246315_folk_guitar.wav` | [Folk Guitar Music Track](https://freesound.org/people/Dvideoguy/sounds/246315/) | Dvideoguy | CC0 1.0 | 0.0 to 2.5, -3.0 dB |
| `none/freesound_51245_piano_and_bass.wav` | [Atmospheric Ambient Short Piano and Bass.wav](https://freesound.org/people/RutgerMuller/sounds/51245/) | RutgerMuller | CC0 1.0 | 0.0 to 2.5, -0.1 dB |
| `none/freesound_369148_music_box.wav` | [Music Box - The Flea Waltz](https://freesound.org/people/Flying_Deer_Fx/sounds/369148/) | Flying_Deer_Fx | CC0 1.0 | 0.0 to 2.5 |
| `none/freesound_426767_room_tone.wav` | [Apartment Room Tone.wav](https://freesound.org/people/cMilan/sounds/426767/) | cMilan | CC0 1.0 | 0.0 to 2.5 |

## LibriVox

Read speech from LibriVox's
[Short Story Collection Vol. 081](https://archive.org/details/ss081_1907_librivox),
read by LibriVox volunteers and marked with the Public Domain Mark 1.0 on the
Internet Archive. The clips are cut from the Ogg Vorbis files of its tracks.

| File | Track | Licence | Excerpt |
|---|---|---|---|
| `none/librivox_ss081_01_speech.wav` | [01, After Twenty Years by O. Henry](https://archive.org/download/ss081_1907_librivox/ss081_01_various.ogg) | Public Domain Mark 1.0 | 30.0 to 32.5 |
| `none/librivox_ss081_11_speech.wav` | [11, A Leaf in the Storm by Ouida](https://archive.org/download/ss081_1907_librivox/ss081_11_various.ogg) | Public Domain Mark 1.0 | 30.0 to 32.5 |

## Rendered for this repository

The robot's own voice, rendered with Kokoro as voice-tts speaks: the model
`kokoro-v0_19.onnx` and its voice `af_sarah` (Apache 2.0), at speed 1.0. The lines
are the robot's, written for this repository. The authors of this repository
dedicate the three renders to the public domain under CC0 1.0. Each clip is trimmed
to its speech, plus 0.1 s on each side.

| File | Line | Licence |
|---|---|---|
| `none/kokoro_hang_your_bag.wav` | "Please hang your bag on my hand, and tell me when it is there." | CC0 1.0 |
| `none/kokoro_close_my_hand.wav` | "I will close my hand now. Please be careful." | CC0 1.0 |
| `none/kokoro_introduce.wav` | "Alex, this is Robin. Robin's favourite drink is tea." | CC0 1.0 |

## The TF reference scores

`yamnet_reference.npz` holds no audio. It holds the scores of the TF-Hub YAMNet
model (google/yamnet/1, Apache 2.0) on each clip above, as the clip arrives from a
16, a 48 and a 44.1 kHz microphone: one `[windows, 521]` float32 array per clip and
rate, keyed `<clip path>@<rate>`. `tools/export_yamnet.py` writes it, and
`tests/test_doorbell.py` checks the ONNX model against it.
