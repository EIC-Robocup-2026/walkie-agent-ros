"""Export YAMNet's core network to ONNX, and the TF reference scores of the doorbell tests.

    python -I tools/export_yamnet.py --saved-model <dir> --out <release dir> \\
        --clips tests/data/audio --reference tests/data/audio/yamnet_reference.npz

walkie_agent.doorbell runs YAMNet's core network (MobileNet v1, 521 sigmoid
scores) on ONNX Runtime, after log-mel features it computes in numpy. This
tool builds that network from the TF-Hub weights, converts it with tf2onnx,
and checks the conversion against the TF-Hub model itself. It writes to
--out the three files of the doorbell-model-v1 release, from which
`pixi run fetch-doorbell-model` downloads the first two:

    yamnet.onnx              input "patches" [n, 96, 64] float32, output "scores" [n, 521]
    yamnet_class_map.csv     the TF-Hub model's class map, copied as it is
    NOTICE                   the licences of the weights and of AudioSet

It also scores every WAV file under --clips with the TF-Hub model, end to
end from the waveform, along each of the eval's paths: the clip at 16 kHz,
and the clip played at 48 kHz and at 44.1 kHz and resampled block by block
by scipy.signal.resample, as RAI's stream does. --reference gets those
scores, one [windows, 521] float32 array per clip and rate, keyed
"<clip path>@<rate>". tests/test_doorbell.py checks that the numpy features
and the ONNX model give the same scores, within 1e-3.

The weights are TF-Hub's google/yamnet/1, now on Kaggle as the TensorFlow2
variation of google/yamnet, version 1 (Apache 2.0):

    https://www.kaggle.com/api/v1/models/google/yamnet/tensorFlow2/yamnet/1/download
    sha256 b80da2a1a56926fb0767205051a200dd7b3beaf3ea1ea126c42a53943996e5e0

Extract the archive and pass its directory as --saved-model. The tool runs in
a throwaway environment of its own, not in pixi's, with Python 3.11,
tensorflow 2.15, tensorflow-hub 0.16, tf2onnx 1.16, onnx 1.16, onnxruntime,
scipy and numpy below 2. Run it with python -I, since it reads downloaded
files.

A new test clip changes only --reference: point --out at a scratch
directory and keep the published release. tf2onnx writes other bytes on
each export, so a new yamnet.onnx would need a new release and new sums in
pixi.toml (docs/setup/07-voice.md).
"""

import argparse
import hashlib
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import scipy.signal
import tensorflow as tf
import tf2onnx

SAMPLE_RATE = 16000
# voice-asr's blocks at 16 kHz, and the device rates of the eval's paths
# (walkie_agent.doorbell.BLOCK_SAMPLES and EVAL_RATES).
BLOCK_SAMPLES = 1280
EVAL_RATES = (16000, 48000, 44100)
PATCH_FRAMES = 96
PATCH_HOP_FRAMES = 48
BATCHNORM_EPSILON = 1e-4
OPSET = 13
# The largest difference allowed between two models' scores.
TOLERANCE = 1e-3

# yamnet.py's _YAMNET_LAYER_DEFS: (kind, stride, filters), all with 3x3 kernels.
LAYERS = (
    ("conv", 2, 32),
    ("separable", 1, 64),
    ("separable", 2, 128),
    ("separable", 1, 128),
    ("separable", 2, 256),
    ("separable", 1, 256),
    ("separable", 2, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 1, 512),
    ("separable", 2, 1024),
    ("separable", 1, 1024),
)

NOTICE = """\
yamnet.onnx and yamnet_class_map.csv

yamnet.onnx is the core network of YAMNet, an audio event classifier by
Google, converted to ONNX by tools/export_yamnet.py in
https://github.com/EIC-Robocup-2026/walkie-agent-ros. Its weights are
those of the TF-Hub model google/yamnet/1, published by Google on Kaggle
(https://www.kaggle.com/models/google/yamnet, TensorFlow2 variation,
version 1) under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0). The conversion changes the
file format only, not the weights. The model's source code is in
https://github.com/tensorflow/models/tree/master/research/audioset/yamnet,
also under the Apache License, Version 2.0.

yamnet_class_map.csv is the class map shipped in the same TF-Hub model,
copied unchanged. Its class names and identifiers are those of the
AudioSet ontology (https://github.com/audioset/ontology), made available
by Google Inc. under the Creative Commons Attribution-ShareAlike 4.0
International license (https://creativecommons.org/licenses/by-sa/4.0/).

YAMNet was trained on the AudioSet dataset
(https://research.google.com/audioset/), made available by Google Inc.
under the Creative Commons Attribution 4.0 International license
(https://creativecommons.org/licenses/by/4.0/).
"""


class ExportError(Exception):
    """The export failed a check."""


def core_network(variables: dict[str, np.ndarray]) -> tf.types.experimental.GenericFunction:
    """yamnet.py's yamnet(): the patches' scores, as a tf.function over the weights as constants."""

    def batch_norm(net: tf.Tensor, name: str) -> tf.Tensor:
        # yamnet.py's batch norm has an offset and no scale.
        return tf.nn.batch_normalization(
            net,
            mean=variables[f"{name}/moving_mean"],
            variance=variables[f"{name}/moving_variance"],
            offset=variables[f"{name}/beta"],
            scale=None,
            variance_epsilon=BATCHNORM_EPSILON,
        )

    @tf.function(
        input_signature=[tf.TensorSpec([None, PATCH_FRAMES, 64], tf.float32, name="patches")]
    )
    def scores(patches: tf.Tensor) -> tf.Tensor:
        net = tf.expand_dims(patches, -1)
        for i, (kind, stride, _) in enumerate(LAYERS):
            name = f"layer{i + 1}"
            if kind == "conv":
                net = tf.nn.conv2d(net, variables[f"{name}/conv/kernel"], stride, "SAME")
                net = tf.nn.relu(batch_norm(net, f"{name}/conv/bn"))
                continue
            net = tf.nn.depthwise_conv2d(
                net,
                variables[f"{name}/depthwise_conv/depthwise_kernel"],
                [1, stride, stride, 1],
                "SAME",
            )
            net = tf.nn.relu(batch_norm(net, f"{name}/depthwise_conv/bn"))
            net = tf.nn.conv2d(net, variables[f"{name}/pointwise_conv/kernel"], 1, "SAME")
            net = tf.nn.relu(batch_norm(net, f"{name}/pointwise_conv/bn"))
        embeddings = tf.reduce_mean(net, axis=[1, 2])
        logits = tf.matmul(embeddings, variables["dense/kernel"]) + variables["dense/bias"]
        return tf.sigmoid(logits)

    return scores


def hub_variables(model: tf.Module) -> dict[str, np.ndarray]:
    """The TF-Hub model's weights by name, without the ":0"."""
    variables = {v.name.removesuffix(":0"): v.numpy() for v in model._yamnet.variables}
    # A conv layer has a kernel and three batch norm values; a separable
    # layer has two kernels and two batch norms; the classifier has two.
    expected = 4 + 8 * (len(LAYERS) - 1) + 2
    if len(variables) != expected:
        raise ExportError(f"The SavedModel holds {len(variables)} weights, not {expected}.")
    return variables


def patches_of(log_mel: np.ndarray) -> np.ndarray:
    """The TF-Hub model's log-mel spectrogram framed into its patches, [n, 96, 64]."""
    count = 1 + (len(log_mel) - PATCH_FRAMES) // PATCH_HOP_FRAMES
    return np.stack(
        [log_mel[i * PATCH_HOP_FRAMES : i * PATCH_HOP_FRAMES + PATCH_FRAMES] for i in range(count)]
    )


def to_float(block: np.ndarray) -> np.ndarray:
    # As walkie_agent.doorbell.to_float.
    return np.clip(np.asarray(block, np.float32) * (1 / 32768), -1.0, 1.0)


def through_rai(clip: np.ndarray, device_rate: int) -> np.ndarray:
    """walkie_agent.doorbell.through_rai, with scipy.signal.resample, as RAI calls it."""
    if device_rate == SAMPLE_RATE:
        return to_float(clip)
    played = scipy.signal.resample(
        np.asarray(clip, np.float64), round(len(clip) * device_rate / SAMPLE_RATE)
    )
    pcm = np.clip(np.round(played), -32768, 32767).astype(np.int16)
    size = int(BLOCK_SAMPLES * device_rate / SAMPLE_RATE)
    blocks = [
        to_float(
            scipy.signal.resample(pcm[start : start + size], int(size / device_rate * SAMPLE_RATE))
        )
        for start in range(0, len(pcm) - size + 1, size)
    ]
    return np.concatenate(blocks)


def read_clip(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ExportError(f"{path} is not a 16 kHz mono 16-bit WAV file.")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").copy()


def export_onnx(scores: tf.types.experimental.GenericFunction, path: Path) -> None:
    """Convert the core network to ONNX, with its input named patches and its output scores."""
    model, _ = tf2onnx.convert.from_function(
        scores, input_signature=scores.input_signature, opset=OPSET
    )
    (output,) = model.graph.output
    for node in model.graph.node:
        node.output[:] = ["scores" if name == output.name else name for name in node.output]
    output.name = "scores"
    (model_input,) = model.graph.input
    if model_input.name != "patches":
        raise ExportError(f"The ONNX model's input is {model_input.name!r}, not 'patches'.")
    model.doc_string = (
        "YAMNet's core network (tensorflow/models research/audioset/yamnet) with the TF-Hub "
        "google/yamnet/1 weights (Apache 2.0), exported by walkie-agent-ros "
        "tools/export_yamnet.py. Input: log-mel patches [n, 96, 64]; output: 521 AudioSet "
        "class scores [n, 521]."
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--saved-model", type=Path, required=True, help="The TF-Hub model.")
    parser.add_argument("--out", type=Path, required=True, help="The release files' directory.")
    parser.add_argument("--clips", type=Path, required=True, help="The WAV files to score.")
    parser.add_argument("--reference", type=Path, required=True, help="The .npz to write.")
    args = parser.parse_args()

    hub = tf.saved_model.load(str(args.saved_model))
    scores = core_network(hub_variables(hub))
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "yamnet.onnx"
    export_onnx(scores, onnx_path)
    class_map = args.saved_model / "assets" / "yamnet_class_map.csv"
    shutil.copyfile(class_map, args.out / "yamnet_class_map.csv")
    (args.out / "NOTICE").write_text(NOTICE)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )

    reference: dict[str, np.ndarray] = {}
    worst = {"core": 0.0, "onnx": 0.0}
    paths = sorted(args.clips.rglob("*.wav"))
    if not paths:
        raise ExportError(f"{args.clips} holds no WAV files.")
    for path in paths:
        clip = read_clip(path)
        for rate in EVAL_RATES:
            waveform = through_rai(clip, rate)
            hub_scores, _, log_mel = hub(waveform)
            hub_scores = hub_scores.numpy()
            patches = patches_of(log_mel.numpy())
            # The rebuilt network and its ONNX export, on the TF-Hub model's
            # own features: what the conversion alone changes.
            core = scores(patches).numpy()
            (converted,) = session.run(["scores"], {"patches": patches})
            worst["core"] = max(worst["core"], float(np.abs(core - hub_scores).max()))
            worst["onnx"] = max(worst["onnx"], float(np.abs(converted - hub_scores).max()))
            reference[f"{path.relative_to(args.clips).as_posix()}@{rate}"] = hub_scores
    np.savez_compressed(args.reference, **reference)

    print(f"{len(paths)} clips at {', '.join(map(str, EVAL_RATES))} Hz, {len(reference)} arrays")
    print(f"rebuilt network against TF-Hub: max |difference| {worst['core']:.2e}")
    print(f"ONNX against TF-Hub, on TF-Hub's features: max |difference| {worst['onnx']:.2e}")
    for name in ("yamnet.onnx", "yamnet_class_map.csv", "NOTICE"):
        print(f"{sha256(args.out / name)}  {name}")
    if max(worst.values()) >= TOLERANCE:
        raise ExportError(f"A difference is not below {TOLERANCE:g}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
