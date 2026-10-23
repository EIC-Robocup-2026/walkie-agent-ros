"""Deterministic face and appearance vectors, the same as walkie-vision's fakes.

hash_vector is copied from walkie-vision's src/walkie_vision/fakes.py and must
stay identical to it: the tests here and oracle_perception publish vectors
that a fake walkie-vision would also make, so a vector stored while one runs
matches the same person seen while the other runs.

The model names are those of walkie-vision's FakeFaceRecognizer and
FakeReIdentifier. No walkie-database vector table has a fake_ name, so these
vectors can never be stored next to a real model's.
"""

import hashlib

import numpy as np

# The face_model of walkie-vision's FakeFaceRecognizer.
FAKE_FACE_MODEL = "fake_face"
# The appearance_model of walkie-vision's FakeReIdentifier.
FAKE_REID_MODEL = "fake_reid"


def hash_vector(data: bytes | str, dim: int) -> np.ndarray:
    """Return a unit-length float32 vector of length dim that depends only on data."""
    if isinstance(data, str):
        data = data.encode()
    seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)
