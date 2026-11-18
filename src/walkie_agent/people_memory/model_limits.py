"""The largest cosine distance at which a stored vector names a person, per vector model.

IdentityConfig (identity.py) takes its default face and appearance limits from here.
A model without a row is never matched, so an unknown model cannot produce a confident
wrong name.

A new face or re-identification model needs a row here, keyed by the vector model name
that walkie-vision publishes. Three repositories change together: walkie-vision's
provider for the model, a walkie-database migration that adds its vector table, and
this row.
"""

from collections.abc import Mapping
from types import MappingProxyType


def face_limits() -> Mapping[str, float]:
    return MappingProxyType(
        {
            # walkie-agent-v2's FACE_MATCH_THRESHOLD for the same ArcFace
            # weights, tuned on the robot.
            "arcface_w600k_r50": 0.4,
            # walkie-vision's fake face model (src/walkie_vision/fakes.py),
            # which oracle_perception and the tests publish. Hash vectors of
            # different inputs are nearly orthogonal, a distance near 1, so
            # 0.2 separates them.
            "fake_face": 0.2,
        }
    )


def appearance_limits() -> Mapping[str, float]:
    return MappingProxyType(
        {
            # From walkie-agent-v2's APPEARANCE_MATCH_THRESHOLD, which gated a
            # score fused from face and appearance rather than appearance
            # alone, so this value is still to be tuned on the robot.
            "osnet_x1_0_msmt17": 0.5,
            # walkie-vision's fake re-identification model, as for fake_face.
            "fake_reid": 0.2,
        }
    )
