"""Extract the convex hulls of the arms' collision meshes from walkie-ros's walkie.xml.

    pixi run python scripts/extract_arm_collision_hulls.py <walkie-ros>/description

tests/test_door_clearance.py measures how near the arms come to the open door
leaf of walkie-ros's chula_pat14_6 world without MuJoCo, which this
environment does not have, so it reads the arms' collision shapes from
tests/data/arm_collision_hulls.npz. MuJoCo collides a mesh geom as the convex
hull of its vertices, so the hull's vertices are the whole collision shape:
keeping only them changes no distance. It shrinks the 24 meshes from 4970 to
942 vertices: each wrist camera bracket, a visual mesh, from 1988 to 150, and
each finger from 134 to 46. The link meshes are convex already.

Reads mujoco/walkie.xml of a description checkout and the STL files it
names, never changing them. For every body of both arms, from
openarm_left_link0 and openarm_right_link0 down, it takes the body's one
collision geom (contype or conaffinity not 0), scales its mesh and places it
in the body's frame by the geom's pos and quat, and writes the hull's
vertices, float32 meters, under the body's name to --output (by default
tests/data/arm_collision_hulls.npz).

The committed file comes from the description at 526c976, the submodule
commit of walkie-ros's feat/chula-pat14-6-simulation (adb9725), the branch
with the world. To regenerate it, check out adb9725 in a walkie-ros clone,
run `git submodule update --init description` there, and pass the clone's
description directory. scipy computes the hulls; the speech packages bring it
into the environment. The test itself needs only numpy.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "tests" / "data" / "arm_collision_hulls.npz"
MODEL_PATH = Path("mujoco") / "walkie.xml"

# The first body of each arm. Every body inside it moves with the arm.
ARM_ROOTS = ("openarm_left_link0", "openarm_right_link0")
# Geom attributes that place a mesh other than by pos and quat.
PLACEMENT_ATTRIBUTES = ("euler", "axisangle", "xyaxes", "zaxis", "fromto")
# The mesh attributes the script applies. Any other one could change the
# shape, so it stops the script.
MESH_ATTRIBUTES = {"name", "content_type", "file", "scale"}
# A binary STL file: an 80-byte header, the triangle count, then the triangles.
STL_HEADER = 84
STL_TRIANGLE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)
# zip entries carry a time stamp. A fixed one keeps the file byte for byte the
# same when the meshes are, so a regeneration shows in git only if they changed.
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class HullError(Exception):
    """The description does not have the shape this script expects."""


def read_floats(element: ET.Element, attribute: str, default: str, count: int) -> np.ndarray:
    text = element.get(attribute, default)
    try:
        values = np.array([float(value) for value in text.split()])
    except ValueError:
        raise HullError(f"{element.get('name')}: {attribute}={text!r} is not numbers.") from None
    if len(values) != count or not np.isfinite(values).all():
        raise HullError(f"{element.get('name')}: {attribute}={text!r} is not {count} numbers.")
    return values


def quat_rotation(quat: np.ndarray) -> np.ndarray:
    """The rotation matrix of a quaternion (w, x, y, z), normalized first as MuJoCo does."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_stl(path: Path) -> np.ndarray:
    """The triangle corners of a binary STL file, one row each, in the file's units."""
    data = path.read_bytes()
    count = int.from_bytes(data[80:STL_HEADER], "little")
    if len(data) != STL_HEADER + count * STL_TRIANGLE.itemsize:
        raise HullError(f"{path} is not a binary STL file.")
    triangles = np.frombuffer(data, STL_TRIANGLE, count=count, offset=STL_HEADER)
    return triangles["vertices"].reshape(-1, 3).astype(float)


def collides(geom: ET.Element) -> bool:
    """MuJoCo's contype and conaffinity default to 1, and a geom with both 0 never collides."""
    return int(geom.get("contype", "1")) != 0 or int(geom.get("conaffinity", "1")) != 0


def body_hulls(description: Path) -> dict[str, np.ndarray]:
    """The hull vertices of each arm body's collision mesh, in the body's frame."""
    model_path = description / MODEL_PATH
    root = ET.parse(model_path).getroot()
    # A class could set any geom or mesh attribute, and the script reads only
    # the elements' own attributes.
    if root.find(".//default") is not None:
        raise HullError(f"{model_path} has <default> classes, which the script does not apply.")
    compiler = root.find("compiler")
    mesh_dir = model_path.parent / ("" if compiler is None else compiler.get("meshdir", ""))
    meshes = {mesh.get("name"): mesh for mesh in root.iter("mesh")}
    hulls: dict[str, np.ndarray] = {}
    for arm in ARM_ROOTS:
        top = root.find(f".//body[@name='{arm}']")
        if top is None:
            raise HullError(f"{model_path} has no body {arm}.")
        for body in top.iter("body"):
            name = body.get("name")
            geoms = [geom for geom in body.findall("geom") if collides(geom)]
            if not geoms:
                continue
            if len(geoms) > 1 or geoms[0].get("type") != "mesh":
                found = [(geom.get("name"), geom.get("type")) for geom in geoms]
                raise HullError(f"{name}: expected one collision geom, a mesh; found {found}.")
            geom = geoms[0]
            placement = [key for key in PLACEMENT_ATTRIBUTES if key in geom.attrib]
            if placement:
                raise HullError(
                    f"{geom.get('name')}: {placement} are not applied, only pos and quat."
                )
            mesh = meshes.get(geom.get("mesh"))
            if mesh is None or mesh.get("file") is None:
                raise HullError(f"{geom.get('name')}: no mesh file for {geom.get('mesh')!r}.")
            unknown = sorted(set(mesh.attrib) - MESH_ATTRIBUTES)
            if unknown:
                raise HullError(f"mesh {mesh.get('name')}: {unknown} are not applied.")
            corners = read_stl(mesh_dir / mesh.get("file")) * read_floats(mesh, "scale", "1 1 1", 3)
            rotation = quat_rotation(read_floats(geom, "quat", "1 0 0 0", 4))
            placed = read_floats(geom, "pos", "0 0 0", 3) + corners @ rotation.T
            hulls[name] = placed[np.sort(ConvexHull(placed).vertices)].astype(np.float32)
    return hulls


def write_hulls(hulls: dict[str, np.ndarray], output: Path) -> None:
    """Write the hulls as an archive that np.load reads, like np.savez_compressed."""
    with zipfile.ZipFile(output, "w") as archive:
        for name, vertices in hulls.items():
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_TIME)
            entry.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(entry, "w") as f:
                np.lib.format.write_array(f, vertices, allow_pickle=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "description", type=Path, help="walkie-ros's description directory, read only"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="the file to write (%(default)s)"
    )
    args = parser.parse_args()
    try:
        hulls = body_hulls(args.description)
    except (OSError, ET.ParseError, ValueError, HullError) as err:
        print(f"extract_arm_collision_hulls: {err}", file=sys.stderr)
        return 1
    write_hulls(hulls, args.output)
    count = sum(len(vertices) for vertices in hulls.values())
    print(f"Wrote {args.output}: {len(hulls)} bodies, {count} hull vertices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
