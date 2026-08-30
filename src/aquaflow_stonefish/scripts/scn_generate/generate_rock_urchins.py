#!/usr/bin/env python3
"""Generate a reproducible rock-associated sea-urchin layout in a Stonefish SCN."""

from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path


BEGIN = "\t<!-- BEGIN AUTO-GENERATED ROCK URCHINS -->"
END = "\t<!-- END AUTO-GENERATED ROCK URCHINS -->"


@dataclass(frozen=True)
class Rock:
    name: str
    mesh: str
    scale: float
    x: float
    y: float
    z: float
    yaw: float
    radius_x: float
    radius_y: float


ROCKS = (
    # These values mirror the active rock definitions in Rock_SeaUrchin.scn.
    # Commented-out RockSurface and Rock5 definitions are intentionally excluded.
    Rock("RockSurface02", "Rock2/Rock2.obj", 0.02, -1.20, 1.20, 3.866, -0.35, 0.82, 1.38),
    Rock("RockSurface03", "Rock3/source/rock_17.obj", 3.0, 1.25, -2.00, 3.922, 1.18, 0.78, 0.86),
    Rock("RockSurface04", "Rock4/source/Rock4.obj", 0.36, -4.25, -1.50, 3.922, -0.62, 1.08, 0.98),
    Rock("RockSurface05", "Rock2/Rock2.obj", 0.02, 4.95, -0.50, 3.866, -1.57, 0.82, 1.38),
    Rock("RockSurface06", "Rock3/source/rock_17.obj", 3.0, -3.85, 2.35, 3.840, 0.00, 0.78, 0.86),
    Rock("RockSurface07", "Rock4/source/Rock4.obj", 0.36, 1.85, 1.35, 3.840, 1.57, 1.08, 0.98),
)


URCHINS = {
    "sea": {
        "mesh": "SeaUrchin/Mesh_SeaUrchin.obj",
        "look": "sea_urchin",
        "rpy": "3.1415926536 0 0",
        "scale": (0.016, 0.022),
        # Rx(pi): world Z = -mesh Z; -min(mesh Z) = 3.912245.
        "bottom_factor": 3.912245,
        "radius_factor": 3.28,
    },
    "red": {
        "mesh": "RedSeaUrchin/RedSeaUrchin.obj",
        "look": "red_sea_urchin",
        "rpy": "-1.5707963268 0 0",
        "scale": (1.80, 2.16),
        # Rx(-pi/2): world Z = -mesh Y; min(mesh Y) is zero.
        "bottom_factor": 0.0,
        "radius_factor": 0.0375,
    },
    "purple": {
        "mesh": "PurpleSeaUrchin/source/PurpleSeaUrchin_Cluster_Med.obj",
        "look": "purple_sea_urchin",
        "rpy": "-1.5707963268 0 0",
        "scale": (0.09, 0.09),
        # One PurpleSeaUrchin asset contains several individual urchins.
        "bottom_factor": 0.4631021023,
        "radius_factor": 7.55,
    },
}

NAME_PREFIX = {"sea": "SeaUrchin", "red": "RedSeaUrchin", "purple": "PurpleSeaUrchin"}
TYPE_LABEL = {"sea": "sea urchin", "red": "red sea urchin", "purple": "purple sea-urchin cluster"}


def load_mesh(mesh_path: Path):
    vertices = []
    triangles = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append(tuple(map(float, fields[1:4])))
            elif line.startswith("f "):
                indices = []
                for field in line.split()[1:]:
                    index = int(field.split("/", 1)[0])
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for i in range(1, len(indices) - 1):
                    triangles.append((vertices[indices[0]], vertices[indices[i]], vertices[indices[i + 1]]))
    return triangles


def surface_height(triangles, rock: Rock, world_x: float, world_y: float):
    """Return the upper rock surface in NED Z using vertical ray projection."""
    dx = world_x - rock.x
    dy = world_y - rock.y
    cosine = math.cos(rock.yaw)
    sine = math.sin(rock.yaw)
    mesh_x = (cosine * dx + sine * dy) / rock.scale
    mesh_z = (-sine * dx + cosine * dy) / rock.scale
    mesh_ys = []
    for a, b, c in triangles:
        x1, z1 = a[0], a[2]
        x2, z2 = b[0], b[2]
        x3, z3 = c[0], c[2]
        denominator = (z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3)
        if abs(denominator) < 1e-12:
            continue
        u = ((z2 - z3) * (mesh_x - x3) + (x3 - x2) * (mesh_z - z3)) / denominator
        v = ((z3 - z1) * (mesh_x - x3) + (x1 - x3) * (mesh_z - z3)) / denominator
        w = 1.0 - u - v
        if u >= -1e-8 and v >= -1e-8 and w >= -1e-8:
            mesh_ys.append(u * a[1] + v * b[1] + w * c[1])
    if not mesh_ys:
        return None
    # The rocks use Rx(-pi/2), so larger mesh Y is visually higher (smaller NED Z).
    return rock.z - max(mesh_ys) * rock.scale


def sample_on_rock(rng, rock: Rock, triangles, occupied, radius):
    for _ in range(400):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        distance = math.sqrt(rng.uniform(0.0, 1.0))
        local_x = math.cos(angle) * rock.radius_x * distance
        local_y = math.sin(angle) * rock.radius_y * distance
        cosine = math.cos(rock.yaw)
        sine = math.sin(rock.yaw)
        x = rock.x + cosine * local_x - sine * local_y
        y = rock.y + sine * local_x + cosine * local_y
        z = surface_height(triangles, rock, x, y)
        if z is None:
            continue
        if all(math.hypot(x - ox, y - oy) >= radius + other_radius for ox, oy, other_radius in occupied):
            occupied.append((x, y, radius))
            return x, y, z
    raise RuntimeError(f"Could not find a free surface point on {rock.name}")


def sample_beside_rock(rng, rock: Rock, occupied, radius):
    for _ in range(200):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        edge = max(rock.radius_x, rock.radius_y)
        distance = edge + rng.uniform(0.18, 0.38)
        x = rock.x + math.cos(angle) * distance
        y = rock.y + math.sin(angle) * distance
        if not (-6.6 < x < 6.6 and -3.6 < y < 3.6):
            continue
        if all(math.hypot(x - ox, y - oy) >= radius + other_radius for ox, oy, other_radius in occupied):
            occupied.append((x, y, radius))
            return x, y, 4.0
    raise RuntimeError(f"Could not find a free point beside {rock.name}")


def static_xml(name, kind, scale, x, y, contact_z, yaw, location):
    spec = URCHINS[kind]
    origin_z = contact_z - spec["bottom_factor"] * scale
    return (
        f'\t<static name="{name}" type="model">\n'
        f'\t\t<!-- {TYPE_LABEL[kind]}: {location}. -->\n'
        f'\t\t<physical><mesh filename="$(find aquaflow_stonefish)/scenarios/models/{spec["mesh"]}" scale="{scale:.5f}"/>'
        f'<origin xyz="0 0 0" rpy="{spec["rpy"]}"/></physical>\n'
        f'\t\t<material name="Neutral"/><look name="{spec["look"]}"/>\n'
        f'\t\t<world_transform xyz="{x:.4f} {y:.4f} {origin_z:.4f}" rpy="0.0 0.0 {yaw:.4f}"/>\n'
        f'\t</static>'
    )


def generate(package_dir: Path, seed: int, on_rock_count: int, purple_count: int, beside_count: int):
    rng = random.Random(seed)
    triangles = {
        rock.name: load_mesh(package_dir / "scenarios" / "models" / rock.mesh)
        for rock in ROCKS
    }
    occupied = []
    output = [BEGIN, f'\t<!-- seed={seed}; on-rock={on_rock_count + purple_count} models (including {purple_count} purple clusters); beside={beside_count}. -->']

    # Large cluster assets are preferentially placed on the broadest rocks.
    purple_rocks = (ROCKS[2], ROCKS[5], ROCKS[1], ROCKS[4], ROCKS[0], ROCKS[3])
    for index in range(1, purple_count + 1):
        rock = purple_rocks[(index - 1) % len(purple_rocks)]
        kind = "purple"
        scale = 0.09  # exactly twice the previous 0.045 scene scale
        radius = URCHINS[kind]["radius_factor"] * scale * 0.72
        x, y, z = sample_on_rock(rng, rock, triangles[rock.name], occupied, radius)
        output.append(static_xml(f"PurpleSeaUrchinOnRock{index:02d}", kind, scale, x, y, z, rng.uniform(-math.pi, math.pi), f"on {rock.name}"))

    kinds = ["sea", "red"]
    for index in range(1, on_rock_count + 1):
        kind = kinds[(index - 1) % len(kinds)]
        spec = URCHINS[kind]
        scale = rng.uniform(*spec["scale"])
        radius = spec["radius_factor"] * scale
        # Cycle through every rock, then shuffle the cycle order with the seed.
        rock = ROCKS[(index - 1 + rng.randrange(len(ROCKS))) % len(ROCKS)]
        x, y, z = sample_on_rock(rng, rock, triangles[rock.name], occupied, radius)
        output.append(static_xml(f"{NAME_PREFIX[kind]}OnRock{index:02d}", kind, scale, x, y, z, rng.uniform(-math.pi, math.pi), f"on {rock.name}"))

    beside_kinds = ("sea", "red", "sea", "red")
    for index in range(1, beside_count + 1):
        kind = beside_kinds[(index - 1) % len(beside_kinds)]
        spec = URCHINS[kind]
        scale = rng.uniform(*spec["scale"])
        radius = spec["radius_factor"] * scale
        rock = ROCKS[(index * 2 - 1) % len(ROCKS)]
        x, y, z = sample_beside_rock(rng, rock, occupied, radius)
        output.append(static_xml(f"{NAME_PREFIX[kind]}BesideRock{index:02d}", kind, scale, x, y, z, rng.uniform(-math.pi, math.pi), f"beside {rock.name}"))

    output.append(END)
    return "\n".join(output)


def update_scenario(scenario: Path, generated: str):
    text = scenario.read_text(encoding="utf-8")
    marked = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if marked.search(text):
        text = marked.sub(generated, text)
    else:
        # Remove the previous hand-authored urchin instances, but retain all rocks.
        old_urchins = re.compile(
            r'\s*<static name="(?:SeaUrchin|RedSeaUrchin|PurpleSeaUrchin)[^"]*" type="model">.*?</static>',
            re.DOTALL,
        )
        text = old_urchins.sub("", text)
        include_at = text.find('\n\t<include file="$(find aquaflow_stonefish)/scenarios/bluerov2_reference.scn">')
        if include_at < 0:
            raise RuntimeError("BlueROV2 include marker was not found in the scenario")
        text = text[:include_at] + "\n\n" + generated + text[include_at:]
    # Path.write_text gained the newline argument only in newer Python
    # versions.  ROS Noetic commonly runs Python 3.8, so use open() here.
    with scenario.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    package_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=package_dir / "scenarios" / "Rock_SeaUrchin.scn")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--on-rock", type=int, default=18, help="single ordinary/red urchin models on rocks")
    parser.add_argument("--purple", type=int, default=2, help="purple multi-urchin cluster models on rocks")
    parser.add_argument("--beside", type=int, default=4)
    args = parser.parse_args()
    if args.on_rock < 0 or args.purple < 0 or args.beside < 0:
        parser.error("counts must be non-negative")
    generated = generate(package_dir, args.seed, args.on_rock, args.purple, args.beside)
    update_scenario(args.scenario.resolve(), generated)
    print(f"Updated {args.scenario} with seed {args.seed}: {args.on_rock + args.purple} on-rock models ({args.purple} purple clusters), {args.beside} beside-rock models")


if __name__ == "__main__":
    main()
