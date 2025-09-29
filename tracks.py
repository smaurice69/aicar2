"""Track definitions for the vector-style racing game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]


def polyline_to_segments(points: Sequence[Point], *, closed: bool = True) -> List[Tuple[Point, Point]]:
    segments: List[Tuple[Point, Point]] = []
    for idx in range(len(points) - 1):
        segments.append((points[idx], points[idx + 1]))
    if closed and points:
        segments.append((points[-1], points[0]))
    return segments


@dataclass(frozen=True)
class TrackDefinition:
    track_id: str
    name: str
    spawn_position: Point
    spawn_heading: float
    start_line: Dict[str, object]
    checkpoints: List[Dict[str, object]]
    surfaces: List[Dict[str, object]]
    walls: List[Tuple[Point, Point]]
    minimap_bounds: Tuple[Point, Point]


def harbor_run() -> TrackDefinition:
    asphalt_loop = [
        (140, 620),
        (120, 520),
        (160, 340),
        (240, 240),
        (360, 180),
        (540, 140),
        (760, 120),
        (980, 140),
        (1160, 200),
        (1260, 300),
        (1280, 420),
        (1240, 560),
        (1120, 660),
        (960, 720),
        (760, 760),
        (560, 740),
        (380, 700),
        (260, 660),
    ]

    inner_harbor = [
        (420, 540),
        (520, 480),
        (620, 450),
        (760, 440),
        (880, 460),
        (960, 500),
        (980, 560),
        (940, 600),
        (860, 620),
        (740, 640),
        (600, 630),
        (480, 600),
    ]

    gravel_cut = [
        (300, 360),
        (380, 320),
        (480, 300),
        (560, 320),
        (600, 360),
        (520, 420),
        (400, 420),
    ]

    walls = []
    walls.extend(polyline_to_segments(asphalt_loop))
    walls.extend(polyline_to_segments(inner_harbor))

    start_line = {
        "points": ((360, 640), (520, 640)),
        "direction": (0.0, -1.0),
    }

    checkpoints = [
        {"points": ((860, 700), (860, 560)), "direction": (-1.0, -0.1)},
        {"points": ((980, 260), (840, 220)), "direction": (-0.8, 0.2)},
        {"points": ((240, 420), (340, 420)), "direction": (0.8, 0.0)},
    ]

    surfaces = [
        {"polygon": asphalt_loop, "type": "asphalt", "mu": 1.0, "color": (72, 78, 92)},
        {"polygon": inner_harbor, "type": "water", "mu": 0.2, "color": (35, 55, 90)},
        {"polygon": gravel_cut, "type": "gravel", "mu": 0.55, "color": (128, 106, 82)},
    ]

    return TrackDefinition(
        track_id="harbor_run",
        name="Harbor Run",
        spawn_position=(430, 680),
        spawn_heading=-1.55,
        start_line=start_line,
        checkpoints=checkpoints,
        surfaces=surfaces,
        walls=walls,
        minimap_bounds=((80, 120), (1320, 780)),
    )


def canyon_loop() -> TrackDefinition:
    outer = [
        (180, 560),
        (140, 460),
        (140, 340),
        (220, 220),
        (360, 160),
        (520, 140),
        (720, 160),
        (900, 220),
        (1080, 340),
        (1180, 460),
        (1180, 560),
        (1080, 660),
        (900, 740),
        (720, 780),
        (520, 780),
        (340, 740),
        (220, 660),
    ]

    inner = [
        (380, 560),
        (360, 500),
        (360, 420),
        (420, 340),
        (520, 300),
        (640, 300),
        (760, 320),
        (860, 360),
        (940, 420),
        (980, 480),
        (940, 540),
        (860, 600),
        (760, 640),
        (640, 660),
        (520, 660),
        (420, 620),
    ]

    gravel_shelf = [
        (220, 280),
        (320, 240),
        (420, 220),
        (520, 220),
        (600, 240),
        (640, 280),
        (540, 320),
        (420, 320),
        (300, 320),
    ]

    walls: List[Tuple[Point, Point]] = []
    walls.extend(polyline_to_segments(outer))
    walls.extend(polyline_to_segments(inner))

    start_line = {
        "points": ((520, 720), (680, 720)),
        "direction": (0.0, -1.0),
    }

    checkpoints = [
        {"points": ((960, 620), (840, 500)), "direction": (-0.6, -0.2)},
        {"points": ((960, 360), (840, 260)), "direction": (-0.6, 0.2)},
        {"points": ((320, 440), (440, 360)), "direction": (0.5, -0.1)},
    ]

    surfaces = [
        {"polygon": outer, "type": "asphalt", "mu": 1.0, "color": (78, 74, 88)},
        {"polygon": inner, "type": "canyon_floor", "mu": 0.15, "color": (52, 46, 58)},
        {"polygon": gravel_shelf, "type": "gravel", "mu": 0.6, "color": (140, 120, 96)},
    ]

    return TrackDefinition(
        track_id="canyon_loop",
        name="Canyon Loop",
        spawn_position=(600, 740),
        spawn_heading=-1.55,
        start_line=start_line,
        checkpoints=checkpoints,
        surfaces=surfaces,
        walls=walls,
        minimap_bounds=((120, 180), (1220, 820)),
    )


def industrial_sprint() -> TrackDefinition:
    outer = [
        (200, 680),
        (160, 520),
        (160, 360),
        (200, 240),
        (320, 160),
        (520, 120),
        (760, 120),
        (960, 160),
        (1100, 240),
        (1180, 360),
        (1180, 520),
        (1100, 660),
        (960, 740),
        (760, 780),
        (520, 780),
        (320, 740),
    ]

    inner_block = [
        (420, 600),
        (420, 500),
        (520, 500),
        (520, 440),
        (620, 440),
        (620, 380),
        (720, 380),
        (720, 320),
        (820, 320),
        (820, 420),
        (720, 420),
        (720, 520),
        (620, 520),
        (620, 580),
        (520, 580),
        (520, 640),
        (420, 640),
    ]

    gravel_patches = [
        [
            (260, 560),
            (340, 560),
            (340, 640),
            (260, 640),
        ],
        [
            (880, 560),
            (960, 560),
            (960, 640),
            (880, 640),
        ],
        [
            (880, 260),
            (960, 260),
            (960, 340),
            (880, 340),
        ],
    ]

    walls: List[Tuple[Point, Point]] = []
    walls.extend(polyline_to_segments(outer))
    walls.extend(polyline_to_segments(inner_block))

    start_line = {
        "points": ((500, 720), (660, 720)),
        "direction": (0.0, -1.0),
    }

    checkpoints = [
        {"points": ((980, 680), (980, 540)), "direction": (-1.0, -0.1)},
        {"points": ((1080, 360), (920, 360)), "direction": (-0.9, 0.0)},
        {"points": ((320, 320), (480, 320)), "direction": (0.8, 0.2)},
    ]

    surfaces: List[Dict[str, object]] = [
        {"polygon": outer, "type": "asphalt", "mu": 0.95, "color": (68, 68, 74)},
        {"polygon": inner_block, "type": "building", "mu": 0.1, "color": (40, 42, 48)},
    ]

    for patch in gravel_patches:
        surfaces.append({"polygon": patch, "type": "gravel", "mu": 0.55, "color": (134, 116, 90)})

    return TrackDefinition(
        track_id="industrial_sprint",
        name="Industrial Sprint",
        spawn_position=(580, 740),
        spawn_heading=-1.55,
        start_line=start_line,
        checkpoints=checkpoints,
        surfaces=surfaces,
        walls=walls,
        minimap_bounds=((140, 160), (1220, 820)),
    )


def all_tracks() -> List[TrackDefinition]:
    return [harbor_run(), canyon_loop(), industrial_sprint()]

