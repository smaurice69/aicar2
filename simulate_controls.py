"""Simple control simulation helpers for the vector racer car physics.

The module runs a few deterministic scenarios so that the driving feel can be
evaluated without launching the pygame window.  They are not meant to be
exhaustive physics tests, but they provide quick feedback when tweaking the
control parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pygame

from main import Car

@dataclass
class SimulationResult:
    label: str
    duration: float
    final_speed: float
    distance_travelled: float
    heading_deg: float


def _run(car: Car, inputs: Dict[str, float], duration: float, surface_mu: float = 1.0, dt: float = 1 / 120) -> None:
    steps = int(duration / dt)
    for _ in range(steps):
        car.update(dt, inputs, surface_mu)


def accelerate_from_rest(duration: float = 4.0) -> SimulationResult:
    car = Car((0.0, 0.0), 0.0)
    _run(car, {"throttle": 1.0, "steer": 0.0, "brake": 0.0}, duration)
    speed = car.state.velocity.length() * 3.6
    distance = car.state.position.length() / 20.0
    heading = car.state.heading
    return SimulationResult("Full throttle", duration, speed, distance, heading * 180.0 / 3.141592653589793)


def high_speed_corner(duration: float = 5.0) -> SimulationResult:
    car = Car((0.0, 0.0), 0.0)
    _run(car, {"throttle": 1.0, "steer": 0.0, "brake": 0.0}, duration * 0.4)
    _run(car, {"throttle": 0.75, "steer": 0.8, "brake": 0.0}, duration * 0.6)
    speed = car.state.velocity.length() * 3.6
    distance = car.state.position.length() / 20.0
    heading = car.state.heading
    return SimulationResult("Sweeping corner", duration, speed, distance, heading * 180.0 / 3.141592653589793)


def braking_test(duration: float = 2.0) -> SimulationResult:
    car = Car((0.0, 0.0), 0.0)
    _run(car, {"throttle": 1.0, "steer": 0.0, "brake": 0.0}, 2.5)
    _run(car, {"throttle": 0.0, "steer": 0.0, "brake": 1.0}, duration)
    speed = car.state.velocity.length() * 3.6
    distance = car.state.position.length() / 20.0
    heading = car.state.heading
    return SimulationResult("Threshold brake", duration, speed, distance, heading * 180.0 / 3.141592653589793)


def handbrake_turn(duration: float = 3.0) -> SimulationResult:
    car = Car((0.0, 0.0), 0.0)
    _run(car, {"throttle": 1.0, "steer": 0.0, "brake": 0.0}, 1.5)
    _run(car, {"throttle": 0.4, "steer": 1.0, "brake": 0.0, "handbrake": True}, duration)
    speed = car.state.velocity.length() * 3.6
    distance = car.state.position.length() / 20.0
    heading = car.state.heading
    return SimulationResult("Handbrake flick", duration, speed, distance, heading * 180.0 / 3.141592653589793)


def main() -> None:
    scenarios = [
        accelerate_from_rest(),
        high_speed_corner(),
        braking_test(),
        handbrake_turn(),
    ]
    for result in scenarios:
        print(
            f"{result.label:18s} | duration: {result.duration:4.1f}s | "
            f"speed: {result.final_speed:6.1f} km/h | "
            f"distance: {result.distance_travelled:5.2f} m | "
            f"heading: {result.heading_deg:7.2f}°"
        )


if __name__ == "__main__":
    main()

