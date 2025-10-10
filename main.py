"""A compact top-down racing game with responsive controls."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

from tracks import TrackDefinition, load_tracks

Vec2 = pygame.math.Vector2


def format_time(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "--:--.---"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes:02}:{remainder:06.3f}"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for idx in range(len(polygon)):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        )
        if intersects:
            inside = not inside
    return inside


def segment_intersection(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool:
    def cross(a: Vec2, b: Vec2) -> float:
        return a.x * b.y - a.y * b.x

    r = p2 - p1
    s = q2 - q1
    denom = cross(r, s)
    if abs(denom) < 1e-9:
        return False
    t = cross(q1 - p1, s) / denom
    u = cross(q1 - p1, r) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def distance_point_to_segment(point: Vec2, start: Vec2, end: Vec2) -> float:
    seg = end - start
    seg_len_sq = seg.length_squared()
    if seg_len_sq == 0:
        return (point - start).length()
    t = clamp((point - start).dot(seg) / seg_len_sq, 0.0, 1.0)
    projection = start + seg * t
    return (point - projection).length()


def expand_line(start: Vec2, end: Vec2, width: float) -> Tuple[Vec2, Vec2]:
    if width <= 0:
        return start, end
    center = (start + end) * 0.5
    direction = end - start
    if direction.length_squared() == 0:
        direction = Vec2(1, 0)
    direction = direction.normalize()
    half = direction * (width * 0.5)
    return center - half, center + half


class Car:
    LENGTH = 48
    WIDTH = 26

    def __init__(self, spawn_position: Tuple[float, float], heading: float) -> None:
        self.max_speed = 440.0
        self.max_reverse_speed = -160.0
        self.acceleration = 320.0
        self.reverse_acceleration = 220.0
        self.brake_force = 480.0
        self.drag = 1.8
        self.turn_speed = 2.9
        self.reset(spawn_position, heading)

    def reset(self, spawn_position: Tuple[float, float], heading: float) -> None:
        self.position = Vec2(spawn_position)
        self.prev_position = Vec2(spawn_position)
        self.heading = heading
        self.speed = 0.0
        self.velocity = Vec2()

    @property
    def forward(self) -> Vec2:
        return Vec2(math.cos(self.heading), math.sin(self.heading))

    @property
    def right(self) -> Vec2:
        fwd = self.forward
        return Vec2(-fwd.y, fwd.x)

    def get_corners(self) -> List[Vec2]:
        forward = self.forward
        right = self.right
        half_length = self.LENGTH * 0.5
        half_width = self.WIDTH * 0.5
        center = self.position
        return [
            center + forward * half_length + right * half_width,
            center + forward * half_length - right * half_width,
            center - forward * half_length - right * half_width,
            center - forward * half_length + right * half_width,
        ]

    def update(self, dt: float, inputs: Dict[str, float]) -> None:
        self.prev_position = self.position.copy()
        throttle = clamp(inputs.get("throttle", 0.0), -1.0, 1.0)
        brake = clamp(inputs.get("brake", 0.0), 0.0, 1.0)
        steer = clamp(inputs.get("steer", 0.0), -1.0, 1.0)

        if throttle > 0.0:
            self.speed += self.acceleration * throttle * dt
        elif throttle < 0.0:
            self.speed += self.reverse_acceleration * throttle * dt

        if brake > 0.0:
            if self.speed > 0.0:
                self.speed = max(0.0, self.speed - self.brake_force * brake * dt)
            else:
                self.speed = min(0.0, self.speed + self.brake_force * brake * dt)

        self.speed -= self.speed * self.drag * dt
        if abs(self.speed) < 2.0:
            self.speed = 0.0

        self.speed = clamp(self.speed, self.max_reverse_speed, self.max_speed)

        if abs(self.speed) > 1.0 and abs(steer) > 0.0:
            speed_factor = clamp(abs(self.speed) / self.max_speed, 0.35, 1.0)
            if self.speed < 0.0:
                speed_factor *= -1.0
            self.heading += steer * self.turn_speed * speed_factor * dt

        self.velocity = self.forward * self.speed
        self.position += self.velocity * dt


class Track:
    def __init__(self, definition: TrackDefinition) -> None:
        self.definition = definition
        self.road_polygons: List[Sequence[Tuple[float, float]]] = []
        self.blockers: List[Sequence[Tuple[float, float]]] = []
        for surface in definition.surfaces:
            surface_type = surface.get("type", "").lower()
            polygon = surface["polygon"]
            if surface_type == "asphalt":
                self.road_polygons.append(polygon)
            else:
                self.blockers.append(polygon)
        self.walls = [(Vec2(start), Vec2(end)) for start, end in definition.walls]

    def _line_points(self, line: Dict[str, object]) -> Tuple[Vec2, Vec2]:
        start, end = line["points"]
        width = float(line.get("width", 220.0))
        return expand_line(Vec2(start), Vec2(end), width)

    def start_line_points(self) -> Tuple[Vec2, Vec2]:
        return self._line_points(self.definition.start_line)

    def checkpoint_points(self) -> List[Tuple[Vec2, Vec2]]:
        return [self._line_points(checkpoint) for checkpoint in self.definition.checkpoints]

    def is_driveable(self, point: Vec2) -> bool:
        px, py = point.x, point.y
        on_road = any(point_in_polygon((px, py), polygon) for polygon in self.road_polygons)
        if not on_road:
            return False
        for polygon in self.blockers:
            if point_in_polygon((px, py), polygon):
                return False
        return True

    def keep_car_on_track(self, car: Car) -> None:
        corners = car.get_corners()
        if all(self.is_driveable(corner) for corner in corners):
            return
        car.position = car.prev_position.copy()
        car.speed *= -0.3

    def draw(self, screen: pygame.Surface) -> None:
        for surface in self.definition.surfaces:
            color = surface.get("color", (80, 80, 80))
            pygame.draw.polygon(screen, color, surface["polygon"])
            pygame.draw.polygon(screen, (20, 20, 24), surface["polygon"], 2)

        for start, end in self.walls:
            pygame.draw.line(screen, (200, 200, 220), start, end, 3)

        start_a, start_b = self.start_line_points()
        pygame.draw.line(screen, (230, 230, 90), start_a, start_b, 8)

        for checkpoint in self.definition.checkpoints:
            a, b = self._line_points(checkpoint)
            pygame.draw.line(screen, (170, 130, 240), a, b, 6)


class LapTimer:
    def __init__(self, track: Track) -> None:
        self.track = track
        self.total_checkpoints = len(track.definition.checkpoints)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.current_time = 0.0
        self.last_lap: Optional[float] = None
        self.best_lap: Optional[float] = None
        self.next_checkpoint = 0
        self.collected = 0
        self.waiting_to_leave_start = True

    def update(self, dt: float, prev_pos: Vec2, current_pos: Vec2, forward: Vec2) -> None:
        if self.active:
            self.current_time += dt

        start_line = self.track.definition.start_line
        start_a, start_b = self.track.start_line_points()

        if self.waiting_to_leave_start:
            distance = distance_point_to_segment(current_pos, start_a, start_b)
            if distance > 24.0:
                self.waiting_to_leave_start = False

        if not self.waiting_to_leave_start and segment_intersection(prev_pos, current_pos, start_a, start_b):
            direction = Vec2(start_line.get("direction", (1.0, 0.0)))
            if direction.length_squared() == 0 or forward.dot(direction.normalize()) > 0.0:
                if self.active and (self.total_checkpoints == 0 or self.collected == self.total_checkpoints):
                    self.last_lap = self.current_time
                    if self.best_lap is None or (self.last_lap and self.last_lap < self.best_lap):
                        self.best_lap = self.last_lap
                self.active = True
                self.current_time = 0.0
                self.next_checkpoint = 0
                self.collected = 0
                self.waiting_to_leave_start = True

        if not self.active or self.total_checkpoints == 0:
            return

        if self.next_checkpoint >= self.total_checkpoints:
            return

        checkpoint = self.track.definition.checkpoints[self.next_checkpoint]
        cp_a, cp_b = self.track._line_points(checkpoint)
        if segment_intersection(prev_pos, current_pos, cp_a, cp_b):
            direction = Vec2(checkpoint.get("direction", (1.0, 0.0)))
            if direction.length_squared() == 0 or forward.dot(direction.normalize()) > -0.05:
                self.next_checkpoint = (self.next_checkpoint + 1) % self.total_checkpoints
                self.collected = min(self.total_checkpoints, self.collected + 1)


class Game:
    def __init__(self) -> None:
        pygame.init()
        self.size = (1280, 800)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Vector Racer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24)
        self.small_font = pygame.font.SysFont("Consolas", 18)

        self.tracks = load_tracks()
        self.track_index = 0
        self.track = Track(self.tracks[self.track_index])
        self.car = Car(self.track.definition.spawn_position, self.track.definition.spawn_heading)
        self.timer = LapTimer(self.track)
        self.running = True

    def switch_track(self, direction: int) -> None:
        self.track_index = (self.track_index + direction) % len(self.tracks)
        self.track = Track(self.tracks[self.track_index])
        self.car.reset(self.track.definition.spawn_position, self.track.definition.spawn_heading)
        self.timer = LapTimer(self.track)

    def gather_inputs(self) -> Dict[str, float]:
        keys = pygame.key.get_pressed()
        throttle = 0.0
        brake = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if self.car.speed > 80.0:
                brake = 1.0
            else:
                throttle = -1.0
        if keys[pygame.K_SPACE]:
            brake = 1.0
            throttle = min(throttle, 0.0)
        steer = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer += 1.0
        return {"throttle": throttle, "brake": brake, "steer": steer}

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_TAB:
                    self.switch_track(1)

    def update(self, dt: float) -> None:
        inputs = self.gather_inputs()
        self.car.update(dt, inputs)
        self.track.keep_car_on_track(self.car)
        self.timer.update(dt, self.car.prev_position, self.car.position, self.car.forward)

    def draw(self) -> None:
        self.screen.fill((18, 18, 26))
        self.track.draw(self.screen)
        self.draw_car()
        self.draw_hud()
        pygame.display.flip()

    def draw_car(self) -> None:
        corners = self.car.get_corners()
        pygame.draw.polygon(self.screen, (230, 90, 70), [(c.x, c.y) for c in corners])
        pygame.draw.polygon(self.screen, (40, 20, 20), [(c.x, c.y) for c in corners], 2)
        nose = (corners[0] + corners[1]) * 0.5
        arrow = nose + self.car.forward * 20
        pygame.draw.line(self.screen, (255, 240, 190), nose, arrow, 3)

    def draw_hud(self) -> None:
        speed = abs(self.car.speed)
        lap_time = self.timer.current_time if self.timer.active else None
        best_time = self.timer.best_lap
        last_time = self.timer.last_lap

        hud_lines = [
            f"Track: {self.track.definition.name}",
            f"Speed: {speed:05.1f} px/s",
            f"Lap: {format_time(lap_time)}",
            f"Best: {format_time(best_time)}",
        ]
        if last_time is not None:
            hud_lines.append(f"Last: {format_time(last_time)}")

        for index, line in enumerate(hud_lines):
            text = self.font.render(line, True, (240, 240, 240))
            self.screen.blit(text, (20, 20 + index * 28))

        instructions = [
            "W/UP accelerate",
            "S/DOWN brake & reverse",
            "A/D or arrows to steer",
            "Space for hard brake",
            "Tab to change track",
        ]
        for index, line in enumerate(instructions):
            text = self.small_font.render(line, True, (200, 200, 220))
            self.screen.blit(text, (20, self.size[1] - 20 - (len(instructions) - index) * 20))


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
