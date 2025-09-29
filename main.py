"""Top-down vector racing game implemented with pygame."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

from tracks import TrackDefinition, all_tracks

Vec2 = pygame.math.Vector2


def format_time(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "--:--.---"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02}:{secs:06.3f}"


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for idx in range(len(polygon)):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        ):
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


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def distance_point_to_segment(point: Vec2, seg_start: Vec2, seg_end: Vec2) -> float:
    seg_vec = seg_end - seg_start
    seg_len_sq = seg_vec.length_squared()
    if seg_len_sq == 0:
        return (point - seg_start).length()
    t = max(0.0, min(1.0, (point - seg_start).dot(seg_vec) / seg_len_sq))
    projection = seg_start + seg_vec * t
    return (point - projection).length()


@dataclass
class CarState:
    position: Vec2
    velocity: Vec2
    heading: float
    angular_velocity: float


class Car:
    LENGTH = 3.8
    WIDTH = 1.8

    def __init__(self, spawn_pos: Tuple[float, float], spawn_heading: float) -> None:
        self.state = CarState(position=Vec2(spawn_pos), velocity=Vec2(0, 0), heading=spawn_heading, angular_velocity=0.0)
        self.prev_position = Vec2(spawn_pos)
        self.mass = 1100.0
        self.engine_force = 5800.0
        self.brake_force = 7600.0
        self.reverse_force = 3200.0
        self.drag_coefficient = 0.32
        self.rolling_resistance = 5.0
        self.lateral_grip = 8.5
        self.max_speed = 22.2  # ~80 km/h
        self.max_reverse_speed = -5.0
        self.max_steer = math.radians(32)
        self.steer_speed = math.radians(240)
        self.steer_angle = 0.0

    @property
    def forward(self) -> Vec2:
        return Vec2(math.cos(self.state.heading), math.sin(self.state.heading))

    @property
    def right(self) -> Vec2:
        fwd = self.forward
        return Vec2(-fwd.y, fwd.x)

    def get_corners(self) -> List[Vec2]:
        half_length = self.LENGTH * 0.5 * 20
        half_width = self.WIDTH * 0.5 * 20
        forward = self.forward
        right = self.right
        center = self.state.position
        return [
            center + forward * half_length + right * half_width,
            center + forward * half_length - right * half_width,
            center - forward * half_length - right * half_width,
            center - forward * half_length + right * half_width,
        ]

    def update(self, dt: float, inputs: Dict[str, float], surface_mu: float) -> None:
        self.prev_position = self.state.position.copy()

        throttle = clamp(inputs.get("throttle", 0.0), -1.0, 1.0)
        brake = clamp(inputs.get("brake", 0.0), 0.0, 1.0)
        steer = clamp(inputs.get("steer", 0.0), -1.0, 1.0)
        handbrake = inputs.get("handbrake", False)

        forward = self.forward
        right = self.right

        velocity = self.state.velocity
        forward_speed = velocity.dot(forward)
        lateral_speed = velocity.dot(right)
        speed = velocity.length()

        # Steering dynamics
        speed_ratio = clamp(abs(forward_speed) / max(self.max_speed, 1e-3), 0.0, 1.0)
        dynamic_max_steer = self.max_steer * (0.35 + 0.65 * (1.0 - speed_ratio))
        target_steer = clamp(steer * dynamic_max_steer, -dynamic_max_steer, dynamic_max_steer)
        steer_change = clamp(target_steer - self.steer_angle, -self.steer_speed * dt, self.steer_speed * dt)
        self.steer_angle += steer_change

        # Acceleration and braking
        acceleration = Vec2(0, 0)
        if throttle > 0.0 and forward_speed < self.max_speed:
            acceleration += forward * (self.engine_force * throttle * surface_mu / self.mass)
        elif throttle < 0.0:
            if forward_speed > 1.0:
                acceleration += forward * (-self.brake_force * -throttle * surface_mu / self.mass)
            else:
                acceleration += forward * (self.reverse_force * throttle * surface_mu / self.mass)
        if brake > 0.0:
            braking_force = -self.brake_force * brake * surface_mu / self.mass
            acceleration += forward * braking_force

        # Drag & rolling resistance
        if speed > 0.1:
            drag = -velocity.normalize() * self.drag_coefficient * speed * speed / self.mass
            acceleration += drag
        acceleration += -velocity * (self.rolling_resistance / self.mass)

        # Lateral grip
        grip = self.lateral_grip * surface_mu
        if handbrake:
            grip *= 0.35
            acceleration += forward * (-self.brake_force * 0.4 * surface_mu / self.mass)
        lateral_force = -right * (lateral_speed * grip)
        acceleration += lateral_force / self.mass

        # Update velocity and position
        velocity += acceleration * dt
        forward_speed = velocity.dot(forward)
        if forward_speed > self.max_speed:
            velocity -= forward * (forward_speed - self.max_speed)
        if velocity.dot(forward) < self.max_reverse_speed:
            forward_component = forward * self.max_reverse_speed
            lateral_component = velocity - forward_component
            velocity = forward_component + lateral_component

        self.state.velocity = velocity

        # Heading update via bicycle model
        if abs(forward_speed) > 0.2:
            turn_rate = forward_speed / self.LENGTH * math.tan(self.steer_angle) * surface_mu
        else:
            turn_rate = 0.0
        self.state.angular_velocity = turn_rate
        self.state.heading += turn_rate * dt

        self.state.position += self.state.velocity * dt * 20  # convert to pixels

    def resolve_collision(self, normal: Vec2) -> None:
        # Remove velocity component along the normal and dampen speed
        vel = self.state.velocity
        normal_component = vel.dot(normal)
        vel -= normal * normal_component
        self.state.velocity = vel * 0.5
        self.state.position = self.prev_position
        self.state.angular_velocity *= 0.5


class Track:
    def __init__(self, definition: TrackDefinition):
        self.definition = definition
        self.surface_lookup = definition.surfaces
        self.walls = [
            (Vec2(start), Vec2(end))
            for start, end in definition.walls
        ]
        self.start_line = definition.start_line
        self.checkpoints = definition.checkpoints
        self.minimap_bounds = definition.minimap_bounds

    def surface_for_polygon(self, points: Sequence[Vec2]) -> Dict[str, object]:
        best_mu: Optional[float] = None
        surface_type = "GRASS"
        sample_points = [(point.x, point.y) for point in points]
        if points:
            center = sum(points, Vec2()) / len(points)
            sample_points.append((center.x, center.y))
        for surface in self.surface_lookup:
            polygon = surface["polygon"]
            for sx, sy in sample_points:
                if point_in_polygon((sx, sy), polygon):
                    mu = float(surface.get("mu", 0.6))
                    if best_mu is None or mu < best_mu:
                        best_mu = mu
                        surface_type = surface.get("type", "ASPHALT").upper()
                    break
            else:
                continue
        if best_mu is None:
            best_mu = 0.35
        return {"mu": best_mu, "type": surface_type}

    def surface_info_at(self, corners: Sequence[Vec2]) -> Dict[str, object]:
        info = self.surface_for_polygon(corners)
        if info["type"] == "GRASS":
            info["mu"] = 0.35
        return info

    def check_collisions(self, car: Car) -> None:
        corners = car.get_corners()
        car_edges = list(zip(corners, corners[1:] + corners[:1]))
        for wall_start, wall_end in self.walls:
            wall_vec = wall_end - wall_start
            wall_normal = Vec2(-wall_vec.y, wall_vec.x)
            if wall_normal.length_squared() > 0:
                wall_normal = wall_normal.normalize()
            collision = False
            for edge_start, edge_end in car_edges:
                if segment_intersection(edge_start, edge_end, wall_start, wall_end):
                    collision = True
                    break
            if not collision:
                mid_point = Vec2((wall_start.x + wall_end.x) * 0.5, (wall_start.y + wall_end.y) * 0.5)
                if point_in_polygon((mid_point.x, mid_point.y), [(corner.x, corner.y) for corner in corners]):
                    collision = True
            if collision:
                car.resolve_collision(wall_normal)
                break

    def minimap_transform(self, size: Tuple[int, int]):
        (min_x, min_y), (max_x, max_y) = self.minimap_bounds
        width = max_x - min_x
        height = max_y - min_y
        scale = min(size[0] / width, size[1] / height)

        def transform(point: Vec2) -> Tuple[int, int]:
            x = (point.x - min_x) * scale
            y = (point.y - min_y) * scale
            return int(x), int(size[1] - y)

        return transform


class LapTracker:
    def __init__(self, track: TrackDefinition) -> None:
        self.track = track
        self.reset()

    def reset(self) -> None:
        self.current_lap_time = 0.0
        self.lap_active = False
        self.next_checkpoint = 0
        self.start_cleared = True
        self.best_lap: Optional[float] = None
        self.last_lap: Optional[float] = None
        self.banner_time = 0.0
        self.checkpoint_progress = 0

    def update(self, dt: float) -> None:
        if self.lap_active:
            self.current_lap_time += dt
        if self.banner_time > 0:
            self.banner_time = max(0.0, self.banner_time - dt)

    def process_crossings(self, prev_pos: Vec2, current_pos: Vec2, car_forward: Vec2) -> None:
        start_points = self.track.start_line["points"]
        start_start, start_end = Vec2(start_points[0]), Vec2(start_points[1])
        crossed_start = self._check_line(prev_pos, current_pos, self.track.start_line, car_forward)
        checkpoint_count = len(self.track.checkpoints)

        if crossed_start and self.start_cleared:
            direction = Vec2(self.track.start_line["direction"])
            if car_forward.dot(direction) > 0.1:
                if self.lap_active and (checkpoint_count == 0 or self.checkpoint_progress >= checkpoint_count):
                    self.last_lap = self.current_lap_time
                    if self.best_lap is None or self.last_lap < self.best_lap:
                        self.best_lap = self.last_lap
                        self.banner_time = 2.5
                    self.current_lap_time = 0.0
                if not self.lap_active:
                    self.lap_active = True
                    self.current_lap_time = 0.0
                self.next_checkpoint = 0
                self.start_cleared = False
                self.checkpoint_progress = 0

        distance_from_line = distance_point_to_segment(current_pos, start_start, start_end)
        if distance_from_line > 20:
            self.start_cleared = True

        if not self.lap_active:
            return

        for idx in range(checkpoint_count):
            if idx != self.next_checkpoint:
                continue
            checkpoint = self.track.checkpoints[idx]
            if self._check_line(prev_pos, current_pos, checkpoint, car_forward):
                direction = Vec2(checkpoint["direction"])
                if car_forward.dot(direction) > -0.1:
                    self.next_checkpoint = (self.next_checkpoint + 1) % checkpoint_count
                    if self.next_checkpoint == 0:
                        self.checkpoint_progress = checkpoint_count
                    else:
                        self.checkpoint_progress = max(self.checkpoint_progress, idx + 1)

    def _check_line(self, prev_pos: Vec2, current_pos: Vec2, line: Dict[str, object], forward: Vec2) -> bool:
        line_start, line_end = line["points"]
        return segment_intersection(prev_pos, current_pos, Vec2(line_start), Vec2(line_end))


class RecordBook:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf8") as fh:
                    raw = json.load(fh)
                for key, value in raw.items():
                    self.records[key] = float(value)
            except (OSError, json.JSONDecodeError, ValueError):
                self.records = {}

    def save(self) -> None:
        try:
            with self.path.open("w", encoding="utf8") as fh:
                json.dump(self.records, fh, indent=2)
        except OSError:
            pass

    def best_for(self, track_id: str) -> Optional[float]:
        return self.records.get(track_id)

    def update(self, track_id: str, lap_time: Optional[float]) -> bool:
        if lap_time is None:
            return False
        best = self.records.get(track_id)
        if best is None or lap_time < best:
            self.records[track_id] = lap_time
            self.save()
            return True
        return False


class Game:
    def __init__(self) -> None:
        pygame.init()
        self.size = (1280, 800)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Vector Racer")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_medium = pygame.font.SysFont("Consolas", 28)
        self.font_large = pygame.font.SysFont("Consolas", 48)
        self.tracks = all_tracks()
        self.track_index = 0
        self.track = Track(self.tracks[self.track_index])
        self.car = Car(self.track.definition.spawn_position, self.track.definition.spawn_heading)
        self.lap_tracker = LapTracker(self.track.definition)
        self.records = RecordBook(Path("records.json"))
        best = self.records.best_for(self.track.definition.track_id)
        if best is not None:
            self.lap_tracker.best_lap = best
        self.state = "menu"

    def change_track(self, new_index: int) -> None:
        self.track_index = new_index % len(self.tracks)
        self.track = Track(self.tracks[self.track_index])
        self.car = Car(self.track.definition.spawn_position, self.track.definition.spawn_heading)
        self.lap_tracker = LapTracker(self.track.definition)
        best = self.records.best_for(self.track.definition.track_id)
        if best is not None:
            self.lap_tracker.best_lap = best

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_p):
                        if self.state == "running":
                            self.state = "paused"
                        elif self.state == "paused":
                            self.state = "running"
                    if self.state == "menu" and event.key == pygame.K_SPACE:
                        self.state = "running"
                    if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        self.change_track(event.key - pygame.K_1)
                        if self.state == "menu":
                            continue
                        self.state = "running"

            if self.state == "running":
                self.update(dt)

            self.draw()

        pygame.quit()

    def handle_input(self) -> Dict[str, float]:
        keys = pygame.key.get_pressed()
        steer = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer += 1.0

        throttle = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle -= 1.0

        brake = 1.0 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0.0
        handbrake = keys[pygame.K_SPACE]

        return {
            "steer": clamp(steer, -1.0, 1.0),
            "throttle": clamp(throttle, -1.0, 1.0),
            "brake": brake,
            "handbrake": handbrake,
        }

    def update(self, dt: float) -> None:
        inputs = self.handle_input()
        corners = self.car.get_corners()
        surface_info = self.track.surface_info_at(corners)
        mu = clamp(surface_info.get("mu", 0.6), 0.05, 1.2)
        self.car.update(dt, inputs, mu)
        self.track.check_collisions(self.car)
        self.lap_tracker.update(dt)
        self.lap_tracker.process_crossings(self.car.prev_position, self.car.state.position, self.car.forward)
        if self.lap_tracker.last_lap is not None:
            if self.records.update(self.track.definition.track_id, self.lap_tracker.last_lap):
                self.lap_tracker.best_lap = self.records.best_for(self.track.definition.track_id)
            self.lap_tracker.last_lap = None

    def draw(self) -> None:
        self.screen.fill((20, 20, 30))
        self.draw_track()
        self.draw_car()
        self.draw_hud()

        if self.state == "menu":
            self.draw_menu()
        elif self.state == "paused":
            self.draw_pause()

        pygame.display.flip()

    def draw_track(self) -> None:
        for surface in self.track.definition.surfaces:
            points = surface["polygon"]
            color = surface.get("color", (60, 60, 60))
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (20, 20, 24), points, 2)

        for start, end in self.track.walls:
            pygame.draw.line(self.screen, (200, 200, 220), start, end, 3)

        start_line = self.track.start_line["points"]
        pygame.draw.line(self.screen, (220, 220, 80), start_line[0], start_line[1], 5)
        for checkpoint in self.track.checkpoints:
            pygame.draw.line(self.screen, (160, 120, 220), checkpoint["points"][0], checkpoint["points"][1], 2)

        # Canyon shading overlay
        if self.track.definition.track_id == "canyon_loop":
            overlay = pygame.Surface(self.size, pygame.SRCALPHA)
            for offset in range(0, 80, 20):
                shade_rect = pygame.Rect(120, 200 + offset, 1120, 20)
                pygame.draw.rect(overlay, (24, 22, 30, 60), shade_rect, 0)
            self.screen.blit(overlay, (0, 0))

    def draw_car(self) -> None:
        corners = self.car.get_corners()
        pygame.draw.polygon(self.screen, (230, 90, 70), corners)
        pygame.draw.polygon(self.screen, (50, 20, 20), corners, 2)
        nose = sum(corners[:2], Vec2()) / 2
        arrow_tip = nose + self.car.forward * 18
        pygame.draw.line(self.screen, (255, 230, 180), nose, arrow_tip, 2)

    def draw_hud(self) -> None:
        info = self.track.surface_info_at(self.car.get_corners())
        mu_text = info.get("type", "ASPHALT").upper()
        speed = self.car.state.velocity.length() * 3.6
        lap_time = format_time(self.lap_tracker.current_lap_time if self.lap_tracker.lap_active else None)
        best = self.lap_tracker.best_lap or self.records.best_for(self.track.definition.track_id)
        best_text = format_time(best)
        lines = [
            f"Track: {self.track.definition.name}",
            f"Speed: {speed:05.1f} km/h",
            f"Surface: {mu_text}",
            f"Lap: {lap_time}",
            f"Best: {best_text}",
        ]
        for idx, line in enumerate(lines):
            text_surf = self.font_medium.render(line, True, (240, 240, 240))
            self.screen.blit(text_surf, (20, 20 + idx * 28))

        self.draw_minimap()

        if self.lap_tracker.banner_time > 0:
            banner = self.font_large.render("NEW RECORD!", True, (255, 240, 120))
            rect = banner.get_rect(center=(self.size[0] // 2, 80))
            self.screen.blit(banner, rect)

    def draw_minimap(self) -> None:
        minimap_size = (220, 160)
        minimap_rect = pygame.Rect(self.size[0] - minimap_size[0] - 20, self.size[1] - minimap_size[1] - 20, *minimap_size)
        pygame.draw.rect(self.screen, (10, 10, 18), minimap_rect)
        pygame.draw.rect(self.screen, (80, 80, 100), minimap_rect, 2)
        transform = self.track.minimap_transform((minimap_rect.width - 20, minimap_rect.height - 20))

        for surface in self.track.definition.surfaces:
            transformed = [transform(Vec2(p)) for p in surface["polygon"]]
            if len(transformed) >= 2:
                pygame.draw.lines(
                    self.screen,
                    (110, 110, 140),
                    True,
                    [(minimap_rect.x + 10 + x, minimap_rect.y + 10 + y) for x, y in transformed],
                    1,
                )

        for start, end in self.track.walls:
            a = transform(start)
            b = transform(end)
            pygame.draw.line(
                self.screen,
                (150, 150, 180),
                (minimap_rect.x + 10 + a[0], minimap_rect.y + 10 + a[1]),
                (minimap_rect.x + 10 + b[0], minimap_rect.y + 10 + b[1]),
                1,
            )

        car_pos = transform(self.car.state.position)
        pygame.draw.circle(
            self.screen,
            (250, 80, 70),
            (minimap_rect.x + 10 + car_pos[0], minimap_rect.y + 10 + car_pos[1]),
            4,
        )

        start_line = [transform(Vec2(pt)) for pt in self.track.start_line["points"]]
        pygame.draw.line(
            self.screen,
            (220, 220, 80),
            (minimap_rect.x + 10 + start_line[0][0], minimap_rect.y + 10 + start_line[0][1]),
            (minimap_rect.x + 10 + start_line[1][0], minimap_rect.y + 10 + start_line[1][1]),
            2,
        )

    def draw_menu(self) -> None:
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        title = self.font_large.render("Vector Racer", True, (255, 255, 255))
        prompt = self.font_medium.render("Press SPACE to drive", True, (255, 255, 255))
        tracks_text = self.font_small.render("Tracks: 1) Harbor Run  2) Canyon Loop  3) Industrial Sprint", True, (230, 230, 230))
        self.screen.blit(title, title.get_rect(center=(self.size[0] // 2, self.size[1] // 2 - 60)))
        self.screen.blit(prompt, prompt.get_rect(center=(self.size[0] // 2, self.size[1] // 2)))
        self.screen.blit(tracks_text, tracks_text.get_rect(center=(self.size[0] // 2, self.size[1] // 2 + 50)))

    def draw_pause(self) -> None:
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        paused = self.font_large.render("Paused", True, (240, 240, 240))
        help_lines = [
            "Controls:",
            "Arrow/WASD to steer",
            "Up/W throttle, Down/S reverse",
            "Shift brake, Space handbrake",
            "Esc/P to pause",
            "1/2/3 change track",
        ]
        self.screen.blit(paused, paused.get_rect(center=(self.size[0] // 2, self.size[1] // 2 - 80)))
        for idx, line in enumerate(help_lines):
            surf = self.font_medium.render(line, True, (240, 240, 240))
            self.screen.blit(surf, surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2 - 20 + idx * 32)))


def main() -> None:
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    game = Game()
    game.run()


if __name__ == "__main__":
    main()

