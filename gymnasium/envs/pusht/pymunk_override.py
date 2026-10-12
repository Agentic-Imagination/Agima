# Adapted from huggingface/gym-pusht, which in turn adapts pymunk's pygame helpers.

from typing import Sequence, Tuple

import numpy as np
import pygame
import pymunk
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

positive_y_is_up: bool = False


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface, *, circle_radius_override: float | None = None) -> None:
        self.surface = surface
        self.circle_radius_override = circle_radius_override
        super().__init__()

    def draw_circle(self, pos: Vec2d, angle: float, radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        p = to_pygame(pos, self.surface)
        draw_radius = float(self.circle_radius_override) if self.circle_radius_override is not None else float(radius)
        outer_radius = max(1, round(draw_radius))
        inner_radius = max(1, round(draw_radius - min(4.0, max(0.0, draw_radius - 1.0))))
        pygame.draw.circle(self.surface, fill_color.as_int(), p, outer_radius, 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, inner_radius, 0)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(self, a: Tuple[float, float], b: Tuple[float, float], radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(self.surface, fill_color.as_int(), (round(p1[0]), round(p1[1])), round(radius))
            pygame.draw.circle(self.surface, fill_color.as_int(), (round(p2[0]), round(p2[1])), round(radius))

    def draw_polygon(self, verts: Sequence[Tuple[float, float]], radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]
        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)
        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(self, size: float, pos: Tuple[float, float], color: SpaceDebugColor) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    return round(p[0]), round(p[1])


def from_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    return to_pygame(p, surface)


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    return SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
