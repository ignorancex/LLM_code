from typing import NamedTuple


class Vec2D[T](NamedTuple):
    x: T
    y: T

    def __add__(self, other: "Vec2D[T]") -> "Vec2D[T]":
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2D[T]") -> "Vec2D[T]":
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other: "Vec2D[T]") -> "Vec2D[T]":
        return Vec2D(self.x * other.x, self.y * other.y)

    def __repr__(self) -> str:
        return f"Vec2D({self.x}, {self.y})"
