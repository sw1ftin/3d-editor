from enum import Enum
from typing import Tuple, List
import numpy as np

class Position:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        raise IndexError("Position index out of range")

    def __sub__(self, other):
        """Subtract two positions"""
        return Position(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __add__(self, other):
        """Add two positions"""
        return Position(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __mul__(self, scalar):
        """Multiply position by scalar"""
        return Position(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )

    def __truediv__(self, scalar):
        """Divide position by scalar"""
        return Position(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar
        )

    def dot(self, other):
        """Dot product"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Cross product"""
        return Position(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def normalize(self):
        """Return normalized vector"""
        length = np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if length < 1e-6:
            return Position(0, 0, 0)
        return self / length

    def to_array(self):
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(arr):
        """Create Position from numpy array"""
        return Position(arr[0], arr[1], arr[2])

class Color(Enum):
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
