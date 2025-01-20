from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import numpy as np
from enums import Color, Position

@dataclass
class Vertex:
    position: Position
    
@dataclass
class Face:
    vertices: List[int]
    color: Color = Color.WHITE

class Shape:
    def __init__(self, name: str, position: Position):
        self._name = name
        self._position = position

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self):
        return self._position

    @property
    def rotation(self) -> np.ndarray:
        if not hasattr(self, '_rotation'):
            self._rotation = np.array([0.0, 0.0, 0.0])
        return self._rotation

    @rotation.setter
    def rotation(self, rot: np.ndarray) -> None:
        self._rotation = rot

class Model(Shape):
    def __init__(self, name: str):
        super().__init__(name, Position(0, 0, 0))
        self.size = 1.0
        self.color = np.array([0.7, 0.7, 0.7])
        self._rotation = np.array([0.0, 0.0, 0.0])
        self._vertices_cache = None
        self._transform_cache = None

    @classmethod
    def from_obj(cls, filename: str, name: str) -> 'Model':
        model = cls(name)
        with open(filename, 'r') as f:
            vertices = []
            faces = []
            for line in f:
                if line.startswith('v '):
                    v = line.split()[1:]
                    vertices.append(Position(float(v[0]), float(v[1]), float(v[2])))
                elif line.startswith('f '):
                    f = line.split()[1:]
                    face = [int(x.split('/')[0])-1 for x in f]
                    faces.append(Face(vertices=face))
            
            model._vertices = [Vertex(pos) for pos in vertices]
            model._faces = faces
            model._normalize_model()
        return model

    def _normalize_model(self):
        vertices = np.array([v.position.to_array() for v in self._vertices])
        center = np.mean(vertices, axis=0)
        max_dim = np.max(np.abs(vertices - center))
        for vertex in self._vertices:
            pos = vertex.position.to_array()
            vertex.position = Position(*(pos - center) / max_dim)

    def set_transform(self, position: np.ndarray, rotation: np.ndarray, size: float):
        self._position = Position(*position)
        self._rotation = rotation
        self.size = size

    def get_transform(self) -> tuple[np.ndarray, np.ndarray, float]:
        return np.array([self._position.x, self._position.y, self._position.z]), \
               self._rotation, \
               self.size

    def get_vertices(self) -> np.ndarray:
        current_transform = (
            tuple(self._position.to_array()),
            tuple(self._rotation),
            self.size
        )
        
        if self._vertices_cache is not None and self._transform_cache == current_transform:
            return self._vertices_cache

        vertices = np.array([v.position.to_array() for v in self._vertices])
        vertices = vertices * self.size
        
        if hasattr(self, '_rotation') and np.any(self._rotation != 0):
            rx, ry, rz = np.radians(self._rotation)
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            vertices = vertices @ Rz @ Ry @ Rx
            
        vertices = vertices + self._position.to_array()
        
        self._vertices_cache = vertices
        self._transform_cache = current_transform
        
        return vertices
        
class Plane(Shape):
    def __init__(self, name: str, points: List[Position]):
        super().__init__(name, points[0])
        self._points = points
        self._normal = self._calculate_normal()
        self._d = -np.dot(self._normal, points[0].to_array())
    
    def _calculate_normal(self) -> np.ndarray:
        v1 = self._points[1].to_array() - self._points[0].to_array()
        v2 = self._points[2].to_array() - self._points[0].to_array()
        return np.cross(v1, v2)

    @classmethod
    def from_points(cls, name: str, p1: Position, p2: Position, p3: Position) -> 'Plane':
        return cls(name, [p1, p2, p3])

    @classmethod
    def from_point_and_line(cls, name: str, point: Position, line_start: Position, line_end: Position) -> 'Plane':
        return cls(name, [point, line_start, line_end])

    @classmethod
    def from_point_and_parallel(cls, name: str, point: Position, other_plane: 'Plane') -> 'Plane':
        plane = cls(name, [point, point, point])
        plane._normal = other_plane._normal
        plane._d = -np.dot(plane._normal, point.to_array())
        plane._points = [
            point,
            Position(* (point.to_array() + np.cross(plane._normal, [1,0,0]))),
            Position(* (point.to_array() + np.cross(plane._normal, [0,1,0])))
        ]
        return plane

    def intersect(self, other: 'Plane') -> Optional[Tuple[Position, Position]]:
        n1 = self._normal
        n2 = other._normal
        direction = np.cross(n1, n2)
        if np.allclose(direction, 0):
            return None

        n1n1 = np.dot(n1, n1)
        n2n2 = np.dot(n2, n2)
        n1n2 = np.dot(n1, n2)
        n1d2 = self._d
        n2d1 = other._d
        det = n1n1 * n2n2 - n1n2 * n1n2
        if np.isclose(det, 0):
            return None

        c1 = (n2n2 * n1d2 - n1n2 * n2d1) / det
        c2 = (n1n1 * n2d1 - n1n2 * n1d2) / det
        point = c1 * n1 + c2 * n2

        direction = direction / np.linalg.norm(direction)
        p1 = Position(*(point - direction * 10))
        p2 = Position(*(point + direction * 10))
        return (p1, p2)

    def clip_by_plane(self, other: 'Plane') -> Optional[List[Position]]:
        intersection = self.intersect(other)
        if intersection is None:
            return None
            
        p1, p2 = intersection
        
        center = np.mean([p.to_array() for p in self._points], axis=0)
        bounds = 5.0
        
        normal = self._normal
        v1 = np.cross(normal, [0, 1, 0])
        if np.all(v1 == 0):
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1) * bounds
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2) * bounds
        
        corners = [
            center - v1 - v2,
            center + v1 - v2,
            center + v1 + v2,
            center - v1 + v2
        ]
        
        return self._clip_polygon(corners, p1.to_array(), p2.to_array())

    def _clip_polygon(self, points: List[np.ndarray], p1: np.ndarray, p2: np.ndarray) -> List[Position]:
        def get_intersection(p1, p2, p3, p4):
            x1, y1 = p1[:2]
            x2, y2 = p2[:2]
            x3, y3 = p3[:2]
            x4, y4 = p4[:2]
            
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denominator == 0:
                return None
                
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return Position(x, y, p1[2] + t * (p2[2] - p1[2]))
            return None

        result = []
        line_dir = p2 - p1
        line_normal = np.array([-line_dir[1], line_dir[0], 0])
        
        n = len(points)
        for i in range(n):
            current = points[i]
            next_point = points[(i + 1) % n]
            
            current_side = np.dot(current - p1, line_normal)
            next_side = np.dot(next_point - p1, line_normal)
            
            if current_side >= 0:
                result.append(Position(*current))
            
            if (current_side >= 0) != (next_side >= 0):
                intersection = get_intersection(current, next_point, p1, p2)
                if intersection:
                    result.append(intersection)
        
        return result

    def is_point_visible(self, point: Position, camera_pos: Position) -> bool:
        view_vector = point.to_array() - np.array([camera_pos.x, camera_pos.y, camera_pos.z])
        view_vector = view_vector / np.linalg.norm(view_vector)
        
        return np.dot(self._normal, view_vector) < 0

class Cube(Shape):
    def __init__(self, name: str, position: Position, size: float, color: np.ndarray):
        super().__init__(name, position)
        self.size = size
        self.color = color
        self._rotation = np.array([0.0, 0.0, 0.0])

    def get_vertices(self):
        half_size = self.size / 2.0
        return np.array([
            [self._position.x - half_size, self._position.y - half_size, self._position.z - half_size],
            [self._position.x + half_size, self._position.y - half_size, self._position.z - half_size],
            [self._position.x + half_size, self._position.y + half_size, self._position.z - half_size],
            [self._position.x - half_size, self._position.y + half_size, self._position.z - half_size],
            [self._position.x - half_size, self._position.y - half_size, self._position.z + half_size],
            [self._position.x + half_size, self._position.y - half_size, self._position.z + half_size],
            [self._position.x + half_size, self._position.y + half_size, self._position.z + half_size],
            [self._position.x - half_size, self._position.y + half_size, self._position.z + half_size]
        ])

    def intersect_plane(self, plane: Plane):
        """Return list of line segments (pairs of points) where this cube is intersected by the plane."""
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]
        vertices = self.get_vertices()
        segments = []
        def edge_plane_intersect(p1, p2):
            d1 = np.dot(plane._normal, p1) + plane._d
            d2 = np.dot(plane._normal, p2) + plane._d
            if d1 * d2 >= 0:
                return None
            t = d1 / (d1 - d2)
            return p1 + t*(p2 - p1)

        for e in edges:
            p_int = edge_plane_intersect(vertices[e[0]], vertices[e[1]])
            if p_int is not None:
                segments.append(p_int)
        result_lines = []
        for i in range(0, len(segments), 2):
            if i+1 < len(segments):
                result_lines.append((segments[i], segments[i+1]))
        return result_lines
