import numpy as np
from enums import Position

class Camera:
    def __init__(self):
        self._position = Position(0, 0, 10) # Camera position
        self._target = Position(0, 0, 0)    # Look at point
        self._up = Position(0, 1, 0)        # Up vector
        self._yaw = -90.0                   # Rotation around Y axis
        self._pitch = 0.0                   # Rotation around X axis
        self._distance = 10.0               # Distance from target

    def get_view_matrix(self):
        view_dir = (self._target - self._position).normalize()
        right = view_dir.cross(self._up).normalize()
        up = right.cross(view_dir).normalize()

        view_matrix = np.array([
            [right.x, right.y, right.z, -right.dot(self._position)],
            [up.x, up.y, up.z, -up.dot(self._position)],
            [-view_dir.x, -view_dir.y, -view_dir.z, view_dir.dot(self._position)],
            [0, 0, 0, 1]
        ])
        return view_matrix

    def orbit(self, dx: float, dy: float) -> None:
        """Orbit camera around target point"""
        self._yaw += dx
        self._pitch = max(-89.0, min(89.0, self._pitch + dy))

        phi = np.radians(self._yaw)
        theta = np.radians(self._pitch)
        
        x = self._distance * np.cos(phi) * np.cos(theta)
        y = self._distance * np.sin(theta)
        z = self._distance * np.sin(phi) * np.cos(theta)
        
        self._position = Position(
            self._target.x + x,
            self._target.y + y,
            self._target.z + z
        )

    def pan(self, dx: float, dy: float) -> None:
        """Pan camera in view plane"""
        view_dir = (self._target - self._position).normalize()
        right = view_dir.cross(self._up).normalize()
        up = right.cross(view_dir).normalize()
        
        offset = right * (-dx) + up * dy
        self._position += offset
        self._target += offset

    def zoom(self, factor: float) -> None:
        """Zoom camera by changing distance to target"""
        self._distance = max(0.1, self._distance * (1.0 - factor))
        self.orbit(0, 0)

    def move(self, direction: str, speed: float) -> None:
        """Move camera in specified direction"""
        view_dir = (self._target - self._position).normalize()
        right = view_dir.cross(self._up).normalize()
        up = right.cross(view_dir).normalize()
        
        if direction == 'forward':
            offset = view_dir * speed
        elif direction == 'backward':
            offset = view_dir * -speed
        elif direction == 'left':
            offset = right * -speed
        elif direction == 'right':
            offset = right * speed
        elif direction == 'up':
            offset = up * speed
        elif direction == 'down':
            offset = up * -speed
        else:
            return
            
        self._position += offset
        self._target += offset
