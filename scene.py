import json
from typing import List, Dict, Any
import numpy as np
from models import Shape, Cube, Plane, Model
from enums import Position, Color

class SceneManager:
    @staticmethod
    def save_scene(shapes: List[Shape], filename: str) -> None:
        scene_data = []
        for shape in shapes:
            if isinstance(shape, Cube):
                position = shape.position if isinstance(shape.position, list) else shape.position.to_array()
                shape_data = {
                    'type': 'cube',
                    'name': shape.name,
                    'position': position.tolist() if isinstance(position, np.ndarray) else position,
                    'size': shape.size,
                    'color': shape.color.tolist(),
                    'rotation': shape.rotation.tolist()
                }
            elif isinstance(shape, Plane):
                shape_data = {
                    'type': 'plane',
                    'name': shape.name,
                    'points': [
                        [p.x, p.y, p.z] for p in shape._points
                    ]
                }
            scene_data.append(shape_data)
            
        with open(filename, 'w') as f:
            json.dump(scene_data, f)

    @staticmethod
    def load_scene(filename: str) -> List[Shape]:
        with open(filename, 'r') as f:
            scene_data = json.load(f)
        
        shapes = []
        for shape_data in scene_data:
            if shape_data['type'] == 'cube':
                shape = Cube(
                    shape_data['name'],
                    np.array(shape_data['position']),
                    shape_data['size'],
                    np.array(shape_data['color'])
                )
                shape.rotation = np.array(shape_data['rotation'])
            elif shape_data['type'] == 'plane':
                points = [
                    Position(p[0], p[1], p[2]) 
                    for p in shape_data['points']
                ]
                shape = Plane(shape_data['name'], points)
            shapes.append(shape)
        return shapes
