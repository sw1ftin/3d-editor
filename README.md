# 3D Editor
## Requirements

- Python 3.8+
- PySide6
- PyOpenGL
- numpy

## Installation

1. Install `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python app.py
```

## Features

- Add, delete, and manipulate 3D shapes (cubes, planes, models).
- Load and save scenes in JSON format.
- Display coordinate axes.
- Toggle visibility of hidden lines.
- Create planes based on points, lines, or parallel to existing planes.
- Highlight intersections between planes and shapes.

## File Structure

- `app.py`: Main application file.
- `models.py`: Definitions for 3D shapes.
- `camera.py`: Camera control.
- `enums.py`: Enumerations and helper classes.
- `scene.py`: Scene management.
- `requirements.txt`: Required Python packages.
- `README.md`: Project documentation.