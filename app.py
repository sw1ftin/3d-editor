import sys
import os
import numpy as np
from typing import List, Optional, Tuple
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from models import Shape, Model, Cube, Plane
from scene import SceneManager
from camera import Camera
from enums import Position, Color
import json

class Renderer(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self._camera: Camera = Camera()
        self._shapes: List[Shape] = []
        self._show_axes: bool = True
        self._show_hidden_lines: bool = False
        self._last_pos = None
        self._camera_speed = 0.5
        self._camera_distance = 10.0
        self._far_plane = 10000.0
        self._is_shift_pressed = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        view_matrix = self._camera.get_view_matrix()
        glMultMatrixf(view_matrix.T)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self._show_axes:
            self._draw_axes()

        sorted_shapes = sorted(
            self._shapes,
            key=lambda s: -np.linalg.norm(
                self._camera._position.to_array() - 
                (s.position.to_array() if isinstance(s.position, Position) else s.position)
            )
        )
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(view_matrix.T)
        
        for shape in sorted_shapes:
            if isinstance(shape, Cube):
                self._draw_cube(shape)
            elif isinstance(shape, Plane):
                self._draw_plane(shape)
            elif isinstance(shape, Model):
                self._draw_model(shape)

    def draw_cube(self, cube):
        glBegin(GL_QUADS)
        for face in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]:
            glColor3f(cube.color[0], cube.color[1], cube.color[2])
            for vertex in face:
                glVertex3fv(cube.get_vertices()[vertex])
        glEnd()

        self.draw_edges(cube)

    def draw_edges(self, cube):
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINES)
        for edge in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
            for vertex in edge:
                glVertex3fv(cube.get_vertices()[vertex])
        glEnd()

    def _draw_model(self, model: Model):
        if not hasattr(self, '_matrix_stack_depth'):
            self._matrix_stack_depth = 0

        if self._matrix_stack_depth < glGetIntegerv(GL_MAX_MODELVIEW_STACK_DEPTH):
            glPushMatrix()
            self._matrix_stack_depth += 1
            try:
                pos = model.position.to_array() if isinstance(model.position, Position) else model.position
                glTranslatef(pos[0], pos[1], pos[2])
                
                rot = model.rotation if hasattr(model, 'rotation') else [0, 0, 0]
                glRotatef(rot[0], 1.0, 0.0, 0.0)
                glRotatef(rot[1], 0.0, 1.0, 0.0)
                glRotatef(rot[2], 0.0, 0.0, 1.0)
                
                glColor3f(*model.color)
                glBegin(GL_TRIANGLES)
                vertices = model.get_vertices()
                for face in model._faces:
                    for vertex_idx in face.vertices:
                        vertex = vertices[vertex_idx] - pos
                        glVertex3fv(vertex)
                glEnd()
                
                glColor3f(0.2, 0.2, 0.2)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glBegin(GL_TRIANGLES)
                for face in model._faces:
                    for vertex_idx in face.vertices:
                        vertex = vertices[vertex_idx] - pos
                        glVertex3fv(vertex)
                glEnd()
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            finally:
                glPopMatrix()
                self._matrix_stack_depth -= 1

    def _draw_plane(self, plane: Plane):
        if not hasattr(self, '_matrix_stack_depth'):
            self._matrix_stack_depth = 0

        if self._matrix_stack_depth < glGetIntegerv(GL_MAX_MODELVIEW_STACK_DEPTH):
            glPushMatrix()
            self._matrix_stack_depth += 1
            try:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glEnable(GL_CULL_FACE)
                glDisable(GL_DEPTH_TEST)

                glColor4f(0.5, 0.5, 0.5, 0.3)

                center = np.mean([p.to_array() for p in plane._points], axis=0)
                normal = plane._normal
                size = 20.0
                
                v1 = np.cross(normal, [0, 1, 0])
                if np.allclose(v1, 0):
                    v1 = np.cross(normal, [1, 0, 0])
                v1 = v1 / np.linalg.norm(v1) * size
                v2 = np.cross(normal, v1)
                v2 = v2 / np.linalg.norm(v2) * size

                glBegin(GL_QUADS)
                
                glNormal3f(*normal)
                vertices = [
                    center - v1 - v2,
                    center + v1 - v2,
                    center + v1 + v2,
                    center - v1 + v2
                ]
                for vertex in vertices:
                    glVertex3fv(vertex)

                glNormal3f(*(-normal))
                for vertex in reversed(vertices):
                    glVertex3fv(vertex)
                    
                glEnd()

                glColor3f(0.2, 0.2, 0.2)
                glLineWidth(2.0)
                glBegin(GL_LINE_LOOP)
                for vertex in vertices:
                    glVertex3fv(vertex)
                glEnd()
                glLineWidth(1.0)

                glEnable(GL_DEPTH_TEST)
                glDisable(GL_CULL_FACE)
                glDisable(GL_BLEND)

                self._draw_plane_intersections(plane)
                self._draw_plane_shape_intersections(plane)
            finally:
                glPopMatrix()
                self._matrix_stack_depth -= 1

    def _draw_plane_intersections(self, plane: Plane):
        if not hasattr(self, '_matrix_stack_depth'):
            self._matrix_stack_depth = 0

        if self._matrix_stack_depth < glGetIntegerv(GL_MAX_MODELVIEW_STACK_DEPTH):
            glPushMatrix()
            self._matrix_stack_depth += 1
            try:
                glColor3f(0.0, 0.0, 0.0)
                glLineWidth(2.0)
                glDisable(GL_DEPTH_TEST)
                
                glBegin(GL_LINES)
                for other in self._shapes:
                    if isinstance(other, Plane) and other != plane:
                        intersection = plane.intersect(other)
                        if intersection:
                            p1, p2 = intersection
                            glVertex3fv(p1.to_array())
                            glVertex3fv(p2.to_array())
                glEnd()
                
                glEnable(GL_DEPTH_TEST)
                glLineWidth(1.0)
            finally:
                glPopMatrix()
                self._matrix_stack_depth -= 1

    def _draw_plane_shape_intersections(self, plane: Plane):
        glColor3f(0.0, 1.0, 0.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for shape in self._shapes:
            if hasattr(shape, 'intersect_plane'):
                segments = shape.intersect_plane(plane)
                for p1, p2 in segments:
                    glVertex3fv(p1)
                    glVertex3fv(p2)
        glEnd()
        glLineWidth(1.0)

    def _draw_axes(self):
        glLineWidth(2.0)
        glDisable(GL_DEPTH_TEST)
        
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-self._far_plane, 0.0, 0.0)
        glVertex3f(self._far_plane, 0.0, 0.0)
        
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, -self._far_plane, 0.0)
        glVertex3f(0.0, self._far_plane, 0.0)
        
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, -self._far_plane)
        glVertex3f(0.0, 0.0, self._far_plane)
        glEnd()

        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        for i in range(1, 6):
            pos = i * 5.0
            glVertex3f(pos, -0.2, 0.0)
            glVertex3f(pos, 0.2, 0.0)
            glVertex3f(-pos, -0.2, 0.0)
            glVertex3f(-pos, 0.2, 0.0)
        
        glColor3f(0.0, 1.0, 0.0)
        for i in range(1, 6):
            pos = i * 5.0
            glVertex3f(-0.2, pos, 0.0)
            glVertex3f(0.2, pos, 0.0)
            glVertex3f(-0.2, -pos, 0.0)
            glVertex3f(0.2, -pos, 0.0)
        
        glColor3f(0.0, 0.0, 1.0)
        for i in range(1, 6):
            pos = i * 5.0
            glVertex3f(0.0, -0.2, pos)
            glVertex3f(0.0, 0.2, pos)
            glVertex3f(0.0, -0.2, -pos)
            glVertex3f(0.0, 0.2, -pos)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(w) / float(h), 0.1, self._far_plane)
        glMatrixMode(GL_MODELVIEW)

    def _draw_cube(self, cube):
        if not hasattr(self, '_matrix_stack_depth'):
            self._matrix_stack_depth = 0

        if self._matrix_stack_depth < glGetIntegerv(GL_MAX_MODELVIEW_STACK_DEPTH):
            glPushMatrix()
            self._matrix_stack_depth += 1
            try:
                pos = cube.position.to_array() if isinstance(cube.position, Position) else cube.position
                glTranslatef(pos[0], pos[1], pos[2])
                glRotatef(cube._rotation[0], 1.0, 0.0, 0.0)
                glRotatef(cube._rotation[1], 0.0, 1.0, 0.0)
                glRotatef(cube._rotation[2], 0.0, 0.0, 1.0)
                
                glBegin(GL_QUADS)
                for face in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]:
                    glColor3f(*cube.color)
                    vertices = cube.get_vertices()
                    for vertex in face:
                        v = vertices[vertex] - pos
                        glVertex3f(v[0], v[1], v[2])
                glEnd()

                glColor3f(0.0, 0.0, 0.0)
                glBegin(GL_LINES)
                for edge in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
                    vertices = cube.get_vertices()
                    for vertex in edge:
                        v = vertices[vertex] - pos
                        glVertex3f(v[0], v[1], v[2])
                glEnd()
            finally:
                glPopMatrix()
                self._matrix_stack_depth -= 1

    def add_cube(self, name: str, position: List[float], size: float, color: List[float]) -> None:
        new_cube = Cube(name=name, 
                       position=Position(*position),
                       size=size, 
                       color=np.array(color))
        self._shapes.append(new_cube)
        self.update()

    def mouseReleaseEvent(self, event):
        self._last_pos = None

    def mouseMoveEvent(self, event):
        if not self._last_pos:
            self._last_pos = event.position()
            return
            
        current_pos = event.position()
        dx = (current_pos.x() - self._last_pos.x()) * 0.5
        dy = (current_pos.y() - self._last_pos.y()) * 0.5
        
        if event.buttons() & Qt.MiddleButton:
            if event.modifiers() & Qt.ShiftModifier:
                self._camera.pan(dx * 0.01, dy * 0.01)
            else:
                self._camera.orbit(dx, dy)
                
        self._last_pos = current_pos
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._camera.zoom(delta * 0.001)
        self.update()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
            
        key = event.key()
        if key == Qt.Key_Shift:
            self._is_shift_pressed = True
            self._camera_speed = 1.0
        elif key == Qt.Key_W:
            self._camera.move('forward', self._camera_speed)
        elif key == Qt.Key_S:
            self._camera.move('backward', self._camera_speed)
        elif key == Qt.Key_A:
            self._camera.move('left', self._camera_speed)
        elif key == Qt.Key_D:
            self._camera.move('right', self._camera_speed)
        elif key == Qt.Key_Q:
            self._camera.move('up', self._camera_speed)
        elif key == Qt.Key_E:
            self._camera.move('down', self._camera_speed)
            
        self.update()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
            
        key = event.key()
        if key == Qt.Key_Shift:
            self._is_shift_pressed = False
            self._camera_speed = 0.5
        
        self.update()

    def update_rotation(self, shape: Shape, axis: int, value: float) -> None:
        if not hasattr(shape, '_rotation'):
            shape._rotation = np.array([0.0, 0.0, 0.0])
        shape._rotation[axis] = value
        if isinstance(shape, Model):
            shape._vertices_cache = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            winY = viewport[3] - pos.y()
            z = glReadPixels(pos.x(), winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
            
            z_val = float(z[0][0])
            world_pos = gluUnProject(pos.x(), winY, z_val, 
                                   modelview, projection, viewport)
            
            min_dist = float('inf')
            selected = None
            
            for shape in self._shapes:
                if isinstance(shape, (Cube, Model)):
                    dist = np.linalg.norm(
                        np.array(world_pos) - (shape.position.to_array() if isinstance(shape.position, Position) else shape.position)
                    )
                    if dist < min_dist:
                        min_dist = dist
                        selected = shape
            
            if selected:
                for i in range(self.object_list.count()):
                    item = self.object_list.item(i)
                    if item.text() == selected.name:
                        self.object_list.setCurrentItem(item)
                        break


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._renderer = Renderer()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        self._configure_window()
        self._create_tabs_and_panels()
        self._finalize_layout()

    def _connect_signals(self) -> None:
        self.object_list.currentItemChanged.connect(self.object_list_selection_changed)
        self.add_button.clicked.connect(self.add_cube)
        self.apply_button.clicked.connect(self.apply_parameters)
        self.delete_button.clicked.connect(self.delete_cube)

        self.rotation_x_slider.valueChanged.connect(self.update_rotation_x)
        self.rotation_y_slider.valueChanged.connect(self.update_rotation_y)
        self.rotation_z_slider.valueChanged.connect(self.update_rotation_z)

        self.load_model_button.clicked.connect(self._load_model)
        self.save_scene_button.clicked.connect(self._save_scene)
        self.load_scene_button.clicked.connect(self._load_scene)
        self.toggle_axes_button.clicked.connect(self._toggle_axes)
        self.toggle_hidden_button.clicked.connect(self._toggle_hidden_lines)

        self.plane_points_button.clicked.connect(self._create_plane_points)
        self.plane_line_button.clicked.connect(self._create_plane_line)
        self.plane_parallel_button.clicked.connect(self._create_plane_parallel)

    def _configure_window(self) -> None:
        self.setWindowTitle("3D Editor")
        self.setGeometry(100, 100, 1280, 800)

    def _create_tabs_and_panels(self) -> None:
        main_splitter = QSplitter(Qt.Horizontal)
        
        tab_widget = QTabWidget()
        
        object_tab = QWidget()
        object_layout = QVBoxLayout(object_tab)
        
        object_scroll = QScrollArea()
        object_scroll.setWidgetResizable(True)
        object_scroll_content = QWidget()
        object_scroll_layout = QVBoxLayout(object_scroll_content)
        
        self.object_list = QListWidget()
        object_scroll_layout.addWidget(self.object_list)
        object_scroll.setWidget(object_scroll_content)
        object_layout.addWidget(object_scroll)
        
        param_group = QGroupBox("Параметры")
        param_layout = QFormLayout()
        
        basic_group = QGroupBox("Основные")
        basic_layout = QFormLayout()
        self.name_input = QLineEdit("Cube1")
        self.size_input = QLineEdit("1.0")
        self.scale_input = QLineEdit("1.0")
        basic_layout.addRow("Имя", self.name_input)
        basic_layout.addRow("Размер", self.size_input)
        basic_layout.addRow("Масштаб", self.scale_input)
        basic_group.setLayout(basic_layout)
        param_layout.addRow(basic_group)

        position_group = QGroupBox("Позиция")
        position_layout = QFormLayout()
        self.position_x_input = QLineEdit("0.0")
        self.position_y_input = QLineEdit("0.0")
        self.position_z_input = QLineEdit("0.0")
        position_layout.addRow("X", self.position_x_input)
        position_layout.addRow("Y", self.position_y_input)
        position_layout.addRow("Z", self.position_z_input)
        position_group.setLayout(position_layout)
        param_layout.addRow(position_group)

        rotation_group = QGroupBox("Вращение")
        rotation_layout = QVBoxLayout()
        
        for axis, name in [('x', 'X'), ('y', 'Y'), ('z', 'Z')]:
            layout = QGridLayout()
            label = QLabel(f"{name}:")
            slider = QSlider(Qt.Horizontal)
            value_label = QLabel("0°")
            value_label.setMinimumWidth(40)
            
            slider.setRange(0, 360)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(45)
            
            tick_labels = QHBoxLayout()
            for i in range(0, 361, 45):
                lbl = QLabel(f"{i}°")
                lbl.setAlignment(Qt.AlignCenter)
                tick_labels.addWidget(lbl, stretch=1)
            
            slider.valueChanged.connect(lambda v, l=value_label: l.setText(f"{v}°"))
            
            layout.addWidget(label, 0, 0)
            layout.addWidget(slider, 0, 1)
            layout.addWidget(value_label, 0, 2)
            layout.addLayout(tick_labels, 1, 1)
            
            rotation_layout.addLayout(layout)
            setattr(self, f"rotation_{axis}_slider", slider)

        rotation_group.setLayout(rotation_layout)
        param_layout.addRow(rotation_group)

        color_group = QGroupBox("Цвет")
        color_layout = QFormLayout()
        self.color_r_input = QLineEdit("255")
        self.color_g_input = QLineEdit("0")
        self.color_b_input = QLineEdit("0")
        color_layout.addRow("R", self.color_r_input)
        color_layout.addRow("G", self.color_g_input)
        color_layout.addRow("B", self.color_b_input)
        color_group.setLayout(color_layout)
        param_layout.addRow(color_group)

        param_group.setLayout(param_layout)
        object_layout.addWidget(param_group)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Добавить")
        self.apply_button = QPushButton("Применить")
        self.delete_button = QPushButton("Удалить")
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.delete_button)
        object_layout.addLayout(button_layout)

        plane_tab = QWidget()
        plane_layout = QVBoxLayout(plane_tab)
        
        self.plane_group = QGroupBox("Создание плоскости")
        plane_buttons_layout = QVBoxLayout()
        self.plane_points_button = QPushButton("По трем точкам")
        self.plane_line_button = QPushButton("По точке и отрезку")
        self.plane_parallel_button = QPushButton("Параллельная плоскость")
        for btn in [self.plane_points_button, self.plane_line_button, self.plane_parallel_button]:
            plane_buttons_layout.addWidget(btn)
        self.plane_group.setLayout(plane_buttons_layout)
        plane_layout.addWidget(self.plane_group)
        plane_layout.addStretch()

        tools_tab = QWidget()
        tools_layout = QVBoxLayout(tools_tab)
        
        self.load_model_button = QPushButton("Загрузить модель")
        self.save_scene_button = QPushButton("Сохранить сцену")
        self.load_scene_button = QPushButton("Загрузить сцену")
        self.toggle_axes_button = QPushButton("Показать/скрыть оси")
        self.toggle_hidden_button = QPushButton("Показать/скрыть невидимые линии")
        
        for btn in [self.load_model_button, self.save_scene_button, self.load_scene_button,
                   self.toggle_axes_button, self.toggle_hidden_button]:
            tools_layout.addWidget(btn)
        tools_layout.addStretch()

        tab_widget.addTab(object_tab, "Объект")
        tab_widget.addTab(plane_tab, "Плоскости")
        tab_widget.addTab(tools_tab, "Инструменты")

        main_splitter.addWidget(tab_widget)
        main_splitter.addWidget(self._renderer)
        
        main_splitter.setSizes([300, 980])

        self.setCentralWidget(main_splitter)

    def _finalize_layout(self) -> None:
        pass

    def add_cube(self) -> None:
        name = self.name_input.text()
        size = float(self.size_input.text())
        position = [float(self.position_x_input.text()), float(self.position_y_input.text()),
                    float(self.position_z_input.text())]
        color = [float(self.color_r_input.text()) / 255.0,
                 float(self.color_g_input.text()) / 255.0,
                 float(self.color_b_input.text()) / 255.0]

        self._renderer.add_cube(name, position, size, color)
        self.object_list.addItem(name)

    def update_ui_from_shape(self, shape: Shape) -> None:
        """Update UI controls based on selected shape"""
        self.name_input.setText(shape.name)
        
        if isinstance(shape, (Cube, Model)):
            pos = shape.position
            self.position_x_input.setText(str(pos[0] if isinstance(pos, np.ndarray) else pos.x))
            self.position_y_input.setText(str(pos[1] if isinstance(pos, np.ndarray) else pos.y))
            self.position_z_input.setText(str(pos[2] if isinstance(pos, np.ndarray) else pos.z))
            
            if hasattr(shape, 'color'):
                self.color_r_input.setText(str(int(shape.color[0] * 255)))
                self.color_g_input.setText(str(int(shape.color[1] * 255)))
                self.color_b_input.setText(str(int(shape.color[2] * 255)))
            
            if hasattr(shape, 'size'):
                self.size_input.setText(str(shape.size))
                self.scale_input.setText(str(shape.size))
            
            if hasattr(shape, '_rotation'):
                rot = shape._rotation
                self.rotation_x_slider.setValue(int(rot[0]))
                self.rotation_y_slider.setValue(int(rot[1]))
                self.rotation_z_slider.setValue(int(rot[2]))
            elif hasattr(shape, 'rotation'):
                rot = shape.rotation
                self.rotation_x_slider.setValue(int(rot[0]))
                self.rotation_y_slider.setValue(int(rot[1]))
                self.rotation_z_slider.setValue(int(rot[2]))
            else:
                self.rotation_x_slider.setValue(0)
                self.rotation_y_slider.setValue(0)
                self.rotation_z_slider.setValue(0)

    def apply_parameters(self) -> None:
        current_item = self.object_list.currentItem()
        if not current_item:
            return
            
        name = current_item.text()
        for shape in self._renderer._shapes:
            if shape.name == name:
                shape._name = self.name_input.text()
                current_item.setText(shape._name)
                
                position = [
                    float(self.position_x_input.text()),
                    float(self.position_y_input.text()),
                    float(self.position_z_input.text())
                ]
                
                if isinstance(shape, (Cube, Model)):
                    shape._position = Position(*position)
                    
                    if hasattr(shape, 'color'):
                        shape.color = np.array([
                            float(self.color_r_input.text()) / 255.0,
                            float(self.color_g_input.text()) / 255.0,
                            float(self.color_b_input.text()) / 255.0
                        ])
                    
                    if hasattr(shape, 'size'):
                        if isinstance(shape, Model):
                            shape.size = float(self.scale_input.text())
                        else:
                            shape.size = float(self.size_input.text())
                
                break
        
        self._renderer.update()

    def delete_cube(self) -> None:
        current_item = self.object_list.currentItem()
        if current_item:
            name = current_item.text()
            for i, cube in enumerate(self._renderer._shapes):
                if cube.name == name:
                    del self._renderer._shapes[i]
                    break
            self.object_list.takeItem(self.object_list.row(current_item))
            self._renderer.update()

    def update_rotation_x(self, value: int) -> None:
        current_item = self.object_list.currentItem()
        if current_item:
            name = current_item.text()
            for shape in self._renderer._shapes:
                if shape.name == name:
                    self._renderer.update_rotation(shape, 0, value)
                    break
            self._renderer.update()

    def update_rotation_y(self, value: int) -> None:
        current_item = self.object_list.currentItem()
        if current_item:
            name = current_item.text()
            for shape in self._renderer._shapes:
                if shape.name == name:
                    self._renderer.update_rotation(shape, 1, value)
                    break
            self._renderer.update()

    def update_rotation_z(self, value: int) -> None:
        current_item = self.object_list.currentItem()
        if current_item:
            name = current_item.text()
            for shape in self._renderer._shapes:
                if shape.name == name:
                    self._renderer.update_rotation(shape, 2, value)
                    break
            self._renderer.update()

    def object_list_selection_changed(self, current: QListWidgetItem, previous: Optional[QListWidgetItem]) -> None:
        current_item = self.object_list.currentItem()
        if current_item:
            for shape in self._renderer._shapes:
                if shape.name == current_item.text():
                    self.update_ui_from_shape(shape)
                    break

    def _load_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель", "", "OBJ Files (*.obj)"
        )
        if filename:
            name = os.path.splitext(os.path.basename(filename))[0]
            model = Model.from_obj(filename, name)
            self._renderer._shapes.append(model)
            item = QListWidgetItem(model.name)
            self.object_list.addItem(item)
            self.object_list.setCurrentItem(item)
            self.update_ui_from_shape(model)
            self._renderer.update()

    def _create_plane_points(self) -> None:
        dialog = PlanePointsDialog(self)
        if dialog.exec():
            p1, p2, p3 = dialog.get_points()
            plane = Plane.from_points(f"Plane_{len(self._renderer._shapes)}", p1, p2, p3)
            self._renderer._shapes.append(plane)
            self.object_list.addItem(plane.name)
            self._renderer.update()

    def _save_scene(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить сцену", "", "JSON Files (*.json)"
        )
        if filename:
            SceneManager.save_scene(self._renderer._shapes, filename)

    def _load_scene(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить сцену", "", "JSON Files (*.json)"
        )
        if not filename:
            return
        
        with open(filename, 'r', encoding="utf-8") as f:
            scene_data = json.load(f)

        self._renderer._shapes.clear()
        self.object_list.clear()

        if isinstance(scene_data, list):
            shape_list = scene_data
        else:
            shape_list = scene_data.get("shapes", [])

        for shape_info in shape_list:
            shape_type = shape_info.get("type", "").lower()
            name = shape_info.get("name", "Unnamed")
            if shape_type == "cube":
                pos = shape_info.get("position", [0, 0, 0])
                size = shape_info.get("size", 1.0)
                color = shape_info.get("color", [1.0, 1.0, 1.0])
                new_cube = Cube(name, Position(*pos), size, np.array(color))
                if "rotation" in shape_info:
                    new_cube._rotation = np.array(shape_info["rotation"])
                self._renderer._shapes.append(new_cube)
                self.object_list.addItem(new_cube.name)
            elif shape_type == "plane":
                points_list = shape_info.get("points", [[0,0,0],[1,0,0],[0,1,0]])
                positions = [Position(*pt) for pt in points_list]
                plane = Plane(name, positions)
                self._renderer._shapes.append(plane)
                self.object_list.addItem(plane.name)
            elif shape_type == "model":
                obj_path = shape_info.get("objPath", "")
                model = Model.from_obj(obj_path, name)
                if "color" in shape_info:
                    model.color = np.array(shape_info["color"])
                if "position" in shape_info:
                    p = shape_info["position"]
                    model._position = Position(*p)
                if "size" in shape_info:
                    model.size = shape_info["size"]
                if "rotation" in shape_info:
                    model.rotation = np.array(shape_info["rotation"])
                self._renderer._shapes.append(model)
                self.object_list.addItem(model.name)
            # ...handle other shape types if needed...

        self._renderer.update()

    def _create_plane_line(self) -> None:
        dialog = PlaneLineDialog(self)
        if dialog.exec():
            point, line_start, line_end = dialog.get_data()
            plane = Plane.from_point_and_line(
                f"Plane_{len(self._renderer._shapes)}", 
                point, line_start, line_end
            )
            self._renderer._shapes.append(plane)
            self.object_list.addItem(plane.name)
            self._renderer.update()

    def _create_plane_parallel(self) -> None:
        current_item = self.object_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Ошибка", "Выберите плоскость для создания параллельной")
            return

        shape = None
        for s in self._renderer._shapes:
            if s.name == current_item.text():
                shape = s
                break

        if not isinstance(shape, Plane):
            QMessageBox.warning(self, "Ошибка", "Выбранный объект не является плоскостью")
            return

        text, ok = QInputDialog.getText(
            self, 'Новая точка', 
            'Введите координаты точки (x,y,z):', 
            text='0,0,0'
        )
        
        if ok:
            try:
                x, y, z = map(float, text.split(','))
                point = Position(x, y, z)
                plane = Plane.from_point_and_parallel(
                    f"Plane_{len(self._renderer._shapes)}", 
                    point, shape
                )
                self._renderer._shapes.append(plane)
                self.object_list.addItem(plane.name)
                self._renderer.update()
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Неверный формат координат")

    def _toggle_axes(self) -> None:
        """Toggle coordinate axes visibility"""
        self._renderer._show_axes = not self._renderer._show_axes
        self._renderer.update()

    def _toggle_hidden_lines(self) -> None:
        """Toggle hidden lines visibility"""
        self._renderer._show_hidden_lines = not self._renderer._show_hidden_lines
        self._renderer.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            pass


class PlanePointsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Создание плоскости по точкам")
        
        layout = QVBoxLayout()
        
        self.point_inputs = []
        for i in range(3):
            point_group = QGroupBox(f"Точка {i+1}")
            point_layout = QFormLayout()
            
            x = QLineEdit("0.0")
            y = QLineEdit("0.0")
            z = QLineEdit("0.0")
            
            point_layout.addRow("X:", x)
            point_layout.addRow("Y:", y)
            point_layout.addRow("Z:", z)
            
            point_group.setLayout(point_layout)
            layout.addWidget(point_group)
            
            self.point_inputs.append((x, y, z))
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_points(self) -> list[Position]:
        return [
            Position(float(x.text()), float(y.text()), float(z.text()))
            for x, y, z in self.point_inputs
        ]


class PlaneLineDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Создание плоскости по точке и отрезку")
        
        layout = QVBoxLayout()
        
        point_group = QGroupBox("Точка")
        point_layout = QFormLayout()
        self.point_inputs = (QLineEdit("0.0"), QLineEdit("0.0"), QLineEdit("0.0"))
        point_layout.addRow("X:", self.point_inputs[0])
        point_layout.addRow("Y:", self.point_inputs[1])
        point_layout.addRow("Z:", self.point_inputs[2])
        point_group.setLayout(point_layout)
        layout.addWidget(point_group)
        
        line_start_group = QGroupBox("Начало отрезка")
        line_start_layout = QFormLayout()
        self.line_start_inputs = (QLineEdit("0.0"), QLineEdit("0.0"), QLineEdit("0.0"))
        line_start_layout.addRow("X:", self.line_start_inputs[0])
        line_start_layout.addRow("Y:", self.line_start_inputs[1])
        line_start_layout.addRow("Z:", self.line_start_inputs[2])
        line_start_group.setLayout(line_start_layout)
        layout.addWidget(line_start_group)
        
        line_end_group = QGroupBox("Конец отрезка")
        line_end_layout = QFormLayout()
        self.line_end_inputs = (QLineEdit("0.0"), QLineEdit("0.0"), QLineEdit("0.0"))
        line_end_layout.addRow("X:", self.line_end_inputs[0])
        line_end_layout.addRow("Y:", self.line_end_inputs[1])
        line_end_layout.addRow("Z:", self.line_end_inputs[2])
        line_end_group.setLayout(line_end_layout)
        layout.addWidget(line_end_group)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_data(self) -> tuple[Position, Position, Position]:
        point = Position(*(float(x.text()) for x in self.point_inputs))
        line_start = Position(*(float(x.text()) for x in self.line_start_inputs))
        line_end = Position(*(float(x.text()) for x in self.line_end_inputs))
        return point, line_start, line_end


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
