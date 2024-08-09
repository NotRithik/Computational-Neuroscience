import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QLineEdit, QFileDialog, QSizePolicy, QCheckBox
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QFont

class CanvasWidget(QWidget):
    def __init__(self, Ne, Ni):
        super().__init__()
        self.Ne = Ne
        self.Ni = Ni
        self.brush_size = 10
        self.brush_strength = 0.0
        self.relative_brush = True  # Default brush mode is relative

        # Set full matrix dimensions according to the Izhikevich model
        self.full_width = Ne + Ni
        self.full_height = Ne + Ni

        # Initialize the synaptic weights with random values
        self.synaptic_weights = np.zeros((self.full_height, self.full_width), dtype=np.float32)
        self.initialize_synaptic_weights()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def initialize_synaptic_weights(self):
        Ne = self.Ne
        Ni = self.Ni

        # Random values for excitatory and inhibitory neurons
        re = np.random.rand(Ne)
        ri = np.random.rand(Ni)

        # Initialize neuron parameters
        self.a = np.concatenate([0.02 * np.ones(Ne), 0.02 + 0.08 * ri])
        self.b = np.concatenate([0.2 * np.ones(Ne), 0.25 - 0.05 * ri])
        self.c = np.concatenate([-65 + 15 * re**2, -65 * np.ones(Ni)])
        self.d = np.concatenate([8 - 6 * re**2, 2 * np.ones(Ni)])

        # Synaptic weight matrix: 0.5 for excitatory, -1 for inhibitory
        excitatory_weights = 0.5 * np.random.rand(Ne + Ni, Ne)
        inhibitory_weights = -np.random.rand(Ne + Ni, Ni)
        self.synaptic_weights = np.hstack([excitatory_weights, inhibitory_weights])

        # Set self-connections to 0
        np.fill_diagonal(self.synaptic_weights, 0)


    def update_display_canvas(self):
        self.update()

    def resizeEvent(self, event):
        self.update_display_canvas()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Calculate the size and position of the canvas based on the current window size
        widget_width = self.width()
        widget_height = self.height()

        aspect_ratio = self.full_width / self.full_height
        if widget_width / widget_height > aspect_ratio:
            canvas_height = widget_height
            canvas_width = int(widget_height * aspect_ratio)
        else:
            canvas_width = widget_width
            canvas_height = int(widget_width / aspect_ratio)

        x_offset = (widget_width - canvas_width) // 2
        y_offset = (widget_height - canvas_height) // 2

        # Draw neuron indices on the top and left
        painter.setPen(QPen(Qt.black))
        font = QFont("Arial", 8)
        painter.setFont(font)
        for i in range(0, self.full_width, 100):
            painter.drawText(x_offset + i * canvas_width // self.full_width, y_offset - 5, str(i))
        for j in range(0, self.full_height, 100):
            painter.drawText(x_offset - 25, y_offset + j * canvas_height // self.full_height + 5, str(j))

        # Draw the synaptic weights
        for y in range(canvas_height):
            for x in range(canvas_width):
                # Map pixel coordinates to the full synaptic weight matrix
                syn_x = int(x * self.full_width / canvas_width)
                syn_y = int(y * self.full_height / canvas_height)
                value = self.synaptic_weights[syn_y, syn_x]
                
                # Map the value to a color
                if value > 0:
                    intensity = int(min(255, 255 * value))
                    color = QColor(255, 255 - intensity, 255 - intensity)  # Darker red as the value increases
                elif value < 0:
                    intensity = int(min(255, 255 * abs(value)))
                    color = QColor(255 - intensity, 255 - intensity, 255)  # Darker blue as the value decreases
                else:
                    color = QColor(255, 255, 255)  # White for zero

                painter.setPen(QPen(color))
                painter.drawPoint(x + x_offset, y + y_offset)

        # Draw the line separating excitatory and inhibitory neuron weights
        sep_line_position = int(self.Ne * canvas_width / self.full_width) + x_offset
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(sep_line_position, y_offset, sep_line_position, y_offset + canvas_height)

    def mouseMoveEvent(self, event):
        self.apply_brush(event.x(), event.y())
        self.update()

    def mousePressEvent(self, event):
        self.apply_brush(event.x(), event.y())
        self.update()

    def apply_brush(self, x, y):
        # Calculate the size and position of the canvas based on the current window size
        widget_width = self.width()
        widget_height = self.height()

        aspect_ratio = self.full_width / self.full_height
        if widget_width / widget_height > aspect_ratio:
            canvas_height = widget_height
            canvas_width = int(widget_height * aspect_ratio)
        else:
            canvas_width = widget_width
            canvas_height = int(widget_width / aspect_ratio)

        x_offset = (widget_width - canvas_width) // 2
        y_offset = (widget_height - canvas_height) // 2

        if x_offset <= x <= x_offset + canvas_width and y_offset <= y <= y_offset + canvas_height:
            # Map the click coordinates to the full synaptic weight matrix
            syn_x = int((x - x_offset) * self.full_width / canvas_width)
            syn_y = int((y - y_offset) * self.full_height / canvas_height)

            for i in range(-self.brush_size, self.brush_size):
                for j in range(-self.brush_size, self.brush_size):
                    if 0 <= syn_x + i < self.full_width and 0 <= syn_y + j < self.full_height:
                        if np.sqrt(i ** 2 + j ** 2) <= self.brush_size:
                            if not (self.window().prevent_self_connection_toggle.isChecked() and syn_x + i == syn_y + j):
                                if self.relative_brush:
                                    self.synaptic_weights[syn_y + j, syn_x + i] += self.brush_strength
                                else:
                                    self.synaptic_weights[syn_y + j, syn_x + i] = self.brush_strength

        self.update_display_canvas()

        if self.window().auto_update:
            self.window().update_simulation()


    def get_canvas_data(self):
        return self.synaptic_weights

    def set_canvas_data(self, data):
        self.synaptic_weights = data
        self.update_display_canvas()

    def get_neuron_params(self):
        return self.a, self.b, self.c, self.d

    def set_brush_size(self, size):
        self.brush_size = size

    def set_brush_strength(self, strength):
        self.brush_strength = strength

    def set_brush_mode(self, relative):
        self.relative_brush = relative

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.auto_update = True
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Synaptic Weight Simulator")
        self.setGeometry(100, 100, 1300, 700)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        Ne, Ni = 800, 200  # Define number of excitatory and inhibitory neurons

        # Left side: Canvas and controls
        canvas_layout = QVBoxLayout()
        self.canvas_widget = CanvasWidget(Ne, Ni)
        canvas_layout.addWidget(self.canvas_widget)

        # Labels to indicate the neuron types
        label_layout = QHBoxLayout()
        excitatory_label = QLabel("Excitatory")
        excitatory_label.setFont(QFont("Arial", 12))
        inhibitory_label = QLabel("Inhibitory")
        inhibitory_label.setFont(QFont("Arial", 12))
        label_layout.addWidget(excitatory_label)
        label_layout.addStretch()
        label_layout.addWidget(inhibitory_label)
        canvas_layout.addLayout(label_layout)

        # Brush Size Controls
        brush_size_layout = QHBoxLayout()
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
        brush_size_layout.addWidget(QLabel("Brush Size"))
        brush_size_layout.addWidget(self.brush_size_slider)

        self.brush_size_input = QLineEdit("10")
        self.brush_size_input.setFixedWidth(40)
        self.brush_size_input.editingFinished.connect(self.update_brush_size_from_input)
        brush_size_layout.addWidget(self.brush_size_input)

        canvas_layout.addLayout(brush_size_layout)

        # Brush Strength Controls
        brush_strength_layout = QHBoxLayout()
        self.brush_strength_slider = QSlider(Qt.Horizontal)
        self.brush_strength_slider.setRange(-100, 100)
        self.brush_strength_slider.setValue(0)
        self.brush_strength_slider.valueChanged.connect(self.update_brush_strength)
        brush_strength_layout.addWidget(QLabel("Brush Strength"))
        brush_strength_layout.addWidget(self.brush_strength_slider)

        self.brush_strength_input = QLineEdit("0")
        self.brush_strength_input.setFixedWidth(40)
        self.brush_strength_input.editingFinished.connect(self.update_brush_strength_from_input)
        brush_strength_layout.addWidget(self.brush_strength_input)

        canvas_layout.addLayout(brush_strength_layout)

        # Brush Mode Toggle
        self.brush_mode_toggle = QCheckBox("Relative Brush Mode")
        self.brush_mode_toggle.setChecked(True)  # Default to relative mode
        self.brush_mode_toggle.toggled.connect(self.toggle_brush_mode)
        canvas_layout.addWidget(self.brush_mode_toggle)

        # Auto-Update Toggle
        self.auto_update_toggle = QCheckBox("Auto-Update Firings")
        self.auto_update_toggle.setChecked(True)  # Default to auto-update enabled
        self.auto_update_toggle.toggled.connect(self.toggle_auto_update)
        canvas_layout.addWidget(self.auto_update_toggle)

        # Self-Connection Toggle
        self.prevent_self_connection_toggle = QCheckBox("Prevent Self-Connections")
        self.prevent_self_connection_toggle.setChecked(True)  # Default to preventing self-connections
        canvas_layout.addWidget(self.prevent_self_connection_toggle)


        # Save and Load Buttons
        file_buttons_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_synaptic_weights)
        file_buttons_layout.addWidget(save_button)

        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_synaptic_weights)
        file_buttons_layout.addWidget(load_button)

        canvas_layout.addLayout(file_buttons_layout)

        layout.addLayout(canvas_layout)

        # Right side: Firing plot
        plot_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_simulation)
        plot_layout.addWidget(self.update_button)

        layout.addLayout(plot_layout)

    def update_brush_size(self, value):
        self.canvas_widget.set_brush_size(value)
        self.brush_size_input.setText(str(value))

    def update_brush_size_from_input(self):
        value = int(self.brush_size_input.text())
        self.canvas_widget.set_brush_size(value)
        self.brush_size_slider.setValue(value)

    def update_brush_strength(self, value):
        self.canvas_widget.set_brush_strength(value / 100.0)
        self.brush_strength_input.setText(str(value))

    def update_brush_strength_from_input(self):
        value = int(self.brush_strength_input.text())
        self.canvas_widget.set_brush_strength(value / 100.0)
        self.brush_strength_slider.setValue(value)

    def toggle_brush_mode(self, checked):
        self.canvas_widget.set_brush_mode(checked)

    def toggle_auto_update(self, checked):
        self.auto_update = checked

    def update_simulation(self):
        synaptic_weights = self.canvas_widget.get_canvas_data()
        a, b, c, d = self.canvas_widget.get_neuron_params()
        self.simulate_firings(synaptic_weights, a, b, c, d)

    def simulate_firings(self, synaptic_weights, a, b, c, d):
        if not self.auto_update:
            return

        Ne, Ni = 800, 200
        S = synaptic_weights

        v = -65 * np.ones(Ne + Ni)
        u = b * v
        firings = []
        t = 0

        for _ in range(1000):
            I = np.concatenate((5 * np.random.randn(Ne), 2 * np.random.randn(Ni)))
            fired = np.where(v >= 30)[0]
            firings.extend([[t, neuron] for neuron in fired])
            v[fired] = c[fired]
            u[fired] += d[fired]
            I += np.sum(S[:, fired], axis=1)
            v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)
            v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)
            u += a * (b * v - u)
            t += 1

        # Plot firings
        self.ax.cla()
        if firings:
            firing_data = np.array(firings)
            self.ax.scatter(firing_data[:, 0], firing_data[:, 1], s=1)
        self.ax.set_xlim([0, 1000])
        self.ax.set_ylim([0, Ne + Ni])
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Neuron Index')
        self.ax.set_title('Neuron Firings')
        self.canvas.draw()

    def save_synaptic_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Synaptic Weights", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'w') as file:
                data = self.canvas_widget.get_canvas_data().tolist()
                json.dump(data, file)

    def load_synaptic_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Synaptic Weights", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                data = json.load(file)
                self.canvas_widget.set_canvas_data(np.array(data))
                if self.auto_update:
                    self.update_simulation()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
