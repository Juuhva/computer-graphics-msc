import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QFrame
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def generate_knot_vector(n_ctrlpts, degree):
    n_knots = n_ctrlpts + degree + 1
    knots = np.zeros(n_knots)
    knots[degree:n_ctrlpts + 1] = np.linspace(0, 1, n_ctrlpts - degree + 1)
    knots[n_ctrlpts + 1:] = 1
    return knots

def bspline_basis(i, p, u, knots):
    if p == 0:
        return 1.0 if (knots[i] <= u < knots[i + 1]) or (
            np.isclose(u, knots[-1]) and np.isclose(u, knots[i + 1])
        ) else 0.0
    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]
    term1 = 0.0 if denom1 == 0 else (u - knots[i]) / denom1 * bspline_basis(i, p - 1, u, knots)
    term2 = 0.0 if denom2 == 0 else (knots[i + p + 1] - u) / denom2 * bspline_basis(i + 1, p - 1, u, knots)
    return term1 + term2

def draw_nurbs_curve(control_points, weights, degree, smoothness = 200):
    n = len(control_points)
    if n <= degree:
        return np.array([])
    knots = generate_knot_vector(n, degree)
    u_start, u_end = knots[degree], knots[-degree - 1]
    u_vals = np.linspace(u_start, u_end, smoothness)
    curve = []
    for u in u_vals:
        numerator = np.zeros(2)
        denominator = 0.0
        for i in range(n):
            Ni = bspline_basis(i, degree, u, knots)
            wNi = weights[i] * Ni
            numerator += wNi * np.array(control_points[i])
            denominator += wNi
        if denominator > 1e-8:
            curve.append(numerator / denominator)
    return np.array(curve)

class NURBSEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NURBS / B-Spline Curve drawing")
        self.resize(1000, 600)
        self.control_points = []
        self.weights = []
        self.degree = 3
        self.dragging_point = None
        self.show_labels = True
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)
        degree_layout = QHBoxLayout()
        degree_layout.addWidget(QLabel("Degree:"))
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 10)
        self.degree_spin.setValue(self.degree)
        self.degree_spin.valueChanged.connect(self.update_plot)
        degree_layout.addWidget(self.degree_spin)
        left_layout.addLayout(degree_layout)
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smoothness:"))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(2, 2000)
        self.smooth_spin.setValue(200)
        self.smooth_spin.setSingleStep(10)
        self.smooth_spin.valueChanged.connect(self.on_smoothness_changed)
        smooth_layout.addWidget(self.smooth_spin)
        left_layout.addLayout(smooth_layout)
        self.smoothness = self.smooth_spin.value()
        self.label_checkbox = QCheckBox("Show Points")
        self.label_checkbox.setChecked(True)
        self.label_checkbox.stateChanged.connect(self.toggle_labels)
        left_layout.addWidget(self.label_checkbox)
        left_layout.addWidget(QLabel("Weights:"))
        scroll = QScrollArea()
        scroll_widget = QFrame()
        self.weights_layout = QVBoxLayout()
        scroll_widget.setLayout(self.weights_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll)
        clear_btn = QPushButton("Clear canvas")
        clear_btn.clicked.connect(self.clear_points)
        left_layout.addWidget(clear_btn)
        left_layout.addStretch()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, 3)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.grid(True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.update_plot()

    def clear_points(self):
        self.control_points.clear()
        self.weights.clear()
        while self.weights_layout.count():
            item = self.weights_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.update_plot()

    def add_weight_input(self, idx, value=1.0):
        layout = QHBoxLayout()
        label = QLabel(f"P{idx+1}:")
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 10.0)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.valueChanged.connect(self.update_weights)
        layout.addWidget(label)
        layout.addWidget(spin)
        wrapper = QWidget()
        wrapper.setLayout(layout)
        self.weights_layout.addWidget(wrapper)
        return spin

    def update_weights(self):
        new_weights = []
        for i in range(self.weights_layout.count()):
            w_widget = self.weights_layout.itemAt(i).widget()
            if w_widget:
                spin = w_widget.findChild(QDoubleSpinBox)
                if spin:
                    new_weights.append(spin.value())
        self.weights = new_weights
        self.update_plot()

    def toggle_labels(self):
        self.show_labels = self.label_checkbox.isChecked()
        self.update_plot()

    def on_smoothness_changed(self, val):
        self.smoothness = val
        self.update_plot()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        for i, (px, py) in enumerate(self.control_points):
            if abs(px - x) < 0.2 and abs(py - y) < 0.2:
                self.dragging_point = i
                return
        self.control_points.append([x, y])
        self.weights.append(1.0)
        self.add_weight_input(len(self.control_points) - 1, 1.0)
        self.degree_spin.setMaximum(max(1, len(self.control_points) - 1))
        self.update_plot()

    def on_motion(self, event):
        if self.dragging_point is None or event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            self.control_points[self.dragging_point] = [x, y]
            self.update_plot()

    def on_release(self, event):
        self.dragging_point = None

    def update_plot(self):
        self.degree = self.degree_spin.value()
        self.degree_spin.setMaximum(max(1, len(self.control_points) - 1))
        self.ax.clear()
        self.ax.grid(False)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        if self.control_points:
            xs, ys = zip(*self.control_points)
            self.ax.plot(xs, ys, 'o--', color='gray', label='Control Polygon')
            if self.show_labels:
                for i, (x, y) in enumerate(self.control_points):
                    self.ax.text(x + 0.15, y + 0.15, f"P{i+1}", color='red', fontsize=9)
        if len(self.control_points) > self.degree:
            w = self.weights if len(self.weights) == len(self.control_points) else [1.0] * len(self.control_points)
            curve = draw_nurbs_curve(self.control_points, w, self.degree, smoothness=self.smoothness)
            if len(curve):
                cx, cy = curve[:, 0], curve[:, 1]
                self.ax.plot(cx, cy, 'b', label=f'Curve (deg={self.degree})')
        self.ax.legend()
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = NURBSEditor()
    w.show()
    sys.exit(app.exec())
