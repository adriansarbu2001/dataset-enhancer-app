import cv2
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QImage, QPainter, QPaintEvent, QMouseEvent
from PySide6.QtCore import Qt, QPoint


class CanvasWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_array: np.array = None
        self.mask_array: np.array = None
        self.brush_size = 20
        self._left_pressed = False
        self._right_pressed = False

        self.setMouseTracking(True)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        if self.image_array is not None:
            image = self._convert_numpy_image_to_qimage(self.image_array)
            painter.drawImage(0, 0, image)
        if self.mask_array is not None:
            mask = self._convert_numpy_mask_to_qimage(self.mask_array)
            painter.drawImage(0, 0, mask)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._left_pressed = True
            self._draw_pothole_circle(event.pos())
        if event.button() == Qt.RightButton:
            self._right_pressed = True
            self._draw_background_circle(event.pos())
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.underMouse():
            if self._left_pressed:
                self._draw_pothole_circle(event.pos())
            elif self._right_pressed:
                self._draw_background_circle(event.pos())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._left_pressed = False
        self._right_pressed = False

    def _draw_pothole_circle(self, pos: QPoint):
        center_col = pos.x().real
        center_row = pos.y().real
        rows, cols, _ = self.mask_array.shape
        r = self.brush_size // 2
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
        self.mask_array[distance <= r] = 255

    def _draw_background_circle(self, pos: QPoint):
        center_col = pos.x().real
        center_row = pos.y().real
        rows, cols, _ = self.mask_array.shape
        r = self.brush_size // 2
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
        self.mask_array[distance <= r] = 0

    def _convert_numpy_image_to_qimage(self, array: np.array):
        height, width, channel = array.shape
        qimage = QImage(array.data, width, height, QImage.Format.Format_RGB888)
        return qimage

    def _convert_numpy_mask_to_qimage(self, array: np.array):
        height, width, channel = array.shape

        mask = array
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        red_channel = mask[:, :, 2]
        mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2] + 1))
        mask[:, :, 0] = red_channel
        mask[:, :, 3] = 0.3 * red_channel
        mask = mask.astype(np.uint8)

        qimage = QImage(mask.data, width, height, QImage.Format_RGBA8888)
        return qimage
