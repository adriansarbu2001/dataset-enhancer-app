import cv2
import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
from PySide6.QtGui import QImage, QPixmap, QPainter, QPaintEvent


class GraphicsVideoWidget(QGraphicsView):
    def __init__(self, parent: QWidget | None):
        super().__init__(parent)

        self._rgb_video = np.zeros((1, 480, 768, 3))
        self._mask_video = np.zeros((1, 480, 768, 1))
        self._frame_index = 0
        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        _, self.video_height, self.video_width, _ = self._rgb_video.shape

        self.threshold = 0.4
        self.morphological_opening = 5
        self.morphological_closing = 5
        self.opacity = 0.3

    def set_rgb_video(self, rgb_video: np.array):
        self._rgb_video = rgb_video
        _, self.video_height, self.video_width, _ = self._rgb_video.shape

    def set_mask_video(self, mask_video: np.array, mask_range: tuple[int, int] = None):
        if mask_range is None:
            self._mask_video = mask_video
        else:
            self._mask_video[mask_range[0]:mask_range[1]] = mask_video

    def set_frame_index(self, index: int):
        self._frame_index = index

    def get_frame_index(self):
        return self._frame_index

    def get_grayscale_mask_frame_processed(self, index: int):
        mask = self._mask_video[index]

        # apply threshold
        mask = np.where(mask >= self.threshold, 1.0, 0.0)

        # remove background noise
        kernel = np.ones((self.morphological_opening, self.morphological_opening), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # remove foreground noise
        kernel = np.ones((self.morphological_closing, self.morphological_closing), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask

    def get_argb_mask_frame_processed(self, index: int):
        mask = self.get_grayscale_mask_frame_processed(index)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        red_channel = mask[:, :, 2]
        mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2] + 1))
        mask[:, :, 0] = red_channel
        mask[:, :, 3] = self.opacity * red_channel
        mask = mask.astype(np.uint8)
        return mask

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)

        video_image = QImage(
            self._rgb_video[self._frame_index].data,
            self.video_width,
            self.video_height,
            QImage.Format.Format_RGB888
        )
        video_pixmap = QPixmap.fromImage(video_image)

        if self._frame_index < self._mask_video.shape[0]:
            mask_image = QImage(
                self.get_argb_mask_frame_processed(self._frame_index).data,
                self.video_width,
                self.video_height,
                QImage.Format.Format_RGBA8888
            )

            painter = QPainter(video_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawImage(0, 0, mask_image)
            painter.end()

        self._scene.clear()
        self._scene.addPixmap(video_pixmap)
