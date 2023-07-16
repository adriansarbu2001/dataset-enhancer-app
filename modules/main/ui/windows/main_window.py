import cv2
import functools

import tensorflow as tf
import numpy as np

from PySide6.QtWidgets import QWidget, QListView, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, \
    QSizePolicy, QProgressBar, QMessageBox
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, QModelIndex, QRect, QThread, Signal, QThreadPool, QRunnable

from modules.main.exceptions.repo_exception import RepoException
from modules.main.exceptions.service_exception import ServiceException
from modules.main.model.ai_model import AIModel
from modules.main.service.ai_service import AIService
from modules.main.service.dataset_service import DatasetService
from modules.main.ui.windows.edit_window import EditWindow
from modules.main.ui.widgets.graphics_video_widget import GraphicsVideoWidget


class MainWindow(QWidget):
    def __init__(self, ai_service: AIService, dataset_service: DatasetService) -> None:
        super().__init__()
        self.setWindowTitle("Dataset Enhancer App")
        self.setGeometry(QRect(50, 50, 1100, 850))
        self._disabled = False

        self._threadpool = QThreadPool(self)
        self._threadpool.setMaxThreadCount(2)
        self._worker = None

        self._selected_ai: str | None = None
        self._selected_dataset: str | None = None
        self._ai_serv: AIService = ai_service
        self._dataset_serv: DatasetService = dataset_service

        self.rgb_video = np.zeros((1, 480, 768, 3))
        self.mask_video = np.zeros((1, 480, 768, 1))
        self._index_in_prediction = -1

        self._edit_window: EditWindow | None = None

        self._w_file_dialog = QFileDialog(self)

        self._m_item_model_ai = QStandardItemModel(self)
        self._update_ai_list()
        self._w_listview_ai = QListView(self)
        self._w_listview_ai.setModel(self._m_item_model_ai)
        self._w_listview_ai.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._m_item_model_dataset = QStandardItemModel(self)
        self._update_dataset_list()
        self._w_listview_dataset = QListView(self)
        self._w_listview_dataset.setModel(self._m_item_model_dataset)
        self._w_listview_dataset.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._w_video_widget = GraphicsVideoWidget(self)

        self._w_left_button = QPushButton("<", self)
        self._w_right_button = QPushButton(">", self)
        self._w_add_ai_button = QPushButton("Adaugă", self)
        self._w_remove_ai_button = QPushButton("Șterge", self)
        self._w_remove_ai_button.setEnabled(False)
        self._w_add_dataset_button = QPushButton("Adaugă", self)
        self._w_remove_dataset_button = QPushButton("Șterge", self)
        self._w_remove_dataset_button.setEnabled(False)
        self._w_select_button = QPushButton("Selectează videoclip", self)
        self._w_edit_button = QPushButton("Modifică și salvează", self)
        self._w_predict_video_button = QPushButton("Detectează pe videoclip", self)
        self._w_predict_video_button.setEnabled(False)
        self._w_predict_frame_button = QPushButton("Detectează pe cadru", self)
        self._w_predict_frame_button.setEnabled(False)

        # self._w_video_player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._w_slider_frame = QSlider(Qt.Horizontal, self)
        self._w_slider_frame.setRange(0, 0)
        self._w_slider_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._w_slider_threshold = QSlider(Qt.Horizontal, self)
        self._w_slider_threshold.setRange(0, 100)
        self._w_slider_threshold.setSliderPosition(int(self._w_video_widget.threshold * 100))
        self._w_slider_morphological_opening = QSlider(Qt.Horizontal, self)
        self._w_slider_morphological_opening.setRange(1, 30)
        self._w_slider_morphological_opening.setSliderPosition(self._w_video_widget.morphological_opening)
        self._w_slider_morphological_closing = QSlider(Qt.Horizontal, self)
        self._w_slider_morphological_closing.setRange(1, 30)
        self._w_slider_morphological_closing.setSliderPosition(self._w_video_widget.morphological_closing)
        self._w_slider_opacity = QSlider(Qt.Horizontal, self)
        self._w_slider_opacity.setRange(0, 100)
        self._w_slider_opacity.setSliderPosition(int(self._w_video_widget.opacity * 100))

        self._w_progress_bar = QProgressBar()
        self._w_progress_bar.setTextVisible(False)

        self._init_layout()
        self._init_connections()

    def _disable_sliders(self):
        self._w_left_button.setEnabled(False)
        self._w_right_button.setEnabled(False)
        self._w_slider_frame.setEnabled(False)
        self._w_slider_threshold.setEnabled(False)
        self._w_slider_morphological_opening.setEnabled(False)
        self._w_slider_morphological_closing.setEnabled(False)
        self._w_slider_opacity.setEnabled(False)

    def _disable_buttons(self):
        self._disabled = True
        self._w_select_button.setEnabled(False)
        self._w_predict_video_button.setEnabled(False)

    def _restore_buttons_state(self):
        self._w_left_button.setEnabled(True)
        self._w_right_button.setEnabled(True)
        self._w_slider_frame.setEnabled(True)
        self._w_select_button.setEnabled(True)
        if self._selected_ai is not None:
            self._w_predict_video_button.setEnabled(True)
        self._w_slider_threshold.setEnabled(True)
        self._w_slider_morphological_opening.setEnabled(True)
        self._w_slider_morphological_closing.setEnabled(True)
        self._w_slider_opacity.setEnabled(True)
        self._disabled = False

    @staticmethod
    def _show_exception_dialog(exception):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Eroare")
        error_dialog.setText("S-a produs o eroare:")
        error_dialog.setInformativeText(str(exception))
        error_dialog.exec()

    @staticmethod
    def _exception_handling(input_function):
        @functools.wraps(input_function)
        def execute_with_exception_handling(*args, **kwargs):
            try:
                return input_function(*args, **kwargs)
            except ServiceException as service_exception:
                MainWindow._show_exception_dialog(service_exception)
            except RepoException as repo_exception:
                MainWindow._show_exception_dialog(repo_exception)
            except Exception as exception:
                MainWindow._show_exception_dialog(exception)

        return execute_with_exception_handling

    @_exception_handling
    def _update_ai_list(self):
        self._m_item_model_ai.clear()
        for ai_model in self._ai_serv.get_all():
            self._m_item_model_ai.appendRow(QStandardItem(ai_model.get_name()[:-3]))

    @_exception_handling
    def _update_dataset_list(self):
        self._m_item_model_dataset.clear()
        for dataset in self._dataset_serv.get_all():
            self._m_item_model_dataset.appendRow(QStandardItem(dataset.get_name()))

    @_exception_handling
    def _on_list_view_ai_index_changed(self, index: QModelIndex):
        self._selected_ai = index.data() + ".h5"
        self._w_remove_ai_button.setEnabled(True)
        self._w_predict_frame_button.setEnabled(True)
        if not self._disabled:
            self._w_predict_video_button.setEnabled(True)

    @_exception_handling
    def _on_list_view_dataset_index_changed(self, index: QModelIndex):
        self._selected_dataset = index.data()
        self._w_remove_dataset_button.setEnabled(True)

    @_exception_handling
    def _on_position_slider_threshold_moved(self, position: int):
        self._w_video_widget.threshold = position / 100.0

    @_exception_handling
    def _on_position_slider_morphological_opening_moved(self, position: int):
        self._w_video_widget.morphological_opening = position

    @_exception_handling
    def _on_position_slider_morphological_closing_moved(self, position: int):
        self._w_video_widget.morphological_closing = position

    @_exception_handling
    def _on_position_slider_opacity_moved(self, position: int):
        self._w_video_widget.opacity = position / 100.0

    @_exception_handling
    def _on_position_slider_frame_moved(self, position: int):
        self._w_video_widget.set_frame_index(position)

    @_exception_handling
    def _on_edit_button_click(self):
        if self._edit_window is None:
            index = self._w_slider_frame.sliderPosition()
            image = self.rgb_video[index]
            mask = self._w_video_widget.get_grayscale_mask_frame_processed(index)
            self._edit_window = EditWindow(self._dataset_serv, image, mask)
            self._edit_window.closed.connect(self._on_edit_window_closed)
            self.setDisabled(True)
            self._edit_window.show()

    @_exception_handling
    def _on_select_button_click(self):
        self._w_file_dialog = QFileDialog(self)
        self._w_file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        self._w_file_dialog.setNameFilter("MP4 files (*.mp4)")
        if self._w_file_dialog.exec() == QFileDialog.Accepted:
            self._w_slider_frame.setValue(0)
            self._w_slider_frame.setRange(0, 0)
            video_capture = cv2.VideoCapture(self._w_file_dialog.selectedFiles()[0])
            no_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self._w_progress_bar.setRange(0, no_frames - 1)
            self._w_progress_bar.setValue(0)

            self._disable_sliders()
            self._disable_buttons()
            self._worker = SelectButtonWorker(video_capture)
            self._worker.updateProgress.connect(self._w_progress_bar.setValue)
            self._worker.updateRgbVideo.connect(lambda rgb_video: (
                setattr(self, "rgb_video", rgb_video),
                setattr(self, "mask_video", np.zeros((*rgb_video.shape[:-1], 1))),
                self._w_slider_frame.setValue(0),
                self._w_slider_frame.setRange(0, self.rgb_video.shape[0] - 1),
                self._w_video_widget.set_rgb_video(self.rgb_video),
                self._w_video_widget.set_mask_video(self.mask_video),
                self._w_video_widget.set_frame_index(0),
                self._w_progress_bar.setValue(0),
                # setattr(self, "worker", None),
                self._restore_buttons_state()
            ))
            self._threadpool.start(ThreadRunnable(self._worker))

    @_exception_handling
    def _on_predict_frame_button_click(self):
        model = self._ai_serv.get(model_name=self._selected_ai)
        self._index_in_prediction = self._w_video_widget.get_frame_index()
        rgb_frame = self.rgb_video[self._index_in_prediction]

        self._disable_buttons()
        self._worker = PredictFrameButtonWorker(model, rgb_frame)
        self._worker.updateRgbVideoFrame.connect(lambda mask_frame: (
            self.mask_video.__setitem__(self._index_in_prediction, mask_frame),
            self._w_video_widget.set_mask_video(self.mask_video),
            self._restore_buttons_state()
        ))
        self._threadpool.start(ThreadRunnable(self._worker))

    @_exception_handling
    def _on_predict_video_button_click(self):
        self._w_progress_bar.setRange(0, self.rgb_video.shape[0] - 1)
        self._w_progress_bar.setValue(0)
        model = self._ai_serv.get(model_name=self._selected_ai)
        batch_size = 4

        self._disable_buttons()
        self._worker = PredictVideoButtonWorker(batch_size, model, self.rgb_video)
        self._worker.updateRgbVideoSlice.connect(lambda params: (
            i := params[0],
            mask_slice := params[1],
            self.mask_video.__setitem__(slice(i, i + batch_size), mask_slice),
            self._w_video_widget.set_mask_video(self.mask_video[i: i + batch_size], mask_range=(i, i + batch_size)),
            self._w_progress_bar.setValue(i + batch_size - 1)
        ))
        self._worker.end.connect(lambda: (
            self._w_progress_bar.setValue(0),
            self._restore_buttons_state()
        ))
        self._threadpool.start(ThreadRunnable(self._worker))

    @_exception_handling
    def _on_left_button_click(self):
        position = self._w_video_widget.get_frame_index()
        position -= 1
        if position >= self._w_slider_frame.minimum().real:
            self._w_video_widget.set_frame_index(position)
            self._w_slider_frame.setSliderPosition(position)

    @_exception_handling
    def _on_right_button_click(self):
        position = self._w_video_widget.get_frame_index()
        position += 1
        if position <= self._w_slider_frame.maximum().real:
            self._w_video_widget.set_frame_index(position)
            self._w_slider_frame.setSliderPosition(position)

    @_exception_handling
    def _on_add_ai_button_click(self):
        self._w_file_dialog = QFileDialog(self)
        self._w_file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self._w_file_dialog.setNameFilter("HDF5 files (*.h5)")
        if self._w_file_dialog.exec() == QFileDialog.Accepted:
            file_paths = self._w_file_dialog.selectedFiles()
            for file_path in file_paths:
                if file_path.endswith(".h5"):
                    self._ai_serv.add(model_path=file_path)
        self._update_ai_list()

    @_exception_handling
    def _on_remove_ai_button_click(self):
        self._ai_serv.remove(model_name=self._selected_ai)
        self._update_ai_list()
        self._w_remove_ai_button.setEnabled(False)
        self._w_predict_frame_button.setEnabled(False)
        self._w_predict_video_button.setEnabled(False)
        self._selected_ai = None

    @_exception_handling
    def _on_add_dataset_button_click(self):
        self._w_file_dialog = QFileDialog(self)
        self._w_file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        self._w_file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        self._w_file_dialog.setOption(QFileDialog.Option.DontResolveSymlinks, True)
        if self._w_file_dialog.exec() == QFileDialog.Accepted:
            folder_paths = self._w_file_dialog.selectedFiles()
            for folder_path in folder_paths:
                self._dataset_serv.add(dataset_path=folder_path)
        self._update_dataset_list()

    @_exception_handling
    def _on_remove_dataset_button_click(self):
        self._dataset_serv.remove(dataset_name=self._selected_dataset)
        self._update_dataset_list()
        self._w_remove_dataset_button.setEnabled(False)
        self._selected_dataset = None

    @_exception_handling
    def _on_item_model_ai_data_changed(self, top_left: QModelIndex, bottom_right: QModelIndex, roles: list[int]):
        new_model_name = top_left.data() + ".h5"
        self._ai_serv.modify(model_name=self._selected_ai, new_model_name=new_model_name)

    @_exception_handling
    def _on_item_model_dataset_data_changed(self, top_left: QModelIndex, bottom_right: QModelIndex, roles: list[int]):
        new_dataset_name = top_left.data()
        self._dataset_serv.modify(dataset_name=self._selected_dataset, new_dataset_name=new_dataset_name)

    @_exception_handling
    def _on_edit_window_closed(self):
        self.setEnabled(True)
        self._edit_window = None

    def _init_layout(self):

        ly1 = QHBoxLayout()
        ly1.addWidget(self._w_select_button)
        ly1.addWidget(self._w_edit_button)
        ly1.addWidget(self._w_predict_frame_button)
        ly1.addWidget(self._w_predict_video_button)

        ly6 = QHBoxLayout()
        ly6.addWidget(self._w_left_button)
        ly6.addWidget(self._w_slider_frame)
        ly6.addWidget(self._w_right_button)

        ly2 = QVBoxLayout()
        ly2.addWidget(self._w_video_widget)
        ly2.addWidget(self._w_progress_bar)
        ly2.addLayout(ly6)
        ly2.addLayout(ly1)

        ly3 = QVBoxLayout()
        ly3.addLayout(ly2)

        ly3.addWidget(QLabel("Pragul măștii:"))
        ly3.addWidget(self._w_slider_threshold)
        ly3.addWidget(QLabel("Transformare morfologică de deschidere:"))
        ly3.addWidget(self._w_slider_morphological_opening)
        ly3.addWidget(QLabel("Transformare morfologică de închidere:"))
        ly3.addWidget(self._w_slider_morphological_closing)
        ly3.addWidget(QLabel("Opacitatea măștii:"))
        ly3.addWidget(self._w_slider_opacity)

        ly6 = QHBoxLayout()
        ly6.addWidget(self._w_add_ai_button)
        ly6.addWidget(self._w_remove_ai_button)

        ly7 = QHBoxLayout()
        ly7.addWidget(self._w_add_dataset_button)
        ly7.addWidget(self._w_remove_dataset_button)

        ly4 = QVBoxLayout()
        ly4.addWidget(self._w_listview_ai)
        ly4.addLayout(ly6)
        ly4.addWidget(QLabel("Seturi de date:"))
        ly4.addWidget(self._w_listview_dataset)
        ly4.addLayout(ly7)

        ly5 = QHBoxLayout()
        ly5.addLayout(ly4)
        ly5.addLayout(ly3)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Modele de AI:"))
        layout.addLayout(ly5)

        self.setLayout(layout)

    def _init_connections(self):
        self._w_listview_ai.clicked.connect(self._on_list_view_ai_index_changed)
        self._m_item_model_ai.dataChanged.connect(self._on_item_model_ai_data_changed)
        self._w_listview_dataset.clicked.connect(self._on_list_view_dataset_index_changed)
        self._m_item_model_dataset.dataChanged.connect(self._on_item_model_dataset_data_changed)

        self._w_add_ai_button.clicked.connect(self._on_add_ai_button_click)
        self._w_remove_ai_button.clicked.connect(self._on_remove_ai_button_click)
        self._w_add_dataset_button.clicked.connect(self._on_add_dataset_button_click)
        self._w_remove_dataset_button.clicked.connect(self._on_remove_dataset_button_click)
        self._w_select_button.clicked.connect(self._on_select_button_click)
        self._w_predict_frame_button.clicked.connect(self._on_predict_frame_button_click)
        self._w_predict_video_button.clicked.connect(self._on_predict_video_button_click)
        self._w_edit_button.clicked.connect(self._on_edit_button_click)
        self._w_left_button.clicked.connect(self._on_left_button_click)
        self._w_right_button.clicked.connect(self._on_right_button_click)

        self._w_slider_frame.sliderMoved.connect(self._on_position_slider_frame_moved)
        self._w_slider_threshold.sliderMoved.connect(self._on_position_slider_threshold_moved)
        self._w_slider_morphological_opening.sliderMoved.connect(self._on_position_slider_morphological_opening_moved)
        self._w_slider_morphological_closing.sliderMoved.connect(self._on_position_slider_morphological_closing_moved)
        self._w_slider_opacity.sliderMoved.connect(self._on_position_slider_opacity_moved)


class ThreadRunnable(QRunnable):
    def __init__(self, qthread):
        super().__init__()
        self._qthread = qthread

    def run(self):
        self._qthread.start()


class SelectButtonWorker(QThread):
    updateProgress = Signal(int)
    updateRgbVideo = Signal(np.ndarray)

    def __init__(self, video_capture):
        QThread.__init__(self)
        self.video_capture = video_capture

    def run(self):
        frames = []
        frame_count = 0
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frame = np.array(frame)
            frame = frame.astype(np.uint8)
            frame = cv2.resize(frame, (768, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
            self.updateProgress.emit(frame_count)
        rgb_video = np.array(frames)
        self.updateRgbVideo.emit(rgb_video)


class PredictVideoButtonWorker(QThread):
    updateRgbVideoSlice = Signal(tuple)
    end = Signal()

    def __init__(self, batch_size, model, rgb_video):
        QThread.__init__(self)
        self.batch_size = batch_size
        self.model = model
        self.rgb_video = rgb_video

    def run(self):
        for i in range(0, self.rgb_video.shape[0], self.batch_size):
            mask_slice = self._predict_mask_on_video(self.model, self.rgb_video[i: i + self.batch_size])
            self.updateRgbVideoSlice.emit((i, mask_slice))
        self.end.emit()

    def _predict_mask_on_video(self, model: AIModel, video: np.array) -> np.array:
        try:
            model = model.get_model()
            generator = tf.data.Dataset \
                .from_tensor_slices(video) \
                .map(lambda frame: tf.numpy_function(func=self._preprocessing, inp=[frame], Tout=[tf.float32])) \
                .batch(4)
            mask_video = []
            for batch in generator:
                mask_video.extend(model.predict(batch, verbose=0))
            return mask_video
        except Exception:
            pass

    def _preprocessing(self, frame: np.array):
        frame = frame.astype(np.float32)
        frame = frame / 255.0
        return frame


class PredictFrameButtonWorker(QThread):
    updateRgbVideoFrame = Signal(np.ndarray)

    def __init__(self, model, rgb_frame):
        QThread.__init__(self)
        self.model = model
        self.rgb_frame = rgb_frame

    def run(self):
        mask_frame = self._predict_mask_on_frame(self.model, self.rgb_frame)
        self.updateRgbVideoFrame.emit(mask_frame)

    def _predict_mask_on_frame(self, model: AIModel, frame: np.array) -> np.array:
        try:
            model = model.get_model()
            frame = self._preprocessing(frame)
            mask_frame = model.predict(np.array([frame]), verbose=0)
            return mask_frame
        except Exception as e:
            print(e)

    def _preprocessing(self, frame: np.array):
        frame = frame.astype(np.float32)
        frame = frame / 255.0
        return frame
