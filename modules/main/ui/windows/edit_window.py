import functools

import numpy as np

from PySide6.QtGui import QStandardItemModel, QStandardItem, QCloseEvent
from PySide6.QtWidgets import QMainWindow, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QListView, \
    QSizePolicy, QMessageBox
from PySide6.QtCore import Qt, Signal, QModelIndex

from modules.main.exceptions.repo_exception import RepoException
from modules.main.exceptions.service_exception import ServiceException
from modules.main.service.dataset_service import DatasetService
from modules.main.ui.widgets.canvas_widget import CanvasWidget


class EditWindow(QMainWindow):
    closed = Signal()

    def __init__(self, dataset_service: DatasetService, image: np.array, mask: np.array):
        super().__init__()
        self.setFixedSize(768 + 18 + 250 + 12, 560)

        self._selected_dataset: str | None = None
        self._dataset_serv = dataset_service

        self._w_canvas = CanvasWidget(self)
        self._w_canvas.image_array = image
        self._w_canvas.mask_array = mask

        self._m_item_model_dataset = QStandardItemModel(self)
        for dataset in self._dataset_serv.get_all():
            self._m_item_model_dataset.appendRow(QStandardItem(dataset.get_name()))
        self._w_listview_dataset = QListView(self)
        self._w_listview_dataset.setModel(self._m_item_model_dataset)
        self._w_listview_dataset.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._w_save_button = QPushButton("Salvează", self)
        self._w_save_button.setEnabled(False)

        self._w_slider = QSlider(Qt.Horizontal, self)
        self._w_slider.setMinimum(1)
        self._w_slider.setMaximum(50)
        self._w_slider.setValue(self._w_canvas.brush_size)

        self._init_layout()
        self._init_connections()

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
                EditWindow._show_exception_dialog(service_exception)
            except RepoException as repo_exception:
                EditWindow._show_exception_dialog(repo_exception)
            except Exception as exception:
                EditWindow._show_exception_dialog(exception)

        return execute_with_exception_handling

    @_exception_handling
    def _on_slider_value_changed(self, size: int):
        self._w_canvas.brush_size = size

    @_exception_handling
    def _on_list_view_ai_index_changed(self, index: QModelIndex):
        self._selected_dataset = index.data()
        self._w_save_button.setEnabled(True)

    @_exception_handling
    def _on_save_button_click(self):
        self._dataset_serv.save_pair_to_dataset(dataset_name=self._selected_dataset,
                                                image_array=self._w_canvas.image_array,
                                                mask_array=self._w_canvas.mask_array)
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self.closed.emit()
        super().closeEvent(event)

    def _init_layout(self):
        ly1 = QHBoxLayout()
        ly1.addWidget(QLabel("Dimensiunea creionului:"))
        ly1.addWidget(self._w_slider)

        ly2 = QVBoxLayout()
        ly2.addWidget(self._w_canvas)
        ly2.addWidget(self._w_save_button)
        ly2.addLayout(ly1)

        layout = QHBoxLayout()
        layout.addWidget(self._w_listview_dataset)
        layout.addLayout(ly2)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _init_connections(self):
        self._w_listview_dataset.clicked.connect(self._on_list_view_ai_index_changed)
        self._w_save_button.clicked.connect(self._on_save_button_click)
        self._w_slider.valueChanged.connect(self._on_slider_value_changed)
