import sys

from PySide6.QtWidgets import QApplication

from modules.main.repository.ai_repository import AIRepository
from modules.main.repository.dataset_repository import DatasetRepository
from modules.main.service.ai_service import AIService
from modules.main.service.dataset_service import DatasetService
from modules.main.ui.windows.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    models_path = "data/ai_models"
    datasets_path = "data/datasets"
    ai_repository = AIRepository(folder_path=models_path)
    dataset_repository = DatasetRepository(folder_path=datasets_path)
    ai_service = AIService(repository=ai_repository)
    dataset_service = DatasetService(repository=dataset_repository)
    window = MainWindow(ai_service=ai_service, dataset_service=dataset_service)

    window.show()
    app.exec()


if __name__ == "__main__":
    main()
