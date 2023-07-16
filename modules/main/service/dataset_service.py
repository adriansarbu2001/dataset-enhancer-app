import numpy as np

from modules.main.model.dataset import Dataset
from modules.main.exceptions.service_exception import ServiceException
from modules.main.repository.dataset_repository import DatasetRepository


class DatasetService(object):
    def __init__(self, repository: DatasetRepository) -> None:
        self._repo = repository

    def save_pair_to_dataset(self, dataset_name: str, image_array: np.array, mask_array: np.array):
        dataset = self._repo.get(dataset_name=dataset_name)
        try:
            dataset.add_image_wih_mask(image_array=image_array, mask_array=mask_array)
        except Exception:
            raise ServiceException("Imagini invalide!")

    def add(self, dataset_path: str):
        self._repo.add(dataset_path=dataset_path)

    def modify(self, dataset_name: str, new_dataset_name: str):
        self._repo.modify(dataset_name=dataset_name, new_dataset_name=new_dataset_name)

    def remove(self, dataset_name: str):
        self._repo.remove(dataset_name=dataset_name)

    def get_all(self) -> list[Dataset]:
        return self._repo.get_all()
