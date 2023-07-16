import shutil
import os

from modules.main.model.dataset import Dataset
from modules.main.exceptions.repo_exception import RepoException


class DatasetRepository(object):
    def __init__(self, folder_path) -> None:
        self._folder_path = folder_path

    def get(self, dataset_name: str) -> Dataset:
        dataset_path = f"{self._folder_path}/{dataset_name}"
        if not os.path.isdir(dataset_path):
            raise RepoException("Setul de date nu există!")
        return Dataset(dataset_path=dataset_path)

    def add(self, dataset_path: str):
        if not os.path.isdir(dataset_path + "/images") or not os.path.isdir(dataset_path + "/masks"):
            raise RepoException("Folderul setului de date trebuie să conțină un folder \"images\" și un folder \"masks\"!")
        dataset_name = dataset_path.split("/")[-1]
        new_dataset_path = self._folder_path + "/" + dataset_name
        try:
            shutil.copytree(dataset_path, new_dataset_path)
        except Exception:
            raise RepoException("Setul de date nu a putut fi adăugat!")

    def modify(self, dataset_name: str, new_dataset_name: str):
        old_model_path = self._folder_path + "/" + dataset_name
        new_model_path = self._folder_path + "/" + new_dataset_name
        try:
            os.rename(old_model_path, new_model_path)
        except Exception:
            raise RepoException("Setul de date nu a putut fi modificat!")

    def remove(self, dataset_name: str):
        dataset_path = f"{self._folder_path}/{dataset_name}"
        if not os.path.isdir(dataset_path):
            raise RepoException("Setul de date nu există!")
        try:
            shutil.rmtree(dataset_path)
        except Exception:
            raise RepoException("Setul de date nu a putut fi șters!")

    def get_all(self) -> list[Dataset]:
        all_elems = []
        try:
            for dataset_name in os.listdir(self._folder_path):
                path = os.path.join(self._folder_path, dataset_name)
                if os.path.isdir(path):
                    all_elems.append(Dataset(dataset_path=f"{self._folder_path}/{dataset_name}"))
        except Exception:
            raise RepoException("Seturile de date nu au putut fi găsite!")
        return all_elems
