import shutil
import os

from modules.main.model.ai_model import AIModel
from modules.main.exceptions.repo_exception import RepoException


class AIRepository(object):
    def __init__(self, folder_path: str) -> None:
        self._folder_path = folder_path

    def get(self, model_name: str) -> AIModel:
        model_path = f"{self._folder_path}/{model_name}"
        if not os.path.isfile(model_path):
            raise RepoException("Modelul nu există!")
        return AIModel(model_path=model_path)

    def add(self, model_path: str):
        model_name = model_path.split("/")[-1]
        if not model_name.endswith(".h5"):
            raise RepoException("Numele trebuie să se termine cu extensia .h5!")
        new_model_path = self._folder_path + "/" + model_name
        try:
            shutil.copy(model_path, new_model_path)
        except Exception:
            raise RepoException("Modelul nu a putut fi adăugat!")

    def modify(self, model_name: str, new_model_name: str):
        if not new_model_name.endswith(".h5"):
            raise RepoException("Numele trebuie să se termine cu extensia .h5!")
        old_model_path = self._folder_path + "/" + model_name
        new_model_path = self._folder_path + "/" + new_model_name
        try:
            os.rename(old_model_path, new_model_path)
        except Exception:
            raise RepoException("Modelul nu a putut fi modificat!")

    def remove(self, model_name: str):
        model_path = f"{self._folder_path}/{model_name}"
        if not os.path.isfile(model_path):
            raise RepoException("Modelul nu există!")
        try:
            os.remove(model_path)
        except Exception:
            raise RepoException("Modelul nu a putut fi șters!")

    def get_all(self) -> list[AIModel]:
        all_elems = []
        try:
            for filename in os.listdir(self._folder_path):
                if filename.endswith(".h5"):
                    all_elems.append(AIModel(model_path=f"{self._folder_path}/{filename}"))
        except Exception:
            raise RepoException("Modelele nu au putut fi găsite!")
        return all_elems
