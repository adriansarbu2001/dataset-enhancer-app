import numpy as np
import tensorflow as tf

from modules.main.model.ai_model import AIModel
from modules.main.exceptions.service_exception import ServiceException
from modules.main.repository.ai_repository import AIRepository


class AIService(object):
    def __init__(self, repository: AIRepository) -> None:
        self._repo = repository

    def add(self, model_path: str):
        self._repo.add(model_path=model_path)

    def modify(self, model_name: str, new_model_name: str):
        self._repo.modify(model_name=model_name, new_model_name=new_model_name)

    def remove(self, model_name: str):
        self._repo.remove(model_name=model_name)

    def get(self, model_name: str):
        return self._repo.get(model_name=model_name)

    def get_all(self) -> list[AIModel]:
        return self._repo.get_all()
