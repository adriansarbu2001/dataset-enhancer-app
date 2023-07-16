from keras.models import Model, load_model


class AIModel(object):
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None

    def get_name(self) -> str:
        return self._model_path.split("/")[-1]

    def get_model(self) -> Model:
        if self._model is None:
            self._model = load_model(self._model_path)
        return self._model
