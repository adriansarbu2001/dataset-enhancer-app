import cv2
import os

import numpy as np


class Dataset(object):
    def __init__(self, dataset_path: str) -> None:
        self._dataset_path = dataset_path

    def get_name(self) -> str:
        return self._dataset_path.split("/")[-1]

    def add_image_wih_mask(self, image_array: np.array, mask_array: np.array):
        filenames = os.listdir(self._dataset_path + "/images")

        contor = 0
        filename = str(contor).zfill(6) + ".png"

        while filename in filenames:
            contor += 1
            filename = str(contor).zfill(6) + ".png"

        bgr_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self._dataset_path + "/images/" + filename, bgr_array)
        cv2.imwrite(self._dataset_path + "/masks/" + filename, mask_array)
