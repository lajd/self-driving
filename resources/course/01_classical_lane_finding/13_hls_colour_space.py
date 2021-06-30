import cv2
import numpy as np


def rgb_to_hls(image: np.ndarray):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return hls
