import os
import pickle
import glob
import sys
import numpy as np
import cv2

import os
import numpy as np
import cv2
import sys
import pickle
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('../'))

from tools.camera import undistort_image
from tools.image_utils import (
    grayscale,
    mask_with_region_of_interest,
    get_canny_pixels,
    gaussian_blur,
    extract_saturation_channel,
    sobel_thresholding
)
from tools.misc import show_images_in_columns
from tools.camera import undistort_image, get_camera_calibration, get_perspective_transform

sys.path.insert(0, os.path.abspath('../'))
import matplotlib.pyplot as plt
from tools.camera import get_camera_calibration as get_camera_calibration_, undistort_image
from tools.misc import show_images_in_columns

PROJECT_ROOT_DIR = '/home/jon/PycharmProjects/self-driving/CarND_Advanced_Lane_Lines_P2'
CAMERA_CALIBRATION_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, 'camera_cal'))
CHESSBOARD_DIMENSIONS = (9, 6)
CALIBRATION_IMAGE_FILE_PATHS = glob.glob(f'{CAMERA_CALIBRATION_DIR}/*.jpg')
IMAGE_FORMAT = 'BGR'
CALIBRATION_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, 'resources', 'camera_calibration.pkl')
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT_DIR, 'test_images')

# ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, calibration_images = get_camera_calibration()

with open(CALIBRATION_FILE_PATH, 'rb') as f:
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, calibration_points = pickle.load(f)


reference_image_fp = os.path.join(TEST_IMAGES_DIR, os.listdir(TEST_IMAGES_DIR)[0])
ref_image = plt.imread(reference_image_fp)

undistorted_image = undistort_image(ref_image, camera_matrix=mtx, distortion_matrix=dist)

M, Minv, gray_img_shape = get_perspective_transform(undistorted_image, ref_image, x_proportion_bottom=0.5 / 10, x_proportion_top=4 / 10)


PROJECT_ROOT_DIR = '/home/jon/PycharmProjects/self-driving/CarND_Advanced_Lane_Lines_P2'
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT_DIR, 'test_images')

VERTICES_FN: Callable[[np.ndarray], List[Tuple[int, int]]] = lambda image: [
    (0, image.shape[0]),
    (image.shape[1] // 2, 0),
    (image.shape[1], image.shape[0])
]


def select_rgb_white_yellow(image, format_='RGB'):
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    if format_ == 'RGB':
        lower = np.uint8([150, 150, 50])
        upper = np.uint8([255, 255, 115])
    elif format_ == 'BGR':
        lower = np.uint8([50, 150, 150])
        upper = np.uint8([115, 255, 255])
    else:
        raise ValueError("Invalid format: {}".format(format_))
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def detect_edges_and_get_arial_view(image: np.ndarray, mtx: np.ndarray, dist: np.ndarray, M, gray_img_shape, vertices,
                                    image_format):
    undistorted_image = undistort_image(image, camera_matrix=mtx, distortion_matrix=dist)

    arial_image = cv2.warpPerspective(undistorted_image, M, gray_img_shape)

    saturation_channel = extract_saturation_channel(arial_image, format_=image_format)
    binary_yellow_white = grayscale(select_rgb_white_yellow(arial_image, format_=image_format), format_=image_format)

    binary_saturation_channel = sobel_thresholding(
        saturation_channel,
        x_thresh_min=10,
        y_thresh_max=30,
        kernel_size=5,
    )

    thresholded_yellow_white_channel = sobel_thresholding(
        binary_yellow_white,
        x_thresh_min=10,
        y_thresh_max=30,
        kernel_size=5,
    )

    combined_channel = gaussian_blur(binary_saturation_channel + thresholded_yellow_white_channel, kernel_size=3)
    return combined_channel
