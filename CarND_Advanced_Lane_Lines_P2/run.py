import os
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import pickle
import glob
import cv2
from collections import deque
import scipy.signal
import numpy as np
from moviepy.editor import VideoFileClip
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error

from sklearn.linear_model.ransac import RegressorMixin
from scipy.interpolate import UnivariateSpline
from tools.video_utils import video_to_numpy
from tools.image_utils import (
    grayscale,
    mask_with_region_of_interest,
    get_canny_pixels,
    gaussian_blur,
    extract_saturation_channel,
    sobel_thresholding
)

from tools.camera import undistort_image, get_camera_calibration, get_perspective_transform
from tools.annotation_utils.histogram_lane_line_pixel_detection import lane_detection
from tools.annotation_utils.misc import draw_lane_area
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression

PROJECT_ROOT_DIR = '/CarND_Advanced_Lane_Lines_P2/'
PROJECT_VIDEO_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, 'raw_videos/project_video.mp4')
POLYNOMIAL_DEGREE = 2
IMAGE_FORMAT = 'BGR'
ARIAL_VIEW_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, 'resources', 'arial_view')
BUFFER_SIZE = 25


class LaneModel:
    def __init__(self):
        pass

    def fit(self, img_shape, left_x, left_y, right_x, right_y):
        pass


class QuadraticLaneModel(LaneModel):
    def __init__(self):
        super().__init__()

    def fit(self, img_shape, leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each with np.polyfit()#
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        # Calc both polynomials using ploty, left_fit and right_fit#
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def predict(self, left_fit, right_fit, y):
        # Calculate points.
        left_fitx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
        right_fitx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
        return left_fitx, right_fitx


class PolynomialRegression(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


class RANSACLaneModel(LaneModel):
    def __init__(self):
        super().__init__()

    def fit(self, img_shape, left_x, left_y, right_x, right_y):
        left_x = np.array(left_x).reshape(-1,)
        right_x = np.array(right_x).reshape(-1,)

        left_y = np.array(left_y).reshape(-1, 1)
        right_y = np.array(right_y).reshape(-1, 1)

        left_fit = RANSACRegressor(
            random_state=0,
            base_estimator=PolynomialRegression(),
            min_samples=25
        ).fit(left_y, left_x)
        right_fit = RANSACRegressor(
            random_state=0,
            base_estimator=PolynomialRegression(),
            min_samples=25
        ).fit(right_y, right_x)
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0]).reshape(-1, 1)
        left_fitx = left_fit.predict(ploty)
        right_fitx = right_fit.predict(ploty)
        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def predict(self, left_fit, right_fit, y):
        y = np.array(y).reshape(-1, 1)
        left_fitx = left_fit.predict(y)
        right_fitx = right_fit.predict(y)
        return left_fitx, right_fitx


# model = QuadraticLaneModel()
model = RANSACLaneModel()

#################################################################
# Convert to Arial view


def convert_image_to_arial_view(image, M, img_size):
    warped = cv2.warpPerspective(image, M, img_size)
    return warped


ASSUMED_LANE_HORIZON_FRAC = 2.5 / 5
VERTICES_FN: Callable[[np.ndarray], List[Tuple[int, int]]] = lambda image: [
    (0, image.shape[0]),
    (image.shape[1] // 2, 0),
    (image.shape[1], image.shape[0])
]

VIDEO_ARRAY: np.ndarray = video_to_numpy(PROJECT_VIDEO_FILE_PATH)
FIRST_IMAGE = VIDEO_ARRAY[0]
MOD_COUNTER = 20

# Get the camera calibration
ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, calibrated_images = get_camera_calibration()


undistorted_image = undistort_image(FIRST_IMAGE, camera_matrix=mtx, distortion_matrix=dist)
M, Minv, gray_img_shape = get_perspective_transform(
    undistorted_image,
    FIRST_IMAGE,
    x_proportion_bottom=0.5 / 10,
    x_proportion_top=4 / 10
)

arial_image = convert_image_to_arial_view(
    undistorted_image,
    M,
    gray_img_shape
)

triangular_vertices = VERTICES_FN(FIRST_IMAGE)

global left_fit, right_fit, image_counter
left_fit = None
right_fit = None
image_counter = 0
IMAGE_BUFFER = deque(maxlen=BUFFER_SIZE)


def pipeline(img_):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679325/
    https://arxiv.org/pdf/1411.7113.pdf
    2. Converting of Image Distortion
        2.1. Camera Calibration
        2.2. Image Distortion Removal
    3. Edge Detection and Inverse Perspective Transformation
        3.1. Edge Detection
        3.2. Inverse Perspective Transformation
            3.2.1. ROI Extraction
            3.2.2. Inverse Perspective Transformation
    4. Lane Detection
        4.1. Mask Operation
        4.2. Lane Detection Algorithm
            4.2.1. Third-Order B-Spline Curve Model
            4.2.2. Lane Line Fitting Based on RANSAC Algorithm
            4.2.3. Lane Line Fitting Evaluation and Curvature Radius Calculation

    :param VIDEO_ARRAY:
    :return:
    """

    masked_binary_combined_channel = detect_edges_and_get_arial_view(
        img_,
        mtx,
        dist,
        M,
        gray_img_shape,
        VERTICES_FN(FIRST_IMAGE),
        IMAGE_FORMAT
    )

    global image_counter
    if image_counter % MOD_COUNTER == 0:
        plt.imsave(os.path.join(ARIAL_VIEW_FILE_PATH, "{}.jpeg".format(image_counter)), masked_binary_combined_channel)

    global left_fit, right_fit
    left_fit, right_fit, result = lane_detection(masked_binary_combined_channel, model, left_fit=left_fit, right_fit=right_fit)

    annotated_image = draw_lane_area(img_, model, left_fit, right_fit, Minv)

    image_counter += 1
    return annotated_image


clip1 = VideoFileClip(PROJECT_VIDEO_FILE_PATH)
annotated_video = clip1.fl_image(pipeline)
annotated_video.write_videofile('output_video.mp4', audio=False)
