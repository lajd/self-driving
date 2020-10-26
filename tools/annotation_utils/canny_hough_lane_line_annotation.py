
from collections import deque
from typing import Callable, List, Tuple, Union

import numpy as np

from tools.image_utils import annotate_lanes_on_image_with_canny_and_hough

# Assume that the lane-horizon is located at this fraction of the y-scale
ASSUMED_LANE_HORIZON_FRAC = 3 / 5
VERTICES_FN: Callable[[np.ndarray],
                      List[Tuple[int, int]]] = lambda image: [(0,
                                                               image.shape[0]),
                                                              (image.shape[1] // 2,
                                                               int(image.shape[0] * ASSUMED_LANE_HORIZON_FRAC)),
                                                              (image.shape[1],
                                                               image.shape[0])]
MIN_LINE_LENGTH_FN: Callable[[np.ndarray], int] = lambda image: int(
    image.shape[0] * 2 * ASSUMED_LANE_HORIZON_FRAC) // 3
MAX_LINE_GAP_FN: Callable[[np.ndarray], int] = lambda image: int(
    image.shape[0] * ASSUMED_LANE_HORIZON_FRAC) // 3
VOTE_THRESHOLD = 20


class LaneFinder:
    """Class for computing lane lines."""

    def __init__(
            self,
            window_size: int = 10,
            assumed_lane_horizon_frac: float = 3 / 5,
            min_abs_slope: float = 0.25):
        self.window_size = window_size
        self.left_slope = deque(maxlen=window_size)
        self.right_slope = deque(maxlen=window_size)
        self.left_intercept = deque(maxlen=window_size)
        self.right_intercept = deque(maxlen=window_size)

        self.assumed_lane_horizon_frac = assumed_lane_horizon_frac
        self.min_abs_slope = min_abs_slope

    def reset(self):
        self.left_slope = deque(maxlen=self.window_size)
        self.right_slope = deque(maxlen=self.window_size)
        self.left_intercept = deque(maxlen=self.window_size)
        self.right_intercept = deque(maxlen=self.window_size)

    def _update_left(self, left_slope, left_intercept):
        if not np.isnan(left_slope) and not np.isnan(left_intercept):
            self.left_slope.append(left_slope)
            self.left_intercept.append(left_intercept)

    def _update_right(self, right_slope, right_intercept):
        if not np.isnan(right_slope) and not np.isnan(right_intercept):
            self.right_slope.append(right_slope)
            self.right_intercept.append(right_intercept)

    @staticmethod
    def _get_median_lane_parameters(lines):
        """Obtain robust lane-line parameters by taking the median
        slope/y-intercept of across all identified lines."""
        left_lane_slopes = []
        right_lane_slopes = []
        left_lane_intercepts = []
        right_lane_intercepts = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]

            # The left lane has a negative slope
            # Slope is negative since origin (0, 0) is upper left
            if slope < 0:
                left_lane_intercepts.append(y_int)
                left_lane_slopes.append(slope)
            # The right lane has a positive slope
            else:
                right_lane_intercepts.append(y_int)
                right_lane_slopes.append(slope)

        # Obtain median for values, filtering out outlier points
        left_lane = np.median(left_lane_slopes), np.median(
            left_lane_intercepts)
        right_lane = np.median(right_lane_slopes), np.median(
            right_lane_intercepts)
        return left_lane, right_lane

    @staticmethod
    def _create_lines_from_parameters(image: np.ndarray,
                                      slope: Union[float,
                                                   np.ndarray],
                                      y_intercept: Union[float,
                                                         np.ndarray],
                                      assumed_lane_horizon_frac: float = 3 / 5,
                                      min_abs_slope: float = 0.25):
        """Get lane-line endpoints given its parameters.

        Uses the assumed_lane_horizon_frac value to extrapolate the line
        to a predetermined point on the y-axis.
        """

        if np.isnan(slope) or np.isnan(y_intercept):
            return None
        elif abs(slope) < min_abs_slope:
            return None
        else:
            y1 = image.shape[0]  # Always the bottom of the image
            # Extrapolate the line to the assumed lane horizon
            y2 = int(y1 * assumed_lane_horizon_frac)
            x1, x2 = int((y1 - y_intercept) /
                         slope), int((y2 - y_intercept) / slope)
            return np.array([x1, y1, x2, y2])

    def compute_lanes_from_lines_fn(self, img: np.ndarray, lines_: np.ndarray):
        (left_slope, left_intercept), (right_slope,
                                       right_intercept) = self._get_median_lane_parameters(lines_)

        self._update_left(left_slope, left_intercept)
        self._update_right(right_slope, right_intercept)

        left_slope = np.median(self.left_slope)
        right_slope = np.median(self.right_slope)

        left_intercept = np.median(self.left_intercept)
        right_intercept = np.median(self.right_intercept)

        left_line = self._create_lines_from_parameters(
            img, left_slope, left_intercept)
        right_line = self._create_lines_from_parameters(
            img, right_slope, right_intercept)

        lanes = []
        if left_line is not None:
            lanes.append(left_line)
        if right_line is not None:
            lanes.append(right_line)

        lines = np.array([lanes])

        return lines


lane_finder = LaneFinder(window_size=100)


def lane_annotator(image):
    annotated_image = annotate_lanes_on_image_with_canny_and_hough(
        image,
        gs_kernel_size=5,
        low_edge_thesh=75,
        high_edge_thresh=3 * 75,
        rho=1,
        theta=np.pi / 180,
        threshold=VOTE_THRESHOLD,
        min_line_length=MIN_LINE_LENGTH_FN(image),
        max_line_gap=MAX_LINE_GAP_FN(image),
        vertices=VERTICES_FN(image),
        compute_lanes_from_lines_fn=lane_finder.compute_lanes_from_lines_fn
    )
    return annotated_image
