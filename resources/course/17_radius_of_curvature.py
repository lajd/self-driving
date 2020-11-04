import numpy as np


def get_radius_of_curvature(left_lane_polyfit, right_lane_polyfit, y_eval):
    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2 * left_lane_polyfit[0] * y_eval + left_lane_polyfit[1]) ** 2) ** 1.5) / np.absolute(2 * left_lane_polyfit[0])
    right_curverad = ((1 + (2 * right_lane_polyfit[0] * y_eval + right_lane_polyfit[1]) ** 2) ** 1.5) / np.absolute(2 * right_lane_polyfit[0])
    return left_curverad, right_curverad
