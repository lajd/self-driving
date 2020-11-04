from typing import List, Tuple
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


def get_obj_points_from_chessboard_dimensions(chessboard_dimensions: Tuple[int, int] = (9, 6)):
    """ Prepare object points for a chessboard pattern """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    x, y = chessboard_dimensions
    object_points: np.ndarray = np.zeros((x * y, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
    return object_points


def get_camera_calibration(
        chessboard_dimensions: Tuple[int, int],
        calibration_image_file_paths: List[str] = glob.glob('*.jpg'),
        show_images: bool = False,
        show_plot_wait_time_ms: int = 500,
        format_: str = 'BGR'
) -> Tuple:
    """ Obtain camera calibration matrices given calibration files"""
    x, y = chessboard_dimensions
    obj_points = get_obj_points_from_chessboard_dimensions(chessboard_dimensions)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    calibration_pointss = []

    # Step through the list and search for chessboard corners
    assert len(calibration_image_file_paths) > 0
    for file_paths in calibration_image_file_paths:
        original_img = cv2.imread(file_paths)
        if format_ == 'BGR':
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        elif format_ == 'RGB':
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Invalid format: {}".format(format_))

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(obj_points)
            imgpoints.append(corners)

            # Draw and display the corners
            calibration_points = cv2.drawChessboardCorners(original_img, (x, y), corners, ret)
            calibration_pointss.append(calibration_points)
            if show_images:
                cv2.imshow('calibration Image', calibration_points)
                cv2.waitKey(show_plot_wait_time_ms)

    if show_images:
        cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, original_img.shape[0:2], None, None
    )

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, calibration_pointss


def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, distortion_matrix: np.ndarray):
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_matrix, None, camera_matrix)
    return undistorted_image


def get_isosceles_trapezoid_mask_vertices(
        m, n,
        x_proportion_bottom=1 / 10,
        x_proportion_top=4 / 10,
        y_proportion_bottom=1,
        y_proportion_top=3/5
):
    """

    Get the vertices of an isosceles trapezoid starting from bottom left and moving clockwise
    :param image:
    :param x_proportion_bottom:
    :param x_proportion_top:
    :param y_proportion_bottom:
    :param y_proportion_top:
    :return:
    """
    bottom_y = int(m * y_proportion_bottom)
    top_y = int(m * y_proportion_top)

    left_lane_bottom_left_pt = (int(n * x_proportion_bottom), bottom_y)
    left_lane_top_left_pt = (int(n * x_proportion_top), top_y)

    right_lane_bottom_right_pt = (n - int(n * x_proportion_bottom), bottom_y)
    right_lane_top_right_pt = (n - int(n * x_proportion_top), top_y)

    # list(left_lane_top_left_pt),
    # list(right_lane_top_right_pt),
    # list(right_lane_bottom_right_pt),
    # list(left_lane_bottom_left_pt)
    return [left_lane_top_left_pt, right_lane_top_right_pt, right_lane_bottom_right_pt, left_lane_bottom_left_pt]

    # return [left_lane_bottom_left_pt, left_lane_top_left_pt, right_lane_bottom_right_pt, right_lane_top_right_pt]


def get_perspective_transform(undist, image, x_proportion_bottom=1 / 10, x_proportion_top=4 / 10, y_proportion_bottom=1, y_proportion_top=3/5, offset: int = 200):

    m, n, _ = undist.shape
    copy = undist.copy()

    bottom_y = int(m * y_proportion_bottom)
    top_y = int(m * y_proportion_top)

    m, n, c = image.shape

    left_lane_bottom_left_pt = (int(n * x_proportion_bottom), bottom_y)
    left_lane_top_left_pt = (int(n * x_proportion_top), top_y)

    right_lane_bottom_right_pt = (n - int(n * x_proportion_bottom), bottom_y)
    right_lane_top_right_pt = (n - int(n * x_proportion_top), top_y)

    color = [255, 0, 0]
    w = 2
    cv2.line(copy, left_lane_bottom_left_pt, left_lane_top_left_pt, color, w)
    cv2.line(copy, left_lane_top_left_pt, right_lane_top_right_pt, color, w)
    cv2.line(copy, right_lane_top_right_pt, right_lane_bottom_right_pt, color, w)
    cv2.line(copy, right_lane_bottom_right_pt, left_lane_bottom_left_pt, color, w)
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.imshow(copy)

    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    src = np.float32([
        list(left_lane_top_left_pt),
        list(right_lane_top_right_pt),
        list(right_lane_bottom_right_pt),
        list(left_lane_bottom_left_pt)
    ])

    nX = gray.shape[1]
    nY = gray.shape[0]
    img_size = (nX, nY)
    dst = np.float32([
        [offset, 0],
        [img_size[0] - offset, 0],
        [img_size[0] - offset, img_size[1]],
        [offset, img_size[1]]
    ])
    img_size = (gray.shape[1], gray.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv, img_size

# def get_perspective_transform(m, n, x_proportion_bottom=1 / 10, x_proportion_top=4 / 10, y_proportion_bottom = 1, y_proportion_top = 3/5):
#     """ Create a perspective transform from an """
#     left_lane_top_left_pt, right_lane_top_right_pt, right_lane_bottom_right_pt, left_lane_bottom_left_pt = get_isosceles_trapezoid_mask_vertices(
#         m, n, x_proportion_bottom, x_proportion_top, y_proportion_bottom, y_proportion_top
#     )
#
#     src = np.float32([
#         list(left_lane_top_left_pt),
#         list(right_lane_top_right_pt),
#         list(right_lane_bottom_right_pt),
#         list(left_lane_bottom_left_pt)
#     ])
#
#     img_size = (m, n)
#     offset = 100
#     dst = np.float32([
#         [offset, 0],
#         [img_size[0] - offset, 0],
#         [img_size[0] - offset, img_size[1]],
#         [offset, img_size[1]]
#     ])
#     M = cv2.getPerspectiveTransform(src, dst)
#     Minv = cv2.getPerspectiveTransform(dst, src)
#
#     return M, Minv
