import cv2
import numpy as np
import sys

# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img: np.ndarray, nx: int, ny: int, mtx: np.ndarray, dist: np.ndarray, offset: int = 0):
    """
    :param img:
    :param nx:
    :param ny:
    :param mtx:
    :param dist:
    :param offset: offset for dst points
    :return:
    """
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # Draw the corners on the undistorted image
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        img_size = (gray.shape[1], gray.shape[0])

        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                             [img_size[0]-offset, img_size[1]-offset],
                             [offset, img_size[1]-offset]])
        # dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
        #                              [img_size[0]-offset, img_size[1]-offset],
        #                              [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M
