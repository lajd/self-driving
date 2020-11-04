import os
import glob
import pickle

import matplotlib.pyplot as plt

import numpy as np
import cv2


def calibrate(distorted_images_glob: str, nx: int = 9, ny: int = 6, camera_calibration_file: str = "calibration.pkl"):
    """
    Iterate over distorted images, extracting object and image points. Use these object/image points
    to perform calibration, and save the calibration parameters to a pickle file

    Note:
        Assumes all images are the same size
    """

    obj_points = np.zeros((nx*ny,3), np.float32)
    obj_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    images_filepaths = glob.glob(os.path.join(distorted_images_glob))

    img_size = None

    for i, filename in enumerate(images_filepaths):

        img = cv2.imread(filename)

        if not img_size:
            # Set the image size
            img_size = (img.shape[1], img.shape[0])

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # Extract object and image points
        if ret is True:
            obj_points.append(obj_points)
            img_points.append(corners)

    # Perform the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    # Save the calibration file
    camera_calibration = {
        "mtx": mtx,
        "dist": dist
    }
    pickle_file = open(camera_calibration_file, "wb")
    pickle.dump(camera_calibration, pickle_file)
    pickle_file.close()
