import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grayscale(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def sobelx(gray: np.ndarray, scaling: bool = True):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    if scaling:
        abs_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    return abs_sobelx


def sobely(gray: np.ndarray, scaling: bool = True):
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    if scaling:
        abs_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    return abs_sobely


def binary_pixel_mask(scaled_sobel: np.ndarray, thresh_min: int = 20, thresh_max: int = 100):
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def show_sobal(sobel_with_mask: np.ndarray):
    plt.imshow(sobel_with_mask, cmap='gray')


def abs_sobel_thresh(img: np.ndarray, orient: str = 'x', thresh_min: int = 0, thresh_max: int = 255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    direction = [1, 0] if orient == 'x' else [0, 1]
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, *direction)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sbinary)  # Remove this line
    return binary_output

