from typing import Callable, List, Optional, Tuple, Union

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def read_image_from_path(image_path: str):
    # Read in and grayscale the image
    image = plt.imread(image_path)
    return image


def show_image(image: np.ndarray):
    plt.imshow(image)


def grayscale(img: np.ndarray, format_='RGB'):
    """Applies the Grayscale transform This will return an image with only one
    color channel but NOTE: to see the returned image as grayscale (assuming
    your grayscaled image is called 'gray') you should call plt.imshow(gray,
    cmap='gray')"""
    if format_ == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif format_ == 'BGR':
        # Or use BGR2GRAY if you read an image with cv2.imread()
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Incalid image format_ {}".format(format_))


def canny(img: np.ndarray, low_threshold: int, high_threshold: int):
    """Applies the Canny transform."""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img: np.ndarray, kernel_size: int):
    """Applies a Gaussian Noise kernel."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def draw_lines(
        img: np.ndarray,
        lines: List[List[int]],
        color: List[int] = [255, 0, 0],
        thickness: int = 2
):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def hough_lines(
        img: np.ndarray,
        rho: float = 1,
        theta: float = np.pi / 180,
        threshold: float = 20,
        min_line_length: float = 50,
        max_line_gap: float = 10,
) -> np.ndarray:
    """`img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines_ = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Detected lines can be None
    # https://stackoverflow.com/questions/16144015/python-typeerror-nonetype-object-has-no-attribute-getitem
    if lines_ is None:
        lines_ = np.array([])
    return lines_


def weighted_img(
        img: np.ndarray,
        initial_img: np.ndarray,
        alpha: float = 0.8,
        beta: float = 1.,
        gamma: float = 0.):
    """`img` is the output of the hough_lines(), An image with lines drawn on
    it. Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def mask_with_region_of_interest(
        img: np.ndarray, vertices: Optional[List[Tuple[int, int]]] = None):
    """Applies an image mask.

    Only keeps the region of the image defined by the polygon formed
    from `vertices`. The rest of the image is set to black. `vertices`
    should be a numpy array of integer points.
    """
    if vertices is None:
        # Make the ROI the entire image
        m, n = img.shape
        vertices = [(0, m), (0, 0), (n, 0), (n, m)]

    vertices = np.array([vertices], dtype=np.int32)
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill
    # color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_canny_pixels(image, gs_kernel_size=5, low_edge_thesh=75, high_edge_thresh=75*3):
    smoothed_image = gaussian_blur(image, gs_kernel_size)
    edges = canny(smoothed_image, low_edge_thesh, high_edge_thresh)
    return edges


def sobel_thresholding(
        single_channel_image: np.ndarray,
        kernel_size: int = 3,
        x_thresh_min: int = 0,
        x_thresh_max: int = 255,
        y_thresh_min: int = 0,
        y_thresh_max: int = 255,
        direction_thresh_min: float = 0,  # 0.7,
        direction_thresh_max: float = 2  # 1.3,
):

    assert single_channel_image.ndim == 2

    sobelx = cv2.Sobel(single_channel_image, cv2.CV_64F, *[1, 0], ksize=kernel_size)
    sobely = cv2.Sobel(single_channel_image, cv2.CV_64F, *[0, 1], ksize=kernel_size)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = abs(sobelx)
    abs_sobely = abs(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)

    abs_sobelx = np.uint8(255.0 * abs_sobelx / np.max(abs_sobelx))
    abs_sobely = np.uint8(255.0 * abs_sobely / np.max(abs_sobely))

    binary = np.zeros_like(abs_sobelx)
    print(np.mean(direction), np.max(direction))
    binary[
        (abs_sobelx >= x_thresh_min) & (abs_sobelx <= x_thresh_max)
        & (abs_sobely >= y_thresh_min) & (abs_sobely <= y_thresh_max)
        & (direction >= direction_thresh_min) & (direction <= direction_thresh_max)
    ] = 1
    return binary


def extract_saturation_channel(undistorted_image: np.ndarray, format_: str = 'RGB'):
    # Edge Detection and Inverse Perspective Transformation
    #  Edge Detection using Threshold superposition binarization

    # Edit this function to create your own pipeline.
    # Convert to HLS color space and separate the V channel
    if format_ == 'RGB':
        hls = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2HLS)
    elif format_ == 'BGR':
        hls = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HLS)
    else:
        raise ValueError("Invalid format: {}".format(format_))
    s_channel = hls[:, :, 2]
    return s_channel


def annotate_lanes_on_image_with_canny_and_hough(
        image: np.ndarray,
        gs_kernel_size: int = 3,
        low_edge_thesh: int = 75,
        high_edge_thresh: int = 3 * 75,
        rho: int = 1,
        theta: float = np.pi / 180,
        threshold: int = 5,
        min_line_length: int = 50,
        max_line_gap: int = 5,
        vertices: Optional[List[Tuple[int, int]]] = None,
        compute_lanes_from_lines_fn: Callable = lambda image, lines: lines
):
    """Annotate lanes on an image using the Canny edge detection + Hough
    transform technique."""
    smoothed_image = gaussian_blur(image, gs_kernel_size)
    edges = canny(smoothed_image, low_edge_thesh, high_edge_thresh)
    masked_edges = mask_with_region_of_interest(edges, vertices)
    lines = hough_lines(
        masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )

    # line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lines = compute_lanes_from_lines_fn(image, lines)
    annotated_image = draw_lines(image, lines)
    return annotated_image
