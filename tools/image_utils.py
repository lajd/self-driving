# Do all the relevant imports
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def read_image_from_path(image_path: str):
    # Read in and grayscale the image
    image = mpimg.imread(image_path)
    return image


def gaussian_smoothing(image_array: np.ndarray, kernel_size: int = 3):
    assert kernel_size % 2 != 0, "Kernel size must be odd, got {}".format(kernel_size)
    grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Define a kernel size for Gaussian smoothing / blurring
    smoothed_image = cv2.GaussianBlur(grayscale_image, (kernel_size, kernel_size), 0)
    return smoothed_image


def canny_edge_detection(image: np.ndarray, low: int = 75, high: int = 75 * 3):
    edges = cv2.Canny(image, low, high)
    return edges


def mask_image_with_polygon(image: np.ndarray, polygon_vertices: List[Tuple[int]]):
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    # This time we are defining a four sided polygon to mask
    vertices = np.array([polygon_vertices], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(image: np.ndarray, rho: int = 1, theta: float = np.pi/180, threshold: int = 5, min_line_length: int = 50, max_line_gap: int = 5):
    """ Apply Hough transform for lines in polar coordinated

    Args:
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 50     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 200 #minimum number of pixels making up a line
        max_line_gap = 5    # maximum gap in pixels between connectable line segments
        line_image = np.copy(image)*0 # creating a blank to draw lines on

    image-space:
        y = m_0x + b_0
        y =
        x =

    HT: image-space -> Hough Space
    """
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    return lines


def get_hough_transform_lines_from_raw_image(
        image: Union[np.ndarray, str],
        gs_kernel_size: int = 3,
        low_edge_thesh: int = 75,
        high_edge_thresh: int = 75,
        rho: int = 1,
        theta: float = np.pi / 180,
        threshold: int = 5,
        min_line_length: int = 50,
        max_line_gap: int = 5,
        polygon_vertices: Optional[List[Tuple[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    smoothed_image = gaussian_smoothing(image, gs_kernel_size)
    edges = canny_edge_detection(smoothed_image, low_edge_thesh, high_edge_thresh)

    if not polygon_vertices:
        m, n = edges.shape
        polygon_vertices = [(0, m), (0, 0), (n, 0), (m, n)]

    masked_edges = mask_image_with_polygon(edges, polygon_vertices=polygon_vertices)

    lines = hough_transform(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    return edges, lines


def draw_lines_on_image(image: np.ndarray, edges: np.ndarray, lines: np.ndarray):
    # Make a blank the same size as our image to draw on
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    plt.imshow(lines_edges)

image = read_image_from_path('/home/jon/PycharmProjects/self-driving/CarND-LaneLines-P1/test_images/solidWhiteCurve.jpg')
edges, lines = get_hough_transform_lines_from_raw_image(image)
draw_lines_on_image(image, edges, lines)
