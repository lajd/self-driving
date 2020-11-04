# **Finding Lane Lines on the Road** 

![til](../resources/P1/highway.gif)

## Introduction
In this project we create a pipeline for annotating lane-lines on a video clip, for the [eventual] purpose of
using these markings to orient a self-driving vehicle. In particular, we are interested in obtaining the lane markings
that the vehicle is currently residing within -- we are not interested in adjacent lane lines, crosswalk/intersection
lane lines, etc.

## Lane Annotation Pipeline

The annotation processing was first developed to be run on individual images, since it can easily be extended to run
on a video by applying the processing on video frames in sequence and stacking the result.

![Alt text](resources/solidWhiteCurve.jpg?raw=true "Raw Image")


To annotate lane-lines on an image, the pipeline is as follows:
###### 1) Grayscale the image
Converting an RGB/BGR (3 channel) images to grayscale (single channel) images is useful preprocessing step prior to edge detection
since the Canny algorithm measures the change in pixel intensity (gradient) within an image, which is most naturally done with a single channel.

![Alt text](resources/grayscale.png?raw=true "Grayscale Image")


###### 2) Gaussian smoothing
The grayscale image is then smoothed/blurred using a 2d convolution between the image and a kernel whos parameters are given
by a Gaussian. The effect of Gaussian smoothing is to reduce the number of edges detected in the subsequent edge-detection step.

![Alt text](resources/blurred.png?raw=true "Blurred Image")

###### 3) Canny Edge Detection
The Canny edge detection algorithm identifies edges in the image by:

1) Finding the intensity gradients of the image (i.e. difference between pixel values)
2) Performing non-maximum suppression. Non-maximum suppression keeps the pixels with the largest intensity, while removing adjacent pixels which have intensity gradients in the same direction.
3) Performing double thresholding with a low_threshold and a high_threshold. Pixels with intensity values above the high_threshold will be considered 'strong edges', between the low_threshold and high_threshold will be considered weak edges, and those lower than the low_threshold will be discarded
4) Edge tracking by hysteresis. In this step, weak edges connected to strong edges are converted to strong edges. Weak edges which are not are discarded.

For more details, see the [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector).

![Alt text](resources/canny.png?raw=true "Grayscale Image")


###### 4) Apply Region-of-Interest mask
In this step we apply a polygonal mask to the edge-detected image, causing pixels outside of the mask to be zeroed-out, and those within the mask to be left unchanged.
In this pipeline, since we are identifying road lanes, we utilize a triangular-shaped mask. </br>

If an image's width and height are given by n and m, respectively, then the bottom_left, top and bottom_right vertices of the triangular mask are given by: </br>
```
vertices = [(0, m), (n//2, m * lane_horizon_fraction), (n, m)].
```

Where the `lane_horizon_fraction` parameter is used to adjust the height of the mask.

![Alt text](resources/triangular_mask.png?raw=true "Grayscale Image")


###### 5) Hough Line Transform
The Hough Transform takes a line in image space represented by `y=mx+b` and transforms it to a point in Hough space <code>(m<sub>0</sub>, b<sub>0</sub>)</code>.
Similarly, a point in cartesian space <code>(x, y)</code> can be represented by a line <code>b=y-mx</code>. Thus, if there are multiple
points in cartesian space which fall on the same line, then this will be represented in Hough space by multiple lines, which share an intersection point 
at the <code>(m<sub>0</sub>, b<sub>0</sub>)</code> coordinates which represent the equation of the line in image space. With this idea, we can use the 
intersection of multiple lines (called votes) in Hough space to identify the equation of a line in image space.

In order to deal with vertical lines in image space (having infinite slope), we use polar coordinates in Hough space.

![Alt text](resources/annotated.png?raw=true "Grayscale Image")


###### 6) Lane Detection from Hough Lines
The Hough transform provides the start and end points for lines which satisfy the transform's parameters -- in general, 
the returned lines may be noisy and not representative of the desired lane-lines. 

In order transform noisy line representations into lane-lines, we perform the following operations:

1) Obtain line equations for all line candidates
    - We obtain slope and y-intercept information for each line
    
2) Eliminate lines with slopes below a threshold
    - Since we are interested only in the lane lines that the vehicle is currently within, we can make assumptions 
    about the slope of these lines. In particular, we know that these lines should not be horizontal, and so we eliminate
    lines which have absolute slope values less than a threshold. For this task, a threshold of `0.25` was empirically 
    found to perform well.

3) Sort lines according to their slopes
    - Intuitively, when the vehicle camera is positioned between lane-lines, the sign of the slopes of the left/right lane lines will
    be consistent. Since we are dealing with the image coordinate system, the slope of the left-lane-line will be negative, while
    the slope of the right-lane-line will be positive (in contrast to a cartesian coordinate system where the slopes signs will be opposite).
    - In this way, we obtain lane-line candidates for both the left-lane and the right-lane
    - Note that it's possible that a lane has no valid line candidates for a particular image frame. This will be addressed in the item 5).
 
4) Find the median slope/y-intercept between candidate lines
    - In order to find the equations of our lane-lines, we take the median of our lane-line parameters -- the slope and the y-intercept. 
    We use the median since it is more robust to outliers than the mean.
    - We obtain line parameters for both the left-lane and right-lane, as `median_left_slope, median_left_intercept`, `median_right_slope, median_right_intercept`, 
    provided that there are valid candidates obtained in 3).

5) Handle null lane-line detections and improve robustness
    - In 3), it is possible for the left/right lane-line to have no candidate lines as outputted by the Hough transform. Since it is desirable
    to always have a lane annotation, we use historical lane-line information to fill this gap. Due to the continuity of the driving experience, we make the assumption that the car is moving sufficiently
    slow such that, between frames, the lane-lines should not change significantly. This also assumes that the curvature of the roads is sufficiently small.
    In practice, we found that this assumption was OK for the data used, but would not be valid in general (for example, during sharp turns at high speeds, etc.)
    - For each frame, after we obtain `median_left_slope, median_left_intercept`, `median_right_slope, median_right_intercept`, we store these values
    in a history buffer with an appropriate window size (we use a `window_size=100` for our experiments). Then, to obtain the lane equations for the next frame, we simply use the 
    median of these historical values to obtain `median_left_slope, median_left_intercept`, `median_right_slope, median_right_intercept` using our historical windows.
    - In practice, we found this method to allowed for improved smoothness of lane-line-annotations between frames, and more robustness to outlier lanes
    (for example, for the `challenge` video in the Jupyter notebook).

6) Obtaining lane-line endpoints from lane-line equations
    - Finding the lane-line end points (towards the horizon)
        - In order to obtain the end-points of the lane-lines from their slopes and intercepts, we make an assumption about the pixel-height of the 
        lane in the image. In particular, we extrapolate the line to a pre-determined value on the y-axis, which we take as the `lane-line-horizon`.
        
        ![Alt text](resources/lane-line-horizon.jpg?raw=true "Grayscale Image")
        
        In the above diagram, the `lane-line-horizon` is the pixel-distance (from the top of the image) on the y-axis to which
        we extrapolate our lane lines. In general, we take this distance to be a fraction of the total height of the image, where we
        use `assumed_lane_horizon_frac=5/3` in our experiments, representing that the lane-line should be extrapolated to a height 5/3s of the
        way down the image. In practice, we find that this method works well for the data experimented on, but has significant shortcomings (expressed in the next section).
    
    - Finding the lane-line start points 
        - The lane-line start points are given by extrapolating the lane-line equations all the way to the bottom of the image.

# Pipeline Hyper-Parameters

The pipeline hyper-parameters used are given by:

```
GAUSSIAN_KERNEL_SIZE=5
LOW_EDGE_THRESHOLD=75
HIGH_EDGE_THRESHOLD=3 * 75
RHO=1
THETA=np.pi / 180

# Assume that the lane-horizon is located at this fraction of the y-scale
ASSUMED_LANE_HORIZON_FRAC = 3 / 5
VERTICES_FN: lambda image: [
    (0, image.shape[0]),
    (image.shape[1] // 2,  int(image.shape[0] * ASSUMED_LANE_HORIZON_FRAC)),
    (image.shape[1], image.shape[0])
]
MIN_LINE_LENGTH_FN = lambda image: int(image.shape[0] * 2 * ASSUMED_LANE_HORIZON_FRAC) // 3
MAX_LINE_GAP_FN = lambda image: int(image.shape[0] * ASSUMED_LANE_HORIZON_FRAC) // 3
VOTE_THRESHOLD = 20
```

These hyper-parameters were found to generalize decently well across the experiments in the jupyter notebook, however
they did require some manual tuning. Changing these parameters can result in drastically different results, suggesting
that this pipeline will not generalize well to arbitrary traffic environments. 

# Potential shortcomings of the method

In this work, we make a number of assumptions, namely:
- Parameters of Hough transform
    - These parameters significantly reduce the range of driving scenarios for which our model applies. In particular, the parameters
        - voting_threshold
        - min_line_length
        - max_line_gap
    significantly affect the quality/performance of the results. For example, in situations where the gap between lane-markings is
    significant, then a large max_line_gap is required. However, a large value of max_line_gap may introduce significant noise in detected lines,
    resulting in poor performance. Similarly, the having a small value for min_line_length results in many noisy points, but is required, for example,
    when the lanes demonstrate significant curvature.
- Assumptions regarding the extrapolation of a line
    - In order to perform lane extrapolation, we always assume that the lane extends up the height of the image a constant amount, which we take to be
    3/5 of the image height (from the top), or 2/5 of the image height (from the bottom). Empirically, using a constant lane-height was found to produce 
    better results than, say, using the median value for the line end-point obtained from the Hough transform, since these calculated values are all noisy.
    Our constant-lane-height assumption is valid in practice because of the angle the camera makes with the road, and the general curvature of the lanes we encountered
    in our experiments. However, if there is significant curvature in our lane, this assumption would become invalid and the lane-extrapolation would result
    in poor results.

Examples of these shortcomings can be seen when we alter the parameters above and run the jupyter notebook. In particular, the experiment marked
`challenging` is much more sensitive that the other two experiments, because of the curvature of the lanes.

Because of the sensitivity of results to the pipeline hyper-parameters, this pipeline is not suitable for real-world 
traffic situations without significant adaptation.


# Improvements and next-steps

The approach covered in this project is quite rudimentary -- it performs decently well in select situations, but lacks generalization
to different traffic environments, lighting/weather conditions, camera occlusions, etc. In the decades since the Canny/Hough transform
method for line detection were established, the field of deep learning has offered many improvements to such classical methods, such as
[LaneNet](https://arxiv.org/abs/1807.01726).

In order to improve the current pipeline, though, some possible improvements are:
1) Use the changes in lane-line equations to select the equation of the next lane-line
    - Currently we are taking the median of the slope/intercept parameters in order to come up with the next
    slope/intercept. Although this method is robust to outliers, it will result in poor performance when the lanes have
    significant curvature. 
    - One way to improve this estimate is to use the historical lane-line equations to predict/approximate the 
    next lane line. For example, if we start to detect that the slope/intercept of the left-lane-line is changing in a 
    linear way, we can fit a line across the slope/intercepts values, and use this line to predict the next slope/intercept value. Since this
    estimate takes into account the change in line equations, it should be more suitable for handling lane curvature.
2) Automatically choose hyper-parameters based on input image and historical information
    - The hardest part of the above pipeline is choosing hyper parameters for the model. It's possible that these hyper-parameters
    can be extracted on a per-image basis (or every N frames basis) dependent only on the input image itself. As a basic example, 
    we could create a table mapping different traffic scenarios of images to a corresponding set of empirically well-performing hyper parameters
    for our pipeline. Then, when a new image is passed into the pipeline, the first step would be to perform a nearest-neighbours-search
    on the image, find the most similar image in our table, and use the corresponding hyper-parameters for processing that image. 
    - This method could be extrapolated to use a NN approach, where a CNN consumes an image and predicts applicable hyper-parameters, however it might
    be better to go with a fully deep NN approach in this case, although there might be reasons (eg. FPS/performance) to use a hybrid approach.
