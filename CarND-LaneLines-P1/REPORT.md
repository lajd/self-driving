# **Finding Lane Lines on the Road** 

## Introduction
In this project we create a pipeline for annotating lane-lines on a video clip, for the [eventual] purpose of
using these markings to orient a self-driving vehicle.

## Lane Annotation Pipeline

The annotation processing was first developed to be run on individual images, since it can easily be extended to run
on a video by applying the processing on video frames in sequence and stacking the result.

![Alt text](resources/solidWhiteCurve.jpg?raw=true "Raw Image")


To annotate lane-lines on an image, the pipeline is as follows:
###### 1) Grayscale the image
Converting an RGB/BGR (3 channel) images to grayscale (single channel) images is useful preprocessing step propr to edge detection
since the Canny algorithm measures the change in pixel intensity (gradient) within an image, which is most naturally done with a single channel.

![Alt text](resources/grayscale.png?raw=true "Grayscale Image")


###### 2) Gaussian smoothing
The grayscale image is then smoothed/blurred using a 2d convolution between the image and a kernel whos parameters are given
by a Gaussian. The effect of Gaussian smoothing is to reduce the number of edges detected in the subsequent edge-detection step.

![Alt text](resources/blurred.png?raw=true "Blurred Image")

###### 3) Canny Edge Detection
The Canny edge detection algorithm identified edges in the image by:

1) Find the intensity gradients of the image (i.e. difference between pixel values)
2) Perform non-maximum suppression. Non-maximum suppression keeps the pixels with the largest intensity, while removing adjacent pixels which have intensity gradients in the same direction.
3) Perform double thresholding with a low_threshold and a high_threshold. Pixels with intensity values above the high_threshold will be considered 'strong edges', between the low_threshold and high_threshold will be considered weak edges, and those lower than the low_threshold will be discareded
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
# TODO



### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
