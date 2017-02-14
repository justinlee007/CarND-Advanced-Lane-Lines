# Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project utilizes a software pipeline to identify the lane boundaries in a video.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are in the `camera_cal` folder.  The images in `test_images` are used in testing the pipeline on single frames.  Examples of the output from each stage of the pipeline are the `ouput_images` folder.  The video `project_video.mp4` is target video for the lane-finding pipeline.  Each rubric step will be documented with output images and usage.

# [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

## Camera Calibration
1) Have the camera matrix and distortion coefficients been computed correctly and checked on one of the calibration images as a test?

The code for this step is contained in `camera_calibration.py`
```
usage: camera_calibration.py [-h] [-show] [-save]

Camera calibration for Udacity Advanced Lane Finding project

optional arguments:
  -h, --help  show this help message and exit
  -show       Visualize data calibration
  -save       Save calibration images
```
This script loads calibration images of chessboards taken at different angles.  Each image is grayscaled and sent into `cv2.drawChessboardCorners`.  The resulting "object points" are the (x, y, z) coordinates of the chessboard corners in the world.

For each set of corners, the output is displayed and shown to the user (if specified).  The image is also saved to disk (if specified).
  
Finally the corner points are sent to `cv2.calibrateCamera` to get resulting image points and object points.  This dictionary is then saved for reuse in undistorting other images in the pipeline.

![](output_images/chessboard1.jpg) ![](output_images/chessboard9.jpg)

##### Example chessboard images with corners drawn

## Pipeline
This sections includes all scripts and functionality to process images throughout the lane-finding pipeline.

###1) Has the distortion correction been correctly applied to each image?

The `camera_calibration.py` script also contains a utility method for undistorting images based on the saved camera calibration dictionary.  This utility calls `cv2.undistort` internally.  The dictionary is loaded with the `load_calibration_data` method.

![](output_images/undistorted_calibration2.png)

![](output_images/undistorted_signs_vehicles_xygrad.png)

##### Example undistorted images next to the original image

###2) Has a perspective transform been applied to rectify the image?
Since perspective transform occurs after distortion correction, the discussion here follows that flow.

The code for this step is contained in `transform.py`
```
usage: transform.py [-h] [-show] [-save]

Perspective transform for Udacity Advanced Lane Finding project

optional arguments:
  -h, --help  show this help message and exit
  -show       Visualize transform
  -save       Save transform image
```
The main method of this script loads all the "test*.jpg" files in the `test_images` directory and processes each one.

The process includes:
* Undistort the image
* Create mappings (a helper method to define four points in a polygon for perspective transform)
* Apply the transform using `cv2.getPerspectiveTransform` and `cv2.warpPerspective` (the inverse perspective is also calculated)
* Show the transform and/or save the image to disk (if specified)

![](output_images/warped_test2.png)
##### Example perspective transform with mappings drawn for reference

###3) Has a binary image been created using color transforms, gradients or other methods?

The code for this step is contained in `imaging.py`
```
usage: imaging.py [-h] [-show] [-save]

Imaging methods for Udacity Advanced Lane Finding project

optional arguments:
  -h, --help  show this help message and exit
  -show       Visualize imaging process
  -save       Save processed image
```

This script expects undistorted images with the perspective transform already applied. 

The main method of this script loads all the "test*.jpg" files in the `test_images` directory and processes each one.

The process includes:
* Apply two Sobel filters in the x and y directions
    * The x direction sobel filter uses a threshold of 35 and 255
    * The y direction sobel filter uses a threshold of 15 and 255
* Do a bitwise AND function for the two sobel filters
* Apply two color masks in HLS and HSV color space
    * The HLS mask uses a threshold of 150 and 255
    * The HSV mask uses a threshold of 200 and 255
* Do a bitwise AND function for the two color filters
* Do a bitwise OR funtion on the combined sobel filters and combined color masks
* Finally apply a gaussian blur on the outputted image with a kernel of 9
* Show the processed image and/or save to disk (if specified)

![](output_images/imaging_test2.png)
##### Example binary image with combined sobel filters, color masks and gaussian blur

###4) Have lane line pixels been identified in the rectified image and fit with a polynomial?

The code for this step is contained in `poly_fit.py`
```
usage: poly_fit.py [-h] [-show] [-save]

Histogram and lane line fitting for Udacity Advanced Lane Finding project

optional arguments:
  -h, --help  show this help message and exit
  -show       Visualize histogram and lane line images
  -save       Save histogram and lane line image
```

![](output_images/histogram_test2.png)
##### Example histogram slicing from warped undistorted image

![](output_images/lane_lines_test2.png)
##### Example lane line painting from warped undistored image
###5) Having identified the lane lines, has the radius of curvature of the road been estimated and the position of the vehicle with respect to center in the lane?

![](output_images/pipeline_test2.png)
##### Example pipeline image with radius of curvature and vehicle position
## Pipeline (video)

###1) Does the pipeline established with the test images work to process the video?

## Discussion


