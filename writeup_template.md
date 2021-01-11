## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./test_images/test3.jpg "Road Transformed"
[image2]: ./test_images/undistort_testImg.png "Road Transformed"
[image3]: ./output_images/binary_output.png "Binary Example"
[image4]: ./output_images/warp_output.png "Warp Example"
[image5]: ./output_images/fit_output.png "Fit Visual"
[image6]: ./output_images/final_output.png "Output"
[image7]: ./output_images/LeftFitThr.png "Threshold LF"
[image8]: ./output_images/RightFitThr.png "Threshold RF"
[video1]: ./project_videoOutput.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "P2.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at 4th cell of the P2.ipynb).  Here's an example of my output for this step. 

LAB color space -> b channel - easy to separate yellow left lane lines (both from shadowing of trees and pixels lost from presence of sun). But does not detect white right lanes
RGB space -> R channel - easy to get some of yellow left lane pixels, white lines are still visible.
HLS space -> S channel - used to get pixels from sun
Sobel for x,y along with gradients are sort of filters from which other previous color space thresholdings are applied. The result is a thresholded binary image that supports shadows, pixels in front of sun and covers all other aspects. Output images in output_images folder contain thresholded binary images of all test images in test_images folder.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 5th code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 70, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 20), img_size[1]],
    [(img_size[0] * 5 / 6) + 150, img_size[1]],
    [(img_size[0] / 2 + 100), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 460      | 320, 0        | 
| 193, 720      | 320, 720      |
| 1067, 720     | 960, 720      |
| 740, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First I used the sliding window method find_lane_pixels() to find the coefficents of the fit for the relevant pixels of left and right lanes. Then I use this information and fixed margin to search for lane pixels which belong to the frame around previous fit polynomial. After finding pixels, they are again fit using second order polynomial. This is done in search_around_poly().



These steps are done in 6th cell of P2.ipynb with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cell 7 in my code in P2.ipynb. I calculated the meters per pixel in x and y dimension and fit the pixels again in that dimension and then calculate the radius of curvature from formula provided in the lesson. The offset of the image center from center of lane (detected at the image end) is the position of vehicle with respect to the center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 8 in my file P2.ipynb. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


Note: Pls check also the html file as autosave did not work and the last code cell output was not saved. (Please help to fix this, ipynb files dont save when we try often). The ouput video below is valid and is direct output of this cell.

For videos, first frame of video: Using sliding window method (find_lane_pixels()), polynomial is fit for lane pixels of frame and the lane pixels are searched around previous fit polynomial and again fit. Final fit from search_around_poly() is taken as an output. 
Everytime new frame is encountered, the sliding window step is skipped and fit by search_around_poly(), a difference in current and previous fit of previous frame (from search_around_poly) are determined (in a,b,c of ax2+bx+c) and initially a correlation is found as a spike in the graph of difference in coefficients of left and right fit between current frame and previous frame - whenever the ego car turns in the highway. A threshold is determined by the correlation graph and can be used to check for bad line. If it is a bad line, again frame is fit using find_lane_pixels() and subsequently searched and fit using search_around_poly(). This is a reset step. To prevent jumping, a smoothing is applied based on mean of 10 frames. The correlation graphs for left and right fit are shown below. The three graphs are a,b,c coefficents of second order polynomial. X axis depicts the number of frames and Y axis depicts the difference in coefficients (a,b,c) between current and previous frame.

![alt text][image7]
![alt text][image8]


Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems:

1) The thresholding step is very important, if unwanted pixels are identified from this step, then the warping step will have an issue. After warping did not achieve desired result, I returned back to thresholding step and fixed it.

2) ROI was difficult to tune manually.

3) Finding which frame to reset to sliding window step was difficult. Spent many days to find a way to fix this. Checked the correlation of coefficients between current and previous frame and found a solution.

Pipeline might fail for other videos as ROI differs. Smoothing will be a problem if even one frame changes very fast and if lane lines are not detected. One way to improve could be to check on degree of polynomial for other challenge videos. 
