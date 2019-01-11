#!/usr/bin/env python
# coding: utf-8

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# ## First step:
# 
# The first step is to compute the camera calibration matrix (mtx) and distortion coefficients (dist) given a set of chessboard calibration images from (../camera_cal/). Camera calibration is done by function cv2.calibrateCamera() to compute the transformation between 3D object points in the world and 2D image points.

# In[13]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import pdb
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

nx = 9
ny = 6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
   
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
 
    # If found, add object points, image points
    if ret == True:
        #print(fname)
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

 # Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

Step1_pickle = {}
Step1_pickle["mtx"] = mtx
Step1_pickle ["dist"] = dist
with open('Step1.pkl', 'wb') as f:
    pickle.dump(Step1_pickle, f)


# ## Second step:
# 
# The second step is to apply distortion correction to the raw images using cv2.undistort(). Distortion correction is used to make sure that the geometrical shape of the objects is represend consistently, no matter where they appear in the image.
#   

# In[14]:


image = cv2.imread("camera_cal/calibration1.jpg")[...,::-1]
    
## Step2 - Distortion correction
undist = cv2.undistort(image, mtx, dist, None, mtx)  
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom= 0.)


# In[ ]:


images = os.listdir("test_images/")

for image in images:
    image = cv2.imread("test_images/"+ image)[...,::-1]
    
    ## Step2 - Distortion correction
    undist = cv2.undistort(image, mtx, dist, None, mtx)  
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom= 0.)


# In[ ]:


images = os.listdir("test_images/")

def abs_sobel_thresh(gray, orient, sobel_kernel, thresh):
    # Calculate directional gradient
    #Compute gradients with sobel operators in x and y direction
    if orient == 'x':
      sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
      sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    sobel = np.int8(255 * np.absolute(sobel)/np.max(sobel))
    grad_binary = np.zeros_like(sobel)
    grad_binary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
    # Apply threshold
    return grad_binary

def mag_thresh(gray, sobel_kernel, mag_thresh):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    mag_gradient = np.absolute(sobelx**2 + sobely**2)
    mag_gradient = np.int8(255* (mag_gradient / np.max(mag_gradient)))
    mag_binary = np.zeros_like(mag_gradient)
    # Apply threshold
    mag_binary[(mag_gradient >= mag_thresh[0]) & (mag_gradient <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(gray, sobel_kernel, thresh):
    # Calculate gradient direction
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    dir_gradient = np.arctan(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(dir_gradient)
    # Apply threshold
    dir_binary[(dir_gradient >= thresh[0]) & (dir_gradient <= thresh[1])] = 1
    return dir_binary

def preprocess(undist):
    ## Step3: Apply gradients
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    grad_binaryx = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(15,255))
    grad_binaryy = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(15, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=3, mag_thresh=(15, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=3, thresh=(0.7, 1.3))
    combined_sobel = np.zeros_like(dir_binary)
    combined_sobel[((grad_binaryx == 1) & (grad_binaryy == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    ## Step4: Apply color thresholding
    s_thresh=(150, 255) 
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(s_binary), combined_sobel, s_binary)) * 255
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (combined_sobel == 1)] = 1
    
    # Use R channel
    R = undist[:,:,0]
    thresh=(210, 255)
    binary = np.zeros_like(s_channel)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    binary_img = np.zeros_like(binary)
    binary_img[((combined_binary == 1) & (binary == 1))] = 1
    
    LAB = cv2.cvtColor(undist, cv2.COLOR_RGB2LAB)
    b = LAB[:,:,2]
    
    thresh=(150, 200)
    binaryB = np.zeros_like(s_channel)
    binaryB[(b > thresh[0]) & (b <= thresh[1])] = 1
    
    binaryImgFinal = np.zeros_like(s_channel)
    binaryImgFinal[(binary_img == 1) | (binaryB == 1)] = 1
    
    return binaryImgFinal

for image in images:
    image = cv2.imread("test_images/"+ image)[...,::-1]
    
    ## Step2 - Distortion correction
    undist = cv2.undistort(image, mtx, dist, None, mtx)  
    ## Step3 - Distortion correction
    binary_image = preprocess(image)  
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(binary_image, cmap = 'gray')
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom= 0.)


# ## Fourth step:
# 
# The fourth step is to apply perspective transform in the image to get bird-eyes view. Perspective transform is used to transform an image
# such that we are effectively viewing object from a different angle or direction.

# In[ ]:


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped, Minv

image = cv2.imread("test_images/test3.jpg")[...,::-1]
img_size = (image.shape[1], image.shape[0])

## Step2 - Distortion correction
undist = cv2.undistort(image, mtx, dist, None, mtx)  
## Step3 - Preprocess image to create binary thresholded image
binary_image = preprocess(image)  

src = np.float32(
              [[(img_size[0] / 2) - 70, (img_size[1] / 2) + 100],
              [((img_size[0] / 6))- 20, img_size[1]],
              [(img_size[0] * 5 / 6) + 150, img_size[1]],
              [(img_size[0] / 2 + 100), (img_size[1] / 2) + 100]])
   
dst = np.float32(
             [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])
## Step4 - Apply perspective transform
binary_warped, Minv = warper(binary_image, src, dst)

line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
cv2.line(line_img, (src[0][0], src[0][1]), (src[1][0], src[1][1]), color=[255, 0, 0], thickness=3)
cv2.line(line_img, (src[2][0], src[2][1]), (src[3][0], src[3][1]), color=[255, 0, 0], thickness=3)
cv2.line(line_img, (src[0][0], src[0][1]), (src[3][0], src[3][1]), color=[255, 0, 0], thickness=3)
lane_onSRC = weighted_img(line_img, image, α=0.8, β=1., γ=0.)

warpedImage = binary_warped*255
color_edges = np.dstack((warpedImage, warpedImage, warpedImage)) 
line_imgdst = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
cv2.line(line_imgdst, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), color=[255, 0, 0], thickness=3)
cv2.line(line_imgdst, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), color=[255, 0, 0], thickness=3)
cv2.line(line_imgdst, (dst[0][0], dst[0][1]), (dst[3][0], dst[3][1]), color=[255, 0, 0], thickness=3)
lane_onDST = weighted_img(line_imgdst, color_edges, α=0.8, β=1., γ=0.)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(lane_onSRC)
ax1.set_title('Source points drawn', fontsize=50)
ax2.imshow(lane_onDST, cmap = 'gray')
ax2.set_title('Destination points drawn', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom= 0.)


# # Fifth step - Detect lane pixels and fit to find the lane boundary
# 
# In this step, histogram values are used to get the areas with high pixel values which are further used to detect the lane pixels and boundaries of left and right side of the lane lines.

# In[ ]:


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #print(binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    return left_fit, right_fit

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose
    #the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fitNew, right_fitNew = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result, left_fitx, right_fitx, ploty, left_fitNew, right_fitNew

left_fit, right_fit = fit_polynomial(binary_warped)
result, left_fitx, right_fitx, ploty, lf, rf = search_around_poly(binary_warped, left_fit, right_fit)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(lane_onSRC)
ax1.set_title('Input image', fontsize=50)
ax2.imshow(result, cmap = 'gray')
ax2.set_title('Lane pixels detected and fit', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom= 0.)


# # Sixth step
# 
# In this step, the identified lane pixel values help to determine the radius of curvature and position of the vehicle from the center.

# In[ ]:


# Seventh step

In this step, the identified lane pixel values help to determine the radius of curvature and position of the vehicle from the center.


# In[ ]:


def measure_curvature(binary_warped, ploty, left_fitx, right_fitx):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 15/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/900 # meters per pixel in x dimension
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix  + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix  + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    average_curverad = (left_curverad+right_curverad)/2

    Rcurvature = "Radius of curvature: %.2f m" % average_curverad

    # compute the offset from the center
    lane_center = (right_fitx[719] + left_fitx[719])/2
    center_offset_pixels = abs((binary_warped.shape[1]/2) - lane_center)
    center_offset_m = xm_per_pix*center_offset_pixels
    C_offset = "Center offset: %.2f m" % center_offset_m

    # compute the offset from the center
    lane_center = (right_fitx[719] + left_fitx[719])/2
    center_offset_pixels = abs((binary_warped.shape[1]/2) - lane_center)
    center_offset_m = xm_per_pix*center_offset_pixels
    C_offset = "Center offset: %.2f m" % center_offset_m
    return Rcurvature, C_offset

Rcurvature, C_Offset = measure_curvature(binary_warped, ploty, left_fitx, right_fitx)
print(Rcurvature)
print(C_Offset)


# # Seventh step
# 
# In this step, the detected lane boundaries are warped back onto the original image using inverse perspective transform

# # Eighth step
# 
# Output visual display of the lane boundaries and display of lane curvature and vehicle position.

# In[ ]:


left_fit, right_fit = fit_polynomial(binary_warped)
result, left_fitx, right_fitx, ploty, lf, rf = search_around_poly(binary_warped, left_fit, right_fit)

R_curvature, C_offset = measure_curvature(binary_warped, ploty, left_fitx, right_fitx)

warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
left_fitx = left_fitx - 5* np.ones(720)
right_fitx = right_fitx + 10* np.ones(720)

#Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

cv2.putText(result, R_curvature , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
cv2.putText(result, C_offset, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Input image', fontsize=50)
ax2.imshow(result, cmap = 'gray')
ax2.set_title('Lane detected image after Inverse perspective transform', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom= 0.)


# In[ ]:


def fitPoly(warped_image):
   
    global lfSearch
    global rfSearch
    global left_fit
    global right_fit
    global value 
    global i
    global j
    global store
    global addLX
    global addX
    global addRX
    global saveLF
    global lfFails
    global saveLF1
    
    lfSearch = []
    rfSearch = []
    addLX = []
    addRX = []
    i = 0
    # first time
    if (i == 0) :
        left_fit, right_fit = fit_polynomial(warped_image)
        result, left_fitx, right_fitx, ploty, lf, rf = search_around_poly(warped_image, left_fit, right_fit)
        lfSearch.append(lf)
        rfSearch.append(rf)

    else :
        result, left_fitx, right_fitx, ploty, lf, rf = search_around_poly(warped_image, left_fit, right_fit)
        lfSearch.append(lf)
        rfSearch.append(rf)
        diffL = np.array(lfSearch[-1][:]) - np.array(lfSearch[-2][:])
        diffR = np.array(rfSearch[-1][:]) - np.array(rfSearch[-2][:])  
        if ((diffL[0] > 0.0001) | (diffL[0] < -0.0001)) & ((diffL[1] > 0.1) | (diffL[1] < -0.1)) & ((diffL[2] > 25) | (diffL[2] < -25)):
            good_linesLeft = False
        else:
            good_linesLeft = True
          
        if ((diffR[0] > 0.0002) | (diffR[0] < -0.0002)) & ((diffR[1] > 0.2) | (diffR[1] < -0.2)) & ((diffR[2] > 25) | (diffR[2] < -25)):
            good_linesRight = False
        else:
            good_linesRight = True
       
        if ((good_linesLeft == False) | (good_linesRight == False)):
            #lfSearch = lfSearch[:-1]
            #rfSearch = rfSearch[:-1]
            lfSearch.append(np.array([0,0,0]))
            rfSearch.append(np.array([0,0,0]))
            left_fit, right_fit = fit_polynomial(warped_image)
            result, left_fitx, right_fitx, ploty, lf, rf = search_around_poly(warped_image, left_fit, right_fit)
            lfSearch.append(lf)
            rfSearch.append(rf)
    i = i+1
    R_curvature, C_offset = measure_curvature(warped_image, ploty, left_fitx, right_fitx)
    if len(addLX) > 9:
        addLX = []
    else:
        addLX.append(left_fitx)
    if len(addRX) > 9:
        addRX = []
    else:
        addRX.append(right_fitx)
    left_fitx = np.mean(addLX, axis = 0)
    right_fitx = np.mean(addRX, axis = 0)
    return left_fitx, right_fitx, ploty, R_curvature, C_offset
        
def process_image(image):
    img_size = (image.shape[1], image.shape[0])
    
    ## Step2 - Distortion correction
    undist = cv2.undistort(image, mtx, dist, None, mtx)      
    combined = preprocess(undist)
    
    src = np.float32(
              [[(img_size[0] / 2) - 70, (img_size[1] / 2) + 100],
              [((img_size[0] / 6))- 20, img_size[1]],
              [(img_size[0] * 5 / 6) + 150, img_size[1]],
              [(img_size[0] / 2 + 100), (img_size[1] / 2) + 100]])
   
    dst = np.float32(
             [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])
    
    
    ## Apply perspective transform
    binary_warped, Minv = warper(combined, src, dst)
    
    # Thresholded image with source points drawn
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cv2.line(line_img, (src[0][0], src[0][1]), (src[1][0], src[1][1]), color=[255, 0, 0], thickness=3)
    cv2.line(line_img, (src[2][0], src[2][1]), (src[3][0], src[3][1]), color=[255, 0, 0], thickness=3)
    cv2.line(line_img, (src[0][0], src[0][1]), (src[3][0], src[3][1]), color=[255, 0, 0], thickness=3)
    lane_onSRC = weighted_img(line_img, image, α=0.8, β=1., γ=0.)
    
    # Warped image with destination points drawn
    warpedImage = binary_warped*255
    color_edges = np.dstack((warpedImage, warpedImage, warpedImage)) 
    line_imgdst = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cv2.line(line_imgdst, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), color=[255, 0, 0], thickness=3)
    cv2.line(line_imgdst, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), color=[255, 0, 0], thickness=3)
    cv2.line(line_imgdst, (dst[0][0], dst[0][1]), (dst[3][0], dst[3][1]), color=[255, 0, 0], thickness=3)
    lane_onDST = weighted_img(line_imgdst, color_edges, α=0.8, β=1., γ=0.)
    
    left_fitx, right_fitx, ploty, R, Offset = fitPoly(binary_warped)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    left_fitx = left_fitx - 5* np.ones(720)
    right_fitx = right_fitx + 10* np.ones(720)
    
    #Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    cv2.putText(result, R , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
    cv2.putText(result, Offset, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
    return result

white_output = 'project_videoOutput.mp4'
#Where start_second and end_second are integer values representing the start and end of the subclip
#You may also uncomment the following line for a subclip of the first 5 secondsclip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("challenge_video.mp4").subclip(0, 1)clip1 = VideoFileClip("project_video.mp4").subclip(20, 25)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


# In[ ]:




