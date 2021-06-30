import cv2
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
import matplotlib.pyplot as plt
from PIL import Image

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def RoICrop(img):
    return img[150:530, 430:1000]

def ExposureCorrection(img):
    return cv2.convertScaleAbs(img, alpha=0.9, beta=0)

def BirdEyeView(img):
    IMAGE_CROP_H, IMAGE_CROP_W, IMAGE_CROP_CHANNELS = img.shape
    pts1 = np.float32([[100, IMAGE_CROP_H], [200, 230], [300, 230], [410, IMAGE_CROP_H]])
    pts2 = np.float32([[125, 300], [100, 0], [200, 0], [175, 300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix_inv = cv2.getPerspectiveTransform(pts2, pts1)
    return cv2.warpPerspective(img, matrix, (300,300))

def HLSConversion(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    Lchannel = hls[:,:,1]
    mask = cv2.inRange(Lchannel, 120, 255)
    return cv2.bitwise_and(img,img, mask= mask)

def GrayImageConversion(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def BnWImageConversion(img):
    (thresh, black_and_white) = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    return black_and_white

def CroppingBeforeSlidingWindow(img):
    return img[:, 90:230]

# https://kushalbkusram.medium.com/advanced-lane-detection-fd39572cfe91
# https://github.com/KushalBKusram/AdvancedLaneDetection/blob/master/src/laneDetection.py
def find_lane_pixels(image):
    NILAI_DARI_POJOK_KIRI = 30
    
    
    crop_for_histogram = image[200:image.shape[0], NILAI_DARI_POJOK_KIRI:120]
    histogram = np.sum(crop_for_histogram[crop_for_histogram.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((image, image, image)) * 255
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint]) + NILAI_DARI_POJOK_KIRI
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint + NILAI_DARI_POJOK_KIRI
    print("left ",leftx_base)
    print("right ",rightx_base)

    nwindows = 20
    margin = 8
    minpix = 8

    window_height = np.int(image.shape[0] // nwindows)

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 4)

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]

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
    

    # return leftx, lefty, rightx, righty, out_img
    return out_img

# initialize the WindowCapture class
wincap = WindowCapture(None)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    img_buffer = RoICrop(screenshot)
    img_buffer = ExposureCorrection(img_buffer)
    bird_eye_view = BirdEyeView(img_buffer)
    black_and_white = HLSConversion(bird_eye_view)
    black_and_white = GrayImageConversion(black_and_white)
    black_and_white = BnWImageConversion(black_and_white)
    cropped = CroppingBeforeSlidingWindow(black_and_white)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    sliding_window = find_lane_pixels(cropped)
    
    combined_image= cv2.addWeighted(sliding_window, 0.3, cropped_rgb, 0.7, 0)

    cv2.imshow('Computer Vision: Bird Eye View', bird_eye_view)
    cv2.imshow('Computer Vision: Bird Eye View with Lines', black_and_white)
    cv2.imshow('Computer Vision: Combined', combined_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

print('Done.')
