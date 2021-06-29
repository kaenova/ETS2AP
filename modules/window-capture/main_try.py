import cv2
import numpy as np
import os
from time import time
from windowcapture import WindowCapture

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def RoICrop(img):
    return img[150:530, 430:1000]

def ExposureCorrection(img):
    return cv2.convertScaleAbs(img, alpha=0.9, beta=0)

def BirdEyeView(img):
    IMAGE_CROP_H, IMAGE_CROP_W, IMAGE_CROP_CHANNELS = img.shape
    pts1 = np.float32([[100, IMAGE_CROP_H], [200, 220], [300, 220], [410, IMAGE_CROP_H]])
    pts2 = np.float32([[125, 300], [100, 0], [200, 0], [175, 300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix_inv = cv2.getPerspectiveTransform(pts2, pts1)
    return cv2.warpPerspective(img, matrix, (300,300))

def HLSConversion(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    Lchannel = hls[:,:,1]
    mask = cv2.inRange(Lchannel, 140, 255)
    return cv2.bitwise_and(img,img, mask= mask)

def GrayImageConversion(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def BnWImageConversion(img):
    (thresh, black_and_white) = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    return black_and_white

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

    cv2.imshow('Computer Vision: Bird Eye View', bird_eye_view)
    cv2.imshow('Computer Vision: Bird Eye View with Lines', black_and_white)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

print('Done.')
