import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#convert into grey scale image
def grey(image):
  image=np.asarray(image)
  return cv.cvtColor(image,cv.COLOR_RGB2GRAY)
#Gaussian blur to reduce noise and smoothen the image
def gauss(image):
  return cv.GaussianBlur(image,(5,5),0)
#Canny edge detection
def canny(image):
    edges = cv.Canny(image,50,150)
    return edges

def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(320, height-90), (width//2, 0+300), (width-320, height-90)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv.fillPoly(mask, triangle, 255)
    mask = cv.bitwise_and(image, mask)
    return mask

def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    print(y1)
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
          
        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1) 
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
              
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = create_coordinates(image, left_fit_average)
    right_line = create_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


# initialize the WindowCapture class
wincap = WindowCapture(None)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    grey_img = grey(screenshot)
    gauss_img = gauss(grey_img)
    canny_img = canny(gauss_img)
    isolated = region(canny_img)
    lines = cv.HoughLinesP(isolated, 2, np.pi / 180, 100, 
                            np.array([]), minLineLength = 40, 
                            maxLineGap = 5) 
    averaged_lines = average_slope_intercept(screenshot, lines) 
    line_image = display_lines(screenshot, averaged_lines)
    combo_image = cv.addWeighted(screenshot, 0.8, line_image, 1, 1)

    cv.imshow('Computer Vision: RAW', combo_image)
    cv.imshow('Computer Vision: Isolated no Canny', isolated)

    # debug the loop rate
    #print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
