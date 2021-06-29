import cv2
import numpy as np
import os
from time import time
import d3dshot

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
d = d3dshot.create()

loop_time = time()
while(True):

    # get an updated image of the game
    d.screenshot()
    open_cv_image = np.array(d) 
    print(open_cv_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    cv2.imshow('Computer Vision', open_cv_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

print('Done.')
