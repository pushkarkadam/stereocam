import numpy as np 
import cv2 
import os
import sys 

sys.path.append('../')

from stereocam import *

# Disparity parameters
baseline = 0.12 # 120 cm or 0.12 m
num_disp_factor = 22
window_size = 5
min_disp = 300
num_disp = 16 * num_disp_factor

sgbm_params = {"minDisparity": min_disp,
               "numDisparities": num_disp,
               "blockSize": window_size,
               "P1": 8*3*window_size**2,
               "P2": 32*3*window_size**2,
               "disp12MaxDiff": 1,
               "preFilterCap": 0,
               "uniquenessRatio": 1,
               "speckleWindowSize": 0,
               "speckleRange": 2,
               "mode": 0
              }

# Loading camera data
data = np.load('../images/case2/stereo.npy', allow_pickle=True).item()

Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], data['R'], data['T'], data['img_size']

# Detect all the possible cameras
cam_available = detect_camera()

# Select stereo camera
camera_number = detect_stereo(cam_available)

# Creating a video capture object using the detected stereo camera
cap = cv2.VideoCapture(camera_number)

# obtaining the width and height of the images calibrated to use them to resize the input
width, height = img_size

# Resizing the image input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)



while cap.isOpened():
    retval, frame = cap.read()

    if not retval:
        print('\033[91m' + "Camera failed to start!")
        break

    left_frame, right_frame = split_stereo(frame)

    left_rect, right_rect = rectify_stereo_image(left_frame, right_frame, data)

    disp_map, depth_map = disparity_depth_map(
        left_img=left_rect,
        right_img=right_rect,
        baseline=baseline,
        data=data,
        algorithm="sgbm",
        save_path=None,
        **sgbm_params
    )

    disp_map = np.uint8(disp_map)
    
    cv2.imshow('left', left_frame)
    cv2.imshow('right', right_frame)

    cv2.imshow('disp', disp_map)
    
    if cv2.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

