import numpy as np 
import cv2
import glob


# pattern dimension
row_p, col_p = (10, 8)

# terminaltion criteria
criteria = (cv2.TERM_CRITEREIA_EPS + CV2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
objp = np.zeros((row_p * col_p, 3), np.float32)
objp[:,:2] = np.mgrid[0:row_p, 0:col_p].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.

# 3d point in real world space
objpoints = []

# 2d points in image plane
imgpoints = []

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corneres
    ret, corners = cv2.findChessboardCorners(gray, (row_p, col_p), None)

    # If found, and object points, image points (after regining them)
    if ret:
        # Appending the objp to 3d points in real world space
        objpoints.append(objp)

        # Refining the corner location
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Appending the refined corners to 2d points in image plane
        imgpoints.append(refined_corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (row_p, col_p), refined_corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration
# ret, mtx, dist, 