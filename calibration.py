import numpy as np 
import cv2
import glob
import os
from tqdm import tqdm


def read_chess_board(file_path, pattern_dim=(7, 3), image_format="png", display_rendered=False):
    """Function to read the features of the chess board for calibration.

    Parameters
    ----------
    file_path: str
        Path to images
    pattern_dim: tuple, default ``(7, 3)``
        Dimension of the chess board.
        Make sure to subtract the border boxes of the chessboard.
    image_format: str, default ``"png"``
        The format to read the images.
    display_rendered: bool, default ``False``
        Displays rendered images.

    Returns
    -------
    objpoints: list
        A list of ``np.array`` of the world coordinates points.
    imgpoints: list
        A list of ``np.array`` of image coordinates corresponding to the world coordinates.
    
    """
    # pattern dimension
    row_p, col_p = pattern_dim

    # terminaltion criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((row_p * col_p, 3), np.float32)
    objp[:,:2] = np.mgrid[0:row_p, 0:col_p].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.

    # 3d point in real world space
    objpoints = []

    # 2d points in image plane
    imgpoints = []

    images = list(sorted(glob.glob(os.path.join(file_path ,'*.' + image_format))))
    
    for fname in tqdm(images, colour="green"):
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
            if display_rendered:
                cv2.drawChessboardCorners(img, (row_p, col_p), refined_corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    return objpoints, imgpoints

# Calibration
# ret, mtx, dist, 

def calibration(file_path, 
                pattern_dim=(7, 3), 
                image_format="png", 
                display_rendered=False, 
                undistort_name="calibresult.png", 
                img_crop=None,
                cam_save_postfix=None
               ):
    """Function to read the features of the chess board for calibration.

    Parameters
    ----------
    file_path: str
        Path to images
    pattern_dim: tuple, default ``(7, 3)``
        Dimension of the chess board.
        Make sure to subtract the border boxes of the chessboard.
    image_format: str, default ``"png"``
        The format to read the images.
    display_rendered: bool, default ``False``
        Displays rendered images.
    undistort_name: str, default ``"calibresult.png"``
        The name of the image to store to check the distortion.
    img_crop: list, default ``None``
        A list of tuples that crops the image.
        e.g. ``[(0,100), (10,100)]`` will crop the image as ``I[0:100, 10:100]``
        If the value is ``None``, then it will not crop the images.
    cam_save_postfix: str, default ``None``
        This is used to save the camera matrix and distortion coefficients.
    
    """
    # pattern dimension
    row_p, col_p = pattern_dim

    # terminaltion criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((row_p * col_p, 3), np.float32)
    objp[:,:2] = np.mgrid[0:row_p, 0:col_p].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.

    # 3d point in real world space
    objpoints = []

    # 2d points in image plane
    imgpoints = []

    images = list(sorted(glob.glob(os.path.join(file_path ,'*.' + image_format))))
    
    for fname in tqdm(images, colour="green"):
        img = cv2.imread(fname)

        if img_crop:
            c = img_crop
            img = img[c[0][0]: c[0][1], c[1][0]: c[1][1], :]
        
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
            if display_rendered:
                cv2.drawChessboardCorners(img, (row_p, col_p), refined_corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistortion
    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image 
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.putText(dst, images[0], (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(undistort_name, dst)

    # Saving camera matrix and distortion coefficients
    if cam_save_postfix:
        cam = cam_save_postfix
        campar_save_path = os.path.join(file_path, f'cam_cal_parameters_{cam}.npz')
        np.savez(campar_save_path,
                 newcameramtx = newcameramtx,
                 mtx = mtx,
                 dist = dist,
                 rvecs = rvecs,
                 tvecs = tvecs,
                 refined_corners = refined_corners
                )

    # reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print(f'total error: {mean_error/len(objpoints)}')