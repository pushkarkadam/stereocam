import numpy as np 
import cv2
import glob
import os


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

    images = glob.glob(os.path.join(file_path ,'*.' + image_format))
    
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