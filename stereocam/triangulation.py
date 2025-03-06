import numpy as np 
import cv2
import glob
import os
import open3d as o3d


def stereo_map(calib_data, image_shape=(1080, 1920)):
    """Generates stereo maps.
    
    Parameters
    ----------
    calib_data: str
        Calibration data.
        This parameter can be either a ``str`` or ``dict``.
        If the parameter is ``str``, then it will be read using ``load_calibration_data`` function
    image_shape: tuple, default ``(1080, 1920)``
        Shape of the image.
        Although the image shape can be manually added, it is wise to extract the shape from the image
        when implementing in the pipeline.

    Returns
    -------
    stereoMapL: tuple
        A tuple of stereo map for remaping left image.
    stereoMapR: tuple
        A tuple of stereo map for remaping right image.
    
    """

    if type(calib_data) == str:
        calib_data = load_calibration_data(calib_data)

    # Extracting the calibration data from the dictionary
    mtxL, distL = calib_data["mtxL"], calib_data["distL"]
    mtxR, distR = calib_data["mtxR"], calib_data["distR"]
    R, T = calib_data["R"], calib_data["T"]
    R1, R2, P1, P2, Q = calib_data["R1"], calib_data["R2"], calib_data["P1"], calib_data["P2"], calib_data["Q"]

    stereoMapL = cv2.initUndistortRectifyMap(cameraMatrix=mtxL, 
                                             distCoeffs=distL, 
                                             R=R1, 
                                             newCameraMatrix=P1, 
                                             size=image_shape, 
                                             m1type=cv2.CV_16SC2)
    
    stereoMapR = cv2.initUndistortRectifyMap(cameraMatrix=mtxR, 
                                             distCoeffs=distR, 
                                             R=R2, 
                                             newCameraMatrix=P2, 
                                             size=image_shape, 
                                             m1type=cv2.CV_16SC2)

    return stereoMapL, stereoMapR

