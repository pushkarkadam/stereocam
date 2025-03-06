import numpy as np 
import cv2


def load_calibration_data(calib_file_path):
    """Loads calibration data.
    
    Parameters
    ----------
    calib_file_path: str
        Path where the calibration data is stored
    
    """

    return np.load(calib_file_path)

def colorspace_transform(imageL, imageR, colorspace=cv2.COLOR_BGR2HSV):
    """Transforms the image into defined colorspace.
    
    Parameters
    ----------
    imageL: str
        Image path or ``numpy.ndarray`` image matrix.
    imageR: str
        Image path or ``numpy.ndarray`` image matrix.
    colorspace: int, default ``cv2.COLOR_BGR2HSV``

    Returns
    -------
    imageL: numpy.ndarray
        Rectified left image.
    imageR: numpy.ndarray
        Rectified right image.
        
    """
    if type(imageL) == str and type(imageR) == str:
        imageL = cv2.imread(imageL)
        imageR = cv2.imread(imageR)

    images = [imageL, imageR]

    rect_images = [cv2.cvtColor(image, colorspace) for image in images]

    return rect_images