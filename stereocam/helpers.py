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

def clahe_filter(imageL, imageR, clipLimit=2.0, tileGridSize=(8,8)):
    """Applying clahe filter.

    Parameters
    ----------
    imageL: str
        Image path or ``numpy.ndarray`` image matrix.
    imageR: str
        Image path or ``numpy.ndarray`` image matrix.
    
    Returns
    -------
    filtered_images: numpy.ndarray
        Filtered image.
    """
    images = [imageL, imageR]

    filtered_images = []

    for image in images:
        c1, c2, c3 = np.split(image, indices_or_sections=3, axis=2)
        clahe = cv2.createCLAHE(clipLimit,  tileGridSize=tileGridSize)

        # applying the filters
        c1, c2, c3 = [clahe.apply(c) for c in [c1, c2, c3]]

        image_filtered = np.stack((c1, c2, c3), axis=2)

        filtered_images.append(image_filtered)

    return filtered_images

def hsv2gray(image):
    """Converts HSV image to grayscale image.
    
    Parameters
    ----------
    image: numpy.ndarray
        Image in the BGR format.
    
    Returns
    -------
    gray: numpy.ndarray
        Grayscale image.
    
    """
    
    # Converting HSV image to BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # Converting BGR to Gray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    return gray