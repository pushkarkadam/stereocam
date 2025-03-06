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