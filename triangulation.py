import numpy as np 
import cv2
import glob
import os
from tqdm import tqdm


def rectify_stereo_image(left_img, right_img, data, save_path=None):
    """Rectifies the image for triangulation.
    
    left_img: numpy.ndarray
        A numpy image or file path to the image.
    right_img: numpy.ndarray
        A numpy image or file path to the image.
    data: dict
        A dictionary of camera parameters or a ``.npy`` file.
    save_path: str, default ``None``
        If path is given, then saves the rectified images to that path.
    
    Returns
    -------
    tuple
        A tuple of ``numpy.ndarray`` left and right rectified stereo image.

    """

    if type(left_img) == str:
        left_img = cv2.imread(left_img)
    
    if type(right_img) == str:
        right_img = cv2.imread(right_img)

    if type(data) == str:
        data = np.load(data, allow_pickle=True).item()

    Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], \
                                 data['R'], data['T'], data['img_size']

    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, img_size, R, T)

    xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, img_size, cv2.CV_32FC1)
    xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, img_size, cv2.CV_32FC1)

    left_img_rectified = cv2.remap(left_img, xmap1, ymap1, cv2.INTER_LINEAR)
    right_img_rectified = cv2.remap(right_img, xmap2, ymap2, cv2.INTER_LINEAR)

    if save_path:
        left_path = os.path.join(save_path, "left_rect.png")
        right_path = os.path.join(save_path, "right_rect.png")
        cv2.imwrite(left_path, left_img_rectified)
        cv2.imwrite(right_path, right_img_rectified)

    return left_img_rectified, right_img_rectified

def disparity_depth_map(left_img, 
                        right_img, 
                        baseline,
                        focal_length,
                        data, 
                        algorithm="bm",  
                        save_path=None,
                        **kwargs):
    """Returns disparity and depth map of the image.

    Parameters
    ----------
    left_img: numpy.ndarray
        A numpy image or file path to the image.
    right_img: numpy.ndarray
        A numpy image or file path to the image.
    data: dict
        A dictionary of camera parameters or a ``.npy`` file.
    save_path: str, default ``None``
        If path is given, then saves the rectified images to that path.
    
    Returns
    -------
    tuple
        A tuple of ``numpy.ndarray`` left and right rectified stereo image.
    
    """
    if type(left_img) == str:
        left_img = cv2.imread(left_img)
    
    if type(right_img) == str:
        right_img = cv2.imread(right_img)

    if type(data) == str:
        data = np.load(data, allow_pickle=True).item()

    Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], \
                                 data['R'], data['T'], data['img_size']


    if algorithm == "bm":
        stereo_alg = cv2.StereoBM_create(**kwargs)
    else:
        stereo_alg = cv2.StereoSGBM_create(**kwargs)

    disp_map = stereo_alg.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), 
                                  cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                                 )

    disparity_map = np.where(disp_map <= 0, 1e-5, disp_map)

    depth_map = (baseline * focal_length) / disparity_map

    if save_path:
        disp_path = os.path.join(save_path, "disp_map.png")
        depth_path = os.path.join(save_path, "depth_map.png")
        cv2.imwrite(disp_path, disp_map)
        cv2.imwrite(depth_path, depth_map)

    return disp_map, depth_map

def main():
    left, right = rectify_stereo_image("images/case2/stereo_left/imageL_14.png",
    "images/case2/stereo_right/imageR_14.png",
    "images/case2/stereo.npy"
    )
    
    disp_map, depth_map = disparity_depth_map(
        left_img=left,
        right_img=right,
        data="images/case2/stereo.npy",
        algorithm="sgbm",
        baseline=0.12,
        focal_length=1066,
        save_path='images',
    )

if __name__ == '__main__':
    main()