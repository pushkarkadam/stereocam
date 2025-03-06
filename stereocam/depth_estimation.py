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

def depth_maps(imageL, 
               imageR, 
               Q,
               dispFactor=1, 
               blockSize=5, 
               minDisparity=500,
               disp12MaxDiff=-1,
               preFilterCap=30,
               uniquenessRatio=0,
               speckleWindowSize=100, # range 50-200
               speckleRange=2, # range 1 or 2
               mode = 0,
               image_type='hsv'
              ):
    """Disparity map generation.

    Some paramters provided in this function are taken from opencv method for
    stereo disparity estimation using SGBM method.

    Refer to the documentations::

        https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
    
    Parameters
    ----------
    imageL: numpy.ndarray
        Left image from the stereo camera. Formats: hsv, bgr, or rgb
    imageR: numpy.ndarray
        Right image from the stereo camera. Formats: hsv, bgr, or rgb
    Q: numpy.ndarray
        Re-projection matrix. Available from camera calibration parameters.
    dispFactor: int, default ``1``
        Disparity factor to be multiplied by ``16`` in the code.
    blockSize: int
        The block size for scanning.
    minDisparity: int
        Minimum number of disparity.
        Can be calculated by knowing the approximate distance of the background object.
    disp12MaxDiff: int, default ``-1``
        Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    preFilterCap: int, default ``30``
        Truncation value for the prefiltered image pixels.
    uniquenessRatio: int, default ``0``
        Enforces the requirement that the match value for the current pixel is more
        than the minimum match value observed by some margin. Range: ``5 ~ 15``.
    speckleWindowSize: int, default ``100``
        Maximum size of smooth disparity region. Range: ``50 ~ 200``
    speckleRange: int, default ``2``
        Maximum disparity variation. Range: ``1 ~ 2``
    mode: int, default ``0``
        The method used to compute.
    image_type: str, default ``'hsv'``
        Image type as the input to use correct conversion to gray scale.
    
    Returns
    -------
    disparity: numpy.ndarray
        Disparity matrix where each element consists of the disparity values.
    camera_projection: numpy.ndarray
        A 3D tensor where each channel consists information about the x, y, and z
        coordinates with respect to the left camera frame.
    depth_map: numpy.ndarray
        A matrix that shows depth value of each pixel in left camera frame.

    """
    stereo_images = [imageL, imageR]
    
    # converting the images
    if image_type == 'hsv':
        grayL, grayR = [hsv2gray(image) for image in stereo_images]
    elif image_type == 'bgr':
        grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in stereo_images]
    elif image_type == 'rgb':
        grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in stereo_images]
    else:
        grayL, grayR = stereo_images

    # creating a stereo object
    stereo = cv2.StereoSGBM.create(
        minDisparity=minDisparity,
        numDisparities=16 * dispFactor,
        blockSize=blockSize,
        P1=8 * 3 * blockSize**2,
        P2=32 * 3 * blockSize**2,
        disp12MaxDiff=disp12MaxDiff,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize, # range 50-200
        speckleRange=speckleRange, # range 1 or 2
        mode=mode
    )
    
    # Computing disparity
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Replacing the zero values with smallest values
    disparity[disparity <=0] = 0.1

    # Reprojecting the disparity map to camera coordinates
    camera_projection = cv2.reprojectImageTo3D(disparity, Q)

    # Extracting last channel of the camera projection which is the z axis for depth
    depth_map = camera_projection[:,:,-1]

    return disparity, camera_projection, depth_map

def point_cloud(image,
                depth_limits,
                depth_map,
                image_type='bgr',
                save_path=None,
                pcd_name="point_cloud.ply",
                visualize=True
               ):
    """Generates and saves point cloud.
    
    Parameters
    ----------
    image: numpy.ndarray
        Color image used. Ideally left image of the stereo camera.
        The image should be of the camera whose coordinate frames are primary frame.
    depth_limits: tuple
        Limits of the depth to be bounded.
    image_type: str, default ``bgr``
        The channels of the image specified.
    save_path, str, default ``None``
        The path to save the point cloud.
    pcd_name: str, default ``'point_cloud.ply'``
        Name of the point cloud file.
    visualize: bool, default ``True``
        Visualizes the point cloud.

    Returns
    -------
    open3d.cpu.pybind.geometry.PointCloud
    
    """

    if image_type == 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    colors = image.reshape(-1, 3) / 255.0
    points = depth_map.reshape(-1, 3)

    colors = colors.astype(np.float32)
    points = colors.astype(np.float32)

    # lower and upper limit
    lower_lim, upper_lim = depth_limits

    # Creating a mask as per the depth limit set
    valid_mask = (depth_map > lower_lim) & (depth_map < upper_lim)

    valid_colors = colors[valid_mask.ravel()]
    valid_points = points[valid_mask.ravel()]

    # Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    if save_path:
        o3d.io.write_point_cloud(os.path.join(save_path, pcd_name), pcd)

    if visualize:
        o3d.visualization.draw_geometries([pcd])

    return pcd
