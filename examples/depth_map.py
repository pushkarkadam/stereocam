import sys 

sys.path.append('../')
from stereocam import *

def main():
    left, right = rectify_stereo_image(
        left_img="../images/case2/stereo_left/imageL_14.png",
        right_img="images/case2/stereo_right/imageR_14.png",
        data="images/case2/stereo.npy"
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