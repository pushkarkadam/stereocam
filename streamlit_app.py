import streamlit as st
import numpy as np
import cv2
import time
import os
import yaml
import pickle
import copy
from stereocam import *


# Data paths
DATA_PATH = 'images/case2/stereo.npy'
data = np.load(DATA_PATH, allow_pickle=True).item()

Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], data['R'], data['T'], data['img_size']


# PARAMS
baseline = 0.12 # 12 cm or 0.12 m
num_disp_factor = 22
window_size = 5
min_disp = 300
wls_lambda = 8000
wls_sigma = 2.5

# Detect all the possible cameras
cam_available = detect_camera()

# Select stereo camera
stereo_retval, camera_number = detect_stereo(cam_available)

# Creating a video capture object using the detected stereo camera
cap = cv2.VideoCapture(camera_number)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# obtaining the width and height of the images calibrated to use them to resize the input
width, height = img_size

# Resizing the image input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# -----------streamlit page ------------------------

st.set_page_config(layout="wide")

with st.container():
    st.title("Disparity Parameters Tuning")

with st.container():
    col1, col2 = st.columns([1, 4])

    with col1:
        with st.container():
            st.markdown("## Parameters")
            st.markdown("""
            Adjust the parameters as needed.
            """)

            num_disp_factor = st.slider("num_disp_factor", min_value=1, max_value=50, value=20, step=1)
            min_dip = st.slider("min_disp", min_value=1, max_value=500, value=300, step=1)
            window_size = st.slider("window_size", min_value=5, max_value=255, value=5, step=1)
            disp12MaxDiff = st.slider("disp12MaxDiff", min_value=1, max_value=10, value=1, step=1)
            preFilterCap = st.slider("preFilterCap", min_value=0, max_value=10, value=0, step=1)
            uniquenessRatio = st.slider("uniquenessRatio", min_value=1, max_value=10, value=1, step=1)
            speckleWindowSize = st.slider("speckleWindowSize", min_value=0, max_value=10, value=0, step=1)
            speckleRange = st.slider("speckleRange", min_value=2, max_value=10, value=2, step=1)
            wls_lambda = st.slider("wls_lambda", min_value=0, max_value=10000, value=8000, step=100)
            wls_sigma = st.slider("wld_sigma", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

            map_type = st.selectbox("Map type", ["disp_map", "filtered_disp_map", "depth_map"])

            num_disp = 16 * num_disp_factor

            sgbm_params = {"minDisparity": min_disp,
                        "numDisparities": num_disp,
                        "blockSize": window_size,
                        "P1": 8*3*window_size**2,
                        "P2": 32*3*window_size**2,
                        "disp12MaxDiff": disp12MaxDiff,
                        "preFilterCap": preFilterCap,
                        "uniquenessRatio": uniquenessRatio,
                        "speckleWindowSize": speckleWindowSize,
                        "speckleRange": speckleRange,
                        "mode": 0
                        }

            if st.button("Save Params", type="primary"):
                param_storage = {"num_disp_factor": num_disp_factor,
                    "min_disp": min_dip,
                    "window_size": window_size,
                    "disp12MaxDiff": disp12MaxDiff,
                    "preFilterCap": preFilterCap,
                    "uniquenessRatio": uniquenessRatio,
                    "speckleWindowSize": speckleWindowSize,
                    "speckleRange": speckleRange,
                    "wls_lambda": wls_lambda,
                    "wld_sigma": wls_sigma,
                    "baseline": baseline,
                }
                if not os.path.exists("images"):
                    os.makedirs("images")

                params_storage_path = os.path.join("images", f"disparity_params_{int(time.time())}.yaml")

                with open(params_storage_path, 'w') as outfile:
                    yaml.dump(param_storage, outfile, default_flow_style=False)    

    with col2:
        with st.container():
            if stereo_retval:
                st.success("Stereo successfully detected.", icon="✅")
            if not stereo_retval:
                st.warning("May not be stereo. Change some parameter or reload the page.")

        with st.container():
            st.markdown("## Video")

            frame_placeholder = st.empty()

            # Add a "Stop" button and store its state in a variable
            stop_button_pressed = st.button("Stop")

            while cap.isOpened() and not stop_button_pressed:
                ret, frame = cap.read()

                if not ret:
                    st.write("The video capture has ended.")
                    break

                # You can process the frame here if needed
                # e.g., apply filters, transformations, or object detection

                left_frame, right_frame = split_stereo(frame)

                left_rect, right_rect = rectify_stereo_image(left_frame, right_frame, data)

                disp_map, filtered_disp_map, depth_map = disparity_depth_map(
                    left_img=left_rect,
                    right_img=right_rect,
                    baseline=baseline,
                    data=data,
                    algorithm="sgbm",
                    save_path=None,
                    wls_lambda=wls_lambda,
                    wls_sigma=wls_sigma,
                    **sgbm_params
                )

                # Converting the image format for streamlit visualisation
                disp_map = np.uint8(disp_map)
                filtered_disp_map = np.uint8(filtered_disp_map)

                # Display the frame using Streamlit's st.image

                if map_type == "disp_map":
                    frame_placeholder.image(disp_map, clamp=True)
                elif map_type == "filtered_disp_map":
                    frame_placeholder.image(filtered_disp_map, clamp=True)
                else:
                    frame_placeholder.image(depth_map, clamp=True)

cap.release()
cv2.destroyAllWindows()