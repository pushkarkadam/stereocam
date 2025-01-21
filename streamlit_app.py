import streamlit as st
import numpy as np
import cv2
from stereocam import *


# Data paths
DATA_PATH = 'images/case2/stereo.npy'
data = np.load(DATA_PATH, allow_pickle=True).item()

Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], data['R'], data['T'], data['img_size']


# PARAMS
baseline = 0.12 # 120 cm or 0.12 m
num_disp_factor = 22
window_size = 5
min_disp = 300


# Detect all the possible cameras
cam_available = detect_camera()

# Select stereo camera
camera_number = detect_stereo(cam_available)

# Creating a video capture object using the detected stereo camera
cap = cv2.VideoCapture(camera_number)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# obtaining the width and height of the images calibrated to use them to resize the input
width, height = img_size

# Resizing the image input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# Use this line to capture video from the webcam
# cap = cv2.VideoCapture(0)
# initialise_camera(cap)


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

            num_disp_factor = st.slider("num_disp_factor", 1, 50, 1)
            min_dip = st.slider("min_disp", 1, 500, 1)
            window_size = st.slider("window_size", 5, 255, 3)
            disp12MaxDiff = st.slider("disp12MaxDiff", 1, 10, 1)
            preFilterCap = st.slider("preFilterCap", 0, 10, 1)
            uniquenessRatio = st.slider("uniquenessRatio", 1, 10, 1)
            speckleWindowSize = st.slider("speckleWindowSize", 0, 10, 1)
            speckleRange = st.slider("speckleRange", 2, 10, 1)

            num_disp = 16 * num_disp_factor

            sgbm_params = {"minDisparity": min_disp,
                        "numDisparities": num_disp,
                        "blockSize": window_size,
                        "P1": 8*3*window_size**2,
                        "P2": 32*3*window_size**2,
                        "disp12MaxDiff": disp12MaxDiff,
                        "preFilterCap": 0,
                        "uniquenessRatio": 1,
                        "speckleWindowSize": 0,
                        "speckleRange": 2,
                        "mode": 0
                        }
    with col2:
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

                disp_map, depth_map = disparity_depth_map(
                    left_img=left_rect,
                    right_img=right_rect,
                    baseline=baseline,
                    data=data,
                    algorithm="sgbm",
                    save_path=None,
                    **sgbm_params
                )

                disp_map = np.uint8(disp_map)


                # Convert the frame from BGR to RGB format
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame using Streamlit's st.image
                frame_placeholder.image(disp_map)

                # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
                # if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
                #     break

cap.release()
cv2.destroyAllWindows()