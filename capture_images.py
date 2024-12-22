import cv2
import argparse
import datetime
import os


def capture_stereo(output_path="images", camera_number=None, width=4416, height=1242):
    """Captures stereo images for calibration"""
    

    # If the camera number index is not provided in the function then perform auto detection
    if not camera_number:
        # Detecting available cameras
        cam_available = detect_camera()

        # Selecting stereo camera
        camera_number = detect_stereo(cam_available)

    # date
    ct = datetime.datetime.now()
    date = ct.strftime("%d-%m-%Y-%H-%M")

    # Make directory for saving both left and right images
    path_left = os.path.join(output_path, date, 'stereo_left')
    path_right = os.path.join(output_path, date, 'stereo_right')

    print(f"Creating directories: \n{path_left}\n{path_right}")
    os.makedirs(path_left)
    os.makedirs(path_right)

    # create a video capture for stereo camera
    cap = cv2.VideoCapture(camera_number)

    # Creating resolution
    resolution = (int(width), int(height))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Performing a five interation count to ensure that the camera starts correctly
    # This is very important because in the first instance, the camera may not be ready
    # If this step is not performed, then a green output is expected in the image.
    count_till = 5

    print("Initialising camera...")
    for i in range(count_till):
        ret, frame = cap.read()
        print(f"Frame count: {i}")

        if not ret:
            print('\033[91m' + "Camera failed to start!")
            break

    num_images = 0

    while cap.isOpened():
        # Read frames from the stereo camera
        retval, frame = cap.read()

        # Checking if retval (return value) is false and if so, then it stops the code
        if not retval:
            print('\033[91m' + "Camera failed to start!")
            break

        # Split the frame into left and right images
        left_frame = frame[:, :width // 2, :]
        right_frame = frame[:, width // 2:, :]

        # Waiting for 5 milliseconds
        key = cv2.waitKey(5)

        # Check if ESC key is pressed whose ASCII value is 27
        if key == 27:
            break 
        elif key == ord('s'):
            cv2.imwrite(os.path.join(path_left, 'imageL_' + str(num_images) + '.png'), left_frame)
            cv2.imwrite(os.path.join(path_right, 'imageR_' + str(num_images) + '.png') , right_frame)
            print('\033[92m' + "Images saved!")
            num_images += 1

        cv2.imshow("Left stereo", left_frame)
        cv2.imshow("Right stereo", right_frame)


    # Release the video capture and video writers
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()


def detect_camera(max_cam=10):
    """Detects camera"""

    cam_available = []

    for cam in range(max_cam):
        try:
            cap = cv2.VideoCapture(cam)
        except Exception as e:
            print(e)
        
        ret, _ = cap.read()

        if ret:
            cam_available.append(cam)

        cap.release()
        cv2.destroyAllWindows()

    return cam_available

def detect_stereo(cam_available, stereo_res=(1242, 4416)):
    """Returns the index of stereo camera.

    Making certain assumptions about what stereo images consist.
    Assuming the stereo camera is connected using a USB and USB is the last camera to be detected.
    Assuming stereo camera has a certain resolution that is expected.
    
    """
    # Using the last value of the cam_available list because it has a potential to be stereo
    stereo_index = cam_available.pop()

    cap = cv2.VideoCapture(stereo_index)

    # Creating resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, stereo_res[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stereo_res[0])

    # Getting image
    ret, frame = cap.read()
    
    if frame.shape[:-1] == stereo_res:
        print('\033[92m' + 'Stereo possible')
    else:
        print('\033[93m' + 'May not be stereo. Please check manually and use correct index!')

    cap.release()
    cv2.destroyAllWindows()

    return stereo_index

def main():
    capture_stereo()

if __name__ == '__main__':
    main()