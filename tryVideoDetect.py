import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import argparse

def ORB_detector(new_image, image_template):
    orb = cv2.ORB_create(1000, 1.2)  # ORB detector of 1000 keypoints, scaling pyramid factor=1.2
    (kp1, des1) = orb.detectAndCompute(new_image, None)  # Detect keypoints on the new image
    (kp2, des2) = orb.detectAndCompute(image_template, None)  # Detect keypoints of the template image
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Matcher
    matches = bf.match(des1, des2)  # Extract matches
    matches = sorted(matches, key=lambda val: val.distance)  # Sort matches
    img2 = cv2.drawKeypoints(image_template, kp2, None, color=(0,255,0), flags=0)
    img1 = cv2.drawKeypoints(new_image, kp2, None, color=(0,255,0), flags=0)
    plt.imshow(img2)
    plt.show()
    return len(matches)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to the input video',
                    required=True)
parser.add_argument('-c', '--consecutive-frames', default=4, type=int,
                    dest='consecutive_frames', help='path to the input video')
args = vars(parser.parse_args())
frame_count = 0
consecutive_frame = args['consecutive_frames']
# Load video file and template image
cap = cv2.VideoCapture("HethenHisPelivideo.mp4")
image_template = cv2.imread('Pelivideo.png', 0)

# Initialize ROI coordinates
top_left_x, top_left_y, bottom_right_x, bottom_right_y = 0, 0, 0, 0

while (cap.isOpened()):
    # Get video frame
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        orig_frame = frame.copy()
        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []
        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray, background)
        # thresholding to convert the frame to binary
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        # append the final result into the `frame_diff_list`
        frame_diff_list.append(dilate_frame)
        # if we have reached `consecutive_frame` number of frames
        if len(frame_diff_list) == consecutive_frame:
            # add all the frames in the `frame_diff_list`
            sum_frames = sum(frame_diff_list)
            # find the contours around the white segmented areas
            contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours, not strictly necessary
            for i, cnt in enumerate(contours):
                cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
            for contour in contours:
                # continue through the loop if contour area is less than 500...
                # ... helps in removing noise detection
                if cv2.contourArea(contour) < 500:
                    continue
                # get the xmin, ymin, width, and height coordinates from the contours
                (x, y, w, h) = cv2.boundingRect(contour)
                # draw the bounding boxes
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            cv2.imshow('Detected Objects', orig_frame)
            out.write(orig_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break
cap.release()
cv2.destroyAllWindows()